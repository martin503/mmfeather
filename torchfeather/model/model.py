import math

import torch
from einops import rearrange, repeat, unpack
from jaxtyping import Float, Int
from torch import Tensor, nn
from torchfeather.model.moe import FeedForward, MoE

from torchfeather.model.model_args import DeepSeekV3ModelArgs
from torchfeather.model.rope import apply_rotary_emb, precompute_freqs_cis


class Attention(nn.Module):
    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__()

        self.dim = model_args.dim
        self.n_heads = model_args.n_heads
        self.q_lora_rank = model_args.q_lora_rank
        self.kv_lora_rank = model_args.kv_lora_rank
        self.qk_nope_head_dim = model_args.qk_nope_head_dim
        self.qk_rope_head_dim = model_args.qk_rope_head_dim
        self.qk_head_dim = model_args.qk_nope_head_dim + model_args.qk_rope_head_dim
        self.v_head_dim = model_args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=model_args.norm_eps)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=model_args.norm_eps)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)
        self.softmax_scale = self.qk_head_dim**-0.5

        if model_args.max_seq_len > model_args.original_seq_len:
            mscale = 0.1 * model_args.mscale * math.log(model_args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale**2

        self.inner_attention = ScaledDotProductAttentionWrapper()

    def forward(
        self, x: Float[Tensor, "B S D"], freqs_cis: Float[Tensor, "S H"]
    ) -> Float[Tensor, "B S D"]:
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_a(x)
            q = self.wq_b(self.q_norm(q))

        q = rearrange(q, "b s (h d) -> b s h d", h=self.n_heads, d=self.qk_head_dim)
        q_nope, q_pe = unpack(q, [[self.qk_nope_head_dim], [self.qk_rope_head_dim]], "b s h *")
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        q = torch.cat([q_nope, q_pe], dim=-1)

        kv = self.wkv_a(x)
        kv, k_pe = unpack(kv, [[self.kv_lora_rank], [self.qk_rope_head_dim]], "b s *")
        k_pe = apply_rotary_emb(rearrange(k_pe, "b s rh -> b s 1 rh"), freqs_cis)
        kv = self.wkv_b(self.kv_norm(kv))
        kv = rearrange(kv, "b s (h kv) -> b s h kv", h=self.n_heads)
        k_nope, v = unpack(kv, [[self.qk_nope_head_dim], [self.v_head_dim]], "b s h *")
        k = torch.cat(
            [k_nope, repeat(k_pe, "b s 1 rh -> b s repeat rh", repeat=self.n_heads)], dim=-1
        )

        q = rearrange(q, "b s h d -> b h s d")
        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")

        output = self.inner_attention(q, k, v, scale=self.softmax_scale)
        output = rearrange(output, "b h s d -> b s (h d)")
        return self.wo(output)

    @torch.no_grad()
    def absorb_mla_weights(self) -> None:
        if self.q_lora_rank != 0:
            raise NotImplementedError()
        n_heads = self.n_heads
        dim = self.dim
        qk_nope_head_dim = self.qk_nope_head_dim
        qk_rope_head_dim = self.qk_rope_head_dim
        v_head_dim = self.v_head_dim
        kv_lora_rank = self.kv_lora_rank

        device = self.wq.weight.device
        dtype = self.wq.weight.dtype

        wq = rearrange(self.wq.weight, "(nh hd) d -> nh hd d", nh=n_heads)
        wq_nope, wq_rope = unpack(wq, [[qk_nope_head_dim], [qk_rope_head_dim]], "nh * d")
        wkv_b = rearrange(self.wkv_b.weight, "(nh hd) l -> nh hd l", nh=n_heads)
        w_uk, w_uv = unpack(wkv_b, [[qk_nope_head_dim], [v_head_dim]], "nh * l")

        wq_abs_nope = torch.bmm(rearrange(w_uk.float(), "nh l d -> nh d l"), wq_nope.float()).to(
            dtype=dtype
        )
        wq_abs = rearrange(torch.cat([wq_abs_nope, wq_rope], dim=1), "nh hd d -> (nh hd) d")
        self.wq_abs = nn.Linear(
            dim,
            n_heads * (kv_lora_rank + qk_rope_head_dim),
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.wq_abs.weight.copy_(wq_abs)
        self.wq_abs.requires_grad_(False)

        w_o = rearrange(self.wo.weight, "d (nh v) -> nh d v", nh=n_heads)
        w_o_abs_per_head = torch.bmm(w_o.float(), w_uv.float()).to(dtype=dtype)
        w_o_abs = rearrange(w_o_abs_per_head, "nh d l -> d (nh l)")
        self.wo_abs = nn.Linear(
            n_heads * kv_lora_rank, dim, bias=False, device=device, dtype=dtype
        )
        self.wo_abs.weight.copy_(w_o_abs)
        self.wo_abs.requires_grad_(False)

    def forward_absorbed(
        self, x: Float[Tensor, "B S D"], freqs_cis: Float[Tensor, "S H"]
    ) -> Float[Tensor, "B S D"]:
        assert self.wq_abs is not None
        assert self.wo_abs is not None

        q = self.wq_abs(x)
        q = rearrange(q, "b s (nh a) -> b s nh a", nh=self.n_heads)
        q_nope, q_rope = unpack(q, [[self.kv_lora_rank], [self.qk_rope_head_dim]], "b s nh *")
        q_rope = apply_rotary_emb(q_rope, freqs_cis)
        q = rearrange(torch.cat([q_nope, q_rope], dim=-1), "b s nh a -> b nh s a")

        latent_raw, k_rope = torch.split(
            self.wkv_a(x), [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        latent = self.kv_norm(latent_raw)
        k_rope = apply_rotary_emb(rearrange(k_rope, "b s rh -> b s 1 rh"), freqs_cis)
        shared_cache = rearrange(
            torch.cat([rearrange(latent, "b s l -> b s 1 l"), k_rope], dim=-1),
            "b s 1 z -> b 1 s z",
        )

        k = shared_cache
        v = shared_cache[..., : self.kv_lora_rank]
        latent_output = rearrange(
            self.inner_attention(q, k, v, scale=self.softmax_scale), "b nh s l -> b s (nh l)"
        )

        return self.wo_abs(latent_output)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, model_args: DeepSeekV3ModelArgs):
        super().__init__()
        self.attention = Attention(model_args)
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        self.moe_enabled = layer_id >= model_args.n_dense_layers
        if self.moe_enabled:
            self.moe = MoE(
                model_args.moe_args, dim=model_args.dim, hidden_dim=model_args.moe_inter_dim
            )
        else:
            self.feed_forward = FeedForward(model_args.dim, model_args.inter_dim)

        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        self.layer_id = layer_id

    def forward(
        self, x: Float[Tensor, "B S D"], freqs_cis: Float[Tensor, "S D"]
    ) -> Float[Tensor, "B S D"]:
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def init_weights(
        self, init_std: float | None = None, buffer_device: torch.device | None = None
    ):
        if buffer_device is None:
            raise ValueError(
                "buffer_device must be provided for TransformerBlock weight initialization"
            )
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        if self.moe_enabled:
            self.moe.init_weights(init_std=self.weight_init_std, buffer_device=buffer_device)
        else:
            self.feed_forward.init_weights(self.weight_init_std)


class DeepSeekV3Model(nn.Module):
    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer("freqs_cis", precompute_freqs_cis(model_args), persistent=False)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(
            model_args.dim, model_args.vocab_size, dtype=torch.get_default_dtype(), bias=False
        )

    def init_weights(
        self, init_std: float | None = None, buffer_device: torch.device | None = None
    ):
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = precompute_freqs_cis(self.model_args)
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(init_std=init_std, buffer_device=buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def forward(self, tokens: Int[Tensor, "B S"]) -> Float[Tensor, "B S V"]:
        h = self.tok_embeddings(tokens)
        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)
        output = self.output(self.norm(h))
        return output
