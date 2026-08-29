import math

import einops
import torch
from jaxtyping import Float
from torch import Tensor

from .model_args import DeepSeekV3ModelArgs


def precompute_freqs_cis(args: DeepSeekV3ModelArgs) -> Float[Tensor, "S D"]:
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    freqs = 1.0 / (base ** (2 * torch.arange(0, dim // 2, dtype=torch.float32) / dim))

    # YaRN
    def find_correction_dim(
        num_rotations: float, dim: int, base: float, max_seq_len: int
    ) -> float:
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(
        low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int
    ) -> tuple[int, int]:
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp(min: float, max: float, dim: int) -> Float[Tensor, "S D"]:
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = einops.einsum(t, freqs, "a, b -> a b")
    return freqs


def apply_rotary_emb(x: Float[Tensor, "B S H D"], freqs: Float[Tensor, "S D//2"]) -> torch.Tensor:
    dtype = x.dtype
    cos = einops.repeat(torch.cos(freqs), "s d -> 1 s 1 (d 2)")
    sin = einops.repeat(torch.sin(freqs), "s d -> 1 s 1 (d 2)")
    x_pairs = einops.rearrange(x, "b s h (d two) -> b s h d two", two=2)
    x_rot = einops.rearrange(
        torch.stack([-x_pairs[..., 1], x_pairs[..., 0]], dim=-1),
        "b s h d two -> b s h (d two)",
    )
    rx = x * cos + x_rot * sin
    return rx.to(dtype)
