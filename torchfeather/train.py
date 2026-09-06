from __future__ import annotations

import contextlib
import os
import signal
import time
from collections.abc import Callable, Iterator
from datetime import timedelta
from typing import cast

import torch
from jaxtyping import Float, Int
from loguru import logger
from torch import Tensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.elastic.multiprocessing.errors import record

from torchfeather.components.checkpoint import CheckpointManager
from torchfeather.components.dataloader import BaseDataLoader, DataloaderExhaustedError
from torchfeather.components.loss import (
    IGNORE_INDEX,
    LossFunction,
    build_cross_entropy_loss,
)
from torchfeather.components.lr_scheduler import (
    LRSchedulersContainer,
    build_lr_schedulers,
)
from torchfeather.components.metrics import (
    MetricsProcessor,
    collect_parameter_norm_metrics,
)
from torchfeather.components.optimizer import (
    OptimizersContainer,
    build_optimizers_with_moe_load_balancing,
)
from torchfeather.components.tokenizer import (
    DeepSeekV3Tokenizer,
)
from torchfeather.config import TORCH_DTYPE_MAP, JobConfig
from torchfeather.config.default_configs import (
    get_config,
)
from torchfeather.config.job_config import Parallelism
from torchfeather.datasets.hf_datasets import build_hf_dataloader
from torchfeather.distributed import ParallelDims
from torchfeather.distributed import utils as dist_utils
from torchfeather.distributed.pipeline_parallel import pipeline_llm
from torchfeather.model.model import DeepSeekV3Model
from torchfeather.model.parallelize import parallelize_deepseekv3
from torchfeather.tools import device_utils, utils
from torchfeather.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)


class Trainer(Stateful):
    job_config: JobConfig
    parallel_dims: ParallelDims

    tokenizer: DeepSeekV3Tokenizer
    dataloader: BaseDataLoader
    model_parts: list[torch.nn.Module]
    loss_fn: LossFunction
    optimizers: OptimizersContainer
    lr_schedulers: LRSchedulersContainer
    metrics_processor: MetricsProcessor
    checkpointer: CheckpointManager

    device: torch.device
    gc_handler: utils.GarbageCollection
    train_context: Callable[..., contextlib.AbstractContextManager]
    maybe_enable_amp: contextlib.AbstractContextManager
    gradient_accumulation_steps: int
    pp_has_first_stage: bool
    pp_has_last_stage: bool

    step: int
    ntokens_seen: int

    @record
    def __init__(self, job_config: JobConfig):
        self.job_config = job_config

        device_module, device_type = device_utils.device_module, device_utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        device_module.set_device(self.device)

        torch.distributed.init_process_group(
            backend="nccl", timeout=timedelta(seconds=job_config.comm.init_timeout_seconds)
        )
        world_size = int(os.environ["WORLD_SIZE"])
        parallelism_config = job_config.parallelism
        self.parallel_dims = parallel_dims = self._create_parallel_dims(
            parallelism_config, world_size
        )

        _ = parallel_dims.world_mesh
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            dp_degree, dp_rank = batch_mesh.size(), batch_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0

        self.gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)
        dist_utils.set_determinism(
            parallel_dims, self.device, job_config.training.seed, job_config.training.deterministic
        )

        self.tokenizer = DeepSeekV3Tokenizer(job_config.model.hf_assets_path)
        self.dataloader = build_hf_dataloader(
            dp_world_size=dp_degree,
            dp_rank=dp_rank,
            tokenizer=self.tokenizer,
            job_config=job_config,
        )

        model_args = job_config.model.args
        model_args.max_seq_len = job_config.training.seq_len
        with (
            torch.device("meta"),
            device_utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]),
        ):
            model = DeepSeekV3Model(model_args)
        self.metrics_processor = MetricsProcessor(job_config, parallel_dims)

        model_param_count, self.metrics_processor.num_flops_per_token = (
            model_args.get_nparams_and_flops(model, job_config.training.seq_len)
        )
        logger.info(f"Model total parameters {model_param_count:,}")

        self.loss_fn = build_cross_entropy_loss(job_config)

        global_batch_size = job_config.training.global_batch_size
        if global_batch_size < 0:
            global_batch_size = job_config.training.local_batch_size * dp_degree
        assert global_batch_size > 0, global_batch_size
        assert global_batch_size % (job_config.training.local_batch_size * dp_degree) == 0, (
            global_batch_size,
            job_config.training.local_batch_size * dp_degree,
        )

        self.gradient_accumulation_steps = global_batch_size // (
            job_config.training.local_batch_size * dp_degree
        )
        assert self.gradient_accumulation_steps > 0, self.gradient_accumulation_steps

        init_device = device_type
        buffer_device = None
        if parallel_dims.pp_enabled:
            self.pp_schedule, self.model_parts, self.pp_has_first_stage, self.pp_has_last_stage = (
                pipeline_llm(
                    model,
                    parallel_dims,
                    job_config,
                    self.device,
                    model_args.n_layers,
                    parallelize_deepseekv3,
                    self.loss_fn,
                )
            )
            del model

            for m in self.model_parts:
                m.to_empty(device=init_device)
                with torch.no_grad():
                    m.init_weights(buffer_device=buffer_device)
                m.train()
        else:
            model = parallelize_deepseekv3(model, parallel_dims, job_config)
            model.to_empty(device=init_device)
            with torch.no_grad():
                model.init_weights(buffer_device=buffer_device)
            model.train()
            self.model_parts = [model]

        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPs used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{device_type.upper()} memory usage for model: {device_mem_stats.max_reserved_gib:.2f}GiB({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        self.optimizers = build_optimizers_with_moe_load_balancing(
            self.model_parts, job_config.optimizer, parallel_dims
        )
        self.lr_schedulers = build_lr_schedulers(
            self.optimizers, job_config.lr_scheduler, job_config.training.steps
        )
        self.metrics_processor.optimizers = self.optimizers
        self.metrics_processor.model_parts = self.model_parts

        self.step = 0
        self.ntokens_seen = 0

        self.checkpointer = CheckpointManager(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            additional_states={"train_state": self},
            checkpoint_config=job_config.checkpoint,
            base_folder=job_config.job.dump_folder,
        )

        loss_parallel_enabled = (
            parallel_dims.tp_enabled and not parallelism_config.disable_loss_parallel
        )
        self.train_context = dist_utils.get_train_context(
            loss_parallel_enabled, parallelism_config.enable_compiled_autograd
        )
        self.maybe_enable_amp = dist_utils.maybe_enable_amp(
            parallel_dims, job_config.training.mixed_precision_param, device_type
        )

        logger.info(
            f"Trainer is initialized with local batch size {job_config.training.local_batch_size}, global batch size {global_batch_size}, gradient_accumulation_steps: {self.gradient_accumulation_steps}, sequence length {job_config.training.seq_len}, total steps {job_config.training.steps} (warmup {job_config.lr_scheduler.warmup_steps})"
        )

    def _create_parallel_dims(
        self, parallelism_config: Parallelism, world_size: int
    ) -> ParallelDims:
        return ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            ep=parallelism_config.expert_parallel_degree,
            etp=parallelism_config.expert_tensor_parallel_degree,
            world_size=world_size,
        )

    def forward_backward_step(
        self,
        input_dict: dict[str, Int[Tensor, "B S"]],
        labels: Int[Tensor, "B S"],
        global_valid_tokens: float | Float[Tensor, 1],
    ) -> Float[Tensor, 1]:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        inputs = input_dict["input"]
        extra_inputs = {k: v for k, v in input_dict.items() if k != "input"}
        extra_kwargs = {}

        optional_context_parallel_ctx = (
            dist_utils.create_context_parallel_ctx(
                cp_mesh=parallel_dims.get_mesh("cp"),
                cp_buffers=cast(
                    list[Tensor], [inputs, labels] + [m.freqs_cis for m in model_parts]
                ),
                cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                cp_no_restore_buffers={inputs, labels},
                cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
                cp_load_balance=self.job_config.parallelism.context_parallel_load_balance,
            )
            if parallel_dims.cp_enabled
            else None
        )

        if parallel_dims.pp_enabled:
            with self.train_context(optional_context_parallel_ctx):
                targets, losses = (labels, []) if self.pp_has_last_stage else (None, None)
                if self.pp_has_first_stage:
                    self.pp_schedule.step(
                        inputs,
                        **extra_inputs,
                        **extra_kwargs,
                        target=targets,
                        losses=losses,
                        return_outputs=False,
                    )
                else:
                    self.pp_schedule.step(
                        **extra_kwargs, target=targets, losses=losses, return_outputs=False
                    )

            if self.pp_has_last_stage:
                loss = (torch.sum(torch.stack(losses)) / global_valid_tokens).to(self.device)
            else:
                loss = torch.tensor([-1.0], device=self.device)
        else:
            with self.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.maybe_enable_amp:
                    pred = model_parts[0](inputs, **extra_inputs, **extra_kwargs)
                    loss_sum = self.loss_fn(pred, labels)
                    loss = loss_sum / global_valid_tokens
                del pred
                loss.backward()

        return loss

    def train_step(
        self, data_iterator: Iterator[tuple[dict[str, Int[Tensor, "B S"]], Int[Tensor, "B S"]]]
    ):
        self.optimizers.zero_grad()
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]
        parallel_dims = self.parallel_dims

        microbatches = []
        local_valid_tokens = torch.tensor(0, dtype=torch.int64, device=self.device)
        for _microbatch in range(self.gradient_accumulation_steps):
            input_dict, labels = next(data_iterator)
            local_valid_tokens += (labels != IGNORE_INDEX).sum()
            microbatches.append((input_dict, labels))
        local_valid_tokens //= self.parallel_dims.cp

        if parallel_dims.dp_cp_enabled:
            global_valid_tokens = dist_utils.dist_sum(
                local_valid_tokens, parallel_dims.get_mesh("loss")
            )
        else:
            global_valid_tokens = local_valid_tokens.float()

        accumulated_losses = []
        for input_dict, labels in microbatches:
            loss = self.forward_backward_step(input_dict, labels, global_valid_tokens)
            accumulated_losses.append(loss.detach())

        if parallel_dims.pp_enabled:
            for m in self.model_parts:
                for p in m.parameters():
                    if p.grad is not None:
                        p.grad.div_(global_valid_tokens)

        should_log = self.metrics_processor.should_log(self.step)
        parameter_metrics = (
            collect_parameter_norm_metrics(
                self.model_parts, pp_mesh=self.parallel_dims.get_optional_mesh("pp")
            )
            if should_log
            else {}
        )
        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
            foreach=True,
            pp_mesh=(parallel_dims.get_optional_mesh("pp")),
            ep_enabled=parallel_dims.ep_enabled,
        )

        self.optimizers.step()
        self.lr_schedulers.step()
        loss = torch.sum(torch.stack(accumulated_losses))

        if not should_log:
            return

        if parallel_dims.dp_cp_enabled:
            loss = loss.detach()
            dp_cp_mesh = parallel_dims.get_mesh("loss")
            global_avg_loss = dist_utils.dist_sum(loss, dp_cp_mesh)
            local_avg_loss = loss * global_valid_tokens / local_valid_tokens
            global_max_loss = dist_utils.dist_max(local_avg_loss, dp_cp_mesh)
            global_ntokens_seen = dist_utils.dist_sum(
                torch.tensor(self.ntokens_seen, dtype=torch.int64, device=self.device), dp_cp_mesh
            )
        else:
            global_avg_loss = global_max_loss = loss.detach().item()
            global_ntokens_seen = self.ntokens_seen

        extra_metrics = {"n_tokens_seen": global_ntokens_seen, "lr": lr}
        extra_metrics.update(parameter_metrics)
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            grad_norm.item(),
            extra_metrics=extra_metrics,
        )

    @record
    def train(self):
        job_config = self.job_config

        self.checkpointer.load(step=job_config.checkpoint.load_step)
        logger.info(f"Training starts at step {self.step + 1}")

        with (
            maybe_enable_profiling(
                job_config.profiling,
                global_step=self.step,
                base_folder=job_config.job.dump_folder,
                leaf_folder="",
            ) as torch_profiler,
            maybe_enable_memory_snapshot(
                job_config.profiling,
                global_step=self.step,
                base_folder=job_config.job.dump_folder,
                leaf_folder="",
            ) as memory_profiler,
        ):
            data_iterator = self.batch_generator(self.dataloader)
            while self.should_continue_training():
                self.step += 1
                self.gc_handler.run(self.step)
                try:
                    self.train_step(data_iterator)
                except DataloaderExhaustedError:
                    logger.warning("Ran out of data; last step was canceled.")
                    break

                self.checkpointer.save(
                    self.step, last_step=(self.step == job_config.training.steps)
                )
                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

                if self.step == 1:
                    dist_utils.set_pg_timeouts(
                        timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                        parallel_dims=self.parallel_dims,
                    )

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)
            logger.info("Training completed")


def _any_successful_shutdown_watchdog(timeout_seconds: int = 30):
    logger.info(f"Arming post-training shutdown watchdog (SIGALRM) for {timeout_seconds} seconds")
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(timeout_seconds)


def _shutdown_after_successful_training(trainer: Trainer):
    _any_successful_shutdown_watchdog()
    trainer.close()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    CONFIG_NAME = os.environ.get("TORCHFEATHER_CONFIG", None)
    if CONFIG_NAME is None:
        raise ValueError("TORCHFEATHER_CONFIG environment variable is not set")

    trainer = None
    try:
        config = get_config(CONFIG_NAME)
        config.job.dump_folder = f"./outputs/{CONFIG_NAME}"
        trainer = Trainer(config)
        trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        _shutdown_after_successful_training(trainer)
