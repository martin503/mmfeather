import copy
import math
from collections.abc import Callable

import torch
from loguru import logger
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    ScheduleZBVZeroBubble,
    _PipelineSchedule,
    get_schedule_class,
)

from torchfeather.components.loss import LossFunction
from torchfeather.config import JobConfig
from torchfeather.distributed import ParallelDims

__all__ = [
    "build_pipeline_schedule",
    "generate_llm_fqn_per_model_part",
    "pipeline_llm",
    "pipeline_module_split",
]


def pipeline_llm(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: torch.device,
    num_layers: int,
    parallelize_fn: Callable[[nn.Module, ParallelDims, JobConfig], nn.Module],
    loss_fn: LossFunction,
) -> tuple[_PipelineSchedule, list[nn.Module], bool, bool]:
    pp_mesh = parallel_dims.get_mesh("pp")
    schedule_class = get_schedule_class(job_config.parallelism.pipeline_parallel_schedule)

    is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)
    layers_per_stage = job_config.parallelism.pipeline_parallel_layers_per_stage

    input_weight = job_config.parallelism.pipeline_parallel_first_stage_less_layers
    output_weight = job_config.parallelism.pipeline_parallel_last_stage_less_layers

    if layers_per_stage is not None:
        num_virtual_stages = math.ceil(
            (num_layers + input_weight + output_weight) / layers_per_stage
        )
        assert num_virtual_stages % parallel_dims.pp == 0, (num_virtual_stages, parallel_dims.pp)

        stages_per_rank = num_virtual_stages // parallel_dims.pp
        assert not is_single_stage_schedule or stages_per_rank == 1, (
            is_single_stage_schedule,
            stages_per_rank,
        )
        assert is_single_stage_schedule or stages_per_rank >= 2, (
            is_single_stage_schedule,
            stages_per_rank,
        )
    else:
        stages_per_rank = 1 if is_single_stage_schedule else 2
        num_virtual_stages = parallel_dims.pp * stages_per_rank

    module_names_per_stage = job_config.parallelism.module_fqns_per_model_part
    if module_names_per_stage is None:
        module_names_per_stage = generate_llm_fqn_per_model_part(
            num_virtual_stages, num_layers, input_weight, output_weight
        )
    for i, stage_ms in enumerate(module_names_per_stage):
        logger.debug(f"Stage {i}: {stage_ms}")

    stages, model_parts = pipeline_module_split(
        model,
        pp_mesh,
        job_config.parallelism.pipeline_parallel_schedule,
        device,
        module_names_per_stage,
    )
    for i, m in enumerate(model_parts):
        m = parallelize_fn(m, parallel_dims, job_config)
        model_parts[i] = m
        stages[i].submod = m
    pp_schedule = build_pipeline_schedule(job_config, stages, loss_fn)

    has_first_stage = False
    has_last_stage = False
    for stage in stages:
        if stage.is_first:
            has_first_stage = True
        if stage.is_last:
            has_last_stage = True

    return pp_schedule, model_parts, has_first_stage, has_last_stage


def generate_llm_fqn_per_model_part(
    num_stages: int, num_layers: int, input_weight: int = 1, output_weight: int = 1
) -> list[list[str]]:
    assert num_stages >= 1, num_stages

    if num_stages == 1:
        layer_names = [f"layers.{i}" for i in range(num_layers)]
        return [["tok_embeddings"] + layer_names + ["norm", "output"]]

    num_effective_layers = num_layers + input_weight + output_weight
    assert num_stages <= num_effective_layers, (num_stages, num_effective_layers)

    layers_per_stage = num_effective_layers // num_stages
    extra_layers = num_effective_layers % num_stages

    assert layers_per_stage > 0, (layers_per_stage, num_stages, num_effective_layers)
    assert input_weight <= layers_per_stage, (input_weight, layers_per_stage)
    assert output_weight <= layers_per_stage, (output_weight, layers_per_stage)

    module_names_per_stage = []
    current_layer = 0

    for stage_idx in range(num_stages):
        effective_layers_for_stage = layers_per_stage + 1 if stage_idx < extra_layers else 0
        num_transformer_layers = effective_layers_for_stage
        if stage_idx == 0:
            num_transformer_layers -= input_weight
        if stage_idx == num_stages - 1:
            num_transformer_layers -= output_weight

        stage_modules = []
        if stage_idx == 0:
            stage_modules.append("tok_embeddings")
        for _ in range(num_transformer_layers):
            if current_layer < num_layers:
                stage_modules.append(f"layers.{current_layer}")
                current_layer += 1

        if stage_idx == num_stages - 1:
            stage_modules.extend(["norm", "output"])
        module_names_per_stage.append(stage_modules)

    return module_names_per_stage


def pipeline_module_split(
    whole_model: nn.Module,
    pp_mesh: DeviceMesh,
    pp_schedule: str,
    device: torch.device,
    module_names_per_stage: list[list[str]],
) -> tuple[list[PipelineStage], list[nn.Module]]:
    pp_rank = pp_mesh.get_local_rank()
    pp_degree = pp_mesh.size()
    num_stages = len(module_names_per_stage)
    stages = []
    models = []

    schedule_class = get_schedule_class(pp_schedule)
    style = "v" if schedule_class is ScheduleZBVZeroBubble else "loop"
    assert num_stages % pp_degree == 0, (num_stages, pp_degree)
    stages_per_rank = num_stages // pp_degree
    if style == "v":
        assert stages_per_rank == 2, stages_per_rank
        stage_v_pairs = list(zip(range(pp_degree), range(num_stages - 1, pp_degree - 1, -1)))
        stage_indices = stage_v_pairs[pp_rank]
    elif style == "loop":
        stage_indices = tuple(pp_rank + s * pp_degree for s in range(stages_per_rank))
    else:
        raise ValueError(f"Unknown schedule style {style}")

    for stage_idx in stage_indices:
        module_names = module_names_per_stage[stage_idx]
        stage, model_chunk = _build_stage_from_modules(
            whole_model, pp_mesh, stage_idx, module_names, num_stages, device
        )
        logger.info(
            f"PP rank {pp_rank} is building stage_idx {stage_idx} with modules {module_names}"
        )
        stages.append(stage)
        models.append(model_chunk)

    return stages, models


def _build_stage_from_modules(
    whole_model: nn.Module,
    pp_mesh: DeviceMesh,
    stage_idx: int,
    module_names: list[str],
    num_stages: int,
    device: torch.device,
) -> tuple[PipelineStage, nn.Module]:
    assert all(x.count(".") <= 1 for x in module_names), (
        f"PP splits only immediate children and module_names seems to have deeper nested entries:\n{module_names}"
    )
    model = copy.deepcopy(whole_model)

    modules_to_keep = set(module_names)
    for module_name, module_value in model.named_children():
        if isinstance(module_value, (nn.ModuleDict, nn.ModuleList)):
            layers_to_keep = {
                name.split(".", 1)[1]
                for name in modules_to_keep
                if name.startswith(f"{module_name}.")
            }
            if layers_to_keep:
                if isinstance(module_value, nn.ModuleDict):
                    for layer_name in list(module_value.keys()):
                        if layer_name not in layers_to_keep:
                            del module_value[layer_name]
                elif isinstance(module_value, nn.ModuleList):
                    indices_to_keep = {int(idx) for idx in layers_to_keep if idx.isdigit()}
                    new_layers = nn.ModuleList(
                        [layer for i, layer in enumerate(module_value) if i in indices_to_keep]
                    )
                    setattr(model, module_name, new_layers)
            else:
                if isinstance(module_value, nn.ModuleDict):
                    setattr(model, module_name, nn.ModuleDict())
                elif isinstance(module_value, nn.ModuleList):
                    setattr(model, module_name, nn.ModuleList())
        elif module_name not in modules_to_keep:
            setattr(model, module_name, None)

    stage = PipelineStage(model, stage_idx, num_stages, device, group=pp_mesh.get_group("pp"))
    return stage, model


def build_pipeline_schedule(
    job_config: JobConfig, stages: list[PipelineStage], loss_fn: Callable
) -> _PipelineSchedule:
    schedule_class = get_schedule_class(job_config.parallelism.pipeline_parallel_schedule)
    looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)
    microbatch_size = job_config.parallelism.pipeline_parallel_microbatch_size
    batch_size = job_config.training.local_batch_size
    assert batch_size % microbatch_size == 0, (batch_size, microbatch_size)
    n_microbatches = batch_size // microbatch_size

    num_total_stages = job_config.parallelism.pipeline_parallel_degree * len(stages)
    if n_microbatches < num_total_stages:
        logger.warning(
            f"Number of microbatches ({n_microbatches}) is less than the total number of stages ({num_total_stages}) which may result in bubble in the pipeline."
        )

    if looped_schedule:
        schedule = schedule_class(
            stages, n_microbatches=n_microbatches, loss_fn=loss_fn, scale_grads=False
        )
    else:
        schedule = schedule_class(
            stages[0], n_microbatches=n_microbatches, loss_fn=loss_fn, scale_grads=False
        )
    logger.info(
        f"Using pipeline schedule {job_config.parallelism.pipeline_parallel_schedule} with {n_microbatches} microbatches and {num_total_stages} stages."
    )

    return schedule
