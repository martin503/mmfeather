from collections.abc import Callable
from functools import partial
from typing import Any

import huggingface_hub
import torch
from datasets import Dataset, DownloadConfig, load_dataset
from datasets.distributed import split_dataset_by_node
from loguru import logger
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchfeather.components.dataloader import ParallelAwareDataloader
from torchfeather.components.tokenizer import DeepSeekV3Tokenizer
from torchfeather.config import JobConfig
from torchfeather.datasets import DatasetConfig
from torchfeather.utils import PathType


def _load_fineweb_dataset(dataset_path: PathType, split: str):
    huggingface_hub.constants.DEFAULT_REQUEST_TIMEOUT = 300
    huggingface_hub.constants.DEFAULT_DOWNLOAD_TIMEOUT = 300
    return load_dataset(
        dataset_path,
        name="default",
        split=split,
        streaming=True,
        download_config=DownloadConfig(max_retries=40),
    )


def _process_pretrain_record(sample: dict[str, Any]) -> str:
    return sample["text"]


DATASETS = {
    "fineweb": DatasetConfig(
        path="HuggingFaceFW/fineweb",
        loader=partial(_load_fineweb_dataset, split="train"),
        sample_processor=_process_pretrain_record,
    )
}


def _validate_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor


class HuggingFaceDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: PathType | None,
        tokenizer: DeepSeekV3Tokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        dataset_name = dataset_name.lower()
        path, dataset_loader, text_processor = _validate_dataset(dataset_name, dataset_path)
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._text_processor = text_processor
        self._sample_idx: int = 0
        self._token_buffer: list[int] = []

    def _get_data_iter(self):
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))
        return iter(self._data)

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                sample_text = self._text_processor(sample)
                sample_tokens = self._tokenizer.encode(sample_text, add_bos=True, add_eos=True)
                self._token_buffer.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield {"input": input}, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                if (
                    not isinstance(self._data, Dataset)
                    and hasattr(self._data, "set_epoch")
                    and hasattr(self._data, "epoch")
                ):
                    self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict: dict[str, Any]):
        self._token_buffer = state_dict["token_buffer"]
        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict: dict[str, Any] = {"token_buffer": self._token_buffer}
        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            _state_dict["data"] = self._data.state_dict()
        return _state_dict


def build_hf_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: DeepSeekV3Tokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len

    hf_ds = HuggingFaceDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )
    return ParallelAwareDataloader(
        dataset=hf_ds, dp_rank=dp_rank, dp_world_size=dp_world_size, batch_size=batch_size
    )
