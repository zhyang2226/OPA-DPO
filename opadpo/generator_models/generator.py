import re
import abc
import copy
import dataclasses
import gc
import json
import logging
import math
import os
import shutil
from pathlib import Path
import random
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import pandas as pd
import einops
import tqdm
from loguru import logger as lg

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import accelerate

import transformers
from transformers.trainer_utils import enable_full_determinism, set_seed

from utils.data_utils_online_gpt4v import QueryResponseDataset
import utils.common_utils as utils
from opadpo.generator_models.prompt.call_openai_API import get_api_service

logger = logging.getLogger(__name__)

if torch.__version__ < "2.0.0":
    LRScheduler = torch.optim.lr_scheduler._LRScheduler  # noqa
else:
    LRScheduler = torch.optim.lr_scheduler.LRScheduler

FIRST_STEP_IDX = 1

class AlpacaAccelerator(accelerate.Accelerator):
    """Thin wrapper for accelerate.Accelerator."""

    def __repr__(self):
        return (
            f"Accelerator(\n"
            f"  state={self.state}, \n"
            f"  gradient_accumulation_steps={self.gradient_accumulation_steps:.6f}, \n"
            f"  split_batches={self.split_batches}, \n"
            f"  step_scheduler_with_optimizer={self.step_scheduler_with_optimizer},\n"
            f")"
        )

    def unwrap_optimizer(self, optimizer: accelerate.accelerator.AcceleratedOptimizer):
        return optimizer.optimizer


class Generator(object):
    def __init__(
        self,
        args,
        train_dataset: QueryResponseDataset,
        eval_dataset: QueryResponseDataset,
        data_collator: Callable,
        tokenizer: transformers.PreTrainedTokenizer,
        policy: nn.Module,
        accelerator: AlpacaAccelerator,
    ):
        super(Generator, self).__init__()
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.policy = policy
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.log_history = []
        self.args.set_truncate_token_ids(self.tokenizer)

        enable_full_determinism(
            self.args.seed
        ) if self.args.full_determinism else set_seed(self.args.seed)
        self.max_step = args.max_step

        global_rank = int(os.environ.get("RANK", 0))
        api_total_num = int(os.getenv("API_NUM", 1))
        api_index = global_rank % api_total_num
        API_key = os.getenv(f"API_KEY{api_index + 1}")
        AZURE_point = os.getenv(f"AZURE_POINT{api_index + 1}")
        self.api_model = os.getenv(f"API_MODEL{api_index + 1}")
        self.openai_API = get_api_service(type='azure', key=API_key, max_retries=30, use_cache=True, azure_endpoint=AZURE_point)

    @abc.abstractmethod
    @torch.inference_mode()
    def rollout(self, queries_data) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def record_step_stats(self, step_idx, response_dict):
        if self.args.output_dir is not None:
            rollouts_to_disk = pd.DataFrame(response_dict).to_dict(
                orient="records"
            )
            rollout_log_dir = os.path.join(self.args.output_dir, "rollouts")
            os.makedirs(rollout_log_dir, exist_ok=True)
            global_rank = int(os.environ.get("RANK", 0))
            with open(
                os.path.join(rollout_log_dir, f"step{step_idx}_rank{global_rank}.json"),
                "w",
            ) as f:
                json.dump(rollouts_to_disk, f, indent=4)
        return None

    def step(self, train_dataloader, step_idx: int):
        queries_batches = [
            next(train_dataloader) for _ in range(self.args.rollout_accumulation_steps)
        ]
        response_dict = self.rollout(queries_batches, step_idx)
        torch.cuda.empty_cache()
        stats = self.record_step_stats(
            step_idx=step_idx, response_dict=response_dict,
        )
        return stats

    def find_max_step(self):
        directory = self.args.output_dir
        directory = directory.rstrip('/') + '/rollouts/'
        pattern = re.compile(r'step(\d+)')
        max_step = None
        os.makedirs(directory, exist_ok=True)
        for filename in os.listdir(directory):
            match = pattern.search(filename)
            if match:
                step = int(match.group(1))
                if max_step is None or step > max_step:
                    max_step = step
        if max_step is None:
            max_step = 0
        return max_step

    def generate(self):
        """Entry point for training."""
        total_epochs = self.args.total_epochs
        total_episodes = len(self.train_dataset) * total_epochs  # noqa
        total_steps = total_episodes // self.args.rollout_batch_size  # noqa

        self.max_step = self.args.sample_num // self.args.rollout_batch_size
        print("Max step:", self.max_step)
        logger.warning(
            f"***Training starts***\n"
            f"Total epochs: {total_epochs} => Total episodes: {total_episodes} => Total steps: {total_steps}"
        )

        skipping_steps = self.find_max_step()
        print("Skipping steps:", skipping_steps)
        infinite_train_dataloader = self.get_train_dataloader()

        for step_idx in tqdm.tqdm(
            range(FIRST_STEP_IDX, total_steps + FIRST_STEP_IDX),
            disable=not self.accelerator.is_main_process,
            desc="steps",
            total=total_steps,
        ):
            if step_idx < skipping_steps:
                for _ in range(self.args.rollout_accumulation_steps):
                    next(infinite_train_dataloader)
                continue
            if step_idx >= self.max_step:
                break

            stats = self.step(infinite_train_dataloader, step_idx)
            self.log_history.append(stats)

        return self.log_history

    @torch.inference_mode()
    def evaluate(self, step_idx: int, unwrapped_policy=None):
        raise NotImplementedError

    @abc.abstractmethod
    @torch.inference_mode()
    def save_model(self, output_dir: Optional[str] = None):
        raise NotImplementedError

    @abc.abstractmethod
    @torch.inference_mode()
    def resume_training(self, checkpoint_dir: str):
        raise NotImplementedError

    def _log_batch_size(self, loader: DataLoader, loader_name):
        batch = next(iter(loader))
        if isinstance(batch, torch.Tensor):
            batch_size = batch.shape[0]
        elif isinstance(batch, (list, tuple)):
            batch_size = batch[0]
        else:
            tensor = list(batch.values())[0]
            batch_size = tensor.size(0)
        logger.warning(
            f"Batch size of {loader_name} dataloader: {batch_size}",
            # main_process_only=True,
        )

    def get_train_dataloader(self):
        logger.warning(
            f"Train dataset size: {len(self.train_dataset)}",
            # main_process_only=True
        )  # noqa
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.data_collator,
            batch_size=self.args.rollout_per_device_batch_size,
            shuffle=True,
            drop_last=True,
        )
        train_dataloader = self.accelerator.prepare(train_dataloader)  # noqa
        self._log_batch_size(train_dataloader, "train_dataloader")
        return utils.InfiniteLoader(train_dataloader)

    def get_rollouts_dataloader(
        self, rollouts: Dict[str, torch.Tensor], shuffle=True, drop_last=True, keys=None
    ):
        if keys is None:
            keys = tuple(key for key in rollouts.keys() if not 'adv_' in key)

        def collate_rollouts(instances: Sequence[tuple]):
            return {
                key: torch.stack([instance[idx] for instance in instances])
                for idx, key in enumerate(keys)
            }

        rollouts_dataset = TensorDataset(*[rollouts[key] for key in keys])
        rollouts_dataloader = DataLoader(
            dataset=rollouts_dataset,
            batch_size=self.args.step_per_device_batch_size,
            collate_fn=collate_rollouts,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        return rollouts_dataloader


def truncate_after_eos_with_padding(
    completions, eos_token_id, pad_token_id, additional_tokens=None
):
    # We truncate tokens after eos_token_id
    clean_completions = completions.tolist()
    for idx, completion in enumerate(clean_completions):
        try:
            end_idx = completion.index(eos_token_id)
        except ValueError:
            end_idx = None

        if additional_tokens is not None:
            for additional_token in additional_tokens:
                try:
                    end_idx = completion.index(additional_token)
                except ValueError:
                    pass

        if end_idx is not None:
            clean_completions[idx] = completion[: end_idx + 1]

            if end_idx + 1 < len(completion):
                clean_completions[idx] = clean_completions[idx] + [pad_token_id] * (
                    len(completion) - end_idx - 1
                )

    clean_completions = torch.tensor(
        clean_completions, dtype=torch.long, device=completions.device
    )
    return clean_completions
