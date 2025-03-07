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

import einops
import tqdm
from loguru import logger as lg

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import accelerate
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.utils import convert_outputs_to_fp32

import transformers
from transformers.trainer_utils import enable_full_determinism, set_seed

import utils.common_utils as utils
import utils.distributed_utils as distributed_utils
from utils.trainer_utils import create_optimizer, create_scheduler

from utils.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


logger = logging.getLogger(__name__)
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

class RLTrainer(object):
    def __init__(
        self,
        args,
        train_dataset,
        eval_dataset,
        data_collator: Callable,
        tokenizer: transformers.PreTrainedTokenizer,
        policy: nn.Module,
        accelerator: AlpacaAccelerator,
        ref_policy: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ):
        super(RLTrainer, self).__init__()
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.policy = policy
        self.ref_policy = ref_policy
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.lr_scheduler = lr_scheduler
        self.log_history = []
        self.args.set_truncate_token_ids(self.tokenizer)
        enable_full_determinism(
            self.args.seed
        ) if self.args.full_determinism else set_seed(self.args.seed)
        self.max_step = args.max_step

        self.reference_free = args.reference_free
        self.f_divergence_type = args.f_divergence_type
        self.loss_type = args.loss_type
        self.beta = args.beta
        self.label_smoothing = args.label_smoothing

    @abc.abstractmethod
    @torch.inference_mode()
    def rollout(self, queries_data) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(
        self, rollouts: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError

    @abc.abstractmethod
    @torch.inference_mode()
    def record_step_stats(self, train_stats, rollouts, step_idx, **kwargs):
        raise NotImplementedError

    @property
    def optimizable_params(self):
        return [
            p
            for p in self.policy.parameters()
            if p.requires_grad and p.grad is not None
        ]

    @torch.inference_mode()
    def _compute_grad_norm(self):
        grad_norm = torch.stack([p.grad.norm(2) for p in self.optimizable_params]).norm(
            2
        )
        return grad_norm

    @torch.inference_mode()
    def _compute_param_norm(self):
        param_norm = torch.stack([p.norm(2) for p in self.optimizable_params]).norm(2)
        return param_norm

    def step_with_rollouts(self, rollouts):
        """Based on fixed rollouts, run PPO for multiple epochs."""
        assert isinstance(self.optimizer, AcceleratedOptimizer), (
            "`optimizer` must be pushed through `accelerator.prepare`. "
            "Otherwise the `accelerator.accumulate` context manager won't correctly disable `zero_grad` or `step`."
        )
        rollouts_dataloader = self.get_rollouts_dataloader(rollouts=rollouts)
        stats_list = []
        for epoch_idx in range(self.args.noptepochs):
            for batch_idx, rollouts_batch in tqdm.tqdm(
                enumerate(rollouts_dataloader, 1),
                total=len(rollouts_dataloader),
                disable=not self.accelerator.is_main_process,
                desc="gradstep",
            ):
                gc.collect()
                torch.cuda.empty_cache()
                with self.accelerator.accumulate(self.policy):
                    stats_for_this_step = {}
                    with self.accelerator.no_sync(self.policy):
                        policy_loss, policy_stats = self.compute_policy_loss(
                            rollouts_batch
                        )
                        stats_for_this_step.update(policy_stats)
                        self.accelerator.backward(policy_loss)

                    if self.accelerator.sync_gradients:
                        if self.args.max_grad_norm is not None:
                            self.accelerator.clip_grad_norm_(
                                self.policy.parameters(), self.args.max_grad_norm
                            )
                        stats_for_this_step[
                            "loss/grad_norm"
                        ] = self._compute_grad_norm()
                        # print(f"grad_norm: {stats_for_this_step['loss/grad_norm']}")
                        stats_list.append(stats_for_this_step)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

        return utils.merge_dict(
            stats_list, torch.stack
        )

    def step(self, train_dataloader, step_idx: int):
        queries_batches = [
            next(train_dataloader) for _ in range(self.args.rollout_accumulation_steps)
        ]
        rollouts = self.rollout(queries_batches)
        torch.cuda.empty_cache()
        train_stats = self.step_with_rollouts(rollouts)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        stats = self.record_step_stats(
            rollouts=rollouts,
            train_stats=train_stats,
            step_idx=step_idx,
        )
        return stats

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        optimizer = create_optimizer(
            args=self.args, model=self.policy, optimizer=self.optimizer
        )
        lr_scheduler = create_scheduler(
            args=self.args,
            optimizer=optimizer,
            lr_scheduler=self.lr_scheduler,
            num_training_steps=num_training_steps,
        )
        self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            optimizer, lr_scheduler
        )
        self.accelerator.register_for_checkpointing(
            self.lr_scheduler
        )
        return self.optimizer, self.lr_scheduler

    def train(self, resume_training_ckpt: Optional[str] = None):
        """Entry point for training."""
        total_epochs = self.args.total_epochs
        total_episodes = len(self.train_dataset) * total_epochs  # noqa
        total_steps = total_episodes // self.args.rollout_batch_size  # noqa
        logger.warning(
            f"***Training starts***\n"
            f"Total epochs: {total_epochs} => Total episodes: {total_episodes} => Total steps: {total_steps}"
        )
        min_total_step = min(total_steps, self.max_step)
        self.create_optimizer_and_scheduler(min_total_step)
        skipping_steps = 0
        if resume_training_ckpt is not None:
            skipping_steps = self.resume_training(resume_training_ckpt)
            print(
                f"Resuming training from {resume_training_ckpt} at step {skipping_steps}."
            )

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

            if (
                step_idx % self.args.save_steps == 0
                or step_idx in self.args.save_steps_extra_list
            ):
                if step_idx > skipping_steps:
                    self.save_model(
                        os.path.join(self.args.output_dir, f"checkpoint-{step_idx}")
                    )

                # # Delete the old checkpoint
                # if step_idx // self.args.save_steps > self.args.save_total_limit:
                #     pre_doc_idx = step_idx - self.args.save_steps * self.args.save_total_limit
                #     pre_doc_path = os.path.join(self.args.output_dir, f"checkpoint-{pre_doc_idx}")
                #     if os.path.exists(pre_doc_path):
                #         lg.warning(f"{pre_doc_path} will be removed")
                #         for item_name in os.listdir(pre_doc_path):
                #             item_path = os.path.join(pre_doc_path, item_name)
                #             if os.path.isfile(item_path) or os.path.islink(item_path):
                #                 os.unlink(item_path)  # Remove files and links
                #             elif os.path.isdir(item_path):
                #                 shutil.rmtree(item_path)  # Remove subdirectories

            if (
                self.args.eval_steps is not None
                and step_idx % self.args.eval_steps == 0
            ):
                self.evaluate(step_idx)

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
            keys = tuple(rollouts.keys())

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

def remove_image_token(completions):
    # We truncate tokens after eos_token_id
    clean_completions = completions.tolist()
    for idx, completion in enumerate(clean_completions):
        completion = [token for token in completion if token != IMAGE_TOKEN_INDEX]
        clean_completions[idx] = completion
    return clean_completions


def truncate_after_eos(completions, eos_token_id):
    # We truncate tokens after eos_token_id
    clean_completions = completions.tolist()
    for idx, completion in enumerate(clean_completions):
        completion = [token for token in completion if token != IMAGE_TOKEN_INDEX]
        clean_completions[idx] = completion
        try:
            end_idx = completion.index(eos_token_id)
            clean_completions[idx] = completion[: end_idx + 1]
        except ValueError:
            pass
    return clean_completions


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
