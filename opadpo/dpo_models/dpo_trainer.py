import json
import gc
import glob
from itertools import chain
import logging
import os
import pathlib
import re
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import accelerate
import pandas as pd
import torch
import torch.nn.functional as F
from enum import Enum
import tqdm
import transformers
from torch.nn.utils.rnn import pad_sequence
from loguru import logger as lg

from peft.utils import WEIGHTS_NAME, get_peft_model_state_dict
import copy

from utils.constants import (
    IMAGE_TOKEN_INDEX,
)

import utils.common_utils as common_utils


import opadpo.dpo_models.rl_models as rl_models

from opadpo.dpo_models.qlora_model import load_nbit_model_for_inference
from opadpo.dpo_models.rl_trainer import (
    AlpacaAccelerator,
    RLTrainer,
    remove_image_token,
    truncate_after_eos_with_padding,
)

AnyPath = Union[str, os.PathLike, pathlib.Path]
AnyPathOrNone = Optional[AnyPath]

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
PURPLE = "\033[95m"
RESET = "\033[0m"

logger = logging.getLogger(__name__)

LRScheduler = torch.optim.lr_scheduler.LRScheduler


# Name of the files used for checkpointing
ADAPTER_MODEL_DIR = "adapter_model"
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
VALUE_HEAD_NAME = "value_head.pt"
SCALER_NAME = "scaler.pt"

class FDivergenceType(Enum):
    REVERSE_KL = "reverse_kl"
    JS_DIVERGENCE = "js_divergence"
    ALPHA_DIVERGENCE = "alpha_divergence"
class FDivergenceConstants:
    ALPHA_DIVERGENCE_COEF_KEY = "alpha_divergence_coef"
    ALPHA_DIVERGENCE_COEF_DEFAULT = 1.0

def get_exp_cap(value, decimal=4):
    vdtype_max = torch.zeros([1]).to(value.dtype) + torch.finfo(value.dtype).max
    vdtype_log_max = torch.log(vdtype_max).to(value.device)
    return torch.floor(vdtype_log_max * 10**decimal) / 10**decimal if decimal > 0 else vdtype_log_max

def cap_exp(value, cap=-1):
    cap = get_exp_cap(value) if cap < 0 else cap
    return torch.exp(torch.clamp(value, max=cap))

def mask_single_image(base_image, mask_percentage, mask_method='random'):
    image = copy.deepcopy(base_image)
    mean_value = image.mean()
    _, C, H, W = image.shape
    if mask_method == 'random':
        total_pixels = H * W
        mask_pixels = int(total_pixels * mask_percentage)
        mask_indices = torch.randperm(total_pixels)[:mask_pixels]
        flat_image = image.view(C, -1)
        flat_image[:, mask_indices] = mean_value
    elif mask_method == 'blockwise':
        block_size = 14
        H_blocks = H // block_size
        W_blocks = W // block_size
        total_blocks = H_blocks * W_blocks
        mask_blocks = int(total_blocks * mask_percentage)
        mask_indices = torch.randperm(total_blocks)[:mask_blocks]

        flat_image = image.view(C, H_blocks, block_size, W_blocks, block_size)
        for idx in mask_indices:
            h = idx // W_blocks
            w = idx % W_blocks
            flat_image[:, h, :, w, :] = mean_value
    else:
        raise NotImplementedError
    masked_image = flat_image.view(1, C, H, W)
    return masked_image

def mask_percentage(matrix: torch.Tensor, percentage: float) -> torch.Tensor:
    num_elements = matrix.numel()
    num_to_mask = int(num_elements * percentage)
    indices = torch.randperm(num_elements)[:num_to_mask]
    flat_matrix = matrix.view(-1)
    flat_matrix[indices] = False
    return flat_matrix.view(matrix.size())

def mask_percentage_per_row(matrix: torch.Tensor, percentage: float) -> torch.Tensor:
    num_columns = matrix.size(1)
    num_to_mask_per_row = int(num_columns * percentage)
    for i in range(matrix.size(0)):
        indices = torch.randperm(num_columns)[:num_to_mask_per_row]
        matrix[i, indices] = False
    return matrix

class DPOTrainer(RLTrainer):
    def __init__(
        self,
        args,
        train_dataset,
        eval_dataset,
        data_collator: Callable,
        policy: rl_models.Policy,
        ref_policy: rl_models.Policy,
        tokenizer: transformers.PreTrainedTokenizer,
        accelerator: AlpacaAccelerator,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ):
        super(DPOTrainer, self).__init__(
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            policy=policy,
            ref_policy=ref_policy,
            tokenizer=tokenizer,
            accelerator=accelerator,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )


    def slice_input_shift_pad(self, input_ids, output_slices, query_len=128, pad_side='left'):
        num_rows, max_len = input_ids.shape
        slices_id = [output_slices[i].start for i in range(num_rows)]
        slices_id_np = np.array(slices_id)

        if slices_id_np.max() > query_len and query_len > 0:
            lg.warning(f"In this batch, there exists at least one query which has length larger than {query_len}")

        if query_len > 0:
            queries = torch.zeros(num_rows, query_len, dtype=input_ids.dtype, device=input_ids.device)
        else:
            queries = torch.zeros(num_rows, slices_id_np.max(), dtype=input_ids.dtype, device=input_ids.device)
        standard_responses = torch.zeros(num_rows, max_len-slices_id_np.min(), dtype=input_ids.dtype, device=input_ids.device)
        query_attn_masks = torch.zeros_like(queries)

        if pad_side == 'left':
            for i in range(num_rows):
                length = slices_id[i]
                rest_length = max_len - length
                # QUERY DOES NOT EXCEED LIMITS / NO LIMITS
                if (length <= query_len and query_len > 0) or query_len<=0:
                    queries[i, -length:] = input_ids[i, :length]
                    standard_responses[i, :rest_length] = input_ids[i, length:]
                    query_attn_masks[i, -length:] = 1
                # QUERY EXCEEDS LIMITS
                else:
                    print(f"QUERY: {input_ids[i, :length].cpu().numpy()}")
                    queries[i, :] = input_ids[i, :query_len]
                    if IMAGE_TOKEN_INDEX not in input_ids[i, :query_len-9]:
                        lg.warning(f"NO IMAGE TOKEN FOUND in {i}th row")
                        queries[i, -9:] = torch.tensor([IMAGE_TOKEN_INDEX, 29889, 29871, 319, 1799, 9047, 13566, 29901, 29871], device=input_ids.device)
                    else:
                        queries[i, -8:] = torch.tensor([29889, 29871, 319, 1799, 9047, 13566, 29901, 29871], device=input_ids.device)
                    standard_responses[i, :rest_length] = input_ids[i, length:]
                    query_attn_masks[i, :] = 1
        elif pad_side == 'right':
            for i in range(num_rows):
                length = slices_id[i]
                rest_length = max_len - length
                # QUERY DOES NOT EXCEED LIMITS / NO LIMITS
                if (length <= query_len and query_len > 0) or query_len<=0:
                    queries[i, :length] = input_ids[i, :length]
                    standard_responses[i, :rest_length] = input_ids[i, length:]
                    query_attn_masks[i, :length] = 1
                # QUERY EXCEEDS LIMITS
                else:
                    queries[i, :] = input_ids[i, :query_len]
                    if IMAGE_TOKEN_INDEX not in input_ids[i, :query_len-9]:
                        lg.warning(f"NO IMAGE TOKEN FOUND in {i}th row")
                        queries[i, -9:] = torch.tensor([IMAGE_TOKEN_INDEX, 29889, 29871, 319, 1799, 9047, 13566, 29901, 29871], device=input_ids.device)
                    else:
                        queries[i, -8:] = torch.tensor([29889, 29871, 319, 1799, 9047, 13566, 29901, 29871], device=input_ids.device)
                    standard_responses[i, :rest_length] = input_ids[i, length:]
                    query_attn_masks[i, :] = 1
        else:
            raise NotImplementedError(f"Note that pad_side '{pad_side}' can only be chosen from 'left' and 'right'!")

        return queries, query_attn_masks, standard_responses

    @torch.inference_mode()
    def rollout(self, queries_data) -> Dict[str, torch.Tensor]:
        
        self.policy.eval()
        self.ref_policy.eval()
        rollouts = []

        for batch_idx, batch in tqdm.tqdm(
            enumerate(queries_data),
            total=len(queries_data),
            disable=not self.accelerator.is_main_process,
            desc="rollout",
        ):
            gc.collect()
            torch.cuda.empty_cache()
            # Sample rollouts.
            (
                images,
                queries,
                queries_attention_mask,
                standard_response,
                standard_response_attention_mask,
                original_generate_response,
                original_generate_response_attention_mask,
                AI_pseudo_response,
                AI_pseudo_response_attention_mask,
            ) = common_utils.unpack_dict(
                common_utils.prepare_inputs(batch, device=self.accelerator.device),
                keys=(
                    "images",
                    "queries",
                    "queries_attention_mask",
                    "standard_response",
                    "standard_response_attention_mask",
                    "original_generate_response",
                    "original_generate_response_attention_mask",
                    "AI_pseudo_response",
                    "AI_pseudo_response_attention_mask",
                ),
            )
            if self.args.detailed_report and (self.args.response_score or self.args.response_image_relation):
                (
                    original_generate_response_scores,
                    AI_pseudo_response_scores,
                    original_generate_response_image_relations,
                    AI_pseudo_response_image_relations,
                ) = common_utils.unpack_dict(
                    common_utils.prepare_inputs(batch, device=self.accelerator.device),
                    keys=(
                        "original_generate_response_scores",
                        "AI_pseudo_response_scores",
                        "original_generate_response_image_relations",
                        "AI_pseudo_response_image_relations",
                    ),
                )

            if self.args.bf16:
                images = images.to(torch.bfloat16)
            elif self.args.fp16:
                images = images.half()


            additional_token1 = self.tokenizer.encode("?", add_special_tokens=False)[0]
            assert additional_token1 == 1577

            additional_token2 = self.tokenizer.encode("\n?")[-1]
            assert additional_token2 == 29973

            rollouts_batch = {
                "images": images,
                "queries": queries,
                "queries_attn_masks": queries_attention_mask,
                "standard_response": standard_response,
                "standard_response_attention_mask": standard_response_attention_mask,
                "original_generate_response": original_generate_response,
                "original_generate_response_attention_mask": original_generate_response_attention_mask,
                "AI_pseudo_response": AI_pseudo_response,
                "AI_pseudo_response_attention_mask": AI_pseudo_response_attention_mask,
            }

            if self.args.CoPO is True:
                if self.args.CoPO_method == "random" or self.args.CoPO_method == "blockwise":
                    masked_images = torch.stack([mask_single_image(images[i].unsqueeze(0), self.args.CoPO_mask_ratio, self.args.CoPO_method) for i in range(images.size(0))]).squeeze(1)
                    rollouts_batch.update({f"masked_images": masked_images})
                    new_rollouts_batch = {
                        "images": masked_images,
                        "queries": queries,
                        "queries_attn_masks": queries_attention_mask,
                        "standard_response": standard_response,
                        "standard_response_attention_mask": standard_response_attention_mask,
                        "AI_pseudo_response": AI_pseudo_response,
                        "AI_pseudo_response_attention_mask": AI_pseudo_response_attention_mask,
                    }
                elif self.args.CoPO_method == "attention":
                    new_masks = torch.clone(queries_attention_mask).detach()
                    assert images.size(-1) % 14 == 0
                    image_token_len = (images.size(-1) // 14) ** 2
                    image_attention_mask = new_masks.new_full((new_masks.size(0), image_token_len), True)
                    image_attention_mask = mask_percentage_per_row(image_attention_mask, self.args.CoPO_mask_ratio)
                    masked_query_attn_masks = torch.cat([image_attention_mask, new_masks], dim=1)
                    rollouts_batch.update({f"masked_query_attn_masks": masked_query_attn_masks})
                    new_rollouts_batch = {
                        "images": images,
                        "queries": queries,
                        "queries_attn_masks": masked_query_attn_masks,
                        "standard_response": standard_response,
                        "standard_response_attention_mask": standard_response_attention_mask,
                        "AI_pseudo_response": AI_pseudo_response,
                        "AI_pseudo_response_attention_mask": AI_pseudo_response_attention_mask,
                    }
                else:
                    raise NotImplementedError


            # * FOR Evaluate logprobs of the samples.
            batch_size_per_device = rollouts_batch["queries"].shape[0]
            sub_batch_size = self.args.reward_model_per_device_batch_size

            if sub_batch_size is None or sub_batch_size == batch_size_per_device:
                ref_policy_outputs = self.ref_policy(
                    **rollouts_batch, temperature=self.args.temperature
                )
                policy_outputs = self.policy(
                    **rollouts_batch, temperature=self.args.temperature
                )
                if self.args.CoPO is True:
                    new_ref_policy_outputs = self.ref_policy(
                        **new_rollouts_batch, temperature=self.args.temperature
                    )
            else:
                assert batch_size_per_device % sub_batch_size == 0

                ref_policy_outputs_list = []
                if self.args.CoPO is True:
                    new_ref_policy_outputs_list = []

                for sub_batch_idx in range(batch_size_per_device // sub_batch_size):
                    sub_batch = {
                        key: value[
                            sub_batch_idx * sub_batch_size : (sub_batch_idx + 1) * sub_batch_size
                        ]
                        for key, value in rollouts_batch.items()
                    }
                    sub_batch_ref_policy_outputs = self.ref_policy(
                        **sub_batch, temperature=self.args.temperature
                    )
                    ref_policy_outputs_list.append(sub_batch_ref_policy_outputs)

                    if self.args.COPO is True:
                        new_sub_batch = {
                            key: value[
                                sub_batch_idx * sub_batch_size : (sub_batch_idx + 1) * sub_batch_size
                            ]
                            for key, value in new_rollouts_batch.items()
                        }
                        new_sub_batch_ref_policy_outputs = self.ref_policy(
                            **new_sub_batch, temperature=self.args.temperature
                        )
                        new_ref_policy_outputs_list.append(new_sub_batch_ref_policy_outputs)

                ref_policy_outputs = common_utils.merge_dict(
                    ref_policy_outputs_list, merge_fn=torch.cat
                )
                del sub_batch_ref_policy_outputs
                del ref_policy_outputs_list
                del sub_batch

                if self.args.CoPO is True:
                    new_ref_policy_outputs = common_utils.merge_dict(
                        new_ref_policy_outputs_list, merge_fn=torch.cat
                    )
                    del new_sub_batch_ref_policy_outputs
                    del new_ref_policy_outputs_list
                    del new_sub_batch

            key_set_logprobs = [key+'_logprobs' for key in rollouts_batch.keys() if "response" in key and "mask" not in key and "scores" not in key and "image_relations" not in key]
            key_set_entropes = [key+'_entropies' for key in rollouts_batch.keys() if "response" in key and "mask" not in key and "scores" not in key and "image_relations" not in key]
            key_set = key_set_logprobs + key_set_entropes
            ref_policy_outputs = common_utils.unpack_dict(
                ref_policy_outputs, keys=key_set, return_type=dict
            )
            rollouts_batch.update(
                {f"ref_base_{key}": value for key, value in ref_policy_outputs.items()}
            )

            if self.args.CoPO is True:
                key_set_logprobs = [key+'_logprobs' for key in new_rollouts_batch.keys() if "response" in key and "mask" not in key and "scores" not in key and "image_relations" not in key]
                key_set_entropes = [key+'_entropies' for key in new_rollouts_batch.keys() if "response" in key and "mask" not in key and "scores" not in key and "image_relations" not in key]
                key_set = key_set_logprobs + key_set_entropes
                new_ref_policy_outputs = common_utils.unpack_dict(
                    new_ref_policy_outputs, keys=key_set, return_type=dict
                )
                rollouts_batch.update(
                    {f"ref_mask_{key}": value for key, value in new_ref_policy_outputs.items()}
                )

            if self.args.detailed_report and (self.args.response_score or self.args.response_image_relation):
                rollouts_batch.update(
                    {
                        "original_generate_response_scores": original_generate_response_scores,
                        "AI_pseudo_response_scores": AI_pseudo_response_scores,
                        "original_generate_response_image_relations": original_generate_response_image_relations,
                        "AI_pseudo_response_image_relations": AI_pseudo_response_image_relations,
                    }
                )

            rollouts_batch_cpu = {
                key: value.cpu() for key, value in rollouts_batch.items()
            }
            rollouts.append(rollouts_batch_cpu)

        # * FOR Items in dict need to be of same shape.
        rollouts = common_utils.merge_dict(rollouts, merge_fn=torch.cat)
        return {**rollouts}

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        chosen_scores: Optional[torch.FloatTensor] = None,
        rejected_scores: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        if chosen_scores is None:
            chosen_scores = torch.ones_like(policy_chosen_logps)
        if rejected_scores is None:
            rejected_scores = torch.ones_like(policy_rejected_logps)

        chosen_logratios = policy_chosen_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_chosen_logps.to(self.accelerator.device)
        rejected_logratios = policy_rejected_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_rejected_logps.to(self.accelerator.device)

        if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE.value:
            alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
            if self.f_divergence_params and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY in self.f_divergence_params:
                alpha_coef = float(self.f_divergence_params[FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY])
            logits = (cap_exp(rejected_logratios * -alpha_coef) - cap_exp(chosen_logratios * -alpha_coef)) / alpha_coef
        else:
            logits = chosen_scores * chosen_logratios - rejected_scores * rejected_logratios


            if self.f_divergence_type == FDivergenceType.JS_DIVERGENCE.value:
                logits -= F.softplus(chosen_logratios) - F.softplus(rejected_logratios)
                
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}."
            )

        return losses, self.beta * chosen_logratios, self.beta * rejected_logratios

    def compute_policy_loss(
        self, rollouts: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:

        (
            queries,
            queries_attention_mask,
            images,
            standard_response,
            original_generate_response,
            AI_pseudo_response,
            ref_base_standard_response_logprobs,
            ref_base_original_generate_response_logprobs,
            ref_base_AI_pseudo_response_logprobs,
        ) = common_utils.prepare_inputs(
            common_utils.unpack_dict(
                rollouts,
                keys=(
                    "queries",
                    "queries_attn_masks",
                    "images",
                    "standard_response",
                    "original_generate_response",
                    "AI_pseudo_response",
                    "ref_base_standard_response_logprobs",
                    "ref_base_original_generate_response_logprobs",
                    "ref_base_AI_pseudo_response_logprobs",
                ),
            ),
            device=self.accelerator.device,
        )
        if self.args.detailed_report and (self.args.response_score or self.args.response_image_relation):
            (
                original_generate_response_scores,
                AI_pseudo_response_scores,
                original_generate_response_image_relations,
                AI_pseudo_response_image_relations,
            ) = common_utils.prepare_inputs(
                common_utils.unpack_dict(
                    rollouts,
                    keys=(
                        "original_generate_response_scores",
                        "AI_pseudo_response_scores",
                        "original_generate_response_image_relations",
                        "AI_pseudo_response_image_relations",
                    ),
                ),
                device=self.accelerator.device,
            )
        else:
            original_generate_response_scores = (ref_base_original_generate_response_logprobs != self.tokenizer.pad_token_id).to(ref_base_original_generate_response_logprobs.dtype)
            original_generate_response_image_relations = original_generate_response_scores
            AI_pseudo_response_scores = (ref_base_AI_pseudo_response_logprobs  != self.tokenizer.pad_token_id).to(ref_base_AI_pseudo_response_logprobs.dtype)
            AI_pseudo_response_image_relations = AI_pseudo_response_scores

        if self.args.CoPO:
            if self.args.CoPO_method == "random" or self.args.CoPO_method == "blockwise":
                (
                    masked_images,
                    ref_mask_standard_response_logprobs,
                    ref_mask_AI_pseudo_response_logprobs,
                ) = common_utils.prepare_inputs(
                    common_utils.unpack_dict(
                        rollouts,
                        keys=(
                            "masked_images",
                            "ref_mask_standard_response_logprobs",
                            "ref_mask_AI_pseudo_response_logprobs",
                        ),
                    ),
                    device=self.accelerator.device,
                )
            elif self.args.CoPO_method == "attention":
                (
                    masked_query_attn_masks,
                    ref_mask_standard_response_logprobs,
                    ref_mask_AI_pseudo_response_logprobs,
                ) = common_utils.prepare_inputs(
                    common_utils.unpack_dict(
                        rollouts,
                        keys=(
                            "masked_query_attn_masks",
                            "ref_mask_standard_response_logprobs",
                            "ref_mask_AI_pseudo_response_logprobs",
                        ),
                    ),
                    device=self.accelerator.device,
                )

        need_sync = False

        self.policy.train()

        responses_dict = {
            "standard_response": standard_response,
            "original_generate_response": original_generate_response,
            "AI_pseudo_response": AI_pseudo_response,
        }
        outputs = self.policy(
            queries=queries,
            queries_attn_masks=queries_attention_mask,
            images=images,
            temperature=self.args.temperature,
            mode="policy",
            **responses_dict,
        )

        # * Preference Pair1 Standard >> Original Generate
        losses1, chosen_rewards1, rejected_rewards1 = self.dpo_loss(
            policy_chosen_logps=outputs["standard_response_logprobs"],
            policy_rejected_logps=outputs["original_generate_response_logprobs"],
            reference_chosen_logps=ref_base_standard_response_logprobs,
            reference_rejected_logps=ref_base_original_generate_response_logprobs,
        )
        chosen_rewards1_mask = ref_base_standard_response_logprobs != self.tokenizer.pad_token_id
        rejected_rewards1_mask = ref_base_original_generate_response_logprobs != self.tokenizer.pad_token_id
        standard_logprobs = outputs["standard_response_logprobs"].detach()
        original_logprobs = outputs["original_generate_response_logprobs"].detach()

        # * Preference Pair2 AI Generate >> Original Generate
        losses2, chosen_rewards2, rejected_rewards2 = self.dpo_loss(
            policy_chosen_logps=outputs["AI_pseudo_response_logprobs"],
            policy_rejected_logps=outputs["original_generate_response_logprobs"],
            reference_chosen_logps=ref_base_AI_pseudo_response_logprobs,
            reference_rejected_logps=ref_base_original_generate_response_logprobs,
            chosen_scores=AI_pseudo_response_scores if (self.args.detailed_report and self.args.response_score) else None,
            rejected_scores=original_generate_response_scores if (self.args.detailed_report and self.args.response_score) else None,
        )
        chosen_rewards2_mask = ref_base_AI_pseudo_response_logprobs != self.tokenizer.pad_token_id
        rejected_rewards2_mask = rejected_rewards1_mask
        AI_logprobs = outputs["AI_pseudo_response_logprobs"].detach()

        if need_sync:
            print("Syncing standard, original, AI_gen logprobs")
            standard_logprobs_list = [torch.zeros_like(standard_logprobs) for _ in range(self.accelerator.num_processes)]
            original_logprobs_list = [torch.zeros_like(original_logprobs) for _ in range(self.accelerator.num_processes)]
            AI_logprobs_list = [torch.zeros_like(AI_logprobs) for _ in range(self.accelerator.num_processes)]
            torch.distributed.all_gather(standard_logprobs_list, standard_logprobs)
            torch.distributed.all_gather(original_logprobs_list, original_logprobs)
            torch.distributed.all_gather(AI_logprobs_list, AI_logprobs)
            standard_logprobs_gather = torch.cat(standard_logprobs_list, dim=0)
            original_logprobs_gather = torch.cat(original_logprobs_list, dim=0)
            AI_logprobs_gather = torch.cat(AI_logprobs_list, dim=0)

        if need_sync and self.args.detailed_report and (self.args.response_score or self.args.response_image_relation):
            print("Syncing scores")
            original_generate_response_scores_list = [torch.zeros_like(original_generate_response_scores) for _ in range(self.accelerator.num_processes)]
            AI_pseudo_response_scores_list = [torch.zeros_like(AI_pseudo_response_scores) for _ in range(self.accelerator.num_processes)]
            torch.distributed.all_gather(original_generate_response_scores_list, original_generate_response_scores)
            torch.distributed.all_gather(AI_pseudo_response_scores_list, AI_pseudo_response_scores)
            original_generate_response_scores_gather = torch.cat(original_generate_response_scores_list, dim=0)
            AI_pseudo_response_scores_gather = torch.cat(AI_pseudo_response_scores_list, dim=0)

        AI_gen_logprobs_mask = (ref_base_AI_pseudo_response_logprobs != self.tokenizer.pad_token_id) * (AI_pseudo_response_scores != 1.0)
        original_gen_mask = (ref_base_original_generate_response_logprobs != self.tokenizer.pad_token_id) * (original_generate_response_scores != 1.0)

        loss = losses1.mean() * self.args.standard_pair_coef + losses2.mean() * self.args.AI_pair_coef

        chosen_rewards3_mask = chosen_rewards1_mask
        rejected_rewards3_mask = chosen_rewards1_mask
        chosen_rewards4_mask = chosen_rewards2_mask
        rejected_rewards4_mask = chosen_rewards2_mask

        if self.args.CoPO:
            new_responses_dict={
                "mask_standard_response": standard_response,
                "mask_AI_pseudo_response": AI_pseudo_response,
            }
            if self.args.CoPO_method == "random" or self.args.CoPO_method == "blockwise":
                outputs_new = self.policy(
                    queries=queries,
                    queries_attn_masks=queries_attention_mask,
                    images=masked_images,
                    temperature=self.args.temperature,
                    mode="policy",
                    **new_responses_dict,
                )
            elif self.args.CoPO_method == "attention":
                outputs_new = self.policy(
                    queries=queries,
                    queries_attn_masks=masked_query_attn_masks,
                    images=images,
                    temperature=self.args.temperature,
                    mode="policy",
                    **new_responses_dict,
                )
            else:
                raise NotImplementedError

            # * Preference Pair3 Standard Origin >> Standard Mask
            losses3, chosen_rewards3, rejected_rewards3 = self.dpo_loss(
                policy_chosen_logps=outputs["standard_response_logprobs"],
                policy_rejected_logps=outputs_new["mask_standard_response_logprobs"],
                reference_chosen_logps=ref_base_standard_response_logprobs,
                reference_rejected_logps=ref_mask_standard_response_logprobs,
            )

            # * Preference Pair4 AI_Gen Origin >> AI_Gen Mask
            losses4, chosen_rewards4, rejected_rewards4 = self.dpo_loss(
                policy_chosen_logps=outputs["AI_pseudo_response_logprobs"],
                policy_rejected_logps=outputs_new["mask_AI_pseudo_response_logprobs"],
                reference_chosen_logps=ref_base_AI_pseudo_response_logprobs,
                reference_rejected_logps=ref_mask_AI_pseudo_response_logprobs,
                chosen_scores=AI_pseudo_response_image_relations if (self.args.detailed_report and self.args.response_image_relation) else None,
                rejected_scores=AI_pseudo_response_image_relations if (self.args.detailed_report and self.args.response_image_relation) else None,
            )
            standard_mask_logprobs = outputs_new["mask_standard_response_logprobs"].detach()
            AI_mask_logprobs = outputs_new["mask_AI_pseudo_response_logprobs"].detach()

            if need_sync:
                print("Syncing standard_mask, AI_mask logprobs")
                standard_mask_logprobs_list = [torch.zeros_like(standard_mask_logprobs) for _ in range(self.accelerator.num_processes)]
                AI_mask_logprobs_list = [torch.zeros_like(AI_mask_logprobs) for _ in range(self.accelerator.num_processes)]
                torch.distributed.all_gather(standard_mask_logprobs_list, standard_mask_logprobs)
                torch.distributed.all_gather(AI_mask_logprobs_list, AI_mask_logprobs)
                standard_mask_logprobs_gather = torch.cat(standard_mask_logprobs_list, dim=0)
                AI_mask_logprobs_gather = torch.cat(AI_mask_logprobs_list, dim=0)

            loss += losses3.mean() * self.args.standard_pair_coef * self.args.CoPO_coef + losses4.mean() * self.args.AI_pair_coef * self.args.CoPO_coef
        else:
            standard_mask_logprobs=AI_mask_logprobs=torch.zeros_like(standard_logprobs)

            if need_sync:
                standard_mask_logprobs_gather = AI_mask_logprobs_gather = torch.zeros_like(standard_logprobs_gather)

            losses3=chosen_rewards3=rejected_rewards3=losses4=chosen_rewards4=rejected_rewards4=torch.zeros_like(loss)

        if self.args.AncPO:
            if self.args.mDPO_anchor:
                AncPO_losses = (-F.logsigmoid(chosen_rewards1 - self.args.Anchor_value) - F.logsigmoid(-chosen_rewards2 + self.args.Anchor_value) -
                                F.logsigmoid(chosen_rewards3 - self.args.Anchor_value) - F.logsigmoid(-chosen_rewards4 + self.args.Anchor_value))
            else:
                AncPO_losses = ((chosen_rewards1 - self.args.Anchor_value)**2 + (chosen_rewards2 - self.args.Anchor_value)**2 +
                            (chosen_rewards3 - self.args.Anchor_value)**2 + (chosen_rewards4 - self.args.Anchor_value)**2)
            AncPO_losses = AncPO_losses.mean()
            loss += AncPO_losses * self.args.Anchor_coef
        else:
            AncPO_losses = torch.zeros_like(loss)

        def compute_masked_mean(rewards, mask):
            return masked_mean(rewards, mask).mean()

        def compute_gap_mean(chosen_rewards, chosen_mask, rejected_rewards, rejected_mask):
            return compute_masked_mean(chosen_rewards, chosen_mask) - compute_masked_mean(rejected_rewards, rejected_mask)

        def compute_masked_min(logprobs, mask):
            large_positive = 1e9
            return (logprobs * mask + ~ mask * large_positive).min(axis=1).values.mean()

        def compute_masked_max(logprobs, mask):
            large_negative = -1e9
            return (logprobs * mask + ~ mask * large_negative).max(axis=1).values.mean()


        if not need_sync:
            original_gen_logprobs_mask = (original_logprobs != 0.0) #* (original_generate_response_scores != 1.0)
            AI_gen_logprobs_mask = (AI_logprobs != 0.0) #* (AI_pseudo_response_scores != 1.0)
            standard_logprobs_mask = (standard_logprobs != 0.0)
            logprobs_dict = dict(
                original_logprobs=compute_masked_mean(original_logprobs, original_gen_logprobs_mask),
                standard_logprobs=compute_masked_mean(standard_logprobs, standard_logprobs_mask),
                AI_logprobs=compute_masked_mean(AI_logprobs, AI_gen_logprobs_mask),
                standard_mask_logprobs=compute_masked_mean(standard_mask_logprobs, standard_logprobs_mask),
                AI_mask_logprobs=compute_masked_mean(AI_mask_logprobs, AI_gen_logprobs_mask),

                original_logprobs_min=compute_masked_min(original_logprobs, original_gen_logprobs_mask),
                standard_logprobs_min=compute_masked_min(standard_logprobs, standard_logprobs_mask),
                AI_logprobs_min=compute_masked_min(AI_logprobs, AI_gen_logprobs_mask),
                standard_mask_logprobs_min=compute_masked_min(standard_mask_logprobs, standard_logprobs_mask),
                AI_mask_logprobs_min=compute_masked_min(AI_mask_logprobs, AI_gen_logprobs_mask),

                original_logprobs_max=compute_masked_max(original_logprobs, original_gen_logprobs_mask),
                standard_logprobs_max=compute_masked_max(standard_logprobs, standard_logprobs_mask),
                AI_logprobs_max=compute_masked_max(AI_logprobs, AI_gen_logprobs_mask),
                standard_mask_logprobs_max=compute_masked_max(standard_mask_logprobs, standard_logprobs_mask),
                AI_mask_logprobs_max=compute_masked_max(AI_mask_logprobs, AI_gen_logprobs_mask),
            )
        else:
            original_gen_logprobs_mask_gather = (original_logprobs_gather != 0.0) * (original_generate_response_scores_gather != 1.0)
            AI_gen_logprobs_mask_gather = (AI_logprobs_gather != 0.0) * (AI_pseudo_response_scores_gather != 1.0)
            standard_logprobs_mask_gather = (standard_logprobs_gather != 0.0)
            logprobs_dict = dict(
                original_logprobs=compute_masked_mean(original_logprobs_gather, original_gen_logprobs_mask_gather),
                standard_logprobs=compute_masked_mean(standard_logprobs_gather, standard_logprobs_mask_gather),
                AI_logprobs=compute_masked_mean(AI_logprobs_gather, AI_gen_logprobs_mask_gather),
                standard_mask_logprobs=compute_masked_mean(standard_mask_logprobs_gather, standard_logprobs_mask_gather),
                AI_mask_logprobs=compute_masked_mean(AI_mask_logprobs_gather, AI_gen_logprobs_mask_gather),

                original_logprobs_min=compute_masked_min(original_logprobs_gather, original_gen_logprobs_mask_gather),
                standard_logprobs_min=compute_masked_min(standard_logprobs_gather, standard_logprobs_mask_gather),
                AI_logprobs_min=compute_masked_min(AI_logprobs_gather, AI_gen_logprobs_mask_gather),
                standard_mask_logprobs_min=compute_masked_min(standard_mask_logprobs_gather, standard_logprobs_mask_gather),
                AI_mask_logprobs_min=compute_masked_min(AI_mask_logprobs_gather, AI_gen_logprobs_mask_gather),

                original_logprobs_max=compute_masked_max(original_logprobs_gather, original_gen_logprobs_mask_gather),
                standard_logprobs_max=compute_masked_max(standard_logprobs_gather, standard_logprobs_mask_gather),
                AI_logprobs_max=compute_masked_max(AI_logprobs_gather, AI_gen_logprobs_mask_gather),
                standard_mask_logprobs_max=compute_masked_max(standard_mask_logprobs_gather, standard_logprobs_mask_gather),
                AI_mask_logprobs_max=compute_masked_max(AI_mask_logprobs_gather, AI_gen_logprobs_mask_gather),
            )

        stats = dict(
            loss=dict(
                stand_gen=losses1.mean(),
                AI_gen=losses2.mean(),
                stand_mask=losses3.mean(),
                AI_mask=losses4.mean(),
                AncPO=AncPO_losses
            ),
            policy=dict(
                stand_gen_chosen_mean=compute_masked_mean(chosen_rewards1, chosen_rewards1_mask),
                stand_gen_reject_mean=compute_masked_mean(rejected_rewards1, rejected_rewards1_mask),
                stand_gen_gap_mean=compute_gap_mean(chosen_rewards1, chosen_rewards1_mask, rejected_rewards1, rejected_rewards1_mask),
                AI_gen_chosen_mean=compute_masked_mean(chosen_rewards2, chosen_rewards2_mask),
                AI_gen_reject_mean=compute_masked_mean(rejected_rewards2, rejected_rewards2_mask),
                AI_gen_gap_mean=compute_gap_mean(chosen_rewards2, chosen_rewards2_mask, rejected_rewards2, rejected_rewards2_mask),
                stand_mask_chosen_mean=compute_masked_mean(chosen_rewards3, chosen_rewards3_mask),
                stand_mask_reject_mean=compute_masked_mean(rejected_rewards3, rejected_rewards3_mask),
                stand_mask_gap_mean=compute_gap_mean(chosen_rewards3, chosen_rewards3_mask, rejected_rewards3, rejected_rewards3_mask),
                AI_mask_chosen_mean=compute_masked_mean(chosen_rewards4, chosen_rewards4_mask),
                AI_mask_reject_mean=compute_masked_mean(rejected_rewards4, rejected_rewards4_mask),
                AI_mask_gap_mean=compute_gap_mean(chosen_rewards4, chosen_rewards4_mask, rejected_rewards4, rejected_rewards4_mask),
            ),
            logprobs=logprobs_dict,
        )
        return loss, common_utils.flatten_dict(
            stats, sep="/", postprocess_fn=lambda x: x.detach()
        )

    def record_step_stats(self, train_stats, rollouts, step_idx, **kwargs):
        keys = rollouts.keys()
        entropy_keys = [key for key in keys if "entropies" in key]
        stats = {
            f"objective/lr": self.optimizer.param_groups[0]["lr"],
        }
        stats.update(
            {
                f"objective/{key}": masked_mean(rollouts[key], rollouts[key]!=0.0, axis=1).mean()
                for key in entropy_keys
            }
        )
        for k, v in train_stats.items():
            if not "logprobs/" in k:
                stats[f"dpo/{k}"] = v.mean(dim=0)
            else:
                stats[f"{k}"] = v.mean(dim=0)

        stats = {
            key: value.item() if torch.is_tensor(value) else value
            for key, value in stats.items()
        }

        stats = {
            key[:key.find('/') + 1] + key[key.find('/') + 1:].replace('/', '-') if '/' in key else key: value
            for key, value in stats.items()
        }

        if self.accelerator.is_main_process:
            self.accelerator.log(stats, step=step_idx)

        return stats

    @torch.inference_mode()
    def save_model(
        self,
        output_dir: Optional[str] = None,
        give_rw_access=True,
        check_corrupted=True,
        for_best_ckpt=False,
    ):
        output_dir = self.args.output_dir if output_dir is None else output_dir

        global_rank = int(os.environ.get("RANK", 0))

        if global_rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            print("Saving model checkpoint to %s" % output_dir)

            # Save policy model.
            unwrapped_policy = self.accelerator.unwrap_model(
                self.policy, keep_fp32_wrapper=True
            )

            policy_model = unwrapped_policy.base_model

            peft_model_path = os.path.join(output_dir, ADAPTER_MODEL_DIR)

            save_adapters(
                policy_model,
                peft_model_path,
                adapter_names=["lora_policy"],
            )

            pytorch_model_paths = glob.glob(
                os.path.join(output_dir, "pytorch_model*.bin")
            )
            for pytorch_model_path in pytorch_model_paths:
                if os.path.exists(pytorch_model_path):
                    os.remove(pytorch_model_path)

            if not for_best_ckpt:
                # Save optimizer.
                torch.save(
                    self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME)
                )
                # Save scheduler.
                torch.save(
                    self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME)
                )

                # Delete other optimizer checkpoints to save disk space.
                # glob pattern to match all optimizer.pt files in the father directory
                pattern = os.path.join(os.path.dirname(output_dir), "*/optimizer.pt")

                # get a list of all matching paths
                optimizer_files = glob.glob(pattern)

                # iterate over the optimizer files
                for file in optimizer_files:
                    # if the file is not in the output_dir, delete it
                    if output_dir not in file:
                        os.remove(file)

        else:
            print("Skipping PEFT checkpoint save on rank %d" % global_rank)

    @torch.inference_mode()
    def resume_training(self, checkpoint_dir):
        # Load optimizer.
        optimizer_path = os.path.join(checkpoint_dir, OPTIMIZER_NAME)
        if os.path.exists(optimizer_path):
            load_paged_optimizer_state_dict(
                self.optimizer.optimizer,
                torch.load(
                    optimizer_path,
                    map_location="cpu",
                ),
            )

        # Unpage optimizer state.
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        # Load scheduler.
        scheduler_path = os.path.join(checkpoint_dir, SCHEDULER_NAME)
        if os.path.exists(scheduler_path):
            self.lr_scheduler.load_state_dict(
                torch.load(
                    scheduler_path,
                    map_location="cpu",
                )
            )

        spattern = re.compile(r"checkpoint-(\d+)")
        skipping_steps = int(spattern.search(checkpoint_dir).group(1))
        return skipping_steps


def smart_tokenizer_and_embedding_resize(
    num_new_tokens: int,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.get_input_embeddings().requires_grad_(True)
        model.get_output_embeddings().requires_grad_(True)


def make_models(
    base_model,
    tokenizer: transformers.PreTrainedTokenizer,
    args,
    accelerator: accelerate.Accelerator,
    num_new_tokens: int = 0,
    resume_from_checkpoint: Optional[str] = None,
) -> dict:
    def make_generative_policy(
        adapter_name, is_trainable, reuse_base_model=True, resume_path=None
    ):
        model = load_nbit_model_for_inference(
            checkpoint_dir=resume_path or args.policy_model_name_or_path,
            image_aspect_ratio=args.image_aspect_ratio,
            image_grid_pinpoints=args.image_grid_pinpoints,
            bits=args.bits,
            fp16=args.fp16,
            bf16=args.bf16,
            gradient_checkpointing=args.gradient_checkpointing,
            adapter_name=adapter_name,
            is_trainable=is_trainable,
            reuse_base_model=reuse_base_model,
            trust_remote_code=args.trust_remote_code,
            use_flash_attention=args.use_flash_attention,
            norm_maintain_32=args.norm_maintain_32,
            lora_with_projector=args.lora_with_projector,
            train_from_sft=args.train_from_sft
        )
        smart_tokenizer_and_embedding_resize(num_new_tokens, tokenizer, model)
        return model

    # * If base model is not loaded, then load it as "load_4bit_model_for_inference"
    # if base_model is None:
    policy_resume_path = None
    if resume_from_checkpoint:
        policy_resume_path = os.path.join(
            resume_from_checkpoint, ADAPTER_MODEL_DIR, "lora_policy"
        )

    policy = rl_models.make_policy_with_base_model(
        args,
        make_generative_policy(
            adapter_name="lora_policy",
            is_trainable=True,
            resume_path=policy_resume_path,
        ),
        tokenizer,
        adapter_name="lora_policy",
    )

    ref_policy = rl_models.make_policy_with_base_model(
        args,
        make_generative_policy(
            adapter_name="lora_ref_policy",
            is_trainable=False,
        ),
        tokenizer,
        adapter_name="lora_ref_policy",
    )

    if args.vision_tower == "different":
        compute_dtype = (
            torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
        )
        policy_vision_tower = policy.base_model.get_vision_tower()
        if not policy_vision_tower.is_loaded:
            policy_vision_tower.load_model()
        policy_vision_tower.to(device="cuda", dtype=compute_dtype)
        policy_vision_tower.requires_grad_(False)

        policy_mm_projector = policy.base_model.get_model().mm_projector
        policy_mm_projector.to(device="cuda", dtype=compute_dtype)
        policy_mm_projector.requires_grad_(False)
    else:
        raise NotImplemented
    from opadpo.opadpo_train import loading_vision_tower_parameter
    loading_vision_tower_parameter(args.base_model_name, policy.base_model.model)
    print(f"Vision Tower Loaded from {args.base_model_name}")
    policy = accelerator.prepare(policy)  # noqa

    return dict(policy=policy, ref_policy=ref_policy)

def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis, keepdim=True) / mask.sum(axis=axis, keepdim=True)
    else:
        return (values * mask).sum() / mask.sum()

def save_adapters(model, save_directory, adapter_names, **kwargs):
    r"""
    This function saves the adapter model and the adapter configuration files to a directory, so that it can be
    reloaded using the [`LoraModel.from_pretrained`] class method, and also used by the [`LoraModel.push_to_hub`]
    method.

    Args:
        model: The model to save.
        save_directory (`str`):
            Directory where the adapter model and configuration files will be saved (will be created if it does not
            exist).
        adapter_name (`str`):
            Name of the adapter to save.
        kwargs (additional keyword arguments, *optional*):
            Additional keyword arguments passed along to the `push_to_hub` method.
    """
    if os.path.isfile(save_directory):
        raise ValueError(
            f"Provided path ({save_directory}) should be a directory, not a file"
        )
    os.makedirs(save_directory, exist_ok=True)
    # model.create_or_update_model_card(save_directory)

    for adapter_name, peft_config in model.peft_config.items():
        if adapter_name in adapter_names:
            # save only the trainable weights
            output_state_dict = get_peft_model_state_dict(
                model,
                state_dict=kwargs.get("state_dict", None),
                adapter_name=adapter_name,
            )
            output_dir = (
                os.path.join(save_directory, adapter_name)
                if adapter_name != "default"
                else save_directory
            )
            os.makedirs(output_dir, exist_ok=True)

            torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            # save the config and change the inference mode to `True`
            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    model.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True
            peft_config.save_pretrained(output_dir)
            peft_config.inference_mode = inference_mode


def load_paged_optimizer_state_dict(optimizer, state_dict):
    """
    Load an optimizer state dict that was saved.
    """

    # Validate the state_dict
    groups = optimizer.param_groups
    saved_groups = state_dict["param_groups"]

    if len(groups) != len(saved_groups):
        raise ValueError(
            "loaded state dict has a different number of " "parameter groups"
        )
    param_lens = (len(g["params"]) for g in groups)
    saved_lens = (len(g["params"]) for g in saved_groups)
    if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
        raise ValueError(
            "loaded state dict contains a parameter group "
            "that doesn't match the size of optimizer's group"
        )

    # Update the state
    id_map = {
        p: old_id
        for old_id, p in zip(
            chain.from_iterable(g["params"] for g in saved_groups),
            chain.from_iterable(g["params"] for g in groups),
        )
    }

    for g in groups:
        for p in g["params"]:
            if p in optimizer.state:
                values = optimizer.state[p]
                for k, v in values.items():
                    if isinstance(v, torch.Tensor):
                        v.copy_(state_dict["state"][id_map[p]][k])
                        optimizer.state[p][k] = v.to("cpu")
                    else:
                        optimizer.state[p][k] = state_dict["state"][id_map[p]][k]


def remove_pad_and_left_pad(completions, pad_token_id):
    # We truncate tokens after eos_token_id
    clean_completions = completions.tolist()
    padded_length = len(clean_completions[0])
    for idx, completion in enumerate(clean_completions):
        completion = [token for token in completion if token != pad_token_id]

        if len(completion) < padded_length:
            completion = [pad_token_id] * (padded_length - len(completion)) + completion

        clean_completions[idx] = completion

    clean_completions = torch.tensor(
        clean_completions, dtype=torch.long, device=completions.device
    )
    return clean_completions

def get_param_value_by_name(model, param_name):
    state_dict = model.state_dict()
    if param_name in state_dict:
        return state_dict[param_name]
    else:
        raise ValueError(f"Parameter '{param_name}' not found in the model.")

def compare_lora_params(policy):
    for name, param in policy.named_parameters():
        if 'lora_policy' in name:
            ref_name = name.replace('lora_policy', 'lora_ref_policy')
            param_ref = get_param_value_by_name(policy, ref_name)
            if not torch.equal(param, param_ref):
                print(f"Parameter values do not match for {name}-{ref_name}")
            else:
                print(f"Parameter match {name}-{ref_name}")
