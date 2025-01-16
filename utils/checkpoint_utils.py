import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import DictConfig
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from utils.constants import LLAVA_CACHE_DIR
from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM

logger = logging.getLogger(__name__)

from enum import Enum
import torch

class FloatingPointType(Enum):
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"

    @property
    def torch_dtype(self) -> torch.dtype:
        match self:
            case FloatingPointType.FP32:
                return torch.float32
            case FloatingPointType.BF16:
                return torch.bfloat16
            case FloatingPointType.FP16:
                return torch.float16
            case _:
                raise NotImplementedError


@dataclass
class PreTrainedModelAndTokenizer:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast


def get_model_checkpoint(checkpoint_config: DictConfig) -> Path:
    return Path(checkpoint_config.base_model)


def get_pretrained_model_and_tokenizer_for_inference(
    checkpoint_config: DictConfig,
    floating_point_dtype: FloatingPointType,
) -> PreTrainedModelAndTokenizer:
    model_download_path = get_model_checkpoint(checkpoint_config)
    tokenizer = AutoTokenizer.from_pretrained(model_download_path, use_fast=False)
    llava_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": floating_point_dtype.torch_dtype,
    }
    if checkpoint_config.skip_loading_weights:
        llava_config = LlavaConfig.from_pretrained(model_download_path, **llava_kwargs)
        model = LlavaLlamaForCausalLM(config=llava_config)
    else:
        # Setting device_map to auto throws an error: RuntimeError: mat1 and mat2 must have the same dtype
        # We just move to cuda() instead
        model = LlavaLlamaForCausalLM.from_pretrained(model_download_path, **llava_kwargs)
    model = model.cuda()

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        logger.info("Starting to load vision tower")
        vision_tower.load_model(skip_loading_weights=checkpoint_config.skip_loading_weights)
        logger.info("Finished loading vision tower")

    # LLaVA loads in fp16 regardless of language model input type, but we will change it based on the LLM dtype, otherwise the forward pass fails
    vision_tower.to(device='cuda', dtype=floating_point_dtype.torch_dtype)

    return PreTrainedModelAndTokenizer(model=model, tokenizer=tokenizer)


def has_image_model_checkpoint(checkpoint_config: DictConfig) -> bool:
    return checkpoint_config.image_model is not None

def get_image_model_checkpoint(checkpoint_config: DictConfig) -> Path | None:
    if not has_image_model_checkpoint(checkpoint_config):
        return None
    return checkpoint_config.image_model
