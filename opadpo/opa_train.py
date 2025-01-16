import argparse
import copy
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import transformers
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import Dataset

from opadpo.opa_models.opa_trainer import LLaVATrainer
from llava import conversation as conversation_lib
from utils.checkpoint_utils import get_model_checkpoint, has_image_model_checkpoint
from utils.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    LLAVA_INITIAL_CHECKPOINT_NAME,
)
from llava.mm_utils import tokenizer_image_token
from llava.model import *
from llava.model.language_model.llava_llama import LlavaConfig
from llava.model.utils import resize_token_embeddings_with_mean, set_reproducibility
from llava.utils import get_max_num_dataloaders

from utils.data_utils_sft import make_sft_data_module

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
OUTPUT_DIR = Path(OUTPUT_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
logger = logging.getLogger(__name__)

def is_local_rank_zero() -> bool:
    local_rank = os.environ.get("LOCAL_RANK", None)
    return local_rank == "0" or local_rank is None
local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=True)
    tune_base_model: bool = field(default=True)
    tune_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    max_sequence_length: Optional[int] = field(default=2048)
    policy_model_name_or_path: Optional[str] = field(default="none")


@dataclass
class DataArguments:
    data_path: str = field(default="", metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    data_dir: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    full_tune: bool = field(default=True)
    lora_tune: bool = field(default=False)

    entropy_loss: bool = field(default=False)
    entropy_mask_ratio: float = field(default=0.2)
    entropy_mask_method: str = field(default="random")
    entropy_loss_coef: float = field(default=1.0)
    entropy_decay_coef: float = field(default=1.0)

    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(
        default=True, metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.00
    lora_weight_path: str = ""
    lora_bias: str = "none"
    use_flash_attention: bool = False
    use_all_available_cores: bool = True
    # The update frequency for TQDM progress bars of the Huggingface trainer
    logging_interval_seconds: int = 60
    increased_reproducibility: bool = False


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, name=k, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_name = names[0] if len(names) == 1 else names[-1]
            if len(lora_name) == 1:
                lora_name = names[-2] + "." + lora_name
            lora_module_names.add(lora_name)

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ["mm_projector"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f"{current_folder}.bin"))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        # return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    tokenizer.add_special_tokens(special_tokens_dict)
    resize_token_embeddings_with_mean(model, len(tokenizer))


def loading_vision_tower_parameter(Path, model):
    import json
    import torch
    Path = Path.rstrip('/')
    with open(f'{Path}/pytorch_model.bin.index.json', 'r') as f:
        index_data = json.load(f)
    weight_map = index_data.get("weight_map", {})
    vision_tower_weights = {k: v for k, v in weight_map.items() if "vision_tower" in k or "mm_projector" in k}

    loaded_state_dict = {}
    for weight_name, file_name in vision_tower_weights.items():
        if file_name not in loaded_state_dict:
            loaded_state_dict[file_name] = torch.load(f'{Path}/{file_name}')
    model_state_dict = model.state_dict()
    for weight_name, file_name in vision_tower_weights.items():
        if weight_name in model_state_dict:
            model_state_dict[weight_name] = loaded_state_dict[file_name][weight_name]
            print(f"Loaded weight: {weight_name} from {file_name}")
    model.load_state_dict(model_state_dict)

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4:
        trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def print_trainable_parameters_name(model):
    """
    Prints all trainable parameters in the model.
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter: {name}, Shape: {param.shape}")

def train(cfg: DictConfig) -> None:
    global local_rank

    model_args = ModelArguments(**OmegaConf.to_object(cfg.llava.model))
    data_args = DataArguments(**OmegaConf.to_object(cfg.llava.data))
    training_args = TrainingArguments(**OmegaConf.to_object(cfg.llava.training))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    set_reproducibility(cfg.llava.training.seed, training_args.increased_reproducibility)

    if training_args.use_all_available_cores:
        training_args.dataloader_num_workers = get_max_num_dataloaders(training_args.dataloader_num_workers)
    print(f"Using {training_args.dataloader_num_workers=} for dataloader workers per device")

    if is_local_rank_zero():
        logger.info("Saving the hydra config")
        OmegaConf.save(cfg, OUTPUT_DIR / "config.yaml")
        # Output the full configuration to the console, so that we can double-check if overrides were applied correctly
        logger.info(f"Configuration for the current job:\n{OmegaConf.to_yaml(cfg)}")

    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    model_checkpoint_path = str(get_model_checkpoint(cfg.llava.checkpoints))
    image_checkpoint_config = cfg.llava.image_checkpoints

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    if has_image_model_checkpoint(image_checkpoint_config):
        if cfg.llava.checkpoints.skip_loading_weights:
            print("Initializing an empty Llava model without loading checkpoints")
            assert not training_args.use_flash_attention, "Cannot debug with Flash Attention enabled"
            config = LlavaConfig.from_pretrained(
                model_checkpoint_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args,
            )
            model = LlavaLlamaForCausalLM(config)
        else:
            config = transformers.AutoConfig.from_pretrained(model_checkpoint_path,
                                                             attn_implementation="flash_attention_2")
            model = LlavaLlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_checkpoint_path,
                # config=config,
                # cache_dir=training_args.cache_dir,
                use_flash_attention_2=training_args.use_flash_attention,
                torch_dtype=compute_dtype,
                **bnb_model_from_pretrained_args,
            )
            # model.to(dtype=compute_dtype, device=training_args.device)
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_checkpoint_path, cache_dir=training_args.cache_dir, **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_checkpoint_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]


    if has_image_model_checkpoint(image_checkpoint_config):
        vision_tower = model.get_vision_tower()
        if vision_tower is None:
            model.get_model().initialize_vision_modules(
                model_args=model_args, image_checkpoint_args=cfg.llava.image_checkpoints, fsdp=training_args.fsdp
            )
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        loading_vision_tower_parameter(model_checkpoint_path, model)

        vision_tower.to(dtype=compute_dtype, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True
        print(f"Vision Tower Info: {vision_tower.image_processor.crop_size}")

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.full_tune:
        if model_args.policy_model_name_or_path != "none":
            raise NotImplementedError
        # * Case1 Only tune the mm-projector
        if model_args.tune_mm_mlp_adapter and not model_args.tune_base_model and not model_args.tune_vision_tower:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
        # * Case2 Tune mm-projector + LLaMA backbone model
        elif model_args.tune_mm_mlp_adapter and model_args.tune_base_model and not model_args.tune_vision_tower:
            model.requires_grad_(True)
            for p in model.get_model().vision_tower.parameters():
                p.requires_grad = False
        # * Case3 Tune All Parameters
        elif model_args.tune_mm_mlp_adapter and model_args.tune_base_model and model_args.tune_vision_tower:
            model.requires_grad_(True)
        # * Case4 Tune mm-projector + Vision_tower
        elif model_args.tune_mm_mlp_adapter and not model_args.tune_base_model and model_args.tune_vision_tower:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
            for p in model.get_model().vision_tower.parameters():
                p.requires_grad = True
        # * Case5 Tune backbone_models
        elif not model_args.tune_mm_mlp_adapter and model_args.tune_base_model and not model_args.tune_vision_tower:
            model.requires_grad_(True)
            for p in model.get_model().vision_tower.parameters():
                p.requires_grad = False
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
        else:
            raise NotImplementedError

    if training_args.lora_tune:
        if model_args.policy_model_name_or_path == "none":
            from peft import LoraConfig, get_peft_model
            model.requires_grad_(False)
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(model),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            if training_args.bits == 16:
                if training_args.bf16:
                    model.to(torch.bfloat16)
                if training_args.fp16:
                    model.to(torch.float16)
            rank0_print("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)
        else:
            if not os.path.exists(model_args.policy_model_name_or_path):
                raise FileNotFoundError(f"Policy model not found at {model_args.policy_model_name_or_path}")
            else:
                from peft import PeftModel
                model = PeftModel.from_pretrained(
                    model,
                    model_args.policy_model_name_or_path,
                    is_trainable=True,
                )
        # * Case1 Tune All Parameters
        if model_args.tune_mm_mlp_adapter and model_args.tune_base_model and model_args.tune_vision_tower:
            pass
        # * Case2 Do not tune mm-projector and Vision_tower
        elif not model_args.tune_mm_mlp_adapter and model_args.tune_base_model and not model_args.tune_vision_tower:
            for name, module in model.named_parameters():
                if ("vision_tower" in name) or ("mm_projector" in name):
                    module.requires_grad_(False)
        else:
            raise NotImplementedError

    data_module = make_sft_data_module(
        tokenizer=tokenizer,
        args=args,
        data_args=data_args,
    )

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
                        
    if "wandb" in training_args.report_to and local_rank == 0:
        import wandb
        wandb.init(project='OPA_Training')

    print_trainable_parameters(training_args, model)
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    
    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    final_checkpoint_dir = Path(training_args.output_dir) / "checkpoint-final"
    final_checkpoint_dir.mkdir(exist_ok=True, parents=True)
    if training_args.lora_tune:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(str(final_checkpoint_dir))
            model.save_pretrained(str(final_checkpoint_dir), state_dict=state_dict)
            torch.save(non_lora_state_dict, str(final_checkpoint_dir / "non_lora_trainables.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=str(final_checkpoint_dir))
