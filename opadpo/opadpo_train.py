import argparse
from dataclasses import dataclass, field
import copy
import json
import os
import shutil
import sys
from typing import Optional, List
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import torch.distributed as dist

import accelerate
from accelerate import DistributedDataParallelKwargs

import transformers
from transformers import set_seed
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import Dataset

try:
    from transformers import LlamaTokenizerFast as LlamaTokenizer

    print("Using fast tokenizer")
except:
    from transformers import LlamaTokenizer

    print("Using slow tokenizer")

from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.data_utils_dpo import make_dpo_data_module
from utils.lora_utils import get_last_checkpoint
from opadpo.dpo_models.dpo_trainer import DPOTrainer, make_models
from opadpo.dpo_models.rl_trainer import AlpacaAccelerator

from llava import conversation as conversation_lib
from llava.model import *


from utils.checkpoint_utils import get_model_checkpoint, has_image_model_checkpoint
from llava.model.utils import resize_token_embeddings_with_mean, set_reproducibility

from llava.utils import get_max_num_dataloaders

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
OUTPUT_DIR = Path(OUTPUT_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
PURPLE = "\033[95m"
RESET = "\033[0m"
torch.backends.cuda.matmul.allow_tf32 = True

from loguru import logger as lg
logger = logging.getLogger(__name__)

def is_local_rank_zero() -> bool:
    local_rank = os.environ.get("LOCAL_RANK", None)
    return local_rank == "0" or local_rank is None

class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


@dataclass
class ModelArguments:
    policy_model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b")
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
    )
    base_model_name: Optional[str] = field(default="EleutherAI/pythia-12b")

    # from LLaVA
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=True)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")

    # Additional info from default MAIRA
    mm_projector_type: Optional[str] = field(default="linear")
    max_sequence_length: Optional[int] = field(default=2048)


@dataclass
class DataArguments:
    data_path: str = field(default="/datasetdrive/llava-chexpert-dataset-balanced/train.json")
    dataset_name: str = field(default="alpaca_instructions")
    train_splits: List[str] = field(default_factory=lambda: ["unlabeled"])
    stop_token: Optional[str] = field(
        default=None,
        metadata={"help": "Token to stop generation with."},
    )
    # From LLaVA
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)

    # Additional info from default MAIRA is NONE


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    detailed_report: bool = field(default=True)
    response_score: bool = field(default=False)
    response_image_relation: bool = field(default=False)
    standard_pair_coef: float = field(default=1.0)
    AI_pair_coef: float = field(default=1.0)

    CoPO: bool = field(default=False)
    CoPO_mask_ratio: float = field(default=0.2)
    CoPO_method: str = field(default='random')
    CoPO_coef: float = field(default=1.0)
    AncPO: bool = field(default=False)
    Anchor_value: float = field(default=0.0)
    mDPO_anchor: bool = field(default=False)
    Anchor_coef: float = field(default=1.0)

    prefer_data: str = field(default="gpt4o_pseudo")
    reference_free: bool = field(default=False)
    f_divergence_type: str = field(default="reverse_kl")
    loss_type: str = field(default="sigmoid")
    beta: float = field(default=0.1)
    label_smoothing: float = field(default=0.0)

    advantage_whiten_all: bool = field(default=True)
    train_from_sft: bool = field(default=True)
    reward_clip_min: float = field(default=-10.0)
    reward_clip_max: float = field(default=10.0)
    max_step: int = field(default=300)
    reward_scale: float = field(default=1.0)
    norm_maintain_32: bool = field(default=True)
    lora_with_projector: bool = field(default=False)
    value_head_mode: str = field(default='linear')
    # for DDP
    ddp_backend: Optional[str] = field(default=None)
    ddp_find_unused_parameters: Optional[bool] = field(default=None)
    # ELSE
    cache_dir: Optional[str] = field(default=None)
    # From AlpacaFarm
    truncate_tokens: Optional[List[str]] = field(
        default_factory=lambda: None,
        metadata={
            "help": "Tokens in strings to truncate at first occurrence. "
            "This was used in original OAI summarization paper to avoid models returning incomplete sentences. "
        },
    )
    truncate_after: Optional[int] = field(
        default=None,
        metadata={
            "help": "Truncate after this number of tokens. Prevents early truncation."
        },
    )
    penalty_reward_value: float = field(
        default=-1.0,
        metadata={
            "help": "Reward assigned to sequences that are truncated, "
            "e.g., due to outputting incomplete sentences for given context window."
        },
    )
    penalize_no_stop_token: bool = field(
        default=False,
        metadata={
            "help": "Whether to penalize sequences that do not contain stop_token."
        },
    )
    length_bonus_score: float = field(
        default=0.0,
        metadata={
            "help": "Add the reward for longer sequences by this amount. "
            "This is useful for encouraging longer sequences."
        },
    )
    correct_bonus_score: float = field(
        default=0.0,
        metadata={
            "help": "Add the reward for denying the user request by this amount. "
        },
    )
    reward_bias: float = field(
        default=0.0,
        metadata={"help": "Add this amount to the reward."},
    )
    diverse_penalty_reward_scale: float = field(
        default=0.0,
        metadata={"help": "Scale the reward for diverse sequences by this amount."},
    )
    penalize_non_diverse_responses: bool = field(
        default=False,
        metadata={
            "help": "Whether to penalize sequences that are not diverse within the batch."
        },
    )
    relative_stop_token_penalty: bool = field(
        default=False,
        metadata={
            "help": "Whether to penalize sequences that do not contain stop_token "
            "with a relative penalty based on the original reward."
        },
    )
    clean_tokens_after_eos: bool = field(
        default=False,
        metadata={
            "help": "Whether to clean up tokens after the first occurrence of stop_token."
        },
    )
    suppress_eos_at_generation: bool = field(
        default=False,
        metadata={
            "help": "Whether to suppress the end-of-sequence token at generation time."
        },
    )
    total_epochs: int = field(default=10)
    rollout_batch_size: int = field(default=512)
    step_batch_size: int = field(default=256)
    rollout_per_device_batch_size: int = field(default=32)
    step_per_device_batch_size: int = field(default=2)
    reward_model_per_device_batch_size: int = field(default=None)
    noptepochs: int = field(default=2)
    vf_coef: float = field(default=0.1)
    cliprange: float = field(default=0.2)
    cliprange_value: float = field(default=0.2)
    gamma: float = field(default=1.0)
    lam: float = field(default=1.0)
    whiten_rewards: bool = field(default=True)
    temperature: float = field(default=1.0)
    kl_coef: float = field(default=0.2)
    kl_approximator: str = field(default="k1")
    target_kl: float = field(default=6.0)
    k_beta: float = field(default=0.1)
    adaptive_kl: bool = field(default=False)
    eval_batches: int = field(
        default=sys.maxsize,
        metadata={"help": "Maximum number of batches to evaluate on."},
    )
    init_value_with_reward: bool = field(
        default=True,
        metadata={"help": "Initialize the value model with the reward model."},
    )
    save_steps_extra: Optional[str] = field(
        default=None,
        metadata={
            "help": "A list of predetermined checkpoints to save, represented in the format 'no1__no2__no3'. "
            "Parse this with str.split('__')."
        },
    )
    query_len: int = field(default=128)
    min_token_limit: int = field(default=None)
    response_len: int = field(default=384)
    model_max_length: int = field(default=1024)
    whitening_async_stats: str = field(
        default="per_gpu",
        metadata={"help": "How to sync statistics for reward whitening."},
    )
    # From QLoRA
    full_finetune: bool = field(
        default=False, metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(default=False, metadata={"help": "Use 8-bit adam."})
    report_to: str = field(
        default="none",
        metadata={"help": "To use wandb or something else for reporting."},
    )
    resume_dir: str = field(
        default=None,
        metadata={
            "help": "The directory to resume from. If None, will start from scratch."
        },
    )
    output_dir: str = field(
        default="./output", metadata={"help": "The output dir for logs and checkpoints"}
    )
    optim: str = field(
        default="paged_adamw_32bit", metadata={"help": "The optimizer to be used"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={
            "help": "The training batch size per GPU. Increase for better speed."
        },
    )
    gradient_accumulation_steps: int = field(
        default=16,
        metadata={
            "help": "How many gradients to accumulate before to perform an optimizer step"
        },
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "The L2 weight decay rate of AdamW"}
    )  # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": "The learnign rate"})
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Removed unused columns. Needed to make this codebase work."},
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={
            "help": "Gradient clipping max norm. This is tuned and works well for all models tested."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing. You want to use this."},
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "To train or not to train, that is the question?"},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "The frequency of update steps after which to log the loss"},
    )
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_strategy: str = field(
        default="steps", metadata={"help": "When to save checkpoints"}
    )
    save_steps: int = field(default=250, metadata={"help": "How often to save a model"})
    save_total_limit: int = field(
        default=40,
        metadata={
            "help": "How many checkpoints to save before the oldest is overwritten"
        },
    )
    resume_from_training: bool = field(
        default=False, metadata={"help": "Resume from training"}
    )
    reward_prompt_file: Optional[str] = field(default=None)
    image_to_caption_file: Optional[str] = field(default=None)
    use_flash_attention: bool = False
    use_all_available_cores: bool = True
    logging_interval_seconds: int = 60
    increased_reproducibility: bool = False
    lora_enable: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.00
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mpt_attn_impl: Optional[str] = field(default="triton")
    double_quant: bool = field(
        default=True, metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    freeze_mm_mlp_adapter: bool = field(default=True)

    def __post_init__(self):
        super().__post_init__()
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Checks on rollout_batch_size only matter for PPO.
        assert (
            self.rollout_batch_size >= self.rollout_per_device_batch_size * world_size
        ), (
            "rollout_batch_size is smaller than rollout_per_device_batch_size * world_size. "
            "Increase the former or decrease the latter to fix this."
        )
        assert (
            self.rollout_batch_size % (self.rollout_per_device_batch_size * world_size)
            == 0
        ), "rollout_batch_size is not a multiple of rollout_per_device_batch_size * world_size. "

        assert self.step_batch_size >= self.step_per_device_batch_size * world_size, (
            "step_batch_size is smaller than step_per_device_batch_size * world_size. "
            "Increase the former or decrease the latter to fix this."
        )
        assert (
            self.step_batch_size % (self.step_per_device_batch_size * world_size) == 0
        ), "step_batch_size is not a multiple of step_per_device_batch_size * world_size. "

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            logger.warning(
                f"Rollout stats:\n"
                f"\trollout_batch_size: {self.rollout_batch_size}\n"
                f"\trollout_per_device_batch_size: {self.rollout_per_device_batch_size}\n"
                f"\tworld_size: {world_size}\n",
            )
        assert (
            self.rollout_batch_size // self.rollout_per_device_batch_size
        ) % world_size == 0
        self.rollout_accumulation_steps = (
            self.rollout_batch_size // self.rollout_per_device_batch_size // world_size
        )

        logger.warning(
            f"Step stats:\n"
            f"\tstep_batch_size: {self.step_batch_size}\n"
            f"\tstep_per_device_batch_size: {self.step_per_device_batch_size}\n"
            f"\tworld_size: {world_size}\n",
        )
        assert (
            self.step_batch_size // self.step_per_device_batch_size
        ) % world_size == 0
        self.gradient_accumulation_steps = (
            self.step_batch_size // self.step_per_device_batch_size // world_size
        )

        logger.warning(
            f"Accumulation steps:\n"
            f"\trollout_accumulation_steps: {self.rollout_accumulation_steps}\n"
            f"\tgradient_accumulation_steps: {self.gradient_accumulation_steps}\n"
        )

        if self.save_steps_extra is not None:
            self.save_steps_extra_list = [
                int(string) for string in self.save_steps_extra.split("__")
            ]
        else:
            self.save_steps_extra_list = []

    def set_truncate_token_ids(self, tokenizer: transformers.PreTrainedTokenizer):
        """Convert truncation token to token ids.

        This is called in RLTrainer.
        """
        truncate_tokens = self.truncate_tokens
        if truncate_tokens is None:
            truncate_token_ids = None
        else:
            truncate_token_ids = tokenizer.convert_tokens_to_ids(truncate_tokens)
        self.truncate_token_ids = truncate_token_ids

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


def rank0_print(*args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)

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

def train(cfg: DictConfig) -> None:
    global local_rank
    model_args = ModelArguments(**OmegaConf.to_object(cfg.llava.model))
    data_args = DataArguments(**OmegaConf.to_object(cfg.llava.data))
    training_args = TrainingArguments(**OmegaConf.to_object(cfg.llava.training))
    set_reproducibility(cfg.llava.training.seed, training_args.increased_reproducibility)
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    training_args.data_config = data_args

    if training_args.use_all_available_cores:
        training_args.dataloader_num_workers = get_max_num_dataloaders(training_args.dataloader_num_workers)
    print(f"Using {training_args.dataloader_num_workers=} for dataloader workers per device")

    if is_local_rank_zero():
        logger.info("Saving the hydra config")
        OmegaConf.save(cfg, OUTPUT_DIR / "config.yaml")
        logger.info(f"Configuration for the current job:\n{OmegaConf.to_yaml(cfg)}")

    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    # ! Prepare the accelerator
    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)

    if checkpoint_dir is None and args.resume_dir is not None:
        checkpoint_dir, _ = get_last_checkpoint(args.resume_dir)
        completed_training = False

    if completed_training:
        rank0_print("Detected that training was already completed!")

    if checkpoint_dir is None:
        rank0_print("Training from scratch.")
    else:
        rank0_print("Loading from checkpoint:", checkpoint_dir)

    accelerator = AlpacaAccelerator(
        log_with=args.report_to,
        project_dir=args.logging_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        even_batches=True,  # Make sure the batch size on each device is the same.
        split_batches=False,  # Don't break a batch into smaller chunks.
        step_scheduler_with_optimizer=False,  # Untie optimizer and scheduler step.
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=args.ddp_find_unused_parameters,
                # find_unused_parameters=True,
            )
        ],
    )
    dict_args = vars(args)
    for k in dict_args:
        if type(dict_args[k]) not in [int, float, str, bool, torch.Tensor]:
            dict_args[k] = str(dict_args[k])
    accelerator.init_trackers(
        project_name="OPA-DPO",
        config=dict_args,
    )
    logger.warning(
        accelerator.state,
        # main_process_only=False,
    )

    # ! Loading Vision-Tower
    model_checkpoint_path = str(get_model_checkpoint(cfg.llava.checkpoints))
    image_checkpoint_config = cfg.llava.image_checkpoints

    # NOTE: bnb for QLoRA
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

    # NOTE Loading the Base-Language-Model for preparing vision_tower
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
            # if hasattr(config, "image_checkpoint"):
            #     del config.image_checkpoint
            model = LlavaLlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_checkpoint_path,
                # config=config,
                # cache_dir=training_args.cache_dir,
                use_flash_attention_2=training_args.use_flash_attention,
                torch_dtype=compute_dtype,
                **bnb_model_from_pretrained_args,
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_checkpoint_path, cache_dir=training_args.cache_dir, **bnb_model_from_pretrained_args
        )

    # * Loading Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint_path,
        cache_dir=args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        truncation_side="right",
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

    # * Loading Image Model (RAD-DINO-518px from local file)
    if has_image_model_checkpoint(image_checkpoint_config):
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(dtype=compute_dtype, device=training_args.device)
        if "mimic-cxr" in os.environ.get("DATA_DIR", ""):
            loading_vision_tower_parameter(args.base_model_name, model)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True
        print(f"Vision Tower Info: {vision_tower.image_processor.crop_size}")

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        
    dpo_data_module = make_dpo_data_module(tokenizer, args, data_args)


    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    node_id = rank // torch.cuda.device_count()

    print(f"Distributed info: rank={rank}, world_size={world_size}, node_id={node_id}")

    lg.info("Model Initialization Starting")
    # * Model Initialization with Peft lora
    print(f"Using {args.base_model_name} as the base model")
    print(f"Using {args.policy_model_name_or_path} as the policy model")
    if args.base_model_name == args.policy_model_name_or_path:
        print("Building LoRA model from scratch")
        if os.path.exists(str(args.output_dir + '/init_model/adapter_model.bin')) and os.path.exists(str(args.output_dir + '/init_model/adapter_config.json')):
            args.policy_model_name_or_path = str(args.output_dir + '/init_model')
        else:
            from peft import LoraConfig, get_peft_model
            import time
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
            if training_args.local_rank == 0 or training_args.local_rank == -1:
                state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
                model.config.save_pretrained(str(args.output_dir + '/init_model'))
                model.save_pretrained(str(args.output_dir + '/init_model'), state_dict=state_dict)
            args.policy_model_name_or_path = str(args.output_dir + '/init_model')
            time.sleep(10)
    del model
    torch.cuda.empty_cache()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    model_module = make_models(
        base_model=None,
        tokenizer=tokenizer,
        args=args,
        accelerator=accelerator,
        resume_from_checkpoint=(
            checkpoint_dir if training_args.resume_from_training else None
        ),
    )
    lg.info("Model Initialization Ending")

    trainer = DPOTrainer(
        args=training_args,
        accelerator=accelerator,
        **dpo_data_module,
        **model_module,
        tokenizer=tokenizer,
    )

    trainer.train(
        resume_training_ckpt=checkpoint_dir
        if training_args.resume_from_training
        else None
    )
    if trainer.accelerator.is_main_process:
        trainer.save_model(os.path.join(trainer.args.output_dir, f"checkpoint-final"))

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
