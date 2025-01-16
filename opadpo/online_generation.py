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
from peft import PeftModel

import torch
import torch.distributed as dist

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
from utils.data_utils_online_gpt4v import make_rlaif_gpt4v_data_module

from opadpo.generator_models.online_generator import Online_Generator
from opadpo.generator_models.generator import AlpacaAccelerator

from llava import conversation as conversation_lib
from llava.model import *
from utils.constants import (
    IMAGE_TOKEN_INDEX,
)
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

def is_local_rank_zero() -> bool:
    local_rank = os.environ.get("LOCAL_RANK", None)
    return local_rank == "0" or local_rank is None

torch.backends.cuda.matmul.allow_tf32 = True

from loguru import logger as lg
logger = logging.getLogger(__name__)


class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


@dataclass
class ModelArguments:
    policy_model_name_or_path: Optional[str] = field(default="./base_models/llava-v1.5-7b")
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
    )
    base_model_name: Optional[str] = field(default="./base_models/llava-v1.5-7b")

    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=True)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")

    mm_projector_type: Optional[str] = field(default="linear")
    max_sequence_length: Optional[int] = field(default=2048)


@dataclass
class DataArguments:
    data_path: str = field(default="./base_datasets/LLaVA-RLAIF-SubData/subset1")
    dataset_name: str = field(default="rlaif_subset")
    train_splits: List[str] = field(default_factory=lambda: ["unlabeled"])
    stop_token: Optional[str] = field(
        default=None,
        metadata={"help": "Token to stop generation with."},
    )
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    top_k: int = field(default=0)
    top_p: float = field(default=1.0)
    phase: int = field(default=0)
    sample_num: int = field(default=5000)

    add_missing: bool = field(default=True)
    max_step: int = field(default=300)
    norm_maintain_32: bool = field(default=True)
    lora_with_projector: bool = field(default=False)
    value_head_mode: str = field(default='linear')

    ddp_backend: Optional[str] = field(default=None)
    ddp_find_unused_parameters: Optional[bool] = field(default=None)
    cache_dir: Optional[str] = field(default=None)

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
    temperature: float = field(default=1.0)
    eval_batches: int = field(
        default=sys.maxsize,
        metadata={"help": "Maximum number of batches to evaluate on."},
    )
    save_steps_extra: Optional[str] = field(
        default=None,
        metadata={
            "help": "A list of predetermined checkpoints to save, represented in the format 'no1__no2__no3'. "
            "Parse this with str.split('__')."
        },
    )
    query_len: int = field(default=128)
    response_len: int = field(default=384)
    model_max_length: int = field(default=1024)

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
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Removed unused columns. Needed to make this codebase work."},
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

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

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

def generate(cfg: DictConfig) -> None:
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


    accelerator = AlpacaAccelerator(
        log_with=args.report_to,
        project_dir=args.logging_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        even_batches=True,
        split_batches=False,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=args.ddp_find_unused_parameters,
            )
        ],
    )
    dict_args = vars(args)
    for k in dict_args:
        if type(dict_args[k]) not in [int, float, str, bool, torch.Tensor]:
            dict_args[k] = str(dict_args[k])
    accelerator.init_trackers(
        project_name="OPA-DPO_online_generation",
        config=dict_args,
    )
    logger.warning(
        accelerator.state,
    )

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

    # * Loading Image Model
    if has_image_model_checkpoint(image_checkpoint_config):
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(dtype=compute_dtype, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True
        print(f"Vision Tower Info: {vision_tower.image_processor.crop_size}")

        data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter

    # * Loading Dataset
    data_module = make_rlaif_gpt4v_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
        sample_size=args.sample_num,
        batch_idx=args.phase,
        seed=args.seed,
    )


    if accelerator.is_main_process:
        training_data = data_module["train_dataset"]
        for i in range(3):
            ex_input_ids_0 = training_data[i]['queries']
            lg.info(f"Example of idx{i}: Non image part token index")
            rank0_print(ex_input_ids_0[ex_input_ids_0 != tokenizer.pad_token_id])
            ex_input_ids_0[ex_input_ids_0 == IMAGE_TOKEN_INDEX] = tokenizer.eos_token_id
            lg.info(f"Example of idx{i}: Full sentance after decoding")
            rank0_print(tokenizer.decode(ex_input_ids_0[[ex_input_ids_0 != tokenizer.pad_token_id]], skip_special_tokens=False))
            rank0_print("=" * 20)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    node_id = rank // torch.cuda.device_count()

    print(f"Distributed info: rank={rank}, world_size={world_size}, node_id={node_id}")

    loading_vision_tower_parameter(model_checkpoint_path, model)

    if (args.policy_model_name_or_path != "none" and
        os.path.exists(args.policy_model_name_or_path.rstrip('/') + "/adapter_model.bin") and
        os.path.exists(args.policy_model_name_or_path.rstrip('/') + "/adapter_config.json")):
        print(f"loading lora adpater form {args.policy_model_name_or_path}")
        model = PeftModel.from_pretrained(
            model,
            args.policy_model_name_or_path,
        )
    else:
        print(f"No lora adapter found in {args.policy_model_name_or_path}, using the original model")

    generator = Online_Generator(
        args=training_args,
        accelerator=accelerator,
        **data_module,
        policy=model.to(accelerator.device),
        tokenizer=tokenizer,
    )

    generator.generate()
