import argparse
import hydra
import sys
from pathlib import Path
import os

from utils.constants import CONFIGS_DIR
CONFIGS_DIR = Path(CONFIGS_DIR)
from omegaconf import OmegaConf
from loguru import logger

from opadpo.opadpo_train import train

logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> - {message}")

HYPERDRIVE_CHILD_ARG_NAME = "random_seed"
YAML_EXTENSION = ".yaml"

def parse_args():
    parser = argparse.ArgumentParser(description='Train PPO RadLLAVA')
    parser.add_argument('--cfg', type=str, default='configs/llava/radllava_dpo.yaml')

    # For env setting.
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--lora_rank', type=int, default=256, help='Rank parameter for LoRA')
    parser.add_argument('--lora_alpha', type=int, default=512, help='Alpha parameter for LoRA')
    parser.add_argument('--lora_drop', type=float, default=0.0, help='Dropout rate for LoRA')

    # For training parameters
    parser.add_argument('--detailed_report', type=str, default="True")
    parser.add_argument('--response_score', type=str, default="True")
    parser.add_argument('--response_image_relation', type=str, default="True")
    parser.add_argument('--standard_pair_coef', type=float, default=1.0)
    parser.add_argument('--AI_pair_coef', type=float, default=1.0)

    parser.add_argument('--CoPO', type=str, default="True")
    parser.add_argument('--CoPO_mask_ratio', type=float, default=0.3)
    parser.add_argument('--CoPO_method', type=str, default='random')
    parser.add_argument('--CoPO_coef', type=float, default=0.2)
    parser.add_argument('--AncPO', type=str, default="True")
    parser.add_argument('--Anchor_value', type=float, default=0.0)
    parser.add_argument('--mDPO_anchor', type=str, default="True")
    parser.add_argument('--Anchor_coef', type=float, default=1.0)
    parser.add_argument('--reference_free', type=str, default="False")
    parser.add_argument('--f_divergence_type', type=str, default="reverse_kl")
    parser.add_argument('--loss_type', type=str, default="sigmoid")
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--label_smoothing', type=float, default=0.0)

    parser.add_argument('--advantage_whiten_all', type=str, default="True")
    parser.add_argument('--train_from_sft', type=str, default="True")
    parser.add_argument('--norm_maintain_32', type=str, default="False")
    parser.add_argument('--lora_with_projector', type=str, default="False")
    parser.add_argument('--value_head_mode', type=str, default="mlp2x_gelu")
    parser.add_argument('--ddp_backend', type=str, default="None")
    parser.add_argument('--ddp_find_unused_parameters', type=str, default="None")
    # PATH
    parser.add_argument('--base_model', type=str, default="./base_models/llava-v1.5-7b")
    parser.add_argument('--output_dir', type=str, default="./output/llava7b_opadpo_model")
    parser.add_argument('--image_folder', type=str, default=None)
    parser.add_argument('--policy_model_name_or_path', type=str, default="./output/llava7b_opa_model/checkpoint-final")

    parser.add_argument('--do_train', action='store_false')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rollout_batch_size', type=int, default=128)
    parser.add_argument('--step_batch_size', type=int, default=32)
    parser.add_argument('--rollout_per_device_batch_size', type=int, default=16)
    parser.add_argument('--reward_model_per_device_batch_size', type=int, default=16)
    parser.add_argument('--step_per_device_batch_size', type=int, default=16)

    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--init_value_with_reward', action='store_true')
    parser.add_argument('--warmup_steps', type=int, default=5)
    parser.add_argument('--total_epochs', type=int, default=1)
    parser.add_argument('--group_by_length', action='store_true')
    parser.add_argument('--evaluation_strategy', type=str, default='no')
    parser.add_argument('--save_strategy', type=str, default='steps')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parser.add_argument('--logging_steps', type=int, default=1)
    parser.add_argument('--report_to', type=str, default='wandb')
    parser.add_argument('--bf16', action='store_false')
    parser.add_argument('--tf32', action='store_false')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--penalty_reward_value', type=float, default=-8.0)
    parser.add_argument('--length_bonus_score', type=float, default=-10.0)
    parser.add_argument('--correct_bonus_score', type=float, default=2.0)
    parser.add_argument('--relative_stop_token_penalty', action='store_true')
    parser.add_argument('--penalize_no_stop_token', action='store_true')
    parser.add_argument('--resume_from_training', action='store_false')
    parser.add_argument('--kl_coef', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--whitening_async_stats', type=str, default='full_batch')
    parser.add_argument('--clean_tokens_after_eos', action='store_false')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--model_max_length', type=int, default=2048)
    parser.add_argument('--query_len', type=int, default=256)
    parser.add_argument('--response_len', type=int, default=256)
    parser.add_argument('--noptepochs', type=int, default=2)
    parser.add_argument('--use_flash_attention', action='store_false')
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--save_steps', type=int, default=10)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--reward_scale', type=float, default=1.0)
    parser.add_argument('--max_step', type=int, default=300)
    parser.add_argument('--reward_clip_min', type=float, default=-10.0)
    parser.add_argument('--reward_clip_max', type=float, default=10.0)
    parser.add_argument('--cliprange', type=float, default=0.2)
    parser.add_argument('--cliprange_value', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--lam', type=float, default=1.0)

    # For data parameters
    parser.add_argument('--data_path', type=str, default="./base_datasets/opadpo_training_data-7B")
    parser.add_argument('--image_aspect_ratio', type=str, default='pad')
    parser.add_argument('--train_splits', type=str, default='train')

    # For model parameters
    parser.add_argument('--base_model_name', type=str, default="./base_models/llava-v1.5-7b")
    parser.add_argument('--vision_tower', type=str, default='different')
    parser.add_argument('--mm_vision_select_layer', type=int, default=-2)
    parser.add_argument('--mm_use_im_start_end', action='store_true')
    parser.add_argument('--mm_use_im_patch_token', action='store_true')
    parser.add_argument('--freeze_mm_mlp_adapter', action='store_true')

    args = parser.parse_args()
    return args


def expand_config_file_name(name: str, expected_subfolder: str):
    """Make some educated guesses about where a Hydra config file might be located.
    In all that follows, the config file must be in a subpath of the configs folder.

    If the given file name is absolute and the file exists, use that. '~' characters for user home are allowed.
    Otherwise, try to locate the file relative to the current working folder. If the file exists there, use it.
    Otherwise, if the given file exists in the Hydra config dir in the folder 'expected_subfolder', use that.
    Otherwise, if the given file exists in the Hydra config dir, without using the `expected_subfolder`, use that.
    If none of that works, raise a FileNotFoundException.
    If the file exists in any of the locations listed before, return the path of the file relative to the Hydra
    configs folder.

    :param name: A string containing the name of a Hydra config file, as an absolute or truncated path.
    :return: The path of the Hydra config file relative to the CONFIGS_DIR.
    :raises FileNotFoundError: If the given file does not exist in any of the locations tried.
    :raises ValueError: If the given file is not located in the Hydra config folder.
    """

    def is_in_config_folder(path: Path) -> Path:
        try:
            path_relative = path.relative_to(CONFIGS_DIR)
        except ValueError as ex:
            raise ValueError(f"Config file is not located in the 'configs' folder: {path}") from ex
        return path_relative

    if not name.endswith(YAML_EXTENSION):
        name = name + YAML_EXTENSION
    path = Path(name)
    path_expanded = path.expanduser()

    if path_expanded.is_absolute():
        if path_expanded.is_file():
            return is_in_config_folder(path_expanded)
        raise FileNotFoundError(f"The specified config file does not exist: {path_expanded}")

    # Resolve the path relative to the current working directory
    path_resolved = path.resolve()
    if path_resolved.is_file():
        return is_in_config_folder(path_resolved)

    # Try in the "configs" folder and the expected_subfolder (e.g., "azure") therein
    for subfolder in [expected_subfolder, ""]:
        path_in_config = CONFIGS_DIR / subfolder / path
        if path_in_config.is_file():
            return is_in_config_folder(path_in_config)

    raise FileNotFoundError(f"The specified config file does not exist in any of the location tried: {path}")


def override_config(cfg, args):
    # path
    cfg.llava.checkpoints.base_model = args.base_model_name
    
    # override training setting.
    cfg.llava.training.detailed_report = True if args.detailed_report=="True" else False
    cfg.llava.training.response_score = True if args.response_score=="True" else False
    cfg.llava.training.response_image_relation = True if args.response_image_relation=="True" else False
    cfg.llava.training.standard_pair_coef = args.standard_pair_coef
    cfg.llava.training.AI_pair_coef = args.AI_pair_coef

    cfg.llava.training.reference_free = True if args.reference_free=="True" else False
    cfg.llava.training.f_divergence_type = args.f_divergence_type
    cfg.llava.training.loss_type = args.loss_type
    cfg.llava.training.beta = args.beta
    cfg.llava.training.label_smoothing = args.label_smoothing

    cfg.llava.training.CoPO = True if args.CoPO=="True" else False
    cfg.llava.training.CoPO_mask_ratio = args.CoPO_mask_ratio
    cfg.llava.training.CoPO_method = args.CoPO_method
    cfg.llava.training.CoPO_coef = args.CoPO_coef
    cfg.llava.training.AncPO = True if args.CoPO=="True" else False
    cfg.llava.training.Anchor_value = args.Anchor_value
    cfg.llava.training.mDPO_anchor = True if args.mDPO_anchor=="True" else False
    cfg.llava.training.Anchor_coef = args.Anchor_coef

    cfg.llava.training.lora_r = args.lora_rank
    cfg.llava.training.lora_alpha = args.lora_alpha
    cfg.llava.training.lora_dropout = args.lora_drop

    cfg.llava.training.advantage_whiten_all = True if args.advantage_whiten_all=="True" else False
    cfg.llava.training.train_from_sft = True if args.train_from_sft=="True" else False
    cfg.llava.training.norm_maintain_32 = True if args.norm_maintain_32=="True" else False
    cfg.llava.training.lora_with_projector = True if args.lora_with_projector=="True" else False
    cfg.llava.training.value_head_mode = args.value_head_mode
    cfg.llava.training.ddp_backend = args.ddp_backend if args.ddp_backend.lower() != 'none' else None
    cfg.llava.training.ddp_find_unused_parameters = args.ddp_find_unused_parameters if args.ddp_find_unused_parameters.lower() != 'none' else None
    cfg.llava.training.do_train = True if args.do_train else False
    cfg.llava.training.seed = args.seed
    cfg.llava.training.rollout_batch_size = args.rollout_batch_size
    cfg.llava.training.step_batch_size = args.step_batch_size
    cfg.llava.training.rollout_per_device_batch_size = args.rollout_per_device_batch_size
    cfg.llava.training.reward_model_per_device_batch_size = args.reward_model_per_device_batch_size
    cfg.llava.training.step_per_device_batch_size = args.step_per_device_batch_size
    cfg.llava.training.learning_rate = args.learning_rate
    cfg.llava.training.init_value_with_reward = True if args.init_value_with_reward else False
    cfg.llava.training.warmup_steps = args.warmup_steps
    cfg.llava.training.output_dir = args.output_dir
    cfg.llava.training.total_epochs = args.total_epochs
    cfg.llava.training.group_by_length = True if not args.group_by_length else False
    cfg.llava.training.evaluation_strategy = args.evaluation_strategy
    cfg.llava.training.save_strategy = args.save_strategy
    cfg.llava.training.save_steps = args.save_steps
    cfg.llava.training.save_total_limit = args.save_total_limit
    cfg.llava.training.weight_decay = args.weight_decay
    cfg.llava.training.lr_scheduler_type = args.lr_scheduler_type
    cfg.llava.training.logging_steps = args.logging_steps
    cfg.llava.training.report_to = args.report_to
    cfg.llava.training.bf16 = True if args.bf16 else False
    cfg.llava.training.tf32 = True if args.tf32 else False
    cfg.llava.training.fp16 = True if args.fp16 else False
    cfg.llava.training.penalty_reward_value = args.penalty_reward_value
    cfg.llava.training.length_bonus_score = args.length_bonus_score
    cfg.llava.training.correct_bonus_score = args.correct_bonus_score
    cfg.llava.training.relative_stop_token_penalty = True if args.relative_stop_token_penalty else False
    cfg.llava.training.penalize_no_stop_token = True if args.penalize_no_stop_token else False
    cfg.llava.training.resume_from_training = True if args.resume_from_training else False
    cfg.llava.training.kl_coef = args.kl_coef
    cfg.llava.training.max_grad_norm = args.max_grad_norm
    cfg.llava.training.whitening_async_stats = args.whitening_async_stats
    cfg.llava.training.clean_tokens_after_eos = True if args.clean_tokens_after_eos else False
    cfg.llava.training.temperature = args.temperature
    cfg.llava.training.model_max_length = args.model_max_length
    cfg.llava.training.query_len = args.query_len
    cfg.llava.training.response_len = args.response_len
    cfg.llava.training.noptepochs = args.noptepochs
    cfg.llava.training.use_flash_attention = True if args.use_flash_attention else False
    cfg.llava.training.save_steps = args.save_steps
    cfg.llava.training.eval_steps = args.eval_steps
    cfg.llava.training.freeze_mm_mlp_adapter = args.freeze_mm_mlp_adapter
    cfg.llava.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    cfg.llava.training.reward_scale = args.reward_scale
    cfg.llava.training.max_step = args.max_step
    cfg.llava.training.reward_clip_min = args.reward_clip_min
    cfg.llava.training.reward_clip_max = args.reward_clip_max
    cfg.llava.training.cliprange = args.cliprange
    cfg.llava.training.cliprange_value = args.cliprange_value
    cfg.llava.training.gamma = args.gamma
    cfg.llava.training.lam = args.lam

    # override data setting.
    cfg.llava.data.data_path = args.data_path
    cfg.llava.data.image_folder = args.image_folder
    cfg.llava.data.image_aspect_ratio =  args.image_aspect_ratio
    cfg.llava.data.train_splits = args.train_splits

    # override model setting
    cfg.llava.model.policy_model_name_or_path = args.policy_model_name_or_path
    cfg.llava.model.base_model_name = args.base_model_name
    cfg.llava.model.vision_tower = args.vision_tower
    return cfg


def main():
    args = parse_args()
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # initialize
    hydra.initialize_config_dir(str(CONFIGS_DIR), version_base="1.2") # initialize
    amlt_cfg_path = expand_config_file_name(args.cfg, expected_subfolder="llava")
    amlt_cfg = hydra.compose(str(amlt_cfg_path))
    amlt_cfg = override_config(amlt_cfg, args)
    OmegaConf.resolve(amlt_cfg)
    logger.info(f"cfg, {OmegaConf.to_yaml(amlt_cfg)}")
    print(f"We are Here")
    try:
        train(amlt_cfg)
    finally:
        pass




if __name__ == '__main__':
    main()
