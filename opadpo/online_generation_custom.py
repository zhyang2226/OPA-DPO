import argparse
import hydra
import sys
from pathlib import Path

from utils.constants import CONFIGS_DIR
CONFIGS_DIR = Path(CONFIGS_DIR)
from omegaconf import OmegaConf
from loguru import logger

from opadpo.online_generation import generate

import torch
torch.autograd.set_detect_anomaly(True)

logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> - {message}")

HYPERDRIVE_CHILD_ARG_NAME = "random_seed"
YAML_EXTENSION = ".yaml"

def parse_args():
    parser = argparse.ArgumentParser(description='Online Generation')
    parser.add_argument('--cfg', type=str, default='configs/llava/llava_online_generation.yaml')

    # For basic env setting.
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rollout_batch_size', type=int, default=16)
    parser.add_argument('--step_batch_size', type=int, default=16)
    parser.add_argument('--rollout_per_device_batch_size', type=int, default=16)
    parser.add_argument('--reward_model_per_device_batch_size', type=int, default=16)
    parser.add_argument('--step_per_device_batch_size', type=int, default=8)
    parser.add_argument('--warmup_steps', type=int, default=5)
    parser.add_argument('--total_epochs', type=int, default=1)
    parser.add_argument('--logging_steps', type=int, default=1)
    parser.add_argument('--report_to', type=str, default='tensorboard')
    parser.add_argument('--bf16', action='store_false')
    parser.add_argument('--tf32', action='store_false')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--resume_from_training', action='store_false')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--model_max_length', type=int, default=2048)
    parser.add_argument('--query_len', type=int, default=256)
    parser.add_argument('--response_len', type=int, default=256)
    parser.add_argument('--noptepochs', type=int, default=2)
    parser.add_argument('--use_flash_attention', action='store_false')
    parser.add_argument('--eval_steps', type=int, default=10000)
    parser.add_argument('--save_steps', type=int, default=5)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--max_step', type=int, default=300)
    parser.add_argument('--ddp_backend', type=str, default="None")
    parser.add_argument('--ddp_find_unused_parameters', type=str, default="None")
    
    # For path settings
    parser.add_argument('--base_model', type=str, default="./base_models/llava-v1.5-7b")
    parser.add_argument('--output_dir', type=str, default="./output/llava7b_online_generation")
    parser.add_argument('--image_folder', type=str, default=None)
    parser.add_argument('--policy_model_name_or_path', type=str, default='none')
    parser.add_argument('--data_path', type=str, default=None)

    # For Generation
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--phase', type=int, default=0)
    parser.add_argument('--sample_num', type=int, default=5120)

    # For data parameters
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
    cfg.llava.checkpoints.base_model = args.base_model
    cfg.llava.training.ddp_backend = args.ddp_backend if args.ddp_backend.lower() != 'none' else None
    cfg.llava.training.ddp_find_unused_parameters = args.ddp_find_unused_parameters if args.ddp_find_unused_parameters.lower() != 'none' else None
    cfg.llava.training.seed = args.seed
    cfg.llava.training.rollout_batch_size = args.rollout_batch_size
    cfg.llava.training.step_batch_size = args.step_batch_size
    cfg.llava.training.rollout_per_device_batch_size = args.rollout_per_device_batch_size
    cfg.llava.training.reward_model_per_device_batch_size = args.reward_model_per_device_batch_size
    cfg.llava.training.step_per_device_batch_size = args.step_per_device_batch_size
    cfg.llava.training.warmup_steps = args.warmup_steps
    cfg.llava.training.output_dir = args.output_dir
    cfg.llava.training.total_epochs = args.total_epochs
    cfg.llava.training.save_steps = args.save_steps
    cfg.llava.training.save_total_limit = args.save_total_limit
    cfg.llava.training.logging_steps = args.logging_steps
    cfg.llava.training.report_to = args.report_to
    cfg.llava.training.bf16 = True if args.bf16 else False
    cfg.llava.training.tf32 = True if args.tf32 else False
    cfg.llava.training.fp16 = True if args.fp16 else False
    cfg.llava.training.resume_from_training = True if args.resume_from_training else False
    cfg.llava.training.temperature = args.temperature
    cfg.llava.training.model_max_length = args.model_max_length
    cfg.llava.training.query_len = args.query_len
    cfg.llava.training.response_len = args.response_len
    cfg.llava.training.noptepochs = args.noptepochs
    cfg.llava.training.use_flash_attention = True if args.use_flash_attention else False
    cfg.llava.training.save_steps = args.save_steps
    cfg.llava.training.eval_steps = args.eval_steps
    cfg.llava.training.freeze_mm_mlp_adapter = args.freeze_mm_mlp_adapter
    cfg.llava.training.max_step = args.max_step
    cfg.llava.training.top_k = args.top_k
    cfg.llava.training.top_p = args.top_p
    cfg.llava.training.phase = args.phase
    cfg.llava.training.sample_num = args.sample_num

    # override data setting.
    cfg.llava.data.data_path = args.data_path
    cfg.llava.data.image_folder = args.image_folder
    cfg.llava.data.image_aspect_ratio =  args.image_aspect_ratio
    cfg.llava.data.train_splits = args.train_splits

    # override model setting
    cfg.llava.model.policy_model_name_or_path = args.policy_model_name_or_path
    cfg.llava.model.base_model_name = args.base_model_name
    cfg.llava.model.vision_tower = args.vision_tower
    cfg.llava.model.mm_vision_select_layer = args.mm_vision_select_layer
    cfg.llava.model.mm_use_im_start_end = args.mm_use_im_start_end
    cfg.llava.model.mm_use_im_patch_token = args.mm_use_im_patch_token
    
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

    try:
        generate(amlt_cfg)
    finally:
        pass

if __name__ == '__main__':
    main()
