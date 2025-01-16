import argparse
import hydra
import sys
from pathlib import Path
from utils.constants import CONFIGS_DIR
CONFIGS_DIR = Path(CONFIGS_DIR)
from omegaconf import OmegaConf
from loguru import logger

from opadpo.opa_train import train

logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> - {message}")

HYPERDRIVE_CHILD_ARG_NAME = "random_seed"
YAML_EXTENSION = ".yaml"


def parse_args():
    parser = argparse.ArgumentParser(description='OPA Training')
    parser.add_argument('--cfg', type=str, default='configs/llava/radllava_dpo_sft.yaml')
    # For Training
    parser.add_argument('--entropy_loss', type=str, default="False")
    parser.add_argument('--entropy_mask_ratio', type=float, default=0.5)
    parser.add_argument('--entropy_mask_method', type=str, default='random')
    parser.add_argument('--entropy_loss_coef', type=float, default=1.0)
    parser.add_argument('--entropy_decay_coef', type=float, default=1.0)
    parser.add_argument('--base_model', type=str, default="./base_models/llava-v1.5-7b")
    parser.add_argument('--policy_model_name_or_path', type=str, default='none')
    parser.add_argument('--output_dir', type=str, default="./output/llava7b_opa_model")
    parser.add_argument('--deepspeed', type=str, default='./opadpo/deepspeed_stage_1_config.json')

    parser.add_argument('--full_tune', type=str, default='True')
    parser.add_argument('--tune_mm_mlp_adapter', type=str, default='True')
    parser.add_argument('--tune_base_model', type=str, default='True')
    parser.add_argument('--tune_vision_tower', type=str, default='False')

    parser.add_argument('--lora_tune', type=str, default='False', help='Enable or disable LoRA tuning')
    parser.add_argument('--lora_rank', type=int, default=128, help='Rank parameter for LoRA')
    parser.add_argument('--lora_alpha', type=int, default=256, help='Alpha parameter for LoRA')
    parser.add_argument('--lora_drop', type=float, default=0.0, help='Dropout rate for LoRA')

    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--per_device_train_batch_size', type=int, default=8)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--tf32', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--use_flash_attention', action='store_true')
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=100)

    # For data
    parser.add_argument('--image_folder', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default="./base_datasets/opa_training_data-7B")
    parser.add_argument('--mm_vision_select_layer', type=int, default=-2)
    parser.add_argument('--mm_projector_type', type=str, default='mlp2x_gelu')

    # For env setting.
    parser.add_argument('--local-rank', type=int, default=0)
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
    cfg.llava.checkpoints.base_model = args.base_model

    # override model setting.
    cfg.llava.model.policy_model_name_or_path = args.policy_model_name_or_path
    cfg.llava.model.mm_vision_select_layer = args.mm_vision_select_layer
    cfg.llava.model.mm_projector_type = args.mm_projector_type
    cfg.llava.training.full_tune = True if args.full_tune=="True" else False
    cfg.llava.model.tune_mm_mlp_adapter = True if args.tune_mm_mlp_adapter=="True" else False
    cfg.llava.model.tune_base_model = True if args.tune_base_model=="True" else False
    cfg.llava.model.tune_vision_tower = True if args.tune_vision_tower=="True" else False
    cfg.llava.training.lora_tune = True if args.lora_tune=="True" else False
    cfg.llava.training.lora_r = args.lora_rank
    cfg.llava.training.lora_alpha = args.lora_alpha
    cfg.llava.training.lora_dropout = args.lora_drop

    # override training setting.
    cfg.llava.training.entropy_loss = True if args.entropy_loss=="True" else False
    cfg.llava.training.entropy_mask_ratio = args.entropy_mask_ratio
    cfg.llava.training.entropy_mask_method = args.entropy_mask_method
    cfg.llava.training.entropy_loss_coef = args.entropy_loss_coef
    cfg.llava.training.entropy_decay_coef = args.entropy_decay_coef
    cfg.llava.training.output_dir = args.output_dir
    cfg.llava.training.deepspeed = args.deepspeed
    cfg.llava.training.num_train_epochs = args.num_train_epochs
    cfg.llava.training.per_device_train_batch_size = args.per_device_train_batch_size
    cfg.llava.training.per_device_eval_batch_size = args.per_device_eval_batch_size
    cfg.llava.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    cfg.llava.training.bf16 = True if args.bf16 else False
    cfg.llava.training.tf32 = True if args.tf32 else False
    cfg.llava.training.fp16 = True if args.fp16 else False
    cfg.llava.training.use_flash_attention = True if args.use_flash_attention else False
    cfg.llava.training.save_steps = args.save_steps
    cfg.llava.training.eval_steps = args.eval_steps

    # override data setting.
    cfg.llava.data.image_folder = args.image_folder
    cfg.llava.data.data_dir = args.data_dir

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
        train(amlt_cfg)
    finally:
        pass

if __name__ == '__main__':
    main()
