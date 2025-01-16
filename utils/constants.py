from enum import Enum

FACTUAL_PROMPT = "Specifically, the AI's response should be fully supported by the combination of the following captions:\n"

class AnswerType(Enum):
    GENERAL = 1
    A_IN_ABCD = 2
    B_IN_ABCD = 3
    C_IN_ABCD = 4
    D_IN_ABCD = 5
    NO_IN_YESNO = 6
    YES_IN_YESNO = 7

from pathlib import Path
import os

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

REPOSITORY_ROOT_DIR = Path(__file__).absolute().parents[1]
LLAVA_CACHE_DIR = REPOSITORY_ROOT_DIR / "llava"
CONFIGS_DIR = REPOSITORY_ROOT_DIR / "configs"

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

LLAVA_INITIAL_CHECKPOINT_NAME = "initial_checkpoint"