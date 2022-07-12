from enum import Enum
import importlib.resources
from pathlib import Path


# constants related to data paths
DATA_URL = "https://tufts.box.com/shared/static/76qijr3y1vcawpbyforw4f235to0q679.zip"
with importlib.resources.path("polycraft_nov_data", "dataset") as dataset_root:
    DATASET_ROOT = Path(dataset_root)
# constants related to shape of data
IMAGE_SHAPE = (3, 256, 256)
PATCH_SHAPE = (3, 32, 32)
# constants for data splits (train, test)


class SplitEnum(str, Enum):
    TRAIN = "train"
    TEST = "test"


# constants related to classes
NORMAL_CLASSES = [
    "normal",
]
TEST_CLASSES = [
    "ArenaBlockTypeHard",
    "fence",
    "tree_easy",
]
ALL_CLASSES = NORMAL_CLASSES + TEST_CLASSES
ALL_CLASS_TO_IDX = {c: i for i, c in enumerate(ALL_CLASSES)}
ALL_IDX_TO_CLASS = {i: c for i, c in enumerate(ALL_CLASSES)}
