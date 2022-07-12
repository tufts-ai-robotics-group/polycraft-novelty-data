from enum import Enum
import importlib.resources
from pathlib import Path


# constants related to data paths
DATA_URL = "https://tufts.box.com/shared/static/76qijr3y1vcawpbyforw4f235to0q679.zip"
with importlib.resources.path("polycraft_nov_data", "dataset") as dataset_root:
    DATASET_ROOT = Path(dataset_root)
# constants for data splits (train, test)


class SplitEnum(str, Enum):
    TRAIN = "train"
    TEST = "test"
    TEST_NORM = "test_norm"
    TEST_NOVEL = "test_novel"
