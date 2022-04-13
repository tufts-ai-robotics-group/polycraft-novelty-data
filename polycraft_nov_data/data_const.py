import importlib.resources
from pathlib import Path


# constants related to data paths
DATA_URL = "https://tufts.box.com/shared/static/fq0awbrahmsr97zetqo1v2uz5rjkvon6.zip"
with importlib.resources.path("polycraft_nov_data", "dataset") as dataset_root:
    DATASET_ROOT = Path(dataset_root)
DATASET_TARGETS = DATASET_ROOT / Path("targets.csv")
# constants related to shape of data
IMAGE_SHAPE = (3, 256, 256)
PATCH_SHAPE = (3, 32, 32)
