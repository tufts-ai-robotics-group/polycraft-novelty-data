import importlib.resources
import os


# constants related to data labels and locations
DATA_LABELS = [
    "normal",
    "item",
    "height",
]
DATA_URLS = {
    "normal": "https://tufts.box.com/shared/static/7jrtn4tssu9palz3x13b6fq9jkleejbk.zip",
    "item": "https://tufts.box.com/shared/static/p3kpy6njrlx6nvg3gh4t1klt5d3sjdp9.zip",
    "height": "https://tufts.box.com/shared/static/3yfjmkm79yq2rl60kbhnq5eey530iqcv.zip",
}

with importlib.resources.path("polycraft_nov_data", "dataset") as dataset_root:
    DATASET_ROOT = dataset_root
DATA_PATHS = {label: os.path.join(DATASET_ROOT, label) for label in DATA_LABELS}
# constants related to shape of data
IMAGE_SHAPE = (3, 256, 256)
PATCH_SHAPE = (3, 32, 32)
