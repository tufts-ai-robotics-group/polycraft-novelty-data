import importlib.resources
from pathlib import Path


# constants related to data labels and locations
DATA_LABELS = [
    "normal",
    "fence_hard",
]
DATA_URLS = {
    "normal": "https://tufts.box.com/shared/static/9okzlpf6eye03d4ycwhvvpovbjfd7x7z.zip",
    "fence_hard": "https://tufts.box.com/shared/static/zly2idcic2t3oc24kaslqzk7asz4gdot.zip",
}
# class split constants
NORMAL_CLASSES = DATA_LABELS[:1]
NOVEL_CLASSES = DATA_LABELS[1:]
# path constants
with importlib.resources.path("polycraft_nov_data", "dataset") as dataset_root:
    DATASET_ROOT = Path(dataset_root)
DATA_PATHS = {label: DATASET_ROOT / Path(label) for label in DATA_LABELS}
# constants related to shape of data
IMAGE_SHAPE = (3, 256, 256)
PATCH_SHAPE = (3, 32, 32)
