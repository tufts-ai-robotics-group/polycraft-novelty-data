import importlib.resources
import os
import urllib.request
import zipfile


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


def download_datasets():
    for label, data_path in DATA_PATHS.items():
        zip_path = os.path.join(data_path, label + ".zip")
        # assume data is downloaded if env_0 folder exists
        if not os.path.isdir(os.path.join(data_path, "env_0")):
            # download, extract, and delete zip of the data
            urllib.request.urlretrieve(DATA_URLS[label], zip_path)
            with zipfile.ZipFile(zip_path) as zip:
                zip.extractall(data_path)
            os.remove(zip_path)
