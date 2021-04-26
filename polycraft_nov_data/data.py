import importlib.resources
import os
import urllib.request
import zipfile

import torch
from torch.utils import data
from torchvision.datasets import ImageFolder

import polycraft_nov_data.transforms as transforms


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


def polycraft_data(batch_size=32, include_classes=None, shuffle=True, patches=False):
    dataset = ImageFolder(
        DATASET_ROOT,
        transform=transforms.ToPatches() if patches else transforms.SamplePatch(),
    )
    # override batch_size if using patches
    if patches:
        batch_size = 1
    # split into datasets
    split_len = len(dataset) // 10
    train_set, valid_set, test_set = data.random_split(
        dataset,
        [len(dataset) - 2 * split_len, split_len, split_len],
        generator=torch.Generator().manual_seed(42),
    )
    # get DataLoaders for datasets
    return (data.DataLoader(train_set, batch_size, shuffle),
            data.DataLoader(valid_set, batch_size, shuffle),
            data.DataLoader(test_set, batch_size, shuffle))
