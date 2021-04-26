import os
import urllib.request
import zipfile

import torch
from torch.utils import data
from torchvision.datasets import ImageFolder

import polycraft_nov_data.data_const as data_const
import polycraft_nov_data.transforms as transforms


def download_datasets():
    for label, data_path in data_const.DATA_PATHS.items():
        zip_path = os.path.join(data_path, label + ".zip")
        # assume data is downloaded if env_0 folder exists
        if not os.path.isdir(os.path.join(data_path, "env_0")):
            # download, extract, and delete zip of the data
            urllib.request.urlretrieve(data_const.DATA_URLS[label], zip_path)
            with zipfile.ZipFile(zip_path) as zip:
                zip.extractall(data_path)
            os.remove(zip_path)


def polycraft_data(batch_size=32, include_classes=None, shuffle=True, all_patches=False):
    download_datasets()
    dataset = ImageFolder(
        data_const.DATASET_ROOT,
        transform=transforms.TestPreprocess() if all_patches else transforms.TrainPreprocess(),
    )
    # override batch_size if using patches
    if all_patches:
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
