import os
import shutil
import urllib.request

from torch.utils import data
from torchvision.datasets import ImageFolder

import polycraft_nov_data.data_const as data_const
import polycraft_nov_data.dataset_transforms as dataset_transforms
import polycraft_nov_data.image_transforms as image_transforms


def download_datasets():
    """Download Polycraft datasets if not downloaded
    """
    for label, data_path in data_const.DATA_PATHS.items():
        zip_path = os.path.join(data_path, label + ".zip")
        # assume data is downloaded if env_0 folder exists
        if not os.path.isdir(os.path.join(data_path, "env_0")):
            # download, extract, and delete zip of the data
            urllib.request.urlretrieve(data_const.DATA_URLS[label], zip_path)
            shutil.unpack_archive(zip_path, data_path)
            os.remove(zip_path)


def polycraft_dataset(transform=None):
    download_datasets()
    return ImageFolder(data_const.DATASET_ROOT, transform=transform)


def polycraft_dataloaders(batch_size=32, include_classes=None, image_scale=1.0, shuffle=True,
                          all_patches=False):
    """torch DataLoaders for Polycraft datasets

    Args:
        batch_size (int, optional): batch_size for DataLoaders. Defaults to 32.
        include_classes (list, optional): List of classes to include.
                                          Defaults to None, including all classes.
        image_scale (float, optional): Scaling applied to images. Defaults to 1.0.
        shuffle (bool, optional): shuffle for DataLoaders. Defaults to True.
        all_patches (bool, optional): Whether to replace batches with all patches from an image.
                                      Defaults to False.

    Returns:
        (DataLoader, DataLoader, DataLoader): Polycraft train, validation, and test sets.
                                              Contains batches of (3, 32, 32) images,
                                              with values 0-1.
    """
    # if using patches, override batch dim to hold the set of patches
    if not all_patches:
        collate_fn = None
        transform = image_transforms.TrainPreprocess(image_scale)
    else:
        batch_size = None
        collate_fn = dataset_transforms.collate_patches
        transform = image_transforms.TestPreprocess(image_scale)
    # get the dataset
    dataset = polycraft_dataset(transform)
    # split into datasets
    train_set, valid_set, test_set = dataset_transforms.filter_split(
        dataset, [.7, .15, .15], include_classes
    )
    # get DataLoaders for datasets
    num_workers = 4
    prefetch_factor = 1 if batch_size is None else max(batch_size//num_workers, 1)
    dataloader_kwargs = {
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "persistent_workers": True,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_fn,
    }
    return (data.DataLoader(train_set, **dataloader_kwargs),
            data.DataLoader(valid_set, **dataloader_kwargs),
            data.DataLoader(test_set, **dataloader_kwargs))
