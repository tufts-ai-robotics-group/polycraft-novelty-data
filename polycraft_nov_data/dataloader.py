from pathlib import Path
import shutil
import urllib.request
import csv

from torch.utils import data
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

import polycraft_nov_data.data_const as data_const
import polycraft_nov_data.dataset_transforms as dataset_transforms
import polycraft_nov_data.image_transforms as image_transforms


class TrippleDataset(Dataset):
    def __init__(self, datasetA, datasetB, datasetC):
        self.datasetA = datasetA
        self.datasetB = datasetB
        self.datasetC = datasetC
        
    def __getitem__(self, index):
        xA = self.datasetA[index]
        xB = self.datasetB[index]
        xC = self.datasetC[index]
        return xA, xB, xC
    
    def __len__(self):
        return len(self.datasetA)


def download_datasets():
    """Download Polycraft datasets if not downloaded
    """
    for label, data_path in data_const.DATA_PATHS.items():
        # assume data is downloaded if env_0 folder exists
        if not (data_path / Path("env_0")).is_dir():
            # download, extract, and delete zip of the data
            zip_path = data_path / Path(label + ".zip")
            urllib.request.urlretrieve(data_const.DATA_URLS[label], zip_path)
            shutil.unpack_archive(zip_path, data_path)
            zip_path.unlink()


def polycraft_dataset(transform=None):
    download_datasets()
    return ImageFolder(data_const.DATASET_ROOT, transform=transform)


def polycraft_dataloaders(batch_size=32, image_scale=1.0, include_novel=False, shuffle=True):
    """torch DataLoaders for Polycraft datasets

    Args:
        batch_size (int, optional): batch_size for DataLoaders. Defaults to 32.
        image_scale (float, optional): Scaling applied to images. Defaults to 1.0.
        include_novel (bool, optional): Whether to include novelties in non-train sets.
                                        Defaults to False.
        shuffle (bool, optional): shuffle for DataLoaders. Defaults to True.

    Returns:
        (DataLoader, DataLoader, DataLoader): Polycraft train, validation, and test sets.
                                              Contains batches of (3, 32, 32) images,
                                              with values 0-1.
    """
    # if using patches, override batch dim to hold the set of patches
    class_splits = {"normal": [.7, .15, .15]}
    if not include_novel:
        collate_fn = None
        transform = image_transforms.TrainPreprocess(image_scale)
    else:
        class_splits.update({"height": [0, .5, .5],
                             "item": [0, .5, .5]})
        batch_size = None
        collate_fn = dataset_transforms.collate_patches
        transform = image_transforms.TestPreprocess(image_scale)
    # get the dataset
    dataset = polycraft_dataset(transform)
    # update class_splits to use indices instead of names
    class_splits = dataset_transforms.folder_name_to_target_key(dataset, class_splits)
    # split into datasets
    train_set, valid_set, test_set = dataset_transforms.filter_split(dataset, class_splits)
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
            
def polycraft_dataset_for_ms(batch_size=32, image_scale=1.0, patch_shape=(3, 32, 32), include_novel=False, shuffle=True):
    """torch DataLoaders for Polycraft datasets

    Args:
        batch_size (int, optional): batch_size for DataLoaders. Defaults to 32.
        image_scale (float, optional): Scaling applied to images. Defaults to 1.0.
        include_novel (bool, optional): Whether to include novelties in non-train sets.
                                        Defaults to False.
        shuffle (bool, optional): shuffle for DataLoaders. Defaults to True.

    Returns:
        (DataLoader, DataLoader, DataLoader): Polycraft train, validation, and test sets.
                                              Contains batches of (3, 32, 32) images,
                                              with values 0-1.
    """
    # if using patches, override batch dim to hold the set of patches
    class_splits = {"normal": [.7, .15, .15]}
    if not include_novel:
        collate_fn = None
        transform = image_transforms.TrainPreprocess(image_scale)
    else:
        class_splits.update({"height": [0, .5, .5],
                             "item": [0, .5, .5]})
        batch_size = None
        collate_fn = dataset_transforms.collate_patches
        transform = image_transforms.TestPreprocess(image_scale, patch_shape)
    # get the dataset
    dataset = polycraft_dataset(transform)
    # update class_splits to use indices instead of names
    class_splits = dataset_transforms.folder_name_to_target_key(dataset, class_splits)
    # split into datasets
    train_set, valid_set, test_set = dataset_transforms.filter_split(dataset, class_splits)
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
    return (train_set, valid_set, test_set)

