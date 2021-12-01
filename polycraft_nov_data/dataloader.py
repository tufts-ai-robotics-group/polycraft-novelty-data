from pathlib import Path
import shutil
import urllib.request

from torch.utils import data
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

import polycraft_nov_data.data_const as data_const
import polycraft_nov_data.dataset_transforms as dataset_transforms
import polycraft_nov_data.image_transforms as image_transforms


class TrippleDataset(Dataset):
    """Combine three datasets (we have one for each scale)
    """
    def __init__(self, dataset1, dataset2, dataset3):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3

    def __getitem__(self, index):
        return self.dataset1[index], self.dataset2[index], self.dataset3[index]

    def __len__(self):
        return len(self.dataset1)


class QuattroDataset(Dataset):
    """Combine four datasets (we have one for scale 0.5 and scale 0.75 and two
       for scale 1 (32x32 patch and 16x16 patch)
    """
    def __init__(self, dataset1, dataset2, dataset3, dataset4):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3
        self.dataset4 = dataset4

    def __getitem__(self, index):
        return (
            self.dataset1[index],
            self.dataset2[index],
            self.dataset3[index],
            self.dataset4[index]
        )

    def __len__(self):
        return len(self.dataset1)


def download_datasets():
    """Download Polycraft datasets if not downloaded
    """
    for label, data_path in data_const.DATA_PATHS.items():
        # assume data is downloaded if folder contains files or dirs
        if not any(data_path.iterdir()):
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
    class_splits = {c: [.7, .15, .15] for c in data_const.NORMAL_CLASSES}
    if not include_novel:
        collate_fn = None
        transform = image_transforms.TrainPreprocess(image_scale)
    else:
        class_splits.update({c: [0, .5, .5] for c in data_const.NOVEL_CLASSES})
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


def polycraft_dataset_for_ms(batch_size=32, image_scale=1.0, patch_shape=(3, 32, 32),
                             include_novel=False, shuffle=True):
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
        transform = image_transforms.TrainPreprocess(image_scale)
    else:
        class_splits.update({"height": [0, .5, .5],
                             "item": [0, .5, .5]})
        transform = image_transforms.TestPreprocess(image_scale, patch_shape)
    # get the dataset
    dataset = polycraft_dataset(transform)
    # update class_splits to use indices instead of names
    class_splits = dataset_transforms.folder_name_to_target_key(dataset, class_splits)
    # split into datasets
    train_set, valid_set, test_set = dataset_transforms.filter_split(dataset, class_splits)
    # get DataLoaders for datasets
    return (train_set, valid_set, test_set)
