import math

import torch
from torch.utils import data


def targets_tensor(dataset):
    """Gets dataset targets as a tensor, handling varying types

    Args:
        dataset (data.Dataset): Dataset to get targets from

    Returns:
        torch.tensor: Tensor of the targets
    """
    dataset_targets = dataset.targets
    if type(dataset_targets) is list:
        dataset_targets = torch.tensor(dataset_targets)
    return dataset_targets


def folder_name_to_target(dataset, class_dict):
    """Converts image folder name to dataset target

    Args:
        dataset (torchvision.datasets.ImageFolder): Dataset with folder names.
        class_dict (dict): Dict with image folder names as keys.

    Returns:
        dict: Dict with targets as keys.
    """
    return {dataset.class_to_idx[key]: val for key, val in class_dict.items()}


def filter_dataset(dataset, include_classes=None):
    """Filter dataset to include only specified classes

    Args:
        dataset (data.Dataset): Dataset to filter
        include_classes (iterable, optional): Classes from dataset.targets to include.
                                              Defaults to None, not filtering any classes.

    Returns:
        data.Dataset: Dataset with only classes from include_classes
    """
    # select only included classes
    if include_classes is not None:
        dataset_targets = targets_tensor(dataset)
        data_include = torch.any(torch.stack([dataset_targets == target
                                              for target in include_classes]),
                                 dim=0)
        dataset = data.Subset(dataset, torch.nonzero(data_include)[:, 0])
    return dataset


def filter_split(dataset, class_splits):
    """Split dataset with different split per class

    Args:
        dataset (data.Dataset): Dataset to filter and split
        class_splits (dict): Dict mapping class to iterable summing to <= 1

    Raises:
        Exception: Split percents should sum to <= 1

    Returns:
        iterable: Iterable of Datasets with desired splits
    """
    include_classes = list(class_splits.keys())
    target_datasets = [filter_dataset(dataset, [target]) for target in include_classes]
    # create list with empty list for each split
    dataset_splits = [[] for _ in range(len(class_splits[include_classes[0]]))]
    for i in range(len(include_classes)):
        target_dataset = target_datasets[i]
        split_percents = class_splits[include_classes[i]]
        # raise exception if splits do not sum to 1
        if sum(split_percents) > 1:
            raise Exception("Split percents should sum to <= 1, instead got percents " +
                            str(split_percents))
        # get lengths with the first non-zero entry resolving rounding errors
        lengths = [math.floor(len(target_dataset) * percent) for percent in split_percents]
        first_non_zero = 0
        for j, length in enumerate(lengths):
            if length > 0:
                first_non_zero = j
                break
        lengths[first_non_zero] += len(target_dataset) - sum(lengths)
        # split each target
        splits = data.random_split(target_dataset, lengths,
                                   generator=torch.Generator().manual_seed(42))
        # put splits with same percentage into a list
        for j, split in enumerate(splits):
            dataset_splits[j].append(split)
    return [data.ConcatDataset(dataset_split) for dataset_split in dataset_splits]


def collate_patches(dataset_entry):
    """Reshape patches produced by ToPatches for DataLoader

    Args:
        dataset_entry (tuple): Tuple containing image tensor produced by ToPatches and target int

    Returns:
        tuple: Tuple containing set of patches with shape (B, C, H, W), where B = PH * PW,
               and target int
    """
    data, target = dataset_entry
    shape = data.shape
    return (torch.reshape(data, (-1,) + shape[2:]), target)
