import math
from pathlib import Path

import pandas as pd
import torch
from torch.utils import data

import polycraft_nov_data.data_const as data_const


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


def folder_name_to_target_list(dataset, class_list):
    """Converts image folder name as list entry to dataset target

    Args:
        dataset (torchvision.datasets.ImageFolder): Dataset with folder names.
        class_list (list): List with image folder names as entries.

    Returns:
        list: List with targets as entries.
    """
    return [dataset.class_to_idx[c] for c in class_list]


def folder_name_to_target_key(dataset, class_dict):
    """Converts image folder name as dict key to dataset target

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
        data_include = torch.any(torch.stack([
                dataset_targets == dataset.class_to_idx[target] for target in include_classes]),
            dim=0)
        dataset = data.Subset(dataset, torch.nonzero(data_include)[:, 0])
    return dataset


def ep_split(dataset):
    episode_labels = torch.Tensor(
        [int(Path(dataset.dataset.imgs[idx][0]).parent.stem) for idx in dataset.indices]).int()
    splits = []
    labels = []
    for label in torch.unique(episode_labels):
        splits += [data.Subset(dataset, torch.nonzero(episode_labels == label)[:, 0])]
        labels += [str(label.item())]
    return splits, labels


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


def filter_ep_split(dataset, include_novel):
    """Split dataset by episodes with different split per class

    Args:
        dataset (data.Dataset): Dataset to filter and split
        include_novel (bool, optional): Whether to include novelties in non-train sets.
                                        Defaults to False.

    Raises:
        Exception: Split percents should sum to <= 1

    Returns:
        iterable: Iterable of Datasets with desired splits
    """
    include_classes = data_const.NORMAL_CLASSES.copy()
    if include_novel:
        include_classes += data_const.NOVEL_VALID_CLASSES + data_const.NOVEL_TEST_CLASSES
    target_datasets = [filter_dataset(dataset, [target]) for target in include_classes]
    # create list with empty list for each split
    dataset_splits = [[], [], []]
    split_df = pd.read_csv(data_const.DATASET_SPLITS)
    for i in range(len(include_classes)):
        target_dataset = target_datasets[i]
        if include_classes[i] in data_const.NOVEL_VALID_CLASSES:
            dataset_splits[1].append(target_dataset)
            continue
        elif include_classes[i] in data_const.NOVEL_TEST_CLASSES:
            dataset_splits[2].append(target_dataset)
            continue
        target_dataset_by_ep, ep_labels = ep_split(target_dataset)
        # put episodes into splits according to CSV
        for j in range(len(ep_labels)):
            split_label = split_df[
                split_df.episode == include_classes[i] + "/" + ep_labels[j]].split
            split_label = split_label.item()
            if split_label == "train":
                dataset_splits[0].append(target_dataset_by_ep[j])
            elif split_label == "valid":
                dataset_splits[1].append(target_dataset_by_ep[j])
            elif split_label == "test":
                dataset_splits[2].append(target_dataset_by_ep[j])
    return [data.ConcatDataset(dataset_split) for dataset_split in dataset_splits]


def collate_patches(dataset_entry):
    """Reshape patches produced by ToPatches for DataLoader

    Args:
        dataset_entry (tuple): Tuple containing image tensor produced by ToPatches and target int

    Returns:
        tuple: Tuple containing set of patches with shape (B, C, H, W), where B = PH * PW,
               and target int as a tensor
    """
    data, target = dataset_entry
    shape = data.shape
    return (torch.reshape(data, (-1,) + shape[2:]), torch.tensor(target))
