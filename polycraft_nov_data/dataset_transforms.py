import math
import numpy as np

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


def filter_split(dataset, split_percents, include_classes=None):
    """Split dataset with consistent and even sets per class.

    Args:
        dataset (data.Dataset): Dataset to filter and split
        split_percents (iterable): Fraction of dataset to use for each split. Sum should be 1.
        include_classes (iterable, optional): Classes from dataset.targets to include.
                                              Defaults to None, not filtering any classes.

    Returns:
        iterable: Iterable of Datasets with only classes from include_classes
    """
    if include_classes is None:
        include_classes = torch.unique(targets_tensor(dataset))
    target_datasets = [filter_dataset(dataset, [target]) for target in include_classes]
    # raise exception if splits do not sum to 1
    if sum(split_percents) != 1:
        raise Exception("Split percents should sum to 1, instead got percents " +
                        str(split_percents))
    # create list with empty list for each split
    dataset_splits = [[] for _ in range(len(split_percents))]
    for target_dataset in target_datasets:
        # get lengths with the first entry resolving rounding errors
        lengths = [math.floor(len(target_dataset) * percent) for percent in split_percents]
        lengths[0] += len(target_dataset) - sum(lengths)
        # split each target
        splits = data.random_split(target_dataset, lengths,
                                   generator=torch.Generator().manual_seed(42))
        # put splits with same percentage into a list
        for i, split in enumerate(splits):
            dataset_splits[i].append(split)
    return [data.ConcatDataset(dataset_split) for dataset_split in dataset_splits]


def filter_split_csv_idcs(dataset, split_percents):
    """Split polycraft dataset according to a pre-shuffled random order of
    indices, the indices are loaded from the csv file.

    Args:
        dataset (data.Dataset): Dataset to filter and split
        split_percents (iterable): Fraction of dataset to use for each split. Sum should be 1.
        
    Returns:
        train-, val-, test_dataset
    """
    # csv file contains randomly ordered indices from 1 to 4500 
    idcs = np.loadtxt('shuffled_idcs.csv', delimiter=',')
    idcs = idcs.astype(int)
    
    train_size = int(split_percents[0] * len(idcs))
    valid_size = int(split_percents[1] * len(idcs))
    
    train_idcs = idcs[0:train_size]
    valid_idcs = idcs[train_size:(train_size + valid_size)]
    test_idcs = idcs[(train_size + valid_size):len(idcs)]
    
    # Split whole dataset in subsets according to their index
    train_dataset = torch.utils.data.Subset(dataset, train_idcs)
    val_dataset = torch.utils.data.Subset(dataset, valid_idcs)
    test_dataset = torch.utils.data.Subset(dataset, test_idcs)
    
    return train_dataset, val_dataset, test_dataset
    
    
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
