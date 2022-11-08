from typing import Callable, Optional

import torch
from torch.utils import data

from polycraft_nov_data.dataset import NovelCraft, EpisodeDataset


def balanced_sampler(train_set, override_len=None):
    # determine class balancing for training
    train_targets = torch.Tensor(train_set.targets)
    train_weight = torch.zeros_like(train_targets)
    for target in torch.unique(train_targets):
        train_weight[train_targets == target] = 1 / torch.sum(train_targets == target)
    if override_len is None:
        num_samples = len(train_set)
    else:
        num_samples = override_len
    return data.WeightedRandomSampler(train_weight, num_samples)


def collate_patches(dataset_entry):
    """Reshape patches produced by ToPatches for DataLoader

    Args:
        dataset_entry (tuple): Tuple containing image tensor produced by ToPatches and target int

    Returns:
        tuple: Tuple containing set of patches with shape (B, C, H, W), where B = PH * PW,
               and target int as a tensor
    """
    data, target = dataset_entry[0]
    shape = data.shape
    if type(target) != str:
        target = torch.tensor(target)
    return (torch.reshape(data, (-1,) + shape[2:]), target)


def default_dataloader_kwargs(
        batch_size: Optional[int] = 1,
        collate_fn=None):
    if batch_size > 1 and collate_fn == collate_patches:
        raise Exception("Attempting to collate patches with batch > 1 will concatenate patch sets")
    num_workers = 4
    prefetch_factor = 1 if batch_size is None else max(batch_size//num_workers, 1)
    dataloader_kwargs = {
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "persistent_workers": True,
        "batch_size": batch_size,
        "collate_fn": collate_fn,
    }
    return dataloader_kwargs


def novelcraft_dataloader(
        split: str,
        transform: Optional[Callable] = None,
        batch_size: Optional[int] = 1,
        balance_classes: Optional[bool] = False,
        collate_fn=None):
    dataset = NovelCraft(split, transform, training_plus=False)
    dataloader_kwargs = default_dataloader_kwargs(batch_size, collate_fn)
    if balance_classes:
        dataloader_kwargs["sampler"] = balanced_sampler(dataset)
    return data.DataLoader(dataset, **dataloader_kwargs)


def novelcraft_plus_dataloader(
        split: str,
        transform: Optional[Callable] = None,
        batch_size: Optional[int] = 1,
        balance_classes: Optional[bool] = False,
        collate_fn=None):
    dataset = NovelCraft(split, transform, training_plus=True)
    dataloader_kwargs = default_dataloader_kwargs(batch_size, collate_fn)
    if balance_classes:
        dataloader_kwargs["sampler"] = balanced_sampler(dataset)
    return data.DataLoader(dataset, **dataloader_kwargs)


def episode_dataloader(
        split: str,
        transform: Optional[Callable] = None,
        batch_size: Optional[int] = 1,
        collate_fn=None):
    dataset = EpisodeDataset(split, transform)
    dataloader_kwargs = default_dataloader_kwargs(batch_size, collate_fn)
    return data.DataLoader(dataset, **dataloader_kwargs)
