from typing import Callable, Optional

import torch
from torch.utils import data

from polycraft_nov_data.dataset import NovelCraft, EpisodeDataset


def balanced_sampler(train_set):
    # determine class balancing for training
    train_targets = torch.Tensor([target for _, target in train_set])
    train_weight = torch.zeros_like(train_targets)
    for target in torch.unique(train_targets):
        train_weight[train_targets == target] = 1 / torch.sum(train_targets == target)
    return data.WeightedRandomSampler(train_weight, len(train_set))


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


def novelcraft_dataloader(
        split: str,
        transform: Optional[Callable] = None,
        batch_size: Optional[int] = 1,
        balance_classes: Optional[bool] = False,
        collate_fn=None):
    dataset = NovelCraft(split, transform)
    # DataLoader args
    num_workers = 4
    prefetch_factor = 1 if batch_size is None else max(batch_size//num_workers, 1)
    dataloader_kwargs = {
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "persistent_workers": True,
        "batch_size": batch_size,
        "collate_fn": collate_fn,
    }
    if balance_classes:
        dataloader_kwargs["sampler"] = balanced_sampler
    return data.DataLoader(dataset, **dataloader_kwargs)


def episode_dataloader(
        split: str,
        transform: Optional[Callable] = None,
        batch_size: Optional[int] = 1):
    dataset = EpisodeDataset(split, transform)
    # DataLoader args
    num_workers = 4
    prefetch_factor = 1 if batch_size is None else max(batch_size//num_workers, 1)
    dataloader_kwargs = {
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "persistent_workers": True,
        "batch_size": batch_size,
    }
    return data.DataLoader(dataset, **dataloader_kwargs)
