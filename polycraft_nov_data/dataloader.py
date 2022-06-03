import torch
from torch.utils import data

from polycraft_nov_data.dataset import polycraft_dataset
import polycraft_nov_data.dataset_transforms as dataset_transforms
import polycraft_nov_data.image_transforms as image_transforms


def balanced_sampler(train_set):
    # determine class balancing for training
    train_targets = torch.Tensor([target for _, target in train_set])
    train_weight = torch.zeros_like(train_targets)
    for target in torch.unique(train_targets):
        train_weight[train_targets == target] = 1 / torch.sum(train_targets == target)
    return data.WeightedRandomSampler(train_weight, len(train_set))


def polycraft_dataloaders(batch_size=32, image_scale=1.0, patch=False, include_novel=False,
                          shuffle=True, ret_class_to_idx=False, quad_full_image=False):
    """torch DataLoaders for Polycraft datasets
    Args:
        batch_size (int, optional): batch_size for DataLoaders. Defaults to 32.
        image_scale (float, optional): Scaling applied to images. Defaults to 1.0.
        patch (bool, optional): Whether to get image patches or full images. Defaults to False.
        include_novel (bool, optional): Whether to include novelties in non-train sets.
                                        Defaults to False.
        shuffle (bool, optional): shuffle for DataLoaders. Defaults to True.
        ret_class_to_idx (bool, optional): Whether to return class_to_idx. Defaults to False.
    Returns:
        (DataLoader, DataLoader, DataLoader): Polycraft train, validation, and test sets.
                                              Contains batches of (3, 32, 32) images,
                                              with values 0-1.
        dict: class_to_idx if ret_class_to_idx is True.
    """
    # if using patches, override batch dim to hold the set of patches
    collate_fn = None
    if not patch:
        transform = image_transforms.VGGPreprocess(image_scale)
    else:
        if not include_novel:
            transform = image_transforms.TrainPreprocess(image_scale)
        else:
            batch_size = None
            collate_fn = dataset_transforms.collate_patches
            transform = image_transforms.TestPreprocess(image_scale)
    if quad_full_image:
        transform = image_transforms.PreprocessFullQuadraticImage(image_scale)
            
    # get the dataset
    dataset = polycraft_dataset(transform)
    # split into datasets
    train_set, valid_set, test_set = dataset_transforms.filter_ep_split(dataset, include_novel)
    # get DataLoaders for datasets
    num_workers = 4
    prefetch_factor = 1 if batch_size is None else max(batch_size//num_workers, 1)
    dataloader_kwargs = {
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "persistent_workers": True,
        "batch_size": batch_size,
        "collate_fn": collate_fn,
    }
    dataloaders = (data.DataLoader(train_set, sampler=balanced_sampler(train_set),
                                   **dataloader_kwargs),
                   data.DataLoader(valid_set, shuffle=shuffle, **dataloader_kwargs),
                   data.DataLoader(test_set, shuffle=shuffle, **dataloader_kwargs))
    if not ret_class_to_idx:
        return dataloaders
    else:
        return (dataloaders, dataset.class_to_idx)


def polycraft_dataloaders_gcd(transform, batch_size=32):
    # get the dataset
    dataset = polycraft_dataset(transform)
    # split into datasets
    labeled_set, unlabeled_set, _ = dataset_transforms.filter_ep_split(dataset, True)
    # get DataLoaders for datasets
    num_workers = 4
    prefetch_factor = 1 if batch_size is None else max(batch_size//num_workers, 1)
    dataloader_kwargs = {
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "persistent_workers": True,
        "batch_size": batch_size,
    }
    labeled_loader = data.DataLoader(labeled_set, sampler=balanced_sampler(labeled_set),
                                     **dataloader_kwargs)
    unlabeled_loader = data.DataLoader(unlabeled_set, shuffle=True, **dataloader_kwargs)
    return labeled_loader, unlabeled_loader