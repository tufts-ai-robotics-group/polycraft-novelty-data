import torch
from torch.utils import data

import polycraft_nov_data.data_const as data_const
from polycraft_nov_data.dataset import polycraft_dataset
import polycraft_nov_data.dataset_transforms as dataset_transforms
import polycraft_nov_data.image_transforms as image_transforms


def balanced_sampler(train_set):
    # determine class balancing for training
    train_targets = torch.Tensor([target for _, target in train_set])
    train_weight = torch.zeros_like(train_targets, dtype=float)
    for target in torch.unique(train_targets):
        train_weight[train_targets == target] = 1 / torch.sum(train_targets == target)
    return data.WeightedRandomSampler(train_weight)


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
    class_splits = {c: [.8, .1, .1] for c in data_const.NORMAL_CLASSES}
    if not include_novel:
        collate_fn = None
        transform = image_transforms.TrainPreprocess(image_scale)
    else:
        class_splits.update({c: [0, 1, 0] for c in data_const.NOVEL_VALID_CLASSES})
        class_splits.update({c: [0, 0, 1] for c in data_const.NOVEL_TEST_CLASSES})
        batch_size = None
        collate_fn = dataset_transforms.collate_patches
        transform = image_transforms.TestPreprocess(image_scale)
    # get the dataset
    dataset = polycraft_dataset(transform)
    # update class_splits to use indices instead of names
    class_splits = dataset_transforms.folder_name_to_target_key(dataset, class_splits)
    # split into datasets
    train_set, valid_set, test_set = dataset_transforms.filter_ep_split(dataset, class_splits)
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
    return (data.DataLoader(train_set, sampler=balanced_sampler(train_set), **dataloader_kwargs),
            data.DataLoader(valid_set, **dataloader_kwargs),
            data.DataLoader(test_set, **dataloader_kwargs))


def polycraft_dataloaders_full_image(batch_size=32, image_scale=1.0, include_novel=False,
                                     shuffle=True):
    """torch DataLoaders for Polycraft datasets
    Args:
        batch_size (int, optional): batch_size for DataLoaders. Defaults to 32.
        image_scale (float, optional): Scaling applied to images. Defaults to 1.0.
        include_novel (bool, optional): Whether to include novelties in non-train sets.
                                        Defaults to False.
        shuffle (bool, optional): shuffle for DataLoaders. Defaults to True.
    Returns:
        (DataLoader, DataLoader, DataLoader): Polycraft train, validation, and test sets.
                                              Contains batches of (image_scale * height)
                                              x (image_scale * width) images,
                                              with values normalized according to vgg16
                                              pre-training.
    """
    # if using patches, override batch dim to hold the set of patches
    class_splits = {c: [.8, .1, .1] for c in data_const.NORMAL_CLASSES}
    transform = image_transforms.VGGPreprocess(image_scale)
    if include_novel:
        class_splits.update({c: [0, 1, 0] for c in data_const.NOVEL_VALID_CLASSES})
        class_splits.update({c: [0, 0, 1] for c in data_const.NOVEL_TEST_CLASSES})
    # get the dataset
    dataset = polycraft_dataset(transform)
    # update class_splits to use indices instead of names
    class_splits = dataset_transforms.folder_name_to_target_key(dataset, class_splits)
    # split into datasets
    train_set, valid_set, test_set = dataset_transforms.filter_ep_split(dataset, class_splits)
    # get DataLoaders for datasets
    num_workers = 4
    prefetch_factor = 1 if batch_size is None else max(batch_size//num_workers, 1)
    dataloader_kwargs = {
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "persistent_workers": True,
        "batch_size": batch_size,
        "shuffle": shuffle,
    }
    return (data.DataLoader(train_set, sampler=balanced_sampler(train_set), **dataloader_kwargs),
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
    train_set, valid_set, test_set = dataset_transforms.filter_ep_split(dataset, class_splits)
    # get DataLoaders for datasets
    return (train_set, valid_set, test_set)
