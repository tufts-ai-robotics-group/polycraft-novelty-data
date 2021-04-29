import math

import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional

import polycraft_nov_data.data_const as data_const


class CropUI:
    def __call__(self, tensor):
        """Crop the UI by removing bottom row of patches

        Args:
            tensor (torch.tensor): Image tensor to remove Polycraft UI from

        Returns:
            torch.tensor: Cropped image tensor
        """
        _, h, w = data_const.IMAGE_SHAPE
        _, patch_h, _ = data_const.PATCH_SHAPE
        return functional.crop(tensor, 0, 0, h - patch_h, w)


class SamplePatch:
    def __call__(self, tensor):
        """Sample a random patch from the set produced by ToPatches

        Args:
            tensor (torch.tensor): Image tensor to sample a patch from

        Returns:
            torch.tensor: Patch image tensor
        """
        patches = ToPatches()(tensor)
        patch_set_h, patch_set_w, _, _, _ = patches.shape
        return patches[torch.randint(patch_set_h, ()), torch.randint(patch_set_w, ())]


class ToPatches:
    def __call__(self, tensor):
        """Divide image into set of patches

        Args:
            tensor (torch.tensor): Image tensor to divide into patches

        Returns:
            torch.tensor: Set of patch image tensors, shape (PH, PW, C, H, W)
                          where PH and PW are number of patches vertically/horizontally
        """
        _, patch_h, patch_w = data_const.PATCH_SHAPE
        patches = tensor.unfold(1, patch_h, patch_h//2).unfold(2, patch_w, patch_w//2)
        return patches.permute(1, 2, 0, 3, 4)


class TrainPreprocess:
    def __init__(self, image_scale=1.0):
        """Image preprocessing for training

        Args:
            image_scale (float, optional): Scaling to apply to image. Defaults to 1.0.
        """
        _, h, w = data_const.IMAGE_SHAPE
        resize_h = int(h * image_scale)
        resize_w = int(w * image_scale)
        self.preprocess = transforms.Compose([
            transforms.Resize((resize_h, resize_w)),
            CropUI(),
            transforms.ToTensor(),
            SamplePatch(),
        ])

    def __call__(self, tensor):
        return self.preprocess(tensor)


class TestPreprocess:
    def __init__(self, image_scale=1.0):
        """Image preprocessing for testing

        Args:
            image_scale (float, optional): Scaling to apply to image. Defaults to 1.0.
        """
        _, h, w = data_const.IMAGE_SHAPE
        resize_h = int(h * image_scale)
        resize_w = int(w * image_scale)
        self.preprocess = transforms.Compose([
            transforms.Resize((resize_h, resize_w)),
            CropUI(),
            transforms.ToTensor(),
            ToPatches(),
        ])

    def __call__(self, tensor):
        return self.preprocess(tensor)


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
        dataset_targets = torch.Tensor(dataset.targets)
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
        include_classes = torch.unique(torch.Tensor(dataset.targets))
    target_datasets = [filter_dataset(dataset, [target]) for target in include_classes]
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
