import torch
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
        resize_h = int(h * .5)
        resize_w = int(w * .5)
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
        resize_h = int(h * .5)
        resize_w = int(w * .5)
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
