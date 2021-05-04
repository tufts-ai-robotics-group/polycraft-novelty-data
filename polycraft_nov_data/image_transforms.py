import torch
from torchvision import transforms
from torchvision.transforms import functional

import polycraft_nov_data.data_const as data_const


class ScaleImage:
    def __init__(self, image_scale=1.0):
        """Image scaling by multiplicative factor

        Args:
            image_scale (float, optional): Scaling to apply to image. Defaults to 1.0.
        """
        self.image_scale = image_scale

    def __call__(self, tensor):
        _, h, w = tensor.shape
        resize_h = int(h * self.image_scale)
        resize_w = int(w * self.image_scale)
        return functional.resize(tensor, (resize_h, resize_w))


class CropUI:
    def __call__(self, tensor):
        """Crop the UI by removing bottom row of patches

        Args:
            tensor (torch.tensor): Image tensor to remove Polycraft UI from

        Returns:
            torch.tensor: Cropped image tensor
        """
        _, h, w = tensor.shape
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
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            CropUI(),
            ScaleImage(image_scale),
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
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            CropUI(),
            ScaleImage(image_scale),
            ToPatches(),
        ])

    def __call__(self, tensor):
        return self.preprocess(tensor)


class GaussianNoise:
    """Dataset transform to apply Gaussian Noise to normalized data
    """
    def __init__(self, std=1/40):
        """Dataset transform to apply Gaussian Noise to normalized data

        Args:
            std (float, optional): STD of noise. Defaults to 1/40.
        """
        self.std = std

    def __call__(self, tensor):
        out = tensor + torch.randn_like(tensor) * self.std
        out[out < 0] = 0
        out[out > 1] = 1
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(std=%f)' % (self.std,)
