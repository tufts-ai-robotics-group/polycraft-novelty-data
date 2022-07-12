import torch
from torchvision import transforms
from torchvision.transforms import functional
from torch.nn.functional import pad

import polycraft_nov_data.novelcraft_const as novelcraft_const


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
        ui_h = 22
        return functional.crop(tensor, 0, 0, h - ui_h, w)


class SamplePatch:
    def __init__(self, patch_shape):
        """Sample a random patch from the set produced by ToPatches
        Args:
            patch_shape (tuple): Shape of patches to output
        """
        self.to_patches = ToPatches(patch_shape)

    def __call__(self, tensor):
        """Sample a random patch from the set produced by ToPatches
        Args:
            tensor (torch.tensor): Image tensor to sample a patch from
        Returns:
            torch.tensor: Patch image tensor
        """
        patches = self.to_patches(tensor)
        patch_set_h, patch_set_w, _, _, _ = patches.shape
        return patches[torch.randint(patch_set_h, ()), torch.randint(patch_set_w, ())]


class ToPatches:
    def __init__(self, patch_shape):
        """Divide image into set of patches
        Args:
            patch_shape (tuple): Shape of patches to output
        """
        _, self.patch_h, self.patch_w = patch_shape

    def __call__(self, tensor):
        """Divide image into set of patches
        Args:
            tensor (torch.tensor): Image tensor to divide into patches
        Returns:
            torch.tensor: Set of patch image tensors, shape (PH, PW, C, H, W)
                          where PH and PW are number of patches vertically/horizontally
        """
        patches = tensor.unfold(1, self.patch_h, self.patch_h//2)
        patches = patches.unfold(2, self.patch_w, self.patch_w//2)
        return patches.permute(1, 2, 0, 3, 4)


class PatchTrainPreprocess:
    def __init__(self, image_scale=1.0):
        """Patch image preprocessing for training
        Args:
            image_scale (float, optional): Scaling to apply to image. Defaults to 1.0.
        """
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            CropUI(),
            ScaleImage(image_scale),
            SamplePatch(novelcraft_const.PATCH_SHAPE),
        ])

    def __call__(self, tensor):
        return self.preprocess(tensor)


class PatchTestPreprocess:
    def __init__(self, image_scale=1.0):
        """Patch image preprocessing for testing
        Args:
            image_scale (float, optional): Scaling to apply to image. Defaults to 1.0.
        """
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            CropUI(),
            ScaleImage(image_scale),
            ToPatches(novelcraft_const.PATCH_SHAPE),
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


def patch_array_shape(patch_shape, tensor_shape):
    """Calculate shape of the output of ToPatches
    Args:
        patch_shape (tuple): Shape of patches to output
        tensor_shape (tuple): Shape of image tensor to divide into patches
    Returns:
        tuple: Shape of the output of ToPatches (PH, PW, C, H, W)
               where PH and PW are number of patches vertically/horizontally
    """
    patch_array = ToPatches(patch_shape)(torch.zeros(tensor_shape))
    return patch_array.shape


class VGGPreprocess:
    """Preprocessing for VGG models
    """
    def __init__(self, image_scale=1.0):
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            CropUI(),
            ScaleImage(image_scale),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, tensor):
        return self.preprocess(tensor)


class CustomPad:
    def __call__(self, tensor):
        """Pad the previusly cropped UI region by adding replicative pixel
        rows in order to get a quadratic tensor.
        Args:
            tensor (torch.tensor): Image tensor to remove Polycraft UI from
        Returns:
            torch.tensor: Padded image tensor
        """
        ui_h = 22
        padding = (0, 0, 0, ui_h)
        return pad(tensor, padding, mode='replicate')


class PreprocessFullQuadraticImage:
    def __init__(self, image_scale=1.0):
        """Image preprocessing for AE training with quadratic full (un-patched)
        images.
        Args:
            image_scale (float, optional): Scaling to apply to image. Defaults to 1.0.
        """

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            CropUI(),
            CustomPad(),
            ScaleImage(image_scale),
        ])

    def __call__(self, tensor):
        return self.preprocess(tensor)
