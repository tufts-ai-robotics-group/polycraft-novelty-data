import torch
from torchvision import transforms
from torchvision.transforms import functional

from polycraft_nov_data.data import IMAGE_SHAPE, PATCH_SHAPE


class CropUI:
    def __call__(self, tensor):
        """Crop the UI by removing bottom row of patches

        Args:
            tensor (torch.tensor): Image tensor to remove Polycraft UI from

        Returns:
            torch.tensor: Cropped image tensor
        """
        _, h, w = IMAGE_SHAPE
        _, patch_h, _ = PATCH_SHAPE
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
        patch_set_h, patch_set_w, _ = patches.shape
        return patches[torch.randint(patch_set_h), torch.randint(patch_set_w)]


class ToPatches:
    def __call__(self, tensor):
        """Divide image into set of patches

        Args:
            tensor (torch.tensor): Image tensor to divide into patches

        Returns:
            torch.tensor: Set of patch image tensors, shape (PH, PW, C, H, W)
                          where PH and PW are number of patches vertically/horizontally
        """
        _, patch_h, patch_w = PATCH_SHAPE
        return tensor.unfold(1, patch_h, patch_h//2).unfold(2, patch_w, patch_w//2)


class TrainPreprocess:
    def __init__(self, image_scale=1.0):
        """Image preprocessing for training

        Args:
            image_scale (float, optional): Scaling to apply to image. Defaults to 1.0.
        """
        _, h, w = IMAGE_SHAPE
        resize_h = int(h * .5)
        resize_w = int(w * .5)
        self.preprocess = transforms.Compose([
            transforms.Resize((resize_h, resize_w)),
            CropUI(),
            SamplePatch(),
            transforms.ToTensor(),
        ])

    def __call__(self, tensor):
        return self.preprocess(tensor)


class TestPreprocess:
    def __init__(self, image_scale=1.0):
        """Image preprocessing for testing

        Args:
            image_scale (float, optional): Scaling to apply to image. Defaults to 1.0.
        """
        _, h, w = IMAGE_SHAPE
        resize_h = int(h * .5)
        resize_w = int(w * .5)
        self.preprocess = transforms.Compose([
            transforms.Resize((resize_h, resize_w)),
            CropUI(),
            ToPatches(),
            transforms.ToTensor(),
        ])

    def __call__(self, tensor):
        return self.preprocess(tensor)
