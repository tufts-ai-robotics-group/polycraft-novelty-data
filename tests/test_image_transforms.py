import torch

import polycraft_nov_data.data_const as data_const
import polycraft_nov_data.image_transforms as image_transforms


def test_crop_on_resized():
    image_scales = [1, .75, .5]
    crop_shapes = [
        (3, 234, 256),
        (3, 175, 192),
        (3, 117, 128),
    ]
    crop_ui = image_transforms.CropUI()
    for i in range(len(image_scales)):
        tensor = torch.zeros(data_const.IMAGE_SHAPE)
        scale_image = image_transforms.ScaleImage(image_scales[i])
        tensor = scale_image(crop_ui(tensor))
        assert tensor.shape == crop_shapes[i]


# GaussianNoise tests
def test_gaussian_noise():
    # check bounds of noisy image and that noise is applied
    image = torch.rand(data_const.IMAGE_SHAPE)
    noisy_image = image_transforms.GaussianNoise()(image)
    assert noisy_image.min() >= 0
    assert noisy_image.max() <= 1
    assert (noisy_image != image).any()
