import torch

import polycraft_nov_data.data_const as data_const
import polycraft_nov_data.image_transforms as image_transforms


def test_crop_on_resized():
    image_scales = [1, .75, .5]
    crop_shapes = [
        (3, 224, 256),
        (3, 168, 192),
        (3, 112, 128),
    ]
    crop_ui = image_transforms.CropUI()
    for i in range(len(image_scales)):
        tensor = torch.zeros(data_const.IMAGE_SHAPE)
        scale_image = image_transforms.ScaleImage(image_scales[i])
        tensor = scale_image(crop_ui(tensor))
        assert tensor.shape == crop_shapes[i]
