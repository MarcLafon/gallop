from typing import Type

import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from PIL.Image import Image

NoneType = Type[None]


def _convert_image_to_rgb(image: Image) -> Image:
    return image.convert("RGB")


def get_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],  # clip mean
                std=[0.26862954, 0.26130258, 0.27577711],  # clip std
            ),
        ]
    )


def get_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC),
            _convert_image_to_rgb,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],  # clip mean
                std=[0.26862954, 0.26130258, 0.27577711],  # clip std
            ),
        ],
    )
