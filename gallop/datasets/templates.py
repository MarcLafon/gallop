# source: https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
from typing import Type, List, Optional
import random

NoneType = Type[None]


BASE_PROMPT = "a photo of a {}."

IMAGENET_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]

IMAGENET_TEMPLATES_SELECT = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]

EUROSAT_TEMPLATE = "A centered satellite photo of {}."
STANFORD_CARS_TEMPLATE = "A photo of a {}."
FLOWERS_102_TEMPLATE = "A photo of a {}, a type of flower."
OXFORD_PETS_TEMPLATE = "A photo of a {}, a type of pet."
UCF_101_TEMPLATE = "A photo of a person doing {}."
AIRCRAFT_TEMPLATE = "A photo of a {}, a type of aircraft."
DTD_TEMPLATE = "{} texture."
CALTECH_101_TEMPLATE = "A photo of a {}."
FOOD_101_TEMPLATE = "A photo of {}, a type of food."
SUN_397_TEMPLATE = "A photo of a {}."


def _get_templates(level: str = 'select') -> List[str]:
    assert level in [
        'base',
        'full',
        'select',
        'eurosat',
        'stanford_cars',
        'flowers_102',
        'oxford_pets',
        'ucf_101',
        'aircraft',
        'dtd',
        'caltech_101',
        'food_101',
        'sun_397',
    ], f'level must be one of base, full, select, got {level}'
    if level == 'base':
        return [BASE_PROMPT]
    elif level == 'full':
        return IMAGENET_TEMPLATES
    elif level == 'select':
        return IMAGENET_TEMPLATES_SELECT
    elif level == 'eurosat':
        return [EUROSAT_TEMPLATE]
    elif level == 'stanford_cars':
        return [STANFORD_CARS_TEMPLATE]
    elif level == 'flowers_102':
        return [FLOWERS_102_TEMPLATE]
    elif level == 'oxford_pets':
        return [OXFORD_PETS_TEMPLATE]
    elif level == 'ucf_101':
        return [UCF_101_TEMPLATE]
    elif level == 'aircraft':
        return [AIRCRAFT_TEMPLATE]
    elif level == 'dtd':
        return [DTD_TEMPLATE]
    elif level == 'caltech_101':
        return [CALTECH_101_TEMPLATE]
    elif level == 'food_101':
        return [FOOD_101_TEMPLATE]
    elif level == 'sun_397':
        return [SUN_397_TEMPLATE]


class RandomTemplate:
    def __init__(self, templates: str) -> NoneType:
        self.templates = _get_templates(templates)

    def __call__(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            random.seed(seed)
        return random.choice(self.templates)


def get_templates(level: str = 'select') -> RandomTemplate:
    if level.startswith('random'):
        return [RandomTemplate(level.replace('random_', '').strip())]

    return _get_templates(level)
