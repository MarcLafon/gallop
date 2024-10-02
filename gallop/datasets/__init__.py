from typing import Callable, Tuple, Dict
import os

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from gallop.datasets.caltech101 import Caltech101Dataset
from gallop.datasets.dtd import DTDDataset
from gallop.datasets.eurosat import EuroSATDataset
from gallop.datasets.fgvc_aircraft import FGVCAircraftDataset
from gallop.datasets.flowers102 import Flowers102Dataset
from gallop.datasets.food101 import Food101Dataset
from gallop.datasets.image_folder import ImageFolder
from gallop.datasets.image_list import ImageList
from gallop.datasets.imagenet_a import ImagenetADataset
from gallop.datasets.imagenet_r import ImagenetRDataset
from gallop.datasets.imagenet_sketch import ImagenetSketchDataset
from gallop.datasets.imagenet import ImagenetDataset
from gallop.datasets.imagenetv2 import ImagenetV2Dataset
from gallop.datasets.oxford_pets import OxfordPetsDataset
from gallop.datasets.stanford_cars import StanfordCarsDataset
from gallop.datasets.sun397 import SUN397Dataset
from gallop.datasets.ucf101 import UCF101Dataset

from gallop.datasets.templates import (
    get_templates,
    BASE_PROMPT,
    IMAGENET_TEMPLATES_SELECT,
    IMAGENET_TEMPLATES,
    RandomTemplate,
    CALTECH_101_TEMPLATE,
    OXFORD_PETS_TEMPLATE,
    STANFORD_CARS_TEMPLATE,
    FLOWERS_102_TEMPLATE,
    FOOD_101_TEMPLATE,
    AIRCRAFT_TEMPLATE,
    SUN_397_TEMPLATE,
    DTD_TEMPLATE,
    EUROSAT_TEMPLATE,
    UCF_101_TEMPLATE,
)

import gallop.datasets.tools as tools


__all__ = [
    "Caltech101Dataset",
    "DTDDataset",
    "EuroSATDataset",
    "FGVCAircraftDataset",
    "Flowers102Dataset",
    "Food101Dataset",
    "ImageFolder",
    "ImageList",
    "ImagenetADataset",
    "ImagenetRDataset",
    "ImagenetSketchDataset",
    "ImagenetDataset",
    "ImagenetV2Dataset",
    "OxfordPetsDataset",
    "StanfordCarsDataset",
    "SUN397Dataset",
    "UCF101Dataset",

    "get_templates",
    "BASE_PROMPT",
    "IMAGENET_TEMPLATES_SELECT",
    "IMAGENET_TEMPLATES",
    "RandomTemplate",
    "CALTECH_101_TEMPLATE",
    "OXFORD_PETS_TEMPLATE",
    "STANFORD_CARS_TEMPLATE",
    "FLOWERS_102_TEMPLATE",
    "FOOD_101_TEMPLATE",
    "AIRCRAFT_TEMPLATE",
    "SUN_397_TEMPLATE",
    "DTD_TEMPLATE",
    "EUROSAT_TEMPLATE",
    "UCF_101_TEMPLATE",

    "tools",
]


def return_train_val_datasets(
    name: str,
    data_dir: str,
    train_transform: Callable[[Image.Image], Tensor],
    val_transform: Callable[[Image.Image], Tensor],
) -> Tuple[Dataset, Dataset, str]:
    if name == "imagenet":
        train_dataset = ImagenetDataset(
            root=os.path.join(data_dir, 'Imagenet', "train"),
            transform=train_transform,
        )
        val_dataset = ImagenetDataset(
            root=os.path.join(data_dir, 'Imagenet', "val"),
            transform=val_transform,
        )
        template = 'A photo of a {}.'
    elif name == 'caltech101':
        train_dataset = Caltech101Dataset(
            root=os.path.join(data_dir, 'dataset_suite'),
            split='train',
            transform=train_transform,
        )
        val_dataset = Caltech101Dataset(
            root=os.path.join(data_dir, 'dataset_suite'),
            split='test',
            transform=val_transform,
        )
        template = CALTECH_101_TEMPLATE
    elif name == 'oxfordpets':
        train_dataset = OxfordPetsDataset(
            root=os.path.join(data_dir, 'dataset_suite'),
            split='train',
            transform=train_transform,
        )
        val_dataset = OxfordPetsDataset(
            root=os.path.join(data_dir, 'dataset_suite'),
            split='test',
            transform=val_transform,
        )
        template = OXFORD_PETS_TEMPLATE
    elif name == 'stanford_cars':
        train_dataset = StanfordCarsDataset(
            root=os.path.join(data_dir, 'dataset_suite'),
            split='train',
            transform=train_transform,
        )
        val_dataset = StanfordCarsDataset(
            root=os.path.join(data_dir, 'dataset_suite'),
            split='test',
            transform=val_transform,
        )
        template = STANFORD_CARS_TEMPLATE
    elif name == 'flowers102':
        train_dataset = Flowers102Dataset(
            root=os.path.join(data_dir, 'dataset_suite'),
            split='train',
            transform=train_transform,
        )
        val_dataset = Flowers102Dataset(
            root=os.path.join(data_dir, 'dataset_suite', ),
            split='test',
            transform=val_transform,
        )
        template = FLOWERS_102_TEMPLATE
    elif name == 'food101':
        train_dataset = Food101Dataset(
            root=os.path.join(data_dir, 'dataset_suite'),
            split='train',
            transform=train_transform,
        )
        val_dataset = Food101Dataset(
            root=os.path.join(data_dir, 'dataset_suite'),
            split='test',
            transform=val_transform,
        )
        template = FOOD_101_TEMPLATE
    elif name == 'fgvc_aircraft':
        train_dataset = FGVCAircraftDataset(
            root=os.path.join(data_dir, 'dataset_suite'),
            split='trainval',
            transform=train_transform,
        )
        val_dataset = FGVCAircraftDataset(
            root=os.path.join(data_dir, 'dataset_suite'),
            split='test',
            transform=val_transform,
        )
        template = AIRCRAFT_TEMPLATE
    elif name == 'sun397':
        train_dataset = SUN397Dataset(
            root=os.path.join(data_dir, 'dataset_suite'),
            split='train',
            transform=train_transform,
        )
        val_dataset = SUN397Dataset(
            root=os.path.join(data_dir, 'dataset_suite'),
            split='test',
            transform=val_transform,
        )
        template = SUN_397_TEMPLATE
    elif name == 'dtd':
        train_dataset = DTDDataset(
            root=os.path.join(data_dir, 'dataset_suite'),
            split='train',
            transform=train_transform,
        )
        val_dataset = DTDDataset(
            root=os.path.join(data_dir, 'dataset_suite'),
            split='test',
            transform=val_transform,
        )
        template = DTD_TEMPLATE
    elif name == 'eurosat':
        train_dataset = EuroSATDataset(
            root=os.path.join(data_dir, 'dataset_suite'),
            split='train',
            transform=train_transform,
        )
        val_dataset = EuroSATDataset(
            root=os.path.join(data_dir, 'dataset_suite'),
            split='test',
            transform=val_transform,
        )
        template = EUROSAT_TEMPLATE
    elif name == 'ucf101':
        train_dataset = UCF101Dataset(
            root=os.path.join(data_dir, 'dataset_suite'),
            split='train',
            transform=train_transform,
        )
        val_dataset = UCF101Dataset(
            root=os.path.join(data_dir, 'dataset_suite'),
            split='test',
            transform=val_transform,
        )
        template = UCF_101_TEMPLATE

    else:
        raise NotImplementedError(f"Dataset {name} not implemented")

    return train_dataset, val_dataset, template


def return_ood_loaders(
    data_dir: str,
    transform: Callable[[Image.Image], Tensor],
) -> Dict[str, DataLoader]:
    return {
        "iNaturalist": tools.get_eval_loader(
            ImageFolder(
                root=os.path.join(data_dir, "ood_data", "iNaturalist"),
                transform=transform,
            )
        ),
        "SUN": tools.get_eval_loader(
            ImageFolder(
                root=os.path.join(data_dir, "ood_data", "SUN"),
                transform=transform,
            )
        ),
        "Places": tools.get_eval_loader(
            ImageFolder(
                root=os.path.join(data_dir, "ood_data", "Places"),
                transform=transform,
            )
        ),
        "Textures": tools.get_eval_loader(
            ImageFolder(
                root=os.path.join(data_dir, "ood_data", "dtd", "images"),
                transform=transform,
            )
        ),
    }


def return_domains_loaders(
    data_dir: str,
    transform: Callable[[Image.Image], Tensor],
) -> Dict[str, DataLoader]:
    return {
        "imganetv2": tools.get_eval_loader(
            ImagenetV2Dataset(
                root=os.path.join(data_dir, "domains"),
                transform=transform,
            )
        ),
        "sketch": tools.get_eval_loader(
            ImagenetSketchDataset(
                root=os.path.join(data_dir, "domains"),
                transform=transform,
            )
        ),
        "imageneta": tools.get_eval_loader(
            ImagenetADataset(
                root=os.path.join(data_dir, "domains"),
                transform=transform,
            )
        ),
        "imagenetr": tools.get_eval_loader(
            ImagenetRDataset(
                root=os.path.join(data_dir, "domains"),
                transform=transform,
            )
        ),
    }
