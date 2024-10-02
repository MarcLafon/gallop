from typing import Type, Dict, List, Any, Union

from torch import Tensor
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import has_file_allowed_extension

from gallop.datasets.tools.wnid_to_name import wnid_to_name

NoneType = Type[None]
ArgsType = List[Any]
KwargsType = Dict[str, Any]


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp", "JPEG")


def is_valid_file(x: str) -> bool:
    return has_file_allowed_extension(x, IMG_EXTENSIONS)  # type: ignore[arg-type]


class ImagenetDataset(ImageFolder):
    def __init__(self, root: str, *args: ArgsType, **kwargs: KwargsType) -> NoneType:
        root = root
        super().__init__(root, *args, is_valid_file=is_valid_file, **kwargs)
        self.level = "in1k"

        self.labels = self.targets
        self.wnid_to_idx = self.class_to_idx
        self.idx_to_wnid = {v: k for k, v in self.wnid_to_idx.items()}
        self.all_names = [wnid_to_name(wnid) for wnid in self.classes]
        self.wnids = [self.idx_to_wnid[i] for i in self.labels]

        self.label_names = [wnid_to_name(self.idx_to_wnid[tgt]) for tgt in self.targets]

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str, int]]:
        img, _ = super().__getitem__(index)
        target = self.targets[index]
        return {
            "image": img,
            "target": target,
            "path": self.imgs[index][0],
            "index": index,
            "name": self.label_names[index],
            "wnid": self.wnids[index],
        }
