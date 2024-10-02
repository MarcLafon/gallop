from typing import Type, Dict, List, Any, Union

from torch import Tensor
from torchvision.datasets import ImageFolder as _ImageFolder

NoneType = Type[None]
ArgsType = List[Any]
KwargsType = Dict[str, Any]


class ImageFolder(_ImageFolder):

    def __init__(self, root: str, *args: ArgsType, **kwargs: KwargsType) -> NoneType:
        root = root
        super().__init__(root, *args, **kwargs)
        self.labels = self.targets

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str, int]]:
        img, target = super().__getitem__(index)
        return {
            "image": img,
            "target": target,
            "path": self.imgs[index][0],
            "index": index,
        }
