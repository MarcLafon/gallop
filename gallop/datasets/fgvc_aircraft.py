from typing import Type, Dict, List, Any, Union

from torch import Tensor
from torchvision.datasets import FGVCAircraft


NoneType = Type[None]
ArgsType = List[Any]
KwargsType = Dict[str, Any]


class FGVCAircraftDataset(FGVCAircraft):

    def __init__(self, root: str, *args: ArgsType, **kwargs: KwargsType) -> NoneType:
        root = root
        super().__init__(root, *args, **kwargs)
        self.labels = self.targets = self._labels
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.all_names = [self.idx_to_class[i] for i in range(len(self.classes))]
        self.level = 'fgvc-aircraft'

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str, int]]:
        img, target = super().__getitem__(index)
        return {
            "image": img,
            "target": target,
            "path": self._image_files[index],
            "index": index,
            "name": self.all_names[target],
        }
