from typing import Type, Optional, Callable, Dict, Union
import os

import PIL
from torch import Tensor
from torch.utils.data import Dataset

import gallop.lib as lib

NoneType = Type[None]


class StanfordCarsDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable[[PIL.Image.Image], Tensor]] = None,
    ) -> NoneType:
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        assert split in ["train", "test"]
        abs_path_dirname = os.path.dirname(os.path.abspath(__file__))
        db = lib.load_json(os.path.join(abs_path_dirname, 'files', 'split_zhou_StanfordCars.json'))
        db = db[split]

        self.paths = []
        self.targets = []
        self.names = []
        for x in db:
            self.paths.append(os.path.join(self.root, "stanford_cars", x[0]))
            self.targets.append(x[1])
            self.names.append(x[2])

        # unique names
        self.all_names = []
        for nm in self.names:
            if nm not in self.all_names:
                self.all_names.append(nm)

        name_to_idx = {nm: i for i, nm in enumerate(self.all_names)}
        self.labels = self.targets = [name_to_idx[nm] for nm in self.names]

        self.level = "stanford_cars"

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str, int]]:
        path = self.paths[index]
        target = self.targets[index]
        img = PIL.Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
            "target": target,
            "path": path,
            "index": index,
            "name": self.names[index],
        }
