from typing import Type, Optional, Dict, Union
import os

import torch
from PIL import Image

NoneType = Type[None]


class ImageList(torch.utils.data.Dataset):

    def __init__(self, root: str, image_list: str, transform: Optional[torch.nn.Module] = None) -> NoneType:
        super().__init__()
        self.level = ''

        self.root = root
        self.transform = transform
        self.level = 1

        with open(image_list) as f:
            lines = f.read().splitlines()
        paths, target = [x.split(' ')[0] for x in lines], [int(x.split(' ')[1]) for x in lines]

        self.paths = [os.path.join(self.root, pth) for pth in paths]
        self.targets = target

    def __len__(self,) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        target = torch.tensor(self.targets[idx])

        if self.transform is not None:
            img = self.transform(img)

        out = {"image": img, "target": target, "path": path, "index": idx}
        return out
