from typing import Type, Dict, List, Any, Union
import os

from torch import Tensor
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

from gallop.datasets.tools.wnid_to_name import wnid_to_name

NoneType = Type[None]
ArgsType = List[Any]
KwargsType = Dict[str, Any]


class ImagenetADataset(ImageFolder):

    folder_dir = "imagenet-a"
    download_url = 'https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar'

    def __init__(self, root: str, *args: ArgsType, download: bool = False, **kwargs: KwargsType) -> NoneType:
        self.root_folder = root
        self.root = os.path.join(self.root_folder, self.folder_dir)
        if download:
            self.download()
        super().__init__(self.root, *args, **kwargs)
        self.level = 'in_a'
        self.labels = self.targets
        self.wnid_to_idx = self.class_to_idx
        self.idx_to_wnid = {v: k for k, v in self.class_to_idx.items()}
        self.wnids = [self.idx_to_wnid[i] for i in self.labels]
        self.all_names = [wnid_to_name(wnid) for wnid in self.classes]

    def _check_integrity(self) -> bool:
        # can be more robust and check hash of files
        return os.path.exists(self.root)

    def download(self) -> NoneType:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            self.download_url,
            self.root_folder,
            filename="imagenet-a.tar",
        )

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str, int]]:
        img, target = super().__getitem__(index)
        return {
            "image": img,
            "target": target,
            "path": self.imgs[index][0],
            "index": index,
            "wnid": self.idx_to_wnid[target],
            "name": wnid_to_name(self.idx_to_wnid[target]),
        }
