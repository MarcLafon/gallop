from typing import Type, Dict, List, Any, Union
import os

from torch import Tensor
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

import gallop.lib as lib
from gallop.datasets.tools.wnid_to_name import wnid_to_name

NoneType = Type[None]
ArgsType = List[Any]
KwargsType = Dict[str, Any]


class ImagenetV2Dataset(ImageFolder):

    folder_dir = "imagenetv2-matched-frequency-format-val"
    download_url = 'https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz'

    def __init__(self, root: str, *args: ArgsType, download: bool = False, **kwargs: KwargsType) -> NoneType:
        self.root_folder = root
        self.root = os.path.join(self.root_folder, self.folder_dir)
        if download:
            self.download()
        super().__init__(self.root, *args, **kwargs)
        self.level = 'in_v2'
        _idx_to_idx = {}  # When ImageFolder is initialized, it sorts the classes by name and does not preserve the original order
        for i in range(len(self.targets)):
            pth = self.imgs[i][0]
            idx = int(pth.split('/')[-2])
            old_idx = self.targets[i]
            _idx_to_idx[old_idx] = idx

        self.targets = [_idx_to_idx[i] for i in self.targets]
        self.classes = range(len(self.classes))
        abs_path_dirname = os.path.dirname(os.path.abspath(__file__))
        self.idx_to_wnid = {int(k): v for k, v in lib.load_json(os.path.join(abs_path_dirname, 'files', 'imagenet_idx_to_wnid.json')).items()}
        self.classes = [self.idx_to_wnid[int(i)] for i in self.classes]
        self.labels = self.targets

        self.wnid_to_idx = self.class_to_idx = {v: k for k, v in self.idx_to_wnid.items()}
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
            filename="imagenetv2-matched-frequency.tar.gz",
        )

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str, int]]:
        img, target = super().__getitem__(index)
        target = self.targets[index]  # We have not modified the targets in the internal ImageFolder class
        return {
            "image": img,
            "target": target,
            "path": self.imgs[index][0],
            "index": index,
            "wnid": self.idx_to_wnid[target],
            "name": wnid_to_name(self.idx_to_wnid[target]),
        }
