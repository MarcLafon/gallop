from typing import Optional

import gallop.lib as lib
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


def get_eval_loader(
    dts: Dataset,
    batch_size: int = 256,
    num_workers: int = 10,
    persistent_workers: bool = False,
    shuffle: bool = False,
) -> DataLoader:
    if shuffle:
        lib.LOGGER.warning("Shuffle is set to True for eval loader")

    return DataLoader(
        dataset=dts,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=False,
    )


def get_train_loader(
    dts: Dataset,
    batch_size: int = 256,
    num_workers: int = 4,
    persistent_workers: bool = False,
    distributed: bool = False,
    sampler: Optional[Sampler] = None,
) -> DataLoader:
    if sampler is None and distributed:
        sampler = DistributedSampler(dts)

    return DataLoader(
        dataset=dts,
        batch_size=batch_size,
        shuffle=sampler is None,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
        persistent_workers=persistent_workers,
    )
