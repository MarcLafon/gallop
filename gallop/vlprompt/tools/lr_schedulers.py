"""
Modified from https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/optim/lr_scheduler.py
"""
from typing import Type, List, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

NoneType = Type[None]


class _BaseWarmupScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        successor: _LRScheduler,
        warmup_epoch: int,
        last_epoch: int = -1,
        verbose: bool = False
    ) -> NoneType:
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> Union[float, List[float]]:
        raise NotImplementedError

    def step(self, epoch: int = None) -> NoneType:
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        successor: _LRScheduler,
        warmup_epoch: int,
        cons_lr: float,
        last_epoch: int = -1,
        verbose: bool = False
    ) -> NoneType:
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self) -> Union[float, List[float]]:
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]
