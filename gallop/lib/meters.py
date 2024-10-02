from typing import Type, Union, Dict, Callable, List
from collections import defaultdict
import math

import numpy as np
from torch import Tensor

from gallop.lib.logger import LOGGER

NoneType = Type[None]


def _handle_types(value: Union[Tensor, float]) -> float:
    if hasattr(value, "detach"):
        try:
            value = value.detach().item()
        except ValueError:
            return None

    if np.isnan(value) or np.isinf(value):
        return None

    return value


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, num_decims: int = 3) -> NoneType:
        self.num_decims = num_decims
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.variance = 0
        self.std = float(0)

    def reset(self) -> NoneType:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.variance = 0
        self.std = float(0)

    def update(self, val: float, n: int = 1) -> NoneType:
        val = _handle_types(val)
        if val is None:
            return
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.count > 1:
            self.variance = ((self.count - 1) * self.variance + (val - self.avg) * (val - self.avg) / self.count) / self.count
            self.std = float(math.sqrt(self.variance))

    def __str__(self) -> str:
        return f'{self.val:.{self.num_decims}f} ({self.avg:.{self.num_decims}f})'

    def summary(self) -> str:
        return f'{self.avg:.{self.num_decims}f}'


class RunningAverageMeter(object):
    """Computes and stores the running average and current value"""

    def __init__(self, momentum: float = 0.99) -> NoneType:
        self.momentum = momentum
        self.val = None
        self.avg = 0

    def reset(self) -> NoneType:
        self.val = None
        self.avg = 0

    def update(self, val: float) -> NoneType:
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


class DictAverage(defaultdict):

    def __init__(self,) -> NoneType:
        super().__init__(AverageMeter)

    def update(self, dict_values: Dict[str, float], n: int = 1) -> NoneType:
        for key, item in dict_values.items():
            self[key].update(item, n)

    @property
    def avg(self,) -> Dict[str, float]:
        return {key: item.avg for key, item in self.items()}

    @property
    def sum(self,) -> Dict[str, float]:
        return {key: item.sum for key, item in self.items()}

    def __str__(self) -> List[str]:
        fmtstr_list = [name + ": " + str(meter) for name, meter in self.items()]
        return fmtstr_list

    def summary(self) -> List[str]:
        fmtstr_list = [name + ": " + meter.summary() for name, meter in self.items()]
        return fmtstr_list


class ProgressMeter(object):
    def __init__(
        self,
        num_batches: int,
        meter: DictAverage,
        prefix: str = "",
    ) -> NoneType:
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meter = meter
        self.prefix = prefix

    def display(self, batch: int) -> NoneType:
        entries = [self.prefix + self.batch_fmtstr(batch)]
        entries += [name + ": " + str(meter) for name, meter in self.meter.items()]
        LOGGER.info('  '.join(entries))

    def display_summary(self) -> NoneType:
        entries = [" *"]
        entries += self.meter.summary()
        LOGGER.info(' '.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches: int) -> Callable[[int], str]:
        num_digits = len(str(num_batches // 1))

        def batch_fmtstr(batch: int) -> str:
            return f"[{batch:{num_digits}d}/{num_batches:{num_digits}d}]"
        return batch_fmtstr
