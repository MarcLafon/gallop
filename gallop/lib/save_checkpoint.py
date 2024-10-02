from typing import Type, Union
import os
from argparse import Namespace

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from gallop.lib import DictAverage

NoneType = Type[None]


def save_checkpoint(
    save_dir: str,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    scaler: Union[GradScaler, NoneType],
    train_meter: DictAverage,
    args: Namespace,
) -> NoneType:
    state = {
        "epoch": epoch,
        "train_meter": {k: v.avg for k, v in train_meter.items()},
        "state_dict": model.state_dict() if not hasattr(model, "trainable_state_dict") else model.trainable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "args": args,
    }

    model_type = model.model_name.split("_")[0]
    dataset_name = args.dataset_name
    archi = "_".join(model.model_name.split("_")[1:])

    os.makedirs(save_dir, exist_ok=True)

    # running checkpoint
    name = "_".join([model_type, dataset_name, archi])
    model_name = f"{name}_seed{args.seed}.ckpt" if args.seed is not None else "{name}.ckpt"
    torch.save(state, os.path.join(save_dir, model_name))
