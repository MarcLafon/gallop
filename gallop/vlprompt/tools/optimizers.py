import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD

import gallop.lib as lib


def get_optimizer(optimizer_name: str, model: nn.Module, learning_rate: float, weight_decay: float = 0.0, momentum: float = 0.9) -> torch.optim.Optimizer:
    params_group = lib.get_params_group(model)
    if optimizer_name == 'adam':
        optimizer = Adam(params_group, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = AdamW(params_group, lr=learning_rate, weight_decay=weight_decay, eps=1e-4)
    elif optimizer_name == 'sgd':
        optimizer = SGD(params_group, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {optimizer_name} not implemented")

    return optimizer
