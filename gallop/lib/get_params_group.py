from typing import Dict

from torch import nn
from gallop.lib.logger import LOGGER


def get_params_group(model: nn.Module) -> Dict[str, nn.Parameter]:
    params_group = [{"params": []}]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        else:
            params_group[0]["params"].append(param)
            LOGGER.info(f"{name} added to params_group")
    return params_group
