from typing import Type, List, Dict, Any, Callable
import random
from functools import wraps

import numpy as np
import torch

from gallop.lib.logger import LOGGER

NoneType = Type[None]
ArgsType = List[Any]
KwargsType = Dict[str, Any]


def random_seed(seed: int, backend: bool = True) -> NoneType:
    LOGGER.info(f"Setting the seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if backend:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_random_state() -> Dict[str, np.ndarray]:
    LOGGER.debug("Getting random state")
    RANDOM_STATE = {}
    RANDOM_STATE["RANDOM_STATE"] = random.getstate()
    RANDOM_STATE["NP_STATE"] = np.random.get_state()
    RANDOM_STATE["TORCH_STATE"] = torch.random.get_rng_state()
    RANDOM_STATE["TORCH_CUDA_STATE"] = torch.cuda.get_rng_state_all()
    return RANDOM_STATE


def set_random_state(RANDOM_STATE: Dict[str, np.ndarray]) -> NoneType:
    LOGGER.debug("Setting random state")
    random.setstate(RANDOM_STATE["RANDOM_STATE"])
    np.random.set_state(RANDOM_STATE["NP_STATE"])
    torch.random.set_rng_state(RANDOM_STATE["TORCH_STATE"])
    torch.cuda.set_rng_state_all(RANDOM_STATE["TORCH_CUDA_STATE"])


def get_set_random_state(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: ArgsType, **kwargs: KwargsType) -> Any:  # noqa ANN401
        RANDOM_STATE = get_random_state()
        output = func(*args, **kwargs)
        set_random_state(RANDOM_STATE)
        return output
    return wrapper
