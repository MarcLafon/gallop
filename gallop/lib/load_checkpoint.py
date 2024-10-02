from typing import Type
import torch
import gallop.lib as lib

NoneType = Type[None]


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
) -> NoneType:
    if (checkpoint_path is not None) and (checkpoint_path.lower() != "none"):
        lib.LOGGER.info(f"Loading checkpoint {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cuda")
        keys = model.load_trainable_state_dict(state_dict["state_dict"], strict=False)
        if len(keys.unexpected_keys) > 0:
            raise RuntimeError(f"Unexpected keys in state_dict: {keys.unexpected_keys}")
        if len(keys.missing_keys) > 0:
            lib.LOGGER.warning(f"Missing keys in state_dict: {keys.missing_keys}")
