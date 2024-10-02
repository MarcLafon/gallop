from typing import Dict

import torch


def get_clip_hyperparams(clip_state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    is_vit = "visual.proj" in clip_state_dict
    if is_vit:
        grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)

        kwargs = {
            "embed_dim": clip_state_dict["text_projection"].shape[1],
            "image_resolution": clip_state_dict["visual.conv1.weight"].shape[-1] * grid_size,

            "vision_layers": len([k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")]),
            "vision_width": clip_state_dict["visual.conv1.weight"].shape[0],
            "vision_patch_size": clip_state_dict["visual.conv1.weight"].shape[-1],

            "context_length": clip_state_dict["positional_embedding"].shape[0],
            "vocab_size": clip_state_dict["token_embedding.weight"].shape[0],
            "transformer_width": clip_state_dict["ln_final.weight"].shape[0],
            'transformer_heads': clip_state_dict["ln_final.weight"].shape[0] // 64,
            "transformer_layers": len(set(k.split(".")[2] for k in clip_state_dict if k.startswith("transformer.resblocks"))),
        }

    else:

        counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]

        kwargs = {
            "embed_dim": clip_state_dict["text_projection"].shape[1],
            "image_resolution": output_width * 32,
            "vision_layers": tuple(counts),
            "vision_width": clip_state_dict["visual.layer1.0.conv1.weight"].shape[0],
            "vision_patch_size": None,

            "context_length": clip_state_dict["positional_embedding"].shape[0],
            "vocab_size": clip_state_dict["token_embedding.weight"].shape[0],
            "transformer_width": clip_state_dict["ln_final.weight"].shape[0],
            "transformer_heads": clip_state_dict["ln_final.weight"].shape[0] // 64,
            "transformer_layers": len(set(k.split(".")[2] for k in clip_state_dict if k.startswith("transformer.resblocks"))),
        }

    return kwargs
