from typing import Type, Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from gallop.vlprompt.clip_local import Transformer, VisionTransformer

NoneType = Type[None]
KwargType = Dict[str, Any]


class PromptedTransformer(Transformer):
    def __init__(
        self,
        is_textual: bool = True,
        segments: Optional[int] = 0,
        **kwargs: KwargType,
    ) -> NoneType:
        super().__init__(**kwargs)
        self.is_textual = is_textual
        self.segments = segments

    def replace_context(self, x: Tensor, ctx: Tensor) -> Tensor:
        n_ctx = ctx.shape[0]

        if self.is_textual:
            prefix = x[:1, :, :]
            suffix = x[1 + n_ctx :, :, :]
        else:
            prefix = x[0 : x.shape[0] - n_ctx, :, :]
            suffix = torch.Tensor([]).to(x.dtype).cuda()

        context = ctx.expand(x.shape[1], -1, -1).permute(1, 0, 2)
        return torch.cat([prefix, context, suffix], dim=0)

    def forward(self, x: Tensor, ctx_vectors: Optional[Tensor] = None, batch_first: bool = False) -> Tensor:
        if batch_first:  # The permute is not done outside the loop. This is usefull for DataParallel
            x = x.permute(1, 0, 2)

        if ctx_vectors is None or len(ctx_vectors) == 0:
            if self.segments > 0:
                for i in range(self.layers):
                    if (i % (self.layers // self.segments) == 0):
                        x, q, k, v = checkpoint(self.resblocks[i], x)
                    else:
                        x, q, k, v = self.resblocks[i](x)
            else:
                for i in range(self.layers):
                    x, q, k, v = self.resblocks[i](x)
        else:
            for i in range(self.layers):
                x, q, k, v = self.resblocks[i](x)
                if i < len(ctx_vectors):
                    x = self.replace_context(x, ctx_vectors[i])

        if batch_first:
            x = x.permute(1, 0, 2)

        return x, q, k, v


class PromptedVisionTransformer(VisionTransformer):
    def __init__(
        self,
        **kwargs: KwargType,
    ) -> NoneType:
        super().__init__(**kwargs)

        self.transformer = PromptedTransformer(width=kwargs["width"], layers=kwargs["layers"], heads=kwargs["heads"], attn_mask=None, is_textual=False)

    def forward(self, x: torch.Tensor, ctx_vectors: Optional[nn.ParameterList] = None) -> torch.Tensor:
        if ctx_vectors is None or len(ctx_vectors) == 0:
            return super().forward(x)
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            zeros = torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat([self.class_embedding.to(x.dtype) + zeros, x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.positional_embedding.to(x.dtype)

            context = ctx_vectors[0].expand(x.shape[0], -1, -1)
            x = torch.cat([x, context], dim=1)

            x = self.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x, q, k, v = self.transformer(x, ctx_vectors[1:])
            x = x.permute(1, 0, 2)  # LND -> NLD
            v = v.permute(1, 0, 2)  # LND -> NLD

            v = self.ln_post(v)

            v = v[:, 1:]
            B, _, C = x[:, 1:].shape
            v = v.reshape(B, -1, C).contiguous()

            x = self.ln_post(x[:, 0, :])  # taking CLS token here

            if self.proj is not None:
                x = x @ self.proj
                x_local = v @ self.proj
            return x, x_local
