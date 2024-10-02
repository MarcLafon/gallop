from typing import Type, List

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch import Tensor

from gallop.vlprompt.tools.topk_reduce import topk_reduce

NoneType = Type[None]


class GlobalLocalLoss(_WeightedLoss):

    def __init__(
        self,
        use_global_loss: bool = True,
        use_local_loss: bool = True,
        topk: List[int] = [5],
        global_dropout_p: float = 0.75,
    ) -> NoneType:
        super().__init__()

        self.use_global_loss = use_global_loss
        self.use_local_loss = use_local_loss
        self.topk = topk
        self.global_dropout_p = global_dropout_p

    def forward(
        self,
        global_logits: Tensor,
        local_logits: Tensor,
        targets: Tensor,
        logit_scale: float,
    ) -> Tensor:
        """
        global_logits is a Tensor of shape (b, k, 1) or (b, k, n)
        local_logits is a Tensor of shape (b, p, k, 1) or (b, p, k, m)
        """
        global_loss = local_loss = 0.

        if self.use_local_loss and local_logits is not None:
            local_logits = topk_reduce(local_logits, self.topk)
            local_loss = F.cross_entropy(logit_scale * local_logits, targets.unsqueeze(-1).expand(-1, local_logits.size(-1)))

        if self.use_global_loss:
            # Dropout:
            keep_number = max(global_logits.size(-1) - int(self.global_dropout_p * global_logits.size(-1)), 1)
            index = torch.randint(global_logits.size(-1), (global_logits.size(0), 1, keep_number), device=global_logits.device).expand(-1, global_logits.size(1), -1)
            global_logits = global_logits.gather(-1, index).mean(-1)
        
            if global_logits.ndim == 2:
                global_loss = F.cross_entropy(logit_scale * global_logits, targets)
            elif global_logits.ndim == 3:
                global_loss = F.cross_entropy(logit_scale * global_logits, targets.unsqueeze(-1).expand(-1, global_logits.size(-1)))
            else:
                raise ValueError(f"Global logits must have 2 or 3 dimensions, but got {global_logits.ndim}.")

        return global_loss + local_loss
