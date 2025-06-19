from typing import Iterable
import torch


def gradient_clipping(params: Iterable[torch.nn.Parameter],
                      max_l2_norm:float,
                      eps: float = 1e-6):
    total_norm = 0.0
    for param in params:
        if param.grad is None:
            continue
        total_norm += param.grad.norm(2) ** 2
    total_norm = total_norm ** 0.5

    if total_norm > max_l2_norm:
            scale_factor = max_l2_norm / (total_norm + eps)
            for param in params:
                if param.grad is not None:
                 param.grad *= scale_factor
    return params
