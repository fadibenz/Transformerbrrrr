import torch
import torch.nn as nn

from Transformer.Utils.linear import Linear


class SwiGLU(nn.Module):
    def __init__(self,
                 d_model: int,
                 dff: int,
                 device:torch.device| None=None,
                 dtype: torch.dtype| None=None):
        super().__init__()

        self.W1 = Linear(d_model,dff, device, dtype)
        self.W2 = Linear(dff, d_model, device, dtype)
        self.W3 = Linear(d_model, dff, device, dtype)

    @staticmethod
    def silu(x):
        return x * torch.sigmoid(x)

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        SilU = self.silu(self.W1(x))
        return self.W2(SilU * self.W3(x))