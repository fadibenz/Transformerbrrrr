import torch.nn as nn
import torch
from einops import einsum

class Linear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 device:torch.device | None=None,
                 dtype:torch.dtype |None=None):
        super().__init__()

        kwargs = {}
        if device is not None:
            kwargs['device'] = device
        if dtype is not None:
            kwargs['dtype'] = dtype

        # We use row-major memory ordering for efficiency
        self.W = nn.Parameter(torch.empty((out_features, in_features),
                                          **kwargs))
        std = 2/(out_features + in_features)

        nn.init.trunc_normal_(self.W.data,0, std, -3 * std ,3 * std)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # einops
        return einsum(x, self.W, "... in_features, out_features in_features -> ... out_features")
