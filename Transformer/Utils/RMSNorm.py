import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                 device: torch.device | None=None,
                 dtype: torch.dtype| None=None):
        super().__init__()

        self.eps = eps
        self.d_model = d_model

        self.g = nn.Parameter(torch.ones(d_model,
                                          device=device,
                                          dtype=dtype))

    def forward(self, x:torch.Tensor)-> torch.Tensor:

        in_dtype = x.dtype
        x = x.to(torch.float32)

        RMS_a = torch.sqrt(
            ((1 / self.d_model) * torch.sum(torch.pow(x, 2), -1)) + self.eps
        )
        result = (x / RMS_a.unsqueeze(-1)) * self.g
        return result.to(in_dtype)