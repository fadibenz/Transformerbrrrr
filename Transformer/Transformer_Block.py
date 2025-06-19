import torch
import torch.nn as nn

from Transformer.Utils.RMSNorm import RMSNorm
from Transformer.Attention.ScaledDotAttention import CausalMultiHeadSelfAttention
from Transformer.FFN.SwiGLU import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads:int,
                 d_ff:int,
                 d_k: int | None = None,
                 d_v: int | None = None,
                 theta: float | None = None,
                 use_rope: bool = False,
                 max_seq_len: int | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()

        self.RMSNorm_1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.RMSNorm_2 = RMSNorm(d_model, device=device, dtype=dtype)

        self.MultiHeadSelfAttention = CausalMultiHeadSelfAttention(d_model, num_heads,d_k, d_v,
                                                                   theta, use_rope, max_seq_len,
                                                                   device,dtype)
        self.FeedForward = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x:torch.Tensor,
                token_positions: torch.Tensor) -> torch.Tensor:

        y = x + self.MultiHeadSelfAttention(self.RMSNorm_1(x), token_positions)
        result = y + self.FeedForward(self.RMSNorm_2(y))

        return result
