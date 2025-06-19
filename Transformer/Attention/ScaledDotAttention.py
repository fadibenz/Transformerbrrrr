import torch
import torch.nn as nn

from Transformer.Utils.linear import Linear
from Transformer.Attention.RoPE import RoPE
from einops import einsum, rearrange


def softmax(x:torch.Tensor, i:int) -> torch.Tensor:

    maximum = torch.max(x, i, keepdim=True).values
    sub_max = x - maximum
    exp = torch.exp(sub_max)
    return exp / torch.sum(exp, i, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor,
                                 K: torch.Tensor,
                                 V: torch.Tensor,
                                 mask:torch.Tensor |None=None,
                                 ) -> torch.Tensor:
    d_k = Q.shape[-1]
    pre_softmax = einsum(
        Q, K,
        "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k"
    ) / (d_k ** 0.5)
    if mask is not None:
        pre_softmax = pre_softmax.masked_fill(mask == 0,  -1e9)

    softmax_scores = softmax(pre_softmax, -1)

    attention = einsum(
        softmax_scores,
        V,
        "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v"
    )

    return attention


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int,
                       num_heads: int,
                       d_k:int | None = None,
                       d_v: int | None = None,
                       theta:float | None = None,
                       use_rope:bool = False,
                       max_seq_len:int | None =None,
                       device: torch.device | None=None,
                       dtype: torch.dtype | None=None,):
        super().__init__()

        if d_v is None:
            d_v = int(d_model / num_heads)
        if d_k is None:
            d_k = int(d_model / num_heads)

        self.RoPE = None

        if use_rope and theta is not None and max_seq_len is not None:
            self.RoPE = RoPE(theta, d_k, max_seq_len, device=device)

        self.Q = Linear(d_k * num_heads, d_model, device=device, dtype=dtype)
        self.K = Linear(d_k * num_heads, d_model, device=device, dtype=dtype)
        self.V = Linear(d_v * num_heads, d_model, device=device, dtype=dtype)
        self.O = Linear(d_model, d_v * num_heads, device=device, dtype=dtype)

        self.num_heads = num_heads
        self.d_model = d_model
        self.device = device

    def forward(self, x:torch.Tensor,
                      token_positions: torch.Tensor | None = None
                )-> torch.Tensor:

        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        seq_len = x.shape[-2]

        mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), 1).bool()

        Q = rearrange(Q,
                    "... seq (h dk) -> ... h seq dk",
                      h = self.num_heads)

        K = rearrange(K,
                    "... seq (h dk) -> ... h seq dk",
                      h = self.num_heads)

        V = rearrange(V,
                    "... seq (h dv) -> ... h seq dv",
                      h = self.num_heads)

        if self.RoPE is not None:
            Q = self.RoPE(Q, token_positions)
            K = self.RoPE(K, token_positions)

        attention = scaled_dot_product_attention(Q, K, V, ~mask)
        attention = rearrange(
            attention,
            "... h seq dv -> ... seq (h dv)",
            h=self.num_heads
        )

        return self.O(attention)