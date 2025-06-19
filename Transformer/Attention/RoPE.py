import torch
import torch.nn as nn
from einops import rearrange

class RoPE(nn.Module):
    def __init__(self,
                 theta: float,
                 d_k: int,
                 max_seq_len:int,
                 device: torch.device | None=None):
        super().__init__()

        self.d_k = d_k
        self.max_len = max_seq_len

        i = torch.arange(max_seq_len, device=device)
        theta_pow = 1.0 / (theta ** (torch.arange(0, d_k, 2,  device=device) /d_k))

        frequencies = torch.outer(i, theta_pow)

        self.register_buffer("cos_values", torch.cos(frequencies))
        self.register_buffer("sin_values", torch.sin(frequencies))

    def forward(self, x:torch.Tensor, token_positions:torch.Tensor)-> torch.Tensor:

        cos_values_token = self.cos_values[token_positions].unsqueeze(1)
        sin_values_token = self.sin_values[token_positions].unsqueeze(1)

        x_rearranged = rearrange(
            x,
            "... (half two) -> ... half two",
            two=2,
        )

        x_0 = x_rearranged[..., 0]
        x_1 = x_rearranged[..., 1]

        real_part = x_0 * cos_values_token - x_1 * sin_values_token
        img_part = x_0 * sin_values_token + x_1 * cos_values_token

        x_rearranged = torch.stack((real_part, img_part), -1)

        x = rearrange(x_rearranged,
                      "... half two -> ... (half two)")
        return x


