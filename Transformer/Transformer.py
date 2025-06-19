import torch
import torch.nn as nn

from Transformer.Utils.Embedding import Embedding
from Transformer.Utils.RMSNorm import RMSNorm
from Transformer.Transformer_Block import TransformerBlock
from Transformer.Utils.linear import Linear


class Transformer(nn.Module):
    def __init__(self,
                 vocab_size:int,
                 context_length: int,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 d_k: int | None = None,
                 d_v: int | None = None,
                 theta: float | None = None,
                 use_rope: bool = False,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None
                 ):
        super().__init__()


        self.Embedding = Embedding(vocab_size, d_model, device, dtype)
        self.transformer_layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, d_k, d_v,
                                                    theta,use_rope, context_length,
                                                    device, dtype)
                                   for _ in range(num_layers)])
        self.RMSNorm =  RMSNorm(d_model, device=device, dtype=dtype)
        self.Output = Linear(d_model, vocab_size, device)
        self.device = device

    def forward(self,
                token_indices: torch.Tensor
                ) -> torch.Tensor:

        batch_size, seq_len = token_indices.shape

        token_positions = torch.arange(seq_len, device=self.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        inputs = self.Embedding(token_indices)

        for block in self.transformer_layers:
            inputs = block(inputs, token_positions)

        outputs = self.Output(self.RMSNorm(inputs))

        return outputs