import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 device: torch.device| None=None,
                 dtype: torch.dtype| None=None):
        super().__init__()
        self.Embedding = nn.Parameter(torch.empty((num_embeddings, embedding_dim),
                                                  device=device,
                                                  dtype=dtype))
        nn.init.trunc_normal_(self.Embedding.data, 0, 1, -3, 3)
        self.dtype = dtype
        self.device = device

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.Embedding[token_ids]
