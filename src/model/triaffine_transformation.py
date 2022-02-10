import torch.nn as nn
import torch

class Triaffine(nn.Module):
    """Triaffine transformation to fuse heterogeneous factors"""
    def __init__(self, n_in: int, std: float):
        super().__init__()
        empty_tensor = torch.empty(n_in + 1, n_in, n_in + 1)
        self.weight = nn.Parameter(nn.init.normal_(empty_tensor, mean=0, std=std))

    def forward(self, u: torch.tensor, v: torch.tensor, w: torch.tensor):
        """

        :param u: [batch_size, seq_length, n_in]
        :param v: [batch_size, seq_length, n_in]
        :param w: [batch_size, seq_length, n_in]
        :return:
        """
        u = torch.cat((u, torch.ones_like(u[..., :1])), dim=-1)
        v = torch.cat((v, torch.ones_like(v[..., :1])), dim=-1)
        w = torch.einsum('bij,b->bozij', self.weight, u)
