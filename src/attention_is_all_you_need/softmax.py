import torch
import torch.nn as nn


class Softmax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()

        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=self.dim)
