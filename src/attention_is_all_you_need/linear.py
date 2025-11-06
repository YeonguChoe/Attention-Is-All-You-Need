import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, d_model: int = 512, vocab_size: int = 10000):
        super().__init__()

        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
