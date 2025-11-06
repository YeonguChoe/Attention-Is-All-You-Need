import torch
import torch.nn as nn


class AddAndNorm(nn.Module):
    def __init__(self, d_model: int = 512, dropout: float = 0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x + self.dropout(sublayer_output))
