import torch
import torch.nn as nn
from multihead_attention import MultiHeadAttention


class MaskedMultiHeadAttention(MultiHeadAttention):
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super().__init__(d_model, num_heads)

    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        seq_len = query.size(1)

        # Create causal mask
        mask = self.create_causal_mask(seq_len).to(query.device)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        # Call parent's forward with mask
        return super().forward(query, key, value, mask)
