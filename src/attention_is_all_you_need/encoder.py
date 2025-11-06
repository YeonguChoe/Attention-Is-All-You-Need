import torch
import torch.nn as nn
import math
from multihead_attention import MultiHeadAttention
from feed_forward import FeedForward
from add_and_norm import AddAndNorm


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.add_norm1 = AddAndNorm(d_model, dropout)
        self.add_norm2 = AddAndNorm(d_model, dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        attn_output = self.self_attention(x, x, x, mask)
        x = self.add_norm1(x, attn_output)
        ff_output = self.feed_forward(x)
        x = self.add_norm2(x, ff_output)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int = 6,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
    ):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.register_buffer(
            "positional_encoding",
            self._create_positional_encoding(max_seq_len, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def _create_positional_encoding(
        self, max_seq_len: int, d_model: int
    ) -> torch.Tensor:
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        seq_len = x.size(1)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.positional_encoding[:, :seq_len, :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
