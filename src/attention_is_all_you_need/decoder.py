import torch
import torch.nn as nn
from masked_multihead_attention import MaskedMultiHeadAttention
from multihead_attention import MultiHeadAttention
from feed_forward import FeedForward
from add_and_norm import AddAndNorm


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.masked_self_attention = MaskedMultiHeadAttention(d_model, num_heads)
        self.add_norm1 = AddAndNorm(d_model, dropout)
        self.encoder_decoder_attention = MultiHeadAttention(d_model, num_heads)
        self.add_norm2 = AddAndNorm(d_model, dropout)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.add_norm3 = AddAndNorm(d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        attn_output = self.masked_self_attention(x, x, x)
        x = self.add_norm1(x, attn_output)
        attn_output = self.encoder_decoder_attention(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.add_norm2(x, attn_output)
        ff_output = self.feed_forward(x)
        x = self.add_norm3(x, ff_output)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask)

        return x
