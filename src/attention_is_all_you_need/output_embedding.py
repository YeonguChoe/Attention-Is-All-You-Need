import torch
import math


def output_embedding(
    sequence: torch.Tensor, vocab_size: int, d_model: int = 512
) -> torch.Tensor:
    embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
    embedded_sequence = embedding(sequence)
    return embedded_sequence * math.sqrt(d_model)
