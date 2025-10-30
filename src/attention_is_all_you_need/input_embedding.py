import torch
import math

def input_embedding(sequence: torch.Tensor, vocabulary_size: int) -> torch.Tensor:
    d_model = 512
    embedding = torch.nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=d_model)
    embedded_sequence = embedding(sequence)
    return embedded_sequence * math.sqrt(d_model)