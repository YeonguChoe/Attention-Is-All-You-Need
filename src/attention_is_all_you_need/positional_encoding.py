import torch
import math

from utils import *
from input_embedding import input_embedding


def positional_encoding(embeddings: torch.Tensor) -> torch.Tensor:
    sequence_length = embeddings.shape[0]  # 5
    d_model = embeddings.shape[1]  # 512
    PE = torch.zeros(sequence_length, d_model)
    for position in range(sequence_length):
        for index in range(d_model):
            if index % 2 == 0:
                PE[position, index] = math.sin(position / (10000 ** (index / d_model)))
            else:
                PE[position, index] = math.cos(
                    position / (10000 ** ((index - 1) / d_model))
                )
    return PE


# Example
tokenizer = CustomTokenizer()
tokenizer.train()

tokens = tokenizer.tokenize("My name is Yeongu Choe")
y = torch.tensor(tokens)
vocab_size = tokenizer.get_vocab_size()

z = input_embedding(y, vocab_size)

w = positional_encoding(z)

print((z + w).shape)
