from utils import *
import torch
import math

def input_embedding(sequence: torch.Tensor, vocab_size: int) -> torch.Tensor:
    d_model = 512
    embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
    embedded_sequence = embedding(sequence)
    return embedded_sequence * math.sqrt(d_model)


# Example
tokenizer = CustomTokenizer()
tokenizer.train()

tokens = tokenizer.tokenize("My name is Yeongu Choe")
y = torch.tensor(tokens)
vocab_size = tokenizer.get_vocab_size()

z = input_embedding(y,vocab_size).shape
print(z)