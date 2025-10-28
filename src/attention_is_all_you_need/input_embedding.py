import torch

default_batch = [
    "Hello, world!",
    "From Canada"
]

def word2vec(word: str):
    return torch.nn.Embedding(word)

def input_embedding(batch_tensor: torch.tensor = torch.tensor(default_batch)):
    """
    Input embedding

    Args:
        batch_tensor (torch.tensor): tensor of sentence list.

    Returns:
        torch.tensor of (batch, sequence, vector) : each vector contains properties of a word within a sentence.
    """

    result = []
    for sentence in batch_tensor:
        vector_list = []
        for word in sentence:
            vector_list.append(word2vec(word))
        result.append(vector_list)

    return torch.tensor(result)