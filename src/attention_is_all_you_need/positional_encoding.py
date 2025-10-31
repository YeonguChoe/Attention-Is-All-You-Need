import torch
import math

def positional_encoding(batch: torch.Tensor) -> torch.Tensor:
    result = []
    d_model = 512
    for position in range(batch.size(1)):
        PE = []
        for i in range(d_model):
            if i % 2 == 0:
                PE.append(math.sin(position / 10000 ** (2 * i / d_model)))
            else:
                PE.append(math.cos(position / 10000 ** (2 * i / d_model)))
        result.append(torch.tensor(PE))
    
    return torch.stack(result, dim=0).unsqueeze(0)