import torch
import math
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        
        self.d_model = d_model
        self.W_query = nn.Linear(d_model, d_model, bias=False)
        self.W_keys = nn.Linear(d_model, d_model, bias=False)
        self.W_values = nn.Linear(d_model, d_model, bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        Query = self.W_query(input)
        Keys = self.W_keys(input)
        Values = self.W_values(input)
        d_k = Query.size(-1)
        
        transposed_Keys = torch.transpose(Keys, -2, -1)
        weighted_sum = torch.matmul(Query, transposed_Keys) / math.sqrt(d_k)
        attention_weights = torch.softmax(weighted_sum, dim=-1)
        attention = torch.matmul(attention_weights, Values)    
        
        return attention