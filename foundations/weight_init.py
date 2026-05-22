import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        w = torch.randn(fan_out, fan_in) * std
        return torch.round(w, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:

        torch.manual_seed(0)
        std = math.sqrt(2.0 / fan_in)
        w = torch.randn(fan_out, fan_in) * std
        return torch.round(w, decimals=4).tolist()


    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)
    
        dims = [input_dim] + [hidden_dim] * num_layers
        stds = []
        weights = []

        for i in range(num_layers):
            if init_type == 'xavier':
                std = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
            elif init_type == 'kaiming':
                std = math.sqrt(2.0 / dims[i])
            else:
                std = 1.0
            
            W = torch.randn(dims[i + 1], dims[i]) * std
            weights.append(W)
        
        x = torch.randn(1, input_dim)
        for w in weights:
            x = x @ w.T
            x = torch.relu(x)
            stds.append(round(float(x.std()), 2))

        return stds

