from dataclasses import dataclass
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from bias_gelu import bias_gelu_impl

class MLP(nn.Module):
    def __init__(
        self, 
        dim, 
        add_bias_linear: bool = False, 
        gated_linear_unit: bool = True, 
        is_expert: bool = False, 
        layer_idx=None,
        device=None,
    ):
        super().__init__()
        
        self.layer = layer_idx
        ffn_hidden_size_1 = 4 * dim
        ffn_hidden_size_2 = 4 * dim
        
        # If this is a gated linear unit we double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        if gated_linear_unit:
            ffn_hidden_size_1 *= 2
            
        self.linear_fc1 = nn.Linear(dim, ffn_hidden_size_1, bias = add_bias_linear, device = device)
        self.linear_fc1.is_expert = is_expert

        if gated_linear_unit:

            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                return F.gelu(x[0]) * x[1]

            self.activation_func = glu
        else:
            self.activation_func = F.gelu

        self.linear_fc2 = nn.Linear(ffn_hidden_size_2, dim, bias = add_bias_linear, device=device)

    def forward(self, hidden_states, inference_params=None):
        intermediate = self.linear_fc1(hidden_states)
        intermediate = self.activation_func(intermediate)
        output = self.linear_fc2(intermediate)
        return output