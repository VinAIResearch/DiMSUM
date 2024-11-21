import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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

        self.linear_fc1 = nn.Linear(dim, ffn_hidden_size_1, bias=add_bias_linear, device=device)
        self.linear_fc1.is_expert = is_expert

        if gated_linear_unit:

            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                return F.gelu(x[0]) * x[1]

            self.activation_func = glu
        else:
            self.activation_func = F.gelu

        self.linear_fc2 = nn.Linear(ffn_hidden_size_2, dim, bias=add_bias_linear, device=device)

    def forward(self, hidden_states, inference_params=None):
        intermediate = self.linear_fc1(hidden_states)
        intermediate = self.activation_func(intermediate)
        output = self.linear_fc2(intermediate)
        return output


class GatedMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer=F.gelu,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.act_layer = act_layer()

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = self.act_layer(x1) * x2
        return self.w3(hidden)
