import torch
import torch.nn as nn
from mlp import MLP


def sinkhorn(cost, tol=0.0001):
    "Sinkhorn based MoE routing function"
    cost = torch.exp(2.0 * cost)
    d0 = torch.ones(cost.size(0), device=cost.device, dtype=cost.dtype)
    # d1 = torch.ones(cost.size(1), device=cost.device, dtype=cost.dtype)
    d1 = 1 / (cost.size(1) * torch.sum(cost, 0))

    eps = 0.00000001
    error = 1e9
    d1_old = d1
    while error > tol:
        d0 = (1 / d0.size(0)) * 1 / (torch.sum(d1 * cost, 1) + eps)
        d1 = (1 / d1.size(0)) * 1 / (torch.sum(d0.unsqueeze(1) * cost, 0) + eps)
        error = torch.mean(torch.abs(d1_old - d1))
        d1_old = d1
    return d1 * cost * d0.unsqueeze(1)


class SwitchMLP(nn.Module):
    """
    Top-1 Mixture of Experts Layer. Routes input to one of N MLP "experts"
    Curently supports Sinkhorn based expert routing.
    """

    def __init__(
        self,
        dim: int,
        layer_idx=None,
        mamba_moe_layers=None,
        num_moe_experts: int = None,
        add_bias_linear: bool = False,
        gated_linear_unit: bool = True,
        routing_mode: str = "top1",
    ):
        super().__init__()

        self.layer = layer_idx
        if mamba_moe_layers:
            self.num_moe_experts = int(mamba_moe_layers[layer_idx - 1][-1])
        else:
            self.num_moe_experts = num_moe_experts
        self.router = torch.nn.Linear(dim, self.num_moe_experts)
        self.routing = routing_mode  # 'sinkhorn', 'top1', 'top2', 'sinkhorn_top2'
        self.route_algo = sinkhorn
        self.router_activation = torch.sigmoid

        self.num_local_experts = self.num_moe_experts
        self.local_expert_indices = [i for i in range(self.num_local_experts)]

        self.local_experts = torch.nn.ModuleList()
        for _ in range(self.num_local_experts):
            expert = MLP(
                dim,
                add_bias_linear=add_bias_linear,
                gated_linear_unit=gated_linear_unit,
                is_expert=True,
                layer_idx=layer_idx,
            )
            self.local_experts.append(expert)

    def gather_indices(self, local_indices):
        return local_indices

    def forward(self, hidden_states, inference_params=None):

        hidden_shape = hidden_states.shape
        route = self.router(hidden_states)
        route = route.view(-1, self.num_moe_experts)

        if self.routing == "sinkhorn":
            route = self.router_activation(route)
            max_prob, max_ind = torch.max(route, dim=1)
        else:
            route = torch.softmax(route, dim=1)
            max_prob, max_ind = torch.max(route, dim=1)

        max_prob = torch.unsqueeze(max_prob, 1)
        hidden_states = hidden_states.view(-1, hidden_shape[-1])

        global_hidden_states = hidden_states
        global_indices = max_ind
        output_total = torch.zeros_like(global_hidden_states)

        for expert_num, expert in enumerate(self.local_experts):
            local_expert_index = self.local_expert_indices[expert_num]
            local_indices = (global_indices == local_expert_index).nonzero()
            hidden = global_hidden_states[local_indices, :]
            output = expert(hidden)
            output_total[local_indices, :] = output

        output_total = output_total * max_prob
        output_total = output_total.view(hidden_shape)

        return output_total
