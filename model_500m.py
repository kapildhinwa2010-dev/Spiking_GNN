# ============================================================
# 500M SPIKING GRAPH 3D MODEL (architecture)
# Extracted from the original 500M training cell (sections 3-5):
#   - SigmoidSpikingNeuron  : surrogate LIF spiking neuron
#   - MultiHeadEdgeConv     : edge-aware multi-head message passing
#   - SpikingGraph3D_500M   : ~503M params at hidden=3648, 12 layers
# Defaults: hidden_dim=3648 | num_layers=12 | heads=16 | timesteps=3
# ============================================================

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class SigmoidSpikingNeuron(nn.Module):
    """
    Differentiable surrogate spiking neuron.
    Learnable decay and threshold.
    """
    def __init__(self, dim, sharpness=5.0):
        super().__init__()
        self.sharpness = sharpness
        self.decay = nn.Parameter(torch.ones(dim) * 0.8)
        self.threshold = nn.Parameter(torch.ones(dim) * 1.0)

    def forward(self, current, membrane):
        decay = torch.sigmoid(self.decay) * 0.85 + 0.1
        threshold = torch.abs(self.threshold) + 0.1

        membrane = decay * membrane + current
        spike = torch.sigmoid(self.sharpness * (membrane - threshold))
        membrane = membrane - spike * threshold

        return spike, membrane


class MultiHeadEdgeConv(nn.Module):
    """
    Multi-head message passing with edge features.
    Edge features contain 3D displacement + distance + chemical/mock neurotransmitter features.
    """
    def __init__(self, node_dim, edge_dim, hidden_dim, num_heads=16, dropout=0.05):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.node_proj = nn.Linear(node_dim, hidden_dim)

        self.head_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + edge_dim, self.head_dim),
                nn.SiLU(),
                nn.Linear(self.head_dim, self.head_dim),
            )
            for _ in range(num_heads)
        ])

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.neuron = SigmoidSpikingNeuron(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.res_proj = (
            nn.Linear(node_dim, hidden_dim)
            if node_dim != hidden_dim
            else nn.Identity()
        )

    def forward(self, x, edge_index, edge_attr, membrane):
        N = x.size(0)
        row, col = edge_index[0], edge_index[1]

        x_norm = self.norm1(x)
        x_proj = self.node_proj(x_norm)
        residual = self.res_proj(x)

        if row.numel() == 0:
            msg = torch.zeros(N, x_proj.size(1), device=x.device, dtype=x_proj.dtype)
        else:
            src_feat = x_proj[row]

            head_outputs = []
            for head_mlp in self.head_mlps:
                head_in = torch.cat([src_feat, edge_attr], dim=-1)
                head_out = head_mlp(head_in)
                head_out = self.dropout(head_out)
                head_outputs.append(head_out)

            msg_per_edge = torch.cat(head_outputs, dim=-1)

            msg = torch.zeros(N, x_proj.size(1), device=x.device, dtype=x_proj.dtype)
            msg.index_add_(0, col, msg_per_edge)

        h = self.norm2(msg + residual.to(msg.dtype))
        spike, new_membrane = self.neuron(h, membrane)

        return spike, new_membrane


class SpikingGraph3D_500M(nn.Module):
    def __init__(
        self,
        node_dim=5,
        edge_dim=10,
        hidden_dim=3648,
        num_layers=12,
        num_heads=16,
        timesteps=3,
        num_classes=10,
        dropout=0.05,
    ):
        super().__init__()

        self.timesteps = timesteps
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.layers = nn.ModuleList([
            MultiHeadEdgeConv(
                node_dim=hidden_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_attr):
        N = x.size(0)

        h = self.input_proj(x)

        membranes = [
            torch.zeros(N, h.size(1), device=x.device, dtype=h.dtype)
            for _ in range(self.num_layers)
        ]

        spike_accum = torch.zeros(N, h.size(1), device=x.device, dtype=h.dtype)

        for t in range(self.timesteps):
            z = h
            for i, layer in enumerate(self.layers):
                if self.training:
                    z, membranes[i] = checkpoint(
                        layer,
                        z,
                        edge_index,
                        edge_attr,
                        membranes[i],
                        use_reentrant=False,
                    )
                else:
                    z, membranes[i] = layer(z, edge_index, edge_attr, membranes[i])

            spike_accum = spike_accum + z

        avg_rate = spike_accum / float(self.timesteps)
        mean_pool = avg_rate.mean(dim=0, keepdim=True)
        logits = self.readout(mean_pool)

        return logits, avg_rate
