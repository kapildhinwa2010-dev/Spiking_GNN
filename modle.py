import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class SigmoidSpikingNeuron(nn.Module):
    """
    Differentiable LIF neuron using Sigmoid surrogate for backprop.
    Learnable decay (tau) and threshold per dimension.
    """
    def __init__(self, dim, sharpness=5.0):
        super().__init__()
        self.sharpness = sharpness
        self.decay = nn.Parameter(torch.ones(dim) * 0.8)
        self.threshold = nn.Parameter(torch.ones(dim) * 1.0)

    def forward(self, current, membrane):
        # Clamp decay for stability
        decay = torch.sigmoid(self.decay) * 0.85 + 0.1
        threshold = torch.abs(self.threshold) + 0.1
        
        membrane = decay * membrane + current
        spike = torch.sigmoid(self.sharpness * (membrane - threshold))
        membrane = membrane - spike * threshold
        
        return spike, membrane

class MultiHeadEdgeConv(nn.Module):
    """
    Multi-Head Message Passing with 3D Spatial + Chemical Edge Features.
    """
    def __init__(self, node_dim, edge_dim, hidden_dim, num_heads=16, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.head_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + edge_dim, self.head_dim),
                nn.SiLU(),
                nn.Linear(self.head_dim, self.head_dim)
            ) for _ in range(num_heads)
        ])
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.neuron = SigmoidSpikingNeuron(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.res_proj = nn.Linear(node_dim, hidden_dim) if node_dim != hidden_dim else nn.Identity()

    def forward(self, x, edge_index, edge_attr, membrane):
        N = x.size(0)
        row, col = edge_index[0], edge_index[1]
        
        x_proj = self.node_proj(self.norm1(x))
        residual = self.res_proj(x)
        
        if row.numel() == 0:
            msg = torch.zeros(N, x_proj.size(1), device=x.device)
        else:
            src_feat = x_proj[row]
            head_outputs = [
                self.dropout(head_mlp(torch.cat([src_feat, edge_attr], dim=-1))) 
                for head_mlp in self.head_mlps
            ]
            msg_per_edge = torch.cat(head_outputs, dim=-1) 
            msg = torch.zeros(N, x_proj.size(1), device=x.device, dtype=x.dtype).index_add_(0, col, msg_per_edge)
            
        h = self.norm2(msg + residual)
        spike, new_membrane = self.neuron(h, membrane)
        return spike, new_membrane

class WholeBrainSpikingNet(nn.Module):
    """
    241M Parameter God-Tier Neuromorphic Network.
    """
    def __init__(self, node_dim=5, edge_dim=10, hidden_dim=3072, 
                 num_layers=8, num_heads=16, timesteps=4, num_classes=10, dropout=0.1):
        super().__init__()
        self.timesteps = timesteps
        
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        
        self.layers = nn.ModuleList([
            MultiHeadEdgeConv(hidden_dim, edge_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, edge_index, edge_attr):
        N = x.size(0)
        h = self.input_proj(x)
        membranes = [torch.zeros(N, h.size(1), device=x.device) for _ in self.layers]
        spike_accum = torch.zeros(N, h.size(1), device=x.device)
        
        for t in range(self.timesteps):
            z = h
            for i, layer in enumerate(self.layers):
                # Gradient Checkpointing to save 80% VRAM
                if self.training:
                    z, membranes[i] = checkpoint(layer, z, edge_index, edge_attr, membranes[i], use_reentrant=False)
                else:
                    z, membranes[i] = layer(z, edge_index, edge_attr, membranes[i])
            spike_accum = spike_accum + z
            
        avg_rate = spike_accum / float(self.timesteps)
        logits = self.readout(avg_rate.mean(dim=0, keepdim=True))
        return logits, avg_rate