# ============================================================
# NEUROSPIKING MEMORY MODULES
# 1. MembraneCache   - the "KV cache" of a spiking network:
#                      membrane potentials persist ACROSS forward
#                      calls, so the brain keeps thinking instead
#                      of resetting on every call.
# 2. SpikingMemory   - working memory: exponential trace of node
#                      activations + top-k recall.
# 3. EligibilityTrace- e-prop style local learning traces
#                      (online learning without backprop).
# ============================================================

import torch
import torch.nn as nn


class MembraneCache:
    """Spiking KV-cache: persists per-layer membrane potentials."""

    def __init__(self, num_nodes, hidden_dim, num_layers, device=None):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.membranes = [
            torch.zeros(num_nodes, hidden_dim, device=device)
            for _ in range(num_layers)
        ]
        self.last_avg_rate = None
        self.updates = 0

    def to(self, device):
        self.membranes = [m.to(device) for m in self.membranes]
        return self

    def reset(self):
        self.membranes = [torch.zeros_like(m) for m in self.membranes]
        self.last_avg_rate = None
        self.updates = 0
        return self


@torch.no_grad()
def forward_cached(model, x, edge_index, edge_attr, cache, timesteps=None):
    """Run the spiking graph model READING/WRITING membrane state
    from the cache instead of resetting it to zero every call.
    This is the spiking analogue of a transformer KV-cache."""
    N = x.size(0)
    h = model.input_proj(x)
    timesteps = timesteps or model.timesteps
    spike_accum = torch.zeros(N, h.size(1), device=x.device)

    for _ in range(timesteps):
        z = h
        for i, layer in enumerate(model.layers):
            z, cache.membranes[i] = layer(z, edge_index, edge_attr, cache.membranes[i])
        spike_accum = spike_accum + z

    avg_rate = spike_accum / float(timesteps)
    cache.last_avg_rate = avg_rate
    cache.updates += 1
    return avg_rate


class SpikingMemory(nn.Module):
    """Working memory: EMA trace of node activations + top-k recall."""

    def __init__(self, num_nodes, decay=0.9):
        super().__init__()
        self.register_buffer("trace", torch.zeros(num_nodes))
        self.register_buffer("decay", torch.tensor(float(decay)))

    def update(self, rates):
        # rates: (N,) per-node firing rates
        self.trace = self.decay * self.trace + (1.0 - self.decay) * rates.detach()
        return self.trace

    def recall(self, k=5):
        """Top-k most-active-memory node indices."""
        return torch.topk(self.trace, k).indices


class EligibilityTrace:
    """e-prop style eligibility trace for local (online) learning."""

    def __init__(self, param, decay=0.95):
        self.trace = torch.zeros_like(param)
        self.decay = decay

    def update(self, grad_like):
        self.trace.mul_(self.decay).add_(grad_like.detach())

    def apply(self, param, lr, learning_signal):
        with torch.no_grad():
            param.add_(self.trace * (lr * float(learning_signal)))



class MultiTimescaleMemory(nn.Module):
    """Working memory on two timescales (fast seconds-like, slow minutes-like).
    The fast trace reacts strongly to recent input; the slow trace outlasts it."""

    def __init__(self, num_nodes, fast_decay=0.8, slow_decay=0.99):
        super().__init__()
        self.register_buffer("fast", torch.zeros(num_nodes))
        self.register_buffer("slow", torch.zeros(num_nodes))
        self.register_buffer("fast_decay", torch.tensor(float(fast_decay)))
        self.register_buffer("slow_decay", torch.tensor(float(slow_decay)))

    def update(self, rates):
        rates = rates.detach()
        self.fast = self.fast_decay * self.fast + (1.0 - self.fast_decay) * rates
        self.slow = self.slow_decay * self.slow + (1.0 - self.slow_decay) * rates
        return self.fast, self.slow

    def recall(self, k=5, which="slow"):
        trace = self.slow if which == "slow" else self.fast
        return torch.topk(trace, k).indices
