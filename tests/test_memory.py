import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model import WholeBrainSpikingNet
from memory import MembraneCache, SpikingMemory, EligibilityTrace, forward_cached


def make_graph(n=10, e=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 5, generator=g)
    ei = torch.randint(0, n, (2, e), generator=g)
    disp = x[ei[0], :3] - x[ei[1], :3]
    dist = disp.norm(dim=1, keepdim=True).clamp(min=1e-6)
    ea = torch.cat([disp, dist, torch.rand(e, 6, generator=g)], dim=1)
    return x, ei, ea


def test_forward_cached_persists_membrane():
    torch.manual_seed(0)
    model = WholeBrainSpikingNet(hidden_dim=32, num_layers=2, num_heads=2, timesteps=2, num_classes=10)
    model.eval()
    x, ei, ea = make_graph()
    cache = MembraneCache(x.size(0), 32, 2)

    r1 = forward_cached(model, x, ei, ea, cache)
    assert cache.updates == 1
    assert torch.isfinite(r1).all()

    before = cache.membranes[0].clone()
    r2 = forward_cached(model, x, ei, ea, cache)
    assert cache.updates == 2
    assert not torch.allclose(before, cache.membranes[0]), "membrane state should persist and evolve"
    assert torch.isfinite(r2).all()


def test_spiking_memory_trace_and_recall():
    mem = SpikingMemory(10, decay=0.5)
    mem.update(torch.ones(10))
    assert torch.allclose(mem.trace, torch.full((10,), 0.5))
    idx = mem.recall(3)
    assert idx.shape == (3,)


def test_eligibility_trace_accumulates():
    param = torch.nn.Parameter(torch.zeros(3))
    trace = EligibilityTrace(param, decay=0.95)
    trace.update(torch.ones(3))
    assert torch.allclose(trace.trace, torch.ones(3))
    trace.update(torch.ones(3))
    assert abs(trace.trace[0].item() - 1.95) < 1e-5
    trace.apply(param, lr=0.1, learning_signal=1.0)
    assert param.data[0].item() != 0.0
