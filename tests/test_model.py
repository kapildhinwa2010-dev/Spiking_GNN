import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model import WholeBrainSpikingNet, SigmoidSpikingNeuron


def make_graph(n=10, e=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 5, generator=g)
    ei = torch.randint(0, n, (2, e), generator=g)
    disp = x[ei[0], :3] - x[ei[1], :3]
    dist = disp.norm(dim=1, keepdim=True).clamp(min=1e-6)
    ea = torch.cat([disp, dist, torch.rand(e, 6, generator=g)], dim=1)
    return x, ei, ea


def test_neuron_output_range():
    neuron = SigmoidSpikingNeuron(dim=8)
    current = torch.randn(4, 8)
    membrane = torch.zeros(4, 8)
    spike, membrane = neuron(current, membrane)
    assert spike.shape == (4, 8)
    assert membrane.shape == (4, 8)
    assert (spike >= 0).all() and (spike <= 1).all()


def test_forward_shapes():
    torch.manual_seed(0)
    model = WholeBrainSpikingNet(hidden_dim=32, num_layers=2, num_heads=2, timesteps=2, num_classes=10)
    x, ei, ea = make_graph()
    logits, avg_rate = model(x, ei, ea)
    assert logits.shape == (1, 10)
    assert avg_rate.shape == (x.size(0), 32)
    assert torch.isfinite(avg_rate).all()


def test_backward_pass_finite():
    torch.manual_seed(0)
    model = WholeBrainSpikingNet(hidden_dim=32, num_layers=2, num_heads=2, timesteps=2, num_classes=10)
    x, ei, ea = make_graph()
    logits, avg_rate = model(x, ei, ea)
    loss = logits.sum() + 0.01 * avg_rate.abs().mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients were produced"
    for g in grads:
        assert torch.isfinite(g).all(), "non-finite gradient found"


def test_training_mode_checkpointing():
    torch.manual_seed(0)
    model = WholeBrainSpikingNet(hidden_dim=32, num_layers=2, num_heads=2, timesteps=2)
    model.train()
    x, ei, ea = make_graph()
    logits, avg_rate = model(x, ei, ea)
    assert torch.isfinite(avg_rate).all()
