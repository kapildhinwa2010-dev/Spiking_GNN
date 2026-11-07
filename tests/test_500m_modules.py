import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from graph_builder import make_3d_graph
from model_500m import SpikingGraph3D_500M
from train_utils import make_optimizer


def test_make_3d_graph_shapes():
    x, ei, ea = make_3d_graph(n=8, e=16, seed=0)
    assert x.shape == (8, 5)
    assert ei.shape[0] == 2
    assert ea.shape[0] == ei.shape[1]
    assert ea.shape[1] == 10
    assert torch.isfinite(x).all()
    assert torch.isfinite(ea).all()
    assert (ei[0] != ei[1]).all()


def test_model_500m_tiny_forward():
    torch.manual_seed(0)
    model = SpikingGraph3D_500M(hidden_dim=32, num_layers=2, num_heads=2, timesteps=2, num_classes=10)
    model.eval()
    x, ei, ea = make_3d_graph(n=8, e=16, seed=1)
    with torch.no_grad():
        logits, avg_rate = model(x, ei, ea)
    assert logits.shape == (1, 10)
    assert avg_rate.shape == (8, 32)
    assert torch.isfinite(avg_rate).all()


def test_make_optimizer_fallback():
    model = SpikingGraph3D_500M(hidden_dim=32, num_layers=1, num_heads=2, timesteps=2)
    opt, name = make_optimizer(model, lr=1e-4)
    assert name in ("AdamW8bit", "AdamW")
    assert opt is not None
