import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from homeostasis import HomeostaticRegulator


def test_threshold_rises_when_overfiring():
    reg = HomeostaticRegulator(4, target_rate=0.2)
    before = reg.threshold.clone()
    reg.step(torch.full((4,), 0.9))
    assert torch.all(reg.threshold > before)


def test_threshold_falls_when_underfiring():
    reg = HomeostaticRegulator(4, target_rate=0.2)
    before = reg.threshold.clone()
    reg.step(torch.zeros(4))
    assert torch.all(reg.threshold < before)


def test_rate_converges_to_target():
    reg = HomeostaticRegulator(1, target_rate=0.2, lr=0.05)
    th = reg.threshold[0].item()
    rate = 1 / (1 + math.exp(-4 * (1.0 - th)))
    for _ in range(400):
        th = reg.step(torch.tensor([rate]))[0].item()
        rate = 1 / (1 + math.exp(-4 * (1.0 - th)))
    assert abs(rate - 0.2) < 0.1
