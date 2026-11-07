import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from stdp import STDPLearner


def test_delta_signs_and_magnitude():
    s = STDPLearner(2, 2)
    assert s.delta(5) > 0
    assert s.delta(-5) < 0
    assert abs(s.delta(200)) < 1e-4
    assert abs(s.delta(-200)) < 1e-4


def test_apply_pair_clamps_at_wmax():
    s = STDPLearner(2, 2)
    for _ in range(300):
        s.apply_pair(0, 0, 5)
    assert s.w[0, 0] == s.wmax


def test_online_stdp_potentiates_causal_pair():
    s = STDPLearner(2, 2)
    s.init_traces()
    s.step_traces(torch.tensor([1, 0]), torch.tensor([0, 0]))  # pre fires first
    s.step_traces(torch.tensor([0, 0]), torch.tensor([1, 0]))  # post fires after
    assert s.w[0, 0] > 0


def test_online_stdp_depresses_anti_causal_pair():
    s = STDPLearner(2, 2)
    s.init_traces()
    s.w.fill_(0.5)
    s.step_traces(torch.tensor([0, 0]), torch.tensor([1, 0]))  # post fires first
    s.step_traces(torch.tensor([1, 0]), torch.tensor([0, 0]))  # pre fires after
    assert s.w[0, 0] < 0.5
