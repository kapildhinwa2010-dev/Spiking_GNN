import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from dopamine import DopamineSignal, RewardModulatedLearner


def run(pre_times, post_times, reward_time, lr=0.05):
    r = RewardModulatedLearner(2, 2, lr=lr)
    pre = torch.zeros(2, dtype=torch.int)
    post = torch.zeros(2, dtype=torch.int)
    for t in range(100):
        pre[0] = 1 if t in pre_times else 0
        post[0] = 1 if t in post_times else 0
        r.step(pre, post, reward=1.0 if t == reward_time else 0.0)
    return r.w[0, 0].item()


def test_reward_after_pairing_potentiates():
    w = run({10}, {12}, 15)
    assert w > 0.5 + 0.01, w


def test_late_reward_is_weaker():
    early = run({10}, {12}, 15) - 0.5
    late = run({10}, {12}, 80) - 0.5
    assert early > late


def test_reward_without_pairing_changes_nothing():
    w = run({10}, set(), 15)
    assert w == 0.5


def test_pairing_without_reward_changes_nothing():
    w = run({10}, {12}, None)
    assert w == 0.5


def test_negative_reward_depresses():
    w = run({10}, {12}, 15)
    # now with negative reward
    r = RewardModulatedLearner(2, 2, lr=0.05)
    pre = torch.zeros(2, dtype=torch.int)
    post = torch.zeros(2, dtype=torch.int)
    for t in range(100):
        pre[0] = 1 if t == 10 else 0
        post[0] = 1 if t == 12 else 0
        r.step(pre, post, reward=-1.0 if t == 15 else 0.0)
    assert r.w[0, 0].item() < 0.5
    assert abs(r.w[0, 0].item() - 0.5 + (w - 0.5)) < 1e-6  # symmetric


def test_dopamine_signal_decay():
    d = DopamineSignal(4)
    d.deliver(1.0)
    assert torch.all(d.signal == 1.0)
    d.decay(0.8)
    assert torch.allclose(d.signal, torch.full((4,), 0.8))
