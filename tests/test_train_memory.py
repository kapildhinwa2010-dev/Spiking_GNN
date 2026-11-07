import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_memory import train_bptt, train_eprop


def test_eprop_local_learning_works():
    _, r = train_eprop(epochs=40, n_hidden=32)
    assert r["loss"] < r["start_loss"] * 0.5, "e-prop should substantially reduce loss"


def test_bptt_learning_works():
    _, r = train_bptt(epochs=40, n_hidden=32)
    assert r["loss"] < r["start_loss"], "BPTT should reduce loss"
