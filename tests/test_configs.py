import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    with open(os.path.join(ROOT, "configs", name)) as f:
        return json.load(f)


def test_all_configs_valid():
    for name in ("241m.json", "500m.json", "800m.json"):
        cfg = load(name)
        assert cfg["hidden_dim"] % cfg["num_heads"] == 0
        assert cfg["num_layers"] >= 2
        assert cfg["timesteps"] >= 1


def test_800m_estimate_consistent():
    c241, c500, c800 = load("241m.json"), load("500m.json"), load("800m.json")
    k1 = c241["estimated_params"] / (c241["hidden_dim"] ** 2 * c241["num_layers"])
    k2 = c500["estimated_params"] / (c500["hidden_dim"] ** 2 * c500["num_layers"])
    k = (k1 + k2) / 2
    est = k * c800["hidden_dim"] ** 2 * c800["num_layers"]
    assert 7e8 < est < 9e8, "800M config should land in 700-900M by scaling estimate"
