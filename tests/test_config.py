import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_config_exists_and_valid():
    path = os.path.join(ROOT, "config.json")
    assert os.path.exists(path), "config.json is missing"
    with open(path) as f:
        cfg = json.load(f)
    for key in ("node_dim", "edge_dim", "hidden_dim", "num_layers", "num_heads", "timesteps"):
        assert key in cfg, "missing config key: " + key
    assert cfg["hidden_dim"] % cfg["num_heads"] == 0, "hidden_dim must be divisible by num_heads"
    assert cfg["timesteps"] >= 1
    assert "weights_file" in cfg


def test_requirements_exists():
    assert os.path.exists(os.path.join(ROOT, "requirements.txt"))


def test_core_files_exist():
    for f in ("model.py", "train.py", "inference.py", "evaluate.py", "LICENSE"):
        assert os.path.exists(os.path.join(ROOT, f)), f + " is missing"
