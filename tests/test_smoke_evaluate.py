import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_evaluate_smoke_tiny_cpu():
    """Bounded smoke test: evaluate.py runs end-to-end on a tiny random brain (< 60 s CPU)."""
    out = subprocess.run(
        [sys.executable, "evaluate.py", "--hidden", "32", "--nodes", "40", "--edges", "120", "--trials", "1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    combined = out.stdout + out.stderr
    assert out.returncode == 0, combined
    results = json.loads((ROOT / "evaluate_results.json").read_text())
    assert "sparsity_fraction_silent" in results
    assert "lesion_study" in results
