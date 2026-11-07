"""CI helper: train_50m_results.json -> README results table.
Runs automatically in the Train-50M workflow after training."""
import json
import sys

MARK_START = "<!-- 50M-RESULTS:START -->"
MARK_END = "<!-- 50M-RESULTS:END -->"


def render(r):
    h = r["history"]
    hist = " -> ".join("{:.0%}".format(x["train_acc"]) for x in h) or "n/a"
    lines = [
        "**Model:** {}  ".format(r["model"]),
        "**Parameters:** {:,}".format(r["parameters"]),
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        "| Test accuracy (before training) | {:.1%} |".format(r["test_start"]["accuracy"]),
        "| Test accuracy (after training) | **{:.1%}** |".format(r["test_final"]["accuracy"]),
        "| Test loss (before -> after) | {:.3f} -> {:.3f} |".format(r["test_start"]["loss"], r["test_final"]["loss"]),
        "| Training algorithm | {} |".format(r["training_algorithm"]),
        "| Training data | {} |".format(r["training_data"]),
        "| Epochs / samples | {} / {} |".format(r["epochs_completed"], r["samples_trained"]),
        "| Train accuracy trajectory | {} |".format(hist),
        "| Spike sparsity (rates < 0.1) | {:.1%} |".format(r["spike_sparsity"]),
        "| STDP plasticity index | {} |".format(r["stdp_plasticity_index"]),
        "| Membrane-cache updates (memory persistence) | {} |".format(r["membrane_cache_updates"]),
        "| CI runtime | {} min |".format(r["runtime_min"]),
        "",
        "_Trained on GitHub Actions (free 2-core CPU runner) with the dopamine-gated 3-factor rule - no backprop through time, fully local updates. Full details: TRAINING_50M.md_",
    ]
    return "\n".join(lines)


def main():
    with open("train_50m_results.json", encoding="utf-8") as f:
        r = json.load(f)
    with open("README.md", encoding="utf-8") as f:
        readme = f.read()
    if MARK_START not in readme or MARK_END not in readme:
        print("ERROR: README markers missing", file=sys.stderr)
        sys.exit(1)
    pre, rest = readme.split(MARK_START, 1)
    _, post = rest.split(MARK_END, 1)
    readme = pre + MARK_START + "\n" + render(r) + "\n" + MARK_END + post
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme)
    print("README updated with 50M training results")


if __name__ == "__main__":
    main()
