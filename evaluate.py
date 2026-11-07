# ============================================================
# NEUROSPIKING EVALUATION SUITE
# Metrics:
#   1. Sparsity      - fraction of silent neurons
#   2. Hub suppression - correlation between node degree and firing rate
#   3. Lesion study - hub removal vs random removal (disinhibition)
#   4. Semantic proxy - connected words fire more similarly than random
#                     pairs (needs vocab.json, saved by train.py)
# Works with trained weights (config.json -> weights_file) or a random brain.
# Usage:
#   python evaluate.py                          # full-size from config.json
#   python evaluate.py --hidden 256 --nodes 120 # quick CPU run
# ============================================================

import argparse
import json
import os
import time

import numpy as np
import torch

from model import WholeBrainSpikingNet


def build_graph(n, e, seed=42):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 5, generator=g)
    src = torch.randint(0, n, (e,), generator=g)
    dst = torch.randint(0, n, (e,), generator=g)
    keep = src != dst
    ei = torch.stack([src[keep], dst[keep]], dim=0)
    disp = x[ei[0], :3] - x[ei[1], :3]
    dist = disp.norm(dim=1, keepdim=True).clamp(min=1e-6)
    chem = torch.rand(ei.size(1), 6, generator=g)
    ea = torch.cat([disp, dist, chem], dim=1)
    return x, ei, ea


@torch.no_grad()
def forward_rates(model, x, ei, ea):
    logits, avg_rate = model(x, ei, ea)
    return avg_rate  # (N, hidden)


def degrees(n, ei):
    return (torch.bincount(ei[0], minlength=n) + torch.bincount(ei[1], minlength=n)).float()


def main():
    ap = argparse.ArgumentParser(description="NeuroSpiking evaluation suite")
    ap.add_argument("--nodes", type=int, default=150)
    ap.add_argument("--edges", type=int, default=600)
    ap.add_argument("--hidden", type=int, default=None, help="override hidden_dim for quick CPU runs")
    ap.add_argument("--hubs", type=int, default=10)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    with open("config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    hidden = args.hidden or cfg["hidden_dim"]

    model = WholeBrainSpikingNet(
        node_dim=cfg.get("node_dim", 5),
        edge_dim=cfg.get("edge_dim", 10),
        hidden_dim=hidden,
        num_layers=cfg["num_layers"],
        num_heads=cfg.get("num_heads", 16),
        timesteps=cfg["timesteps"],
        num_classes=cfg.get("num_classes", 10),
    )
    weights = cfg.get("weights_file", "god_tier_241m.pth")
    try:
        model.load_state_dict(torch.load(weights, map_location="cpu"))
        print("Loaded weights: " + weights)
    except Exception as exc:
        print("Weights not loaded (" + weights + ") -> " + str(exc))
        print("Evaluating a randomly initialized brain (baseline, not trained).")
    model.eval()

    x, ei, ea = build_graph(args.nodes, args.edges)
    n = x.size(0)

    t0 = time.time()
    avg = forward_rates(model, x, ei, ea)
    latency = time.time() - t0
    rates = avg.mean(dim=1)  # per-neuron firing rate

    # 1. Sparsity
    sparsity = float((rates < 0.5).float().mean().item())

    # 2. Hub suppression
    deg = degrees(n, ei)
    hub_corr = 0.0
    if deg.std().item() > 0:
        hub_corr = float(np.corrcoef(deg.numpy(), rates.numpy())[0, 1])

    # 3. Lesion study
    base = rates.mean().item()
    hubs = torch.topk(deg, args.hubs).indices
    keep = (~torch.isin(ei[0], hubs)) & (~torch.isin(ei[1], hubs))
    hub_rates = forward_rates(model, x, ei[:, keep], ea[keep]).mean(dim=1)
    hub_delta = hub_rates.mean().item() - base
    rand_deltas = []
    for t in range(args.trials):
        rnd = torch.randperm(n, generator=torch.Generator().manual_seed(t))[: args.hubs]
        keepr = (~torch.isin(ei[0], rnd)) & (~torch.isin(ei[1], rnd))
        rr = forward_rates(model, x, ei[:, keepr], ea[keepr]).mean(dim=1)
        rand_deltas.append(rr.mean().item() - base)
    rand_delta = float(np.mean(rand_deltas))

    # 4. Semantic proxy (optional, needs vocab.json)
    semantic = {"available": False, "reason": "vocab.json not found - run train.py first"}
    if os.path.exists("vocab.json"):
        with open("vocab.json", encoding="utf-8") as f:
            vocab = json.load(f)
        m = min(n, len(vocab))
        src, dst = ei[0], ei[1]
        in_vocab = (src < m) & (dst < m)
        if int(in_vocab.sum()) >= 2:
            emb = avg[:m]
            a, b = emb[src[in_vocab]], emb[dst[in_vocab]]
            cos = (a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)
            g2 = torch.Generator().manual_seed(7)
            p = torch.randint(0, m, (int(in_vocab.sum()),), generator=g2)
            q = torch.randint(0, m, (int(in_vocab.sum()),), generator=g2)
            mask = p != q
            ra, rb = emb[p[mask]], emb[q[mask]]
            rcos = (ra * rb).sum(dim=1) / (ra.norm(dim=1) * rb.norm(dim=1) + 1e-8)
            semantic = {
                "available": True,
                "vocab_size": len(vocab),
                "connected_pair_similarity": float(cos.mean().item()),
                "random_pair_similarity": float(rcos.mean().item()),
            }

    results = {
        "config_used": {
            "hidden_dim": hidden, "num_layers": cfg["num_layers"],
            "timesteps": cfg["timesteps"], "nodes": n, "edges": int(ei.size(1)),
        },
        "sparsity_fraction_silent": sparsity,
        "mean_firing_rate": float(rates.mean().item()),
        "hub_suppression_correlation": hub_corr,
        "lesion_study": {
            "baseline_mean_firing": base,
            "hub_lesion_delta": hub_delta,
            "random_lesion_delta": rand_delta,
            "disinhibition_detected": bool(hub_delta > 0 and hub_delta > rand_delta),
        },
        "semantic_proxy": semantic,
        "forward_latency_s": latency,
    }

    print(json.dumps(results, indent=2))
    with open("evaluate_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved evaluate_results.json")


if __name__ == "__main__":
    main()
