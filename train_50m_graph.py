# ============================================================
# 🧠 50M SPIKING GRAPH 3D MODEL - TRAINING PIPELINE (500M ka 10x scale-down)
# Author Concept: Kapil Dhinwa
# Hidden=1152 | Layers=12 | Heads=16 | Timesteps=3 -> ~50.3M params
# 500M pipeline (train_500m.py + model_500m.py) ka exact architecture,
# sirf hidden 3648 -> 1152. CPU par bhi REAL training chalti hai
# (50M light hai), isliye sanity-only mode ki zaroorat nahi.
# Positive-result proof: fixed hold-out graphs pe eval MSE
# training se PEHLE vs BAAD me napa jaata hai (train_50m_graph_results.json).
# ============================================================

import gc
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from graph_builder import make_3d_graph
from model_500m import SpikingGraph3D_500M
from train_utils import make_optimizer, get_save_dir, save_checkpoint

USE_GOOGLE_DRIVE = True
SAVE_CHECKPOINT = True
LOCAL_SAVE_DIR = "brainai_50m_graph"
RESULTS_JSON = "train_50m_graph_results.json"

# 50M architecture config (500M ka 10x scale-down)
CFG = dict(
    hidden_dim=1152,     # ~50.3M params with 12 layers, 16 heads
    num_layers=12,
    num_heads=16,
    timesteps=3,
    node_dim=5,          # x, y, z, degree_norm, avg_dist_norm
    edge_dim=10,         # dx, dy, dz, dist + 6 neurotransmitter-like features
    num_classes=10,
    dropout=0.05,
)

# Training config - 50M CPU-friendly
TRAIN = dict(
    steps=60,
    lr=3e-4,
    graph_nodes=32,
    graph_edges=96,
    eval_nodes=24,
    eval_edges=64,
)

# Fixed hold-out graphs: before-vs-after proof ka basis
EVAL_SEEDS = [9998, 9999]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {device}")

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

SAVE_DIR = get_save_dir(USE_GOOGLE_DRIVE, LOCAL_SAVE_DIR)


def eval_mse(model):
    """Hold-out graphs pe self-supervised degree-prediction MSE."""
    model.eval()
    mses = []
    with torch.no_grad():
        for seed in EVAL_SEEDS:
            x, ei, ea = make_3d_graph(
                n=TRAIN["eval_nodes"], e=TRAIN["eval_edges"], seed=seed,
            )
            _, avg_rate = model(x.to(device), ei.to(device), ea.to(device))
            pred = avg_rate.mean(dim=1, keepdim=True)
            mses.append(F.mse_loss(pred, x[:, 3:4].to(device)).item())
    model.train()
    return sum(mses) / len(mses)


def train_50m_graph():
    print("\n" + "=" * 75)
    print("  🧠 BUILDING 50M SPIKING GRAPH 3D MODEL (500M architecture, 10x down)")
    print("=" * 75)

    model = SpikingGraph3D_500M(**CFG).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔥 Parameters: {params:,} ({params / 1e6:.2f}M)")
    print(f"⚙️ Config: {CFG}")
    print(f"📊 Train config: {TRAIN}")

    opt, opt_name = make_optimizer(model, lr=TRAIN["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN["steps"])

    # ---- baseline eval (training se PEHLE) ----
    mse_before = eval_mse(model)
    print(f"\n📐 Hold-out eval MSE BEFORE training: {mse_before:.6f}")

    model.train()
    trained_steps = 0
    loss_first = None
    loss_last = None

    print("\n🚀 Starting training...")
    print("-" * 75)

    for step in range(TRAIN["steps"]):
        x = ei = ea = logits = avg_rate = loss = None
        try:
            x, ei, ea = make_3d_graph(
                n=TRAIN["graph_nodes"], e=TRAIN["graph_edges"], seed=1000 + step,
            )
            x, ei, ea = x.to(device), ei.to(device), ea.to(device)

            opt.zero_grad(set_to_none=True)
            logits, avg_rate = model(x, ei, ea)

            # Self-supervised target: predict normalized degree from avg firing rate
            target = x[:, 3:4]
            pred = avg_rate.mean(dim=1, keepdim=True)
            loss = F.mse_loss(pred, target) + 0.01 * avg_rate.abs().mean()

            if torch.isnan(loss):
                raise ValueError("NaN loss detected")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            trained_steps += 1
            if loss_first is None:
                loss_first = loss.item()
            loss_last = loss.item()

            print(
                f"✅ Step {step + 1:02d}/{TRAIN['steps']:02d} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                f"Opt: {opt_name}"
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\n❌ OUT OF MEMORY.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                break
            raise
        except Exception as e:
            print(f"\n❌ Step {step + 1} error: {str(e)[:200]}")
            break
        finally:
            for var in [x, ei, ea, logits, avg_rate, loss]:
                if var is not None:
                    del var
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("-" * 75)
    print(f"🏁 Training finished. Successful steps: {trained_steps}/{TRAIN['steps']}")

    # ---- hold-out eval (training ke BAAD) ----
    mse_after = eval_mse(model)
    print(f"📐 Hold-out eval MSE AFTER training:  {mse_after:.6f}")

    improvement = 0.0
    if mse_before > 0 and mse_after < mse_before:
        improvement = (mse_before - mse_after) / mse_before * 100.0
        print(f"🎯 Improvement: {improvement:.1f}% (hold-out graphs pe generalizes bhi)")
    elif mse_after >= mse_before:
        print("⚠️ Hold-out MSE improve nahi hua - overfit/underfit check karo.")

    if SAVE_CHECKPOINT and trained_steps > 0:
        save_checkpoint(model, SAVE_DIR, CFG, TRAIN, params, trained_steps)

    # ---- brain scan ----
    model.eval()
    with torch.no_grad():
        x, ei, ea = make_3d_graph(
            n=TRAIN["eval_nodes"], e=TRAIN["eval_edges"], seed=999,
        )
        _, avg_rate = model(x.to(device), ei.to(device), ea.to(device))
    firing_rates = avg_rate.mean(dim=1).cpu().numpy()
    active_fraction = (avg_rate > 0.5).float().mean().item()
    mean_firing = avg_rate.mean().item()
    top5 = np.argsort(firing_rates)[::-1][:5]
    print("\n" + "=" * 75)
    print("  🔬 QUICK 50M BRAIN SCAN")
    print("=" * 75)
    print(f"🔥 Mean firing rate: {mean_firing:.4f}")
    print(f"⚡ Active fraction (>0.5): {active_fraction * 100:.2f}%")
    print(f"🧊 Sparse fraction: {(1 - active_fraction) * 100:.2f}%")
    print("\n🌟 Top 5 most active nodes:")
    for rank, idx in enumerate(top5, 1):
        print(f"   {rank}. Node {idx} | Rate {firing_rates[idx]:.4f}")
    print("=" * 75)

    return model, params, trained_steps, {
        "model": "SpikingGraph3D-50M (500M architecture 10x down: hidden 1152 x 12 layers x 16 heads)",
        "parameters": params,
        "device": str(device),
        "optimizer": opt_name,
        "steps_completed": trained_steps,
        "total_steps": TRAIN["steps"],
        "train_loss_first": loss_first,
        "train_loss_last": loss_last,
        "eval_mse_before": round(mse_before, 6),
        "eval_mse_after": round(mse_after, 6),
        "eval_improvement_pct": round(improvement, 2),
        "hold_out_seeds": EVAL_SEEDS,
        "mean_firing_rate": round(mean_firing, 4),
        "active_fraction_pct": round(active_fraction * 100, 2),
        "task": "self-supervised: predict normalized degree from avg firing rate",
        "note": "500M pipeline ka exact scale-down; hold-out graphs pe before/after MSE = generalization proof",
    }


if __name__ == "__main__":
    t0 = time.time()
    model, params, trained_steps, results = train_50m_graph()
    results["runtime_min"] = round((time.time() - t0) / 60.0, 2)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📝 Results saved: {RESULTS_JSON}")
    print(f"⏱️ Total runtime: {time.time() - t0:.2f} seconds")
