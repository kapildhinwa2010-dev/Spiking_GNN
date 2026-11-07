# ============================================================
# 🧠 500M SPIKING GRAPH 3D MODEL - TRAINING PIPELINE
# Author Concept: Kapil Dhinwa
# Hidden=3648 | Layers=12 | Heads=16 | Timesteps=3 -> ~503M params
# Now modular (content extracted 1:1 from the original cell):
#   - architecture  : model_500m.py (SpikingGraph3D_500M)
#   - graph data    : graph_builder.py (make_3d_graph)
#   - optimizer/save: train_utils.py (make_optimizer, get_save_dir, save_checkpoint)
# Note: the 8-bit optimizer needs bitsandbytes (see requirements.txt);
# a normal AdamW fallback is used automatically if it is unavailable.
# ============================================================

import gc
import time

import numpy as np
import torch
import torch.nn.functional as F

from graph_builder import make_3d_graph
from model_500m import SpikingGraph3D_500M
from train_utils import make_optimizer, get_save_dir, save_checkpoint

USE_GOOGLE_DRIVE = True
SAVE_CHECKPOINT = True
LOCAL_SAVE_DIR = "brainai_500m"

# 500M architecture config
CFG = dict(
    hidden_dim=3648,     # gives ~503M params with 12 layers, 16 heads
    num_layers=12,
    num_heads=16,
    timesteps=3,
    node_dim=5,          # x, y, z, degree_norm, avg_dist_norm
    edge_dim=10,         # dx, dy, dz, dist + 6 neurotransmitter-like features
    num_classes=10,
    dropout=0.05,
)

# Training config. Keep graph tiny for 500M on T4.
TRAIN = dict(
    steps=20,
    lr=1e-4,
    graph_nodes=32,
    graph_edges=96,
    eval_nodes=16,
    eval_edges=32,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {device}")

if device.type == "cuda":
    # Speed help on T4/A100
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("⚠️ CPU detected. 500M training practical nahi hai.")
    print("⚠️ Sirf 1 micro-step sanity check chalaya jayega.")
    TRAIN["steps"] = 1
    TRAIN["graph_nodes"] = 8
    TRAIN["graph_edges"] = 16
    TRAIN["eval_nodes"] = 8
    TRAIN["eval_edges"] = 16

SAVE_DIR = get_save_dir(USE_GOOGLE_DRIVE, LOCAL_SAVE_DIR)


def train_500m():
    print("\n" + "=" * 75)
    print("  🧠 BUILDING 500M SPIKING GRAPH 3D MODEL")
    print("=" * 75)

    model = SpikingGraph3D_500M(**CFG).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔥 Parameters: {params:,} ({params / 1e6:.2f}M)")
    print(f"⚙️ Config: {CFG}")
    print(f"📊 Train config: {TRAIN}")

    if device.type == "cpu":
        print("⚠️ CPU par 500M model sirf sanity check ke liye hai.")
        print("⚠️ Real training ke liye GPU chahiye: Colab GPU / Kaggle T4x2 / A100.")

    opt, opt_name = make_optimizer(model, lr=TRAIN["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=TRAIN["steps"],
    )

    model.train()
    trained_steps = 0

    print("\n🚀 Starting training...")
    print("-" * 75)

    for step in range(TRAIN["steps"]):
        x = ei = ea = logits = avg_rate = loss = None

        try:
            x, ei, ea = make_3d_graph(
                n=TRAIN["graph_nodes"],
                e=TRAIN["graph_edges"],
                seed=1000 + step,
            )

            x = x.to(device)
            ei = ei.to(device)
            ea = ea.to(device)

            opt.zero_grad(set_to_none=True)

            logits, avg_rate = model(x, ei, ea)

            # Self-supervised target:
            # Predict normalized degree from average firing rate.
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

            if device.type == "cuda":
                mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            else:
                mem_gb = 0.0

            print(
                f"✅ Step {step + 1:02d}/{TRAIN['steps']:02d} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                f"VRAM: {mem_gb:.2f} GB | "
                f"Opt: {opt_name}"
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\n❌ CUDA OUT OF MEMORY.")
                print("👉 Reduce karo: TRAIN['graph_nodes'], TRAIN['graph_edges'], CFG['timesteps']")
                print("👉 Ya Kaggle A100 / Colab Pro use karo.")
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

    return model, params, trained_steps


def save_model(model, params, trained_steps):
    """Thin wrapper honoring SAVE_CHECKPOINT (see train_utils.save_checkpoint)."""
    if not SAVE_CHECKPOINT:
        print("⚠️ SAVE_CHECKPOINT=False, checkpoint skip.")
        return None
    return save_checkpoint(model, SAVE_DIR, CFG, TRAIN, params, trained_steps)


def quick_eval(model):
    print("\n" + "=" * 75)
    print("  🔬 QUICK 500M BRAIN SCAN")
    print("=" * 75)

    model.eval()

    x, ei, ea = make_3d_graph(
        n=TRAIN["eval_nodes"],
        e=TRAIN["eval_edges"],
        seed=999,
    )

    x = x.to(device)
    ei = ei.to(device)
    ea = ea.to(device)

    with torch.no_grad():
        logits, avg_rate = model(x, ei, ea)

    firing_rates = avg_rate.mean(dim=1).cpu().numpy()
    active_fraction = (avg_rate > 0.5).float().mean().item()
    mean_firing = avg_rate.mean().item()

    print(f"🔢 Eval nodes: {x.size(0)}")
    print(f"🔗 Eval edges: {ei.size(1)}")
    print(f"🔥 Mean firing rate: {mean_firing:.4f}")
    print(f"⚡ Active fraction (>0.5): {active_fraction * 100:.2f}%")
    print(f"🧊 Sparse fraction: {(1 - active_fraction) * 100:.2f}%")

    top5 = np.argsort(firing_rates)[::-1][:5]
    print("\n🌟 Top 5 most active nodes:")
    for rank, idx in enumerate(top5, 1):
        print(f"   {rank}. Node {idx} | Rate {firing_rates[idx]:.4f}")

    print("=" * 75)


# ============================================================
# 11. RUN EVERYTHING
# ============================================================
if __name__ == "__main__":
    t0 = time.time()

    model, params, trained_steps = train_500m()

    if trained_steps > 0:
        save_model(model, params, trained_steps)
        quick_eval(model)
    else:
        print("\n❌ Koi successful training step nahi hua. Checkpoint save nahi kiya.")

    total_time = time.time() - t0
    print(f"\n⏱️ Total runtime: {total_time:.2f} seconds")
