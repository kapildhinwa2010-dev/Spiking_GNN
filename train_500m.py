# ============================================================
# 🧠 500M SPIKING GRAPH 3D MODEL
# Author Concept: Kapil Dhinwa
# Hidden=3648 | Layers=12 | Heads=16 | Timesteps=3
# Approx Parameters: ~503 Million
# ============================================================

!pip install -q bitsandbytes

import os
import gc
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ============================================================
# 1. SETTINGS
# ============================================================
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


# ============================================================
# 2. GOOGLE DRIVE MOUNT
# ============================================================
if USE_GOOGLE_DRIVE:
    try:
        from google.colab import drive
        if not os.path.exists("/content/drive/MyDrive"):
            drive.mount("/content/drive")
        SAVE_DIR = "/content/drive/MyDrive/BrainAI_500M"
    except Exception as e:
        print(f"⚠️ Drive mount failed: {e}")
        SAVE_DIR = LOCAL_SAVE_DIR
else:
    SAVE_DIR = LOCAL_SAVE_DIR

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"💾 Save directory: {SAVE_DIR}")


# ============================================================
# 3. SPIKING NEURON
# ============================================================
class SigmoidSpikingNeuron(nn.Module):
    """
    Differentiable surrogate spiking neuron.
    Learnable decay and threshold.
    """
    def __init__(self, dim, sharpness=5.0):
        super().__init__()
        self.sharpness = sharpness
        self.decay = nn.Parameter(torch.ones(dim) * 0.8)
        self.threshold = nn.Parameter(torch.ones(dim) * 1.0)

    def forward(self, current, membrane):
        decay = torch.sigmoid(self.decay) * 0.85 + 0.1
        threshold = torch.abs(self.threshold) + 0.1

        membrane = decay * membrane + current
        spike = torch.sigmoid(self.sharpness * (membrane - threshold))
        membrane = membrane - spike * threshold

        return spike, membrane


# ============================================================
# 4. MULTI-HEAD EDGE-AWARE GRAPH CONV
# ============================================================
class MultiHeadEdgeConv(nn.Module):
    """
    Multi-head message passing with edge features.
    Edge features contain 3D displacement + distance + chemical/mock neurotransmitter features.
    """
    def __init__(self, node_dim, edge_dim, hidden_dim, num_heads=16, dropout=0.05):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.node_proj = nn.Linear(node_dim, hidden_dim)

        self.head_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + edge_dim, self.head_dim),
                nn.SiLU(),
                nn.Linear(self.head_dim, self.head_dim),
            )
            for _ in range(num_heads)
        ])

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.neuron = SigmoidSpikingNeuron(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.res_proj = (
            nn.Linear(node_dim, hidden_dim)
            if node_dim != hidden_dim
            else nn.Identity()
        )

    def forward(self, x, edge_index, edge_attr, membrane):
        N = x.size(0)
        row, col = edge_index[0], edge_index[1]

        x_norm = self.norm1(x)
        x_proj = self.node_proj(x_norm)
        residual = self.res_proj(x)

        if row.numel() == 0:
            msg = torch.zeros(N, x_proj.size(1), device=x.device, dtype=x_proj.dtype)
        else:
            src_feat = x_proj[row]

            head_outputs = []
            for head_mlp in self.head_mlps:
                head_in = torch.cat([src_feat, edge_attr], dim=-1)
                head_out = head_mlp(head_in)
                head_out = self.dropout(head_out)
                head_outputs.append(head_out)

            msg_per_edge = torch.cat(head_outputs, dim=-1)

            msg = torch.zeros(N, x_proj.size(1), device=x.device, dtype=x_proj.dtype)
            msg.index_add_(0, col, msg_per_edge)

        h = self.norm2(msg + residual.to(msg.dtype))
        spike, new_membrane = self.neuron(h, membrane)

        return spike, new_membrane


# ============================================================
# 5. 500M SPIKING GRAPH 3D NET
# ============================================================
class SpikingGraph3D_500M(nn.Module):
    def __init__(
        self,
        node_dim=5,
        edge_dim=10,
        hidden_dim=3648,
        num_layers=12,
        num_heads=16,
        timesteps=3,
        num_classes=10,
        dropout=0.05,
    ):
        super().__init__()

        self.timesteps = timesteps
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.layers = nn.ModuleList([
            MultiHeadEdgeConv(
                node_dim=hidden_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_attr):
        N = x.size(0)

        h = self.input_proj(x)

        membranes = [
            torch.zeros(N, h.size(1), device=x.device, dtype=h.dtype)
            for _ in range(self.num_layers)
        ]

        spike_accum = torch.zeros(N, h.size(1), device=x.device, dtype=h.dtype)

        for t in range(self.timesteps):
            z = h
            for i, layer in enumerate(self.layers):
                if self.training:
                    z, membranes[i] = checkpoint(
                        layer,
                        z,
                        edge_index,
                        edge_attr,
                        membranes[i],
                        use_reentrant=False,
                    )
                else:
                    z, membranes[i] = layer(z, edge_index, edge_attr, membranes[i])

            spike_accum = spike_accum + z

        avg_rate = spike_accum / float(self.timesteps)
        mean_pool = avg_rate.mean(dim=0, keepdim=True)
        logits = self.readout(mean_pool)

        return logits, avg_rate


# ============================================================
# 6. SYNTHETIC 3D GRAPH GENERATOR
# ============================================================
def make_3d_graph(n=32, e=96, seed=0):
    """
    Creates a small 3D spatial graph.
    Node features: x, y, z, degree_norm, avg_dist_norm  -> 5 dims
    Edge features: dx, dy, dz, dist, 6 chemical features -> 10 dims
    """
    rng = np.random.default_rng(seed)

    # 3D positions
    pos = rng.standard_normal((n, 3)).astype(np.float32)
    pos = pos - pos.mean(axis=0, keepdims=True)
    pos = pos / (np.linalg.norm(pos, axis=1, keepdims=True).max() + 1e-6)

    src = []
    dst = []

    # k-NN style local connections
    for i in range(n):
        d = np.linalg.norm(pos - pos[i], axis=1)
        d[i] = 1e9
        nearest = np.argsort(d)[:2]
        for j in nearest:
            src.append(i)
            dst.append(int(j))

    # random long-range connections
    while len(src) < e:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i != j:
            src.append(i)
            dst.append(j)

    src = np.array(src[:e], dtype=np.int64)
    dst = np.array(dst[:e], dtype=np.int64)

    # remove self loops if any
    mask = src != dst
    src = src[mask]
    dst = dst[mask]

    if len(src) == 0:
        raise ValueError("No edges generated. Increase n/e.")

    # Edge geometry
    disp = pos[src] - pos[dst]
    dist = np.linalg.norm(disp, axis=1, keepdims=True).astype(np.float32)
    dist = np.clip(dist, 1e-6, None)

    # Mock neurotransmitter / chemical features
    chem = np.zeros((len(src), 6), dtype=np.float32)

    # GABA-like local inhibition
    chem[:, 0] = np.exp(-dist[:, 0] * 2.0)

    # Acetylcholine-like learning/modulation
    chem[:, 1] = rng.random(len(src)).astype(np.float32) * 0.5

    # Glutamate-like excitatory long-range
    chem[:, 2] = np.exp(-dist[:, 0] * 0.5)

    # Octopamine-like alertness
    chem[:, 3] = rng.random(len(src)).astype(np.float32) * 0.2

    # Serotonin-like mood/modulation
    chem[:, 4] = rng.random(len(src)).astype(np.float32) * 0.2

    # Dopamine-like reward/local sharp signal
    chem[:, 5] = np.exp(-dist[:, 0] * 3.0) * 0.5

    edge_attr = np.concatenate([disp, dist, chem], axis=1).astype(np.float32)

    # Node features
    degree = np.bincount(src, minlength=n).astype(np.float32)
    degree_norm = (degree / (degree.max() + 1e-6)).reshape(-1, 1)

    sum_dist = np.zeros(n, dtype=np.float32)
    np.add.at(sum_dist, src, dist[:, 0])

    avg_dist = sum_dist / np.maximum(degree, 1.0)
    avg_dist_norm = (avg_dist / (avg_dist.max() + 1e-6)).reshape(-1, 1)

    x = np.concatenate([pos, degree_norm, avg_dist_norm], axis=1).astype(np.float32)

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    return (
        torch.tensor(x, dtype=torch.float32),
        edge_index,
        torch.tensor(edge_attr, dtype=torch.float32),
    )


# ============================================================
# 7. OPTIMIZER
# ============================================================
def make_optimizer(model):
    lr = TRAIN["lr"]

    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )
        print("✅ Using bitsandbytes AdamW8bit optimizer")
        return opt, "AdamW8bit"
    except Exception as e:
        print(f"⚠️ bitsandbytes 8-bit optimizer failed: {e}")
        print("⚠️ Falling back to normal AdamW. VRAM pressure badh sakti hai.")
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4,
            foreach=False,
        )
        return opt, "AdamW"


# ============================================================
# 8. TRAINING LOOP
# ============================================================
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

    opt, opt_name = make_optimizer(model)
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


# ============================================================
# 9. SAVE MODEL
# ============================================================
def save_model(model, params, trained_steps):
    if not SAVE_CHECKPOINT:
        print("⚠️ SAVE_CHECKPOINT=False, checkpoint skip.")
        return None

    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": CFG,
        "train_config": TRAIN,
        "params": params,
        "trained_steps": trained_steps,
        "model_name": "SpikingGraph3D_500M_KapilDhinwa",
    }

    path = os.path.join(SAVE_DIR, "brain_spiking_500m.pth")

    print(f"\n💾 Saving 500M checkpoint to: {path}")
    print("⏳ This may take some time because file is ~2 GB.")

    torch.save(ckpt, path)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"✅ Saved! File size: {size_mb:.1f} MB")

    return path


# ============================================================
# 10. QUICK EVAL / BRAIN SCAN
# ============================================================
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