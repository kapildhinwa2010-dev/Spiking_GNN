# ============================================================================
# ============================================================================
#  SINGLE CELL: FULL TEACHING + DEMO OF NEUROSPIKING GRAPH 3D BRAIN
#  Author Concept: Kapil Dhinwa
#
#  Ye cell ek chhota "teaching brain" banata hai, lekin saare concepts wahi hain
#  jo tumhare 241M / 500M / 800M models mein use hue the.
#
#  IMPORTANT:
#  - Ye 500M model ko train nahi karega, kyunki 500M CPU/Colab free par practical nahi hai.
#  - Ye model architecture samjhane ke liye chhota hai: hidden_dim=32, layers=2.
#  - Bade model ke liye bas Config mein hidden_dim, num_layers, num_heads badhana hai,
#    aur GPU + gradient checkpointing + 8-bit optimizer use karna hai.
#
#  Concepts covered:
#  1. Graph = Neurons + Synapses
#  2. Node features = 3D position + degree + average connection distance
#  3. Edge features = 3D displacement + distance + 6 chemical/neurotransmitter-like features
#  4. Spiking neuron = membrane potential + threshold + spike + reset
#  5. Sparse firing = sirf kuch neurons active, baaki silent/dormant
#  6. Multi-head graph convolution = kai parallel "teams" graph ko samajhti hain
#  7. Edge-aware message passing = connection ki strength/distance/chemical bhi dekha jata hai
#  8. Residual connection = purani information shortcut se bachayi jati hai
#  9. LayerNorm = signal volume control
# 10. Timesteps = brain kai baar sochta hai, average firing rate nikalta hai
# 11. Readout = graph-level prediction
# 12. Loss = prediction error + sparsity penalty
# 13. Dormant neurons = silent neurons dead nahi, reserve capacity hain
# 14. Error-gated dormant recruitment = jab error badhe, tab silent neurons ko chance do
# 15. Random noise baseline = sab neurons ko randomly hilana, ye often disturb karta hai
# 16. Continual learning task switch = Task A seekho, phir Task B seekho
# 17. Forgetting check = Task A kitna bhula
# 18. Lesion scan = important hubs hatao, dekho activity kam hoti hai ya "disinhibition" hoti hai
# 19. Gradient checkpointing concept = bade model mein VRAM bachane ka trick
# 20. 8-bit optimizer concept = AdamW states ko compress karke VRAM bachana
# ============================================================================
# ============================================================================

import math
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 0. DEVICE
# ----------------------------------------------------------------------------
# Agar GPU mile to GPU, warna CPU. Teaching model chhota hai isliye CPU par bhi chalega.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 90)
print("  NEUROSPIKING GRAPH 3D BRAIN - FULL TEACHING CELL")
print("=" * 90)
print(f"Device: {DEVICE}")
print("Note: Ye teaching-size model hai. Bade 241M/500M model ke liye config badhao.")
print("=" * 90)


# ----------------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------------
# Ye config model ki "size" aur experiment ka "rule book" hai.
# Chhote teaching model ke liye hidden_dim=32 rakha hai.
# Bade model ke liye example:
#   hidden_dim=3072, num_layers=8, num_heads=16  -> ~241M
#   hidden_dim=3648, num_layers=12, num_heads=16 -> ~500M
# ----------------------------------------------------------------------------
@dataclass
class TeachConfig:
    # ------------------------------------
    # Model architecture
    # ------------------------------------
    hidden_dim: int = 32          # Har neuron ka "thought vector" size
    num_layers: int = 2           # Kitni baar graph par message passing hogi
    num_heads: int = 2            # Kitni parallel teams graph ko samjhengi
    timesteps: int = 2            # Brain kitni baar sochega (spike accumulation)
    node_dim: int = 5             # x, y, z, degree_norm, avg_dist_norm
    edge_dim: int = 10            # dx, dy, dz, dist + 6 chemical features
    out_dim: int = 1              # Teaching demo regression ke liye 1 output
    dropout: float = 0.0

    # Gradient checkpointing:
    # Chhote teaching model mein zaroorat nahi.
    # Bade model mein True karo aur GPU use karo.
    use_checkpoint: bool = False

    # ------------------------------------
    # Synthetic 3D graph size
    # ------------------------------------
    graph_nodes: int = 12         # Kitne neurons/students
    graph_edges: int = 24         # Kitne synapses/friendships

    # ------------------------------------
    # Real-time continual learning experiment
    # ------------------------------------
    steps: int = 80               # Total online training steps
    task_switch: int = 40         # Pehle 40 steps Task A, baaki 40 steps Task B
    lr: float = 2e-3
    sparsity_lambda: float = 0.01 # Model ko sparse rehne ki सजा/इनाम
    eval_graphs: int = 12         # Held-out evaluation ke liye kitne graphs

    # ------------------------------------
    # Experiment seeds and modes
    # ------------------------------------
    seeds: Tuple[int, ...] = (0, 1)
    modes: Tuple[str, ...] = ("none", "random", "smart")

    # ------------------------------------
    # Dormant recruitment hyperparameters
    # ------------------------------------
    base_noise: float = 0.18      # Shuruwat ka wake-up noise
    min_noise: float = 0.03       # End tak noise kam ho kar itna rahega
    dormant_threshold: float = 0.05
    ema_beta: float = 0.90        # Activity EMA smoothing
    gate_min: float = 0.5
    gate_max: float = 2.5


# ----------------------------------------------------------------------------
# 2. SPIKING NEURON
# ----------------------------------------------------------------------------
# Normal neural network mein neuron turant output deta hai.
# Spiking neuron mein:
#   - membrane potential accumulate hota hai
#   - jab threshold cross hota hai, spike banta hai
#   - fir membrane reset hota hai
#
# Bucket analogy:
#   current = bucket mein girta paani
#   membrane = bucket mein existing paani
#   decay = paani ka dheere-dheere kam hona
#   threshold = bucket ki capacity
#   spike = overflow/alarm
#
# Silent/dormant neurons ke liye optional noise di ja sakta hai.
# Noise sirf spike decision ko affect karta hai, membrane ko permanently kharab nahi karta.
# ----------------------------------------------------------------------------
class SpikingNeuron(nn.Module):
    def __init__(self, dim: int, sharpness: float = 5.0):
        super().__init__()
        self.sharpness = sharpness

        # Learnable decay: har neuron kitni jaldi bhoolta hai
        self.decay = nn.Parameter(torch.ones(dim) * 0.8)

        # Learnable threshold: har neuron ko fire hone ke liye kitna signal chahiye
        self.threshold = nn.Parameter(torch.ones(dim) * 1.0)

    def forward(self, current: torch.Tensor, membrane: torch.Tensor, noise: torch.Tensor = None):
        # Decay ko 0.1 se 0.95 ke beech stable rakho
        decay = torch.sigmoid(self.decay) * 0.85 + 0.1

        # Threshold hamesha positive
        threshold = torch.abs(self.threshold) + 0.1

        # Membrane update: leaky integration
        membrane = decay * membrane + current

        # Spike decision:
        # membrane - threshold > 0 hone par spike strong hota hai
        voltage = membrane - threshold

        # Dormant recruitment noise:
        # Ye silent neurons ko kabhi-kabhi chance deta hai fire karne ka.
        if noise is not None:
            voltage = voltage + noise

        # Differentiable surrogate spike:
        # Hard spike (0/1) gradient nahi deta, isliye sigmoid use karte hain.
        spike = torch.sigmoid(self.sharpness * voltage)

        # Reset: spike ke baad threshold jitna charge hata do
        membrane = membrane - spike * threshold

        return spike, membrane


# ----------------------------------------------------------------------------
# 3. MULTI-HEAD EDGE-AWARE GRAPH CONVOLUTION
# ----------------------------------------------------------------------------
# Ye model ka main "group discussion" layer hai.
#
# Graph convolution ka matlab:
#   Har node apne neighbors se message leta hai.
#
# Edge-aware ka matlab:
#   Message sirf node ki baat nahi, edge/connection ki bhi baat sunta hai:
#   - distance kitna hai
#   - chemical/mood kya hai
#   - connection strong hai ya weak
#
# Multi-head ka matlab:
#   Ek saath kai teams discussion karti hain.
#   Team 1 distance dekh sakti hai, Team 2 GABA dekh sakti hai, etc.
#
# Residual connection:
#   Naya message + purani information = knowledge kam khooti hai.
#
# LayerNorm:
#   Signal ka volume stable rakhta hai.
# ----------------------------------------------------------------------------
class MultiHeadEdgeConv(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Node features ko hidden thought space mein lao
        self.node_proj = nn.Linear(node_dim, hidden_dim)

        # Har head/source team edge features bhi dekhti hai
        self.head_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + edge_dim, self.head_dim),
                nn.SiLU(),
                nn.Linear(self.head_dim, self.head_dim),
            )
            for _ in range(num_heads)
        ])

        # Sab heads ke outputs ko combine karke final message banao
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Normalization
        self.norm_in = nn.LayerNorm(node_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)

        # Spiking neuron layer ke end mein fire karta hai
        self.neuron = SpikingNeuron(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # Residual projection agar input/output dimensions alag hon
        self.res_proj = (
            nn.Linear(node_dim, hidden_dim)
            if node_dim != hidden_dim
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        membrane: torch.Tensor,
        noise: torch.Tensor = None,
    ):
        N = x.size(0)
        row, col = edge_index[0], edge_index[1]

        # Input normalize + project
        x_norm = self.norm_in(x)
        x_proj = self.node_proj(x_norm)

        # Residual shortcut
        residual = self.res_proj(x)

        if row.numel() == 0:
            # Koi edge nahi: message zero
            msg = torch.zeros(N, x_proj.size(1), device=x.device, dtype=x_proj.dtype)
        else:
            # Source nodes ke thoughts
            src_feat = x_proj[row]

            # Har head source thought + edge features dekhti hai
            head_outputs = []
            for head_mlp in self.head_mlps:
                head_in = torch.cat([src_feat, edge_attr], dim=-1)
                head_out = head_mlp(head_in)
                head_out = self.dropout(head_out)
                head_outputs.append(head_out)

            # Sab heads ko concatenate karo: [edges, hidden_dim]
            msg_per_edge = torch.cat(head_outputs, dim=-1)

            # Target nodes par messages aggregate karo
            msg = torch.zeros(N, x_proj.size(1), device=x.device, dtype=x_proj.dtype)
            msg.index_add_(0, col, msg_per_edge)

        # Combined message project karo
        msg = self.out_proj(msg)

        # Residual add + normalize
        h = self.norm_out(msg + residual.to(msg.dtype))

        # Spiking neuron fire karta hai
        spike, new_membrane = self.neuron(h, membrane, noise=noise)

        return spike, new_membrane


# ----------------------------------------------------------------------------
# 4. FULL SPIKING GRAPH 3D NETWORK
# ----------------------------------------------------------------------------
# Ye poora brain hai:
#
# Input 3D graph
#   -> input projection
#   -> N spiking graph convolution layers
#   -> repeat for T timesteps
#   -> average firing rate
#   -> readout prediction
#
# Timesteps ka matlab:
#   Brain ek baar mein final decision nahi leta.
#   Wo kai rounds mein signal ghumaata hai.
#   Har round mein neurons spike karte hain.
#   Final average firing rate = neuron kitna active raha.
# ----------------------------------------------------------------------------
class SpikingGraph3DNet(nn.Module):
    def __init__(self, cfg: TeachConfig):
        super().__init__()

        self.cfg = cfg

        # Raw node features ko hidden thought space mein convert karo
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.node_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dim),
        )

        # Stacked spiking graph layers
        self.layers = nn.ModuleList([
            MultiHeadEdgeConv(
                node_dim=cfg.hidden_dim,
                edge_dim=cfg.edge_dim,
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.num_layers)
        ])

        # Graph-level readout
        self.readout = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.out_dim),
        )

        # Stable initialization
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        noise: torch.Tensor = None,
    ):
        N = x.size(0)

        # Initial hidden representation
        h = self.input_proj(x)

        # Har layer ka apna membrane state hota hai
        membranes = [
            torch.zeros(N, h.size(1), device=x.device, dtype=h.dtype)
            for _ in range(self.cfg.num_layers)
        ]

        # Spike accumulator: kai timesteps ke spikes jama karo
        spike_accum = torch.zeros(N, h.size(1), device=x.device, dtype=h.dtype)

        for _t in range(self.cfg.timesteps):
            z = h

            for i, layer in enumerate(self.layers):
                # Gradient checkpointing concept:
                # Bade model mein backward pass ke liye activations dobara compute karke
                # VRAM bachate hain. Chhote teaching model mein off rakha hai.
                if self.training and self.cfg.use_checkpoint and noise is None:
                    z, membranes[i] = torch.utils.checkpoint.checkpoint(
                        layer,
                        z,
                        edge_index,
                        edge_attr,
                        membranes[i],
                        use_reentrant=False,
                    )
                else:
                    z, membranes[i] = layer(
                        z,
                        edge_index,
                        edge_attr,
                        membranes[i],
                        noise=noise,
                    )

            spike_accum = spike_accum + z

        # Average firing rate = total spikes / timesteps
        avg_rate = spike_accum / float(self.cfg.timesteps)

        # Graph-level summary: sab nodes ka average thought vector
        graph_vec = avg_rate.mean(dim=0, keepdim=True)

        # Final prediction
        logits = self.readout(graph_vec)

        return logits, avg_rate


# ----------------------------------------------------------------------------
# 5. SYNTHETIC 3D GRAPH GENERATOR
# ----------------------------------------------------------------------------
# Real Flywire brain data mein:
#   nodes = neurons
#   edges = synapses
#   node features = 3D position + structural stats
#   edge features = displacement + distance + neurotransmitters
#
# Language graph mein:
#   nodes = words
#   edges = co-occurrence
#   node features = random/embedded 3D position + degree
#   edge features = syntax/chemical-like mock features
#
# Yahan hum synthetic 3D graph bana rahe hain taki cell guaranteed chale.
# ----------------------------------------------------------------------------
def make_3d_graph(cfg: TeachConfig, seed: int, task: str = "A"):
    rng = np.random.default_rng(seed)
    n = cfg.graph_nodes

    # ------------------------
    # 3D positions
    # ------------------------
    pos = rng.standard_normal((n, 3)).astype(np.float32)
    pos = pos - pos.mean(axis=0, keepdims=True)
    pos = pos / (np.linalg.norm(pos, axis=1, keepdims=True).max() + 1e-6)

    # ------------------------
    # Edges: local k-NN + random long-range
    # ------------------------
    src_list = []
    dst_list = []

    # Local connections: paas wale neurons/jo words saath aate hain
    for i in range(n):
        d = np.linalg.norm(pos - pos[i], axis=1)
        d[i] = 1e9
        nearest = np.argsort(d)[:min(2, n - 1)]
        for j in nearest:
            src_list.append(i)
            dst_list.append(int(j))

    # Long-range connections
    while len(src_list) < cfg.graph_edges:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i != j:
            src_list.append(i)
            dst_list.append(j)

    src = np.array(src_list[:cfg.graph_edges], dtype=np.int64)
    dst = np.array(dst_list[:cfg.graph_edges], dtype=np.int64)

    # Self loops hatao
    mask = src != dst
    src = src[mask]
    dst = dst[mask]

    # Fallback ring graph
    if len(src) == 0:
        src = np.arange(n, dtype=np.int64)
        dst = np.roll(np.arange(n, dtype=np.int64), 1)

    # ------------------------
    # Edge geometric features
    # ------------------------
    disp = pos[src] - pos[dst]
    dist = np.linalg.norm(disp, axis=1, keepdims=True).astype(np.float32)
    dist = np.clip(dist, 1e-6, None)

    # ------------------------
    # 6 chemical/neurotransmitter-like features
    # ------------------------
    # Real Flywire mein ye actual columns hote:
    # gaba, ach, glut, oct, ser, da
    # Yahan synthetic patterns banaye hain.
    chem = np.zeros((len(src), 6), dtype=np.float32)

    # GABA-like: local inhibition, paas wale connections par zyada
    chem[:, 0] = np.exp(-dist[:, 0] * 2.0)

    # Acetylcholine-like: learning/modulation, random moderate
    chem[:, 1] = rng.random(len(src)).astype(np.float32) * 0.5

    # Glutamate-like: excitatory long-range, door wale connections par bhi signal
    chem[:, 2] = np.exp(-dist[:, 0] * 0.5)

    # Octopamine-like: alertness, low random
    chem[:, 3] = rng.random(len(src)).astype(np.float32) * 0.2

    # Serotonin-like: mood/modulation, low random
    chem[:, 4] = rng.random(len(src)).astype(np.float32) * 0.2

    # Dopamine-like: sharp local reward signal
    chem[:, 5] = np.exp(-dist[:, 0] * 3.0) * 0.5

    # Final edge features: [dx, dy, dz, dist, 6 chemicals] = 10 dims
    edge_attr = np.concatenate([disp, dist, chem], axis=1).astype(np.float32)

    # ------------------------
    # Node features
    # ------------------------
    degree = np.bincount(src, minlength=n).astype(np.float32)
    degree_norm = (degree / (degree.max() + 1e-6)).reshape(-1, 1)

    sum_dist = np.zeros(n, dtype=np.float32)
    np.add.at(sum_dist, src, dist[:, 0])

    avg_dist = sum_dist / np.maximum(degree, 1.0)
    avg_dist_norm = (avg_dist / (avg_dist.max() + 1e-6)).reshape(-1, 1)

    # Node features: [x, y, z, degree_norm, avg_dist_norm] = 5 dims
    x = np.concatenate([pos, degree_norm, avg_dist_norm], axis=1).astype(np.float32)

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    # ------------------------
    # Task targets
    # ------------------------
    # Task A: feature 0 aur feature 1 par based
    # Task B: feature 3, 4, 2 par based
    # Ye continual learning task switch ke liye hai.
    if task == "A":
        target_value = (
            2.0 * float(x[:, 0].mean())
            + 0.3 * float(np.mean(x[:, 1] ** 2))
        )
    else:
        target_value = (
            3.0 * float(x[:, 3].mean())
            - 1.2 * float(x[:, 4].mean())
            + 0.2 * float(np.mean(x[:, 2] ** 2))
        )

    target = torch.tensor(target_value, dtype=torch.float32)

    return (
        torch.tensor(x, dtype=torch.float32),
        edge_index,
        torch.tensor(edge_attr, dtype=torch.float32),
        target,
    )


# ----------------------------------------------------------------------------
# 6. DORMANT MANAGER
# ----------------------------------------------------------------------------
# Silent/dormant neurons ka management.
#
# Core idea:
#   Jo neurons ek input par kam fire kar rahe hain, wo dead nahi hain.
#   Wo reserve capacity hain.
#
# Modes:
#   none   = koi extra noise nahi
#   random = sab neurons ko random noise (dumb exploration)
#   smart  = sirf dormant neurons ko, error-gated aur annealed noise
#
# Error gate:
#   Jab loss/ error badh raha ho, tab dormant neurons ko zyada chance.
#   Jab loss kam ho, tab noise kam.
#
# Annealing:
#   Shuru mein exploration zyada, end mein stability zyada.
# ----------------------------------------------------------------------------
class DormantManager:
    def __init__(self, cfg: TeachConfig):
        self.cfg = cfg

        # EMA = Exponential Moving Average
        # Har hidden dimension ki average activity track karo
        self.activity_ema = torch.full(
            (cfg.hidden_dim,),
            0.20,
            device=DEVICE,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def update(self, avg_rate: torch.Tensor):
        # avg_rate: [nodes, hidden_dim]
        # Unit activity: har hidden dimension ka average firing
        unit_activity = avg_rate.mean(dim=0).detach()

        self.activity_ema = (
            self.cfg.ema_beta * self.activity_ema
            + (1.0 - self.cfg.ema_beta) * unit_activity
        )

    def dormant_mask(self) -> torch.Tensor:
        # Jo dimensions bahut kam fire kar rahe hain, unhe dormant bolo
        return self.activity_ema < self.cfg.dormant_threshold

    def make_noise(
        self,
        mode: str,
        n_nodes: int,
        prev_loss: float,
        running_loss: float,
        step: int,
    ):
        if mode == "none":
            return None

        # Annealing: noise time ke saath kam hoga, lekin min_noise se neeche nahi jayega
        progress = step / max(1, self.cfg.steps - 1)
        scale = max(
            self.cfg.min_noise,
            self.cfg.base_noise * (1.0 - progress),
        )

        if mode == "random":
            # Sab neurons ko random noise: ye purani learning disturb kar sakta hai
            return torch.randn(
                (n_nodes, self.cfg.hidden_dim),
                device=DEVICE,
                dtype=torch.float32,
            ) * scale

        if mode == "smart":
            # Sirf dormant neurons ko mask karo
            mask = self.dormant_mask().to(DEVICE).float().unsqueeze(0)  # [1, hidden]

            # Error gate
            if prev_loss is None or running_loss is None:
                gate = 1.0
            else:
                raw_gate = float(prev_loss) / (float(running_loss) + 1e-8)
                if not np.isfinite(raw_gate):
                    raw_gate = 1.0
                gate = float(np.clip(raw_gate, self.cfg.gate_min, self.cfg.gate_max))

            noise = torch.randn(
                (n_nodes, self.cfg.hidden_dim),
                device=DEVICE,
                dtype=torch.float32,
            ) * (scale * gate) * mask

            return noise

        return None


# ----------------------------------------------------------------------------
# 7. DORMANT GRADIENT MEASUREMENT
# ----------------------------------------------------------------------------
# Ye check karta hai ki silent/dormant neurons ko seekhne ka signal (gradient) mil raha hai ya nahi.
# Agar gradient zero hai, toh neuron permanently dead jaisa hai.
# Agar gradient non-zero hai, toh neuron reserve capacity ki tarah usable hai.
# ----------------------------------------------------------------------------
def dormant_gradient_norm(model: SpikingGraph3DNet, dormant_mask: torch.Tensor):
    dormant_sq = 0.0
    total_sq = 0.0

    has_dormant = bool(dormant_mask.any().item())

    for layer in model.layers:
        # Threshold aur decay neurons ke important parameters hain
        for param in (layer.neuron.threshold, layer.neuron.decay):
            if param.grad is not None:
                g = param.grad.detach()
                total_sq += float(torch.sum(g * g).item())

                if has_dormant:
                    gd = g[dormant_mask]
                    dormant_sq += float(torch.sum(gd * gd).item())

    return math.sqrt(max(dormant_sq, 0.0)), math.sqrt(max(total_sq, 0.0))


# ----------------------------------------------------------------------------
# 8. HELD-OUT EVALUATION
# ----------------------------------------------------------------------------
# Training ke baad model ko naye graphs par test karo.
# Eval A = Task A kitna accha kar raha hai
# Eval B = Task B kitna accha kar raha hai
# Forgetting = Eval A - training ke end ka Task A loss
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_task(
    model: SpikingGraph3DNet,
    cfg: TeachConfig,
    task: str,
    seed_base: int,
) -> float:
    was_training = model.training
    model.eval()

    losses = []

    for i in range(cfg.eval_graphs):
        x, ei, ea, target = make_3d_graph(cfg, seed=seed_base + i, task=task)

        x = x.to(DEVICE)
        ei = ei.to(DEVICE)
        ea = ea.to(DEVICE)
        target = target.to(DEVICE)

        logits, _ = model(x, ei, ea)
        pred = logits.squeeze()

        loss = F.mse_loss(pred, target)
        losses.append(float(loss.item()))

    if was_training:
        model.train()

    return float(np.mean(losses))


# ----------------------------------------------------------------------------
# 9. TRAIN ONE MODE + ONE SEED
# ----------------------------------------------------------------------------
# Ye real-time online training jaisa hai:
#   har step par ek naya graph aata hai
#   model ek hi sample/step par update hota hai
#   Task A -> Task B switch hota hai
#
# Modes:
#   none   = baseline, koi dormant recruitment nahi
#   random = random wake-up noise
#   smart  = dormant-only + error-gated + annealed noise
# ----------------------------------------------------------------------------
def train_one(mode: str, seed: int, cfg: TeachConfig):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SpikingGraph3DNet(cfg).to(DEVICE)
    model.train()

    # Chhote teaching model ke liye normal Adam.
    # Bade model ke liye:
    #   import bitsandbytes as bnb
    #   opt = bnb.optim.AdamW8bit(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    dm = DormantManager(cfg)

    history = []

    prev_loss = None
    running_loss = None

    switch = cfg.task_switch

    for step in range(cfg.steps):
        task = "A" if step < switch else "B"

        # Fresh live graph
        x, ei, ea, target = make_3d_graph(
            cfg,
            seed=100000 + seed * 10000 + step,
            task=task,
        )

        x = x.to(DEVICE)
        ei = ei.to(DEVICE)
        ea = ea.to(DEVICE)
        target = target.to(DEVICE)

        # Dormant recruitment noise
        noise = dm.make_noise(
            mode=mode,
            n_nodes=x.size(0),
            prev_loss=prev_loss,
            running_loss=running_loss,
            step=step,
        )

        opt.zero_grad(set_to_none=True)

        logits, avg_rate = model(x, ei, ea, noise=noise)

        # Prediction:
        # out_dim=1 hone ki wajah se logits.squeeze() scalar ban jata hai.
        pred = logits.squeeze()

        # Loss:
        # 1) prediction error
        # 2) sparsity penalty: model ko bolo zyada bekar fire mat karo
        pred_loss = F.mse_loss(pred, target)
        sparsity_loss = avg_rate.abs().mean()
        loss = pred_loss + cfg.sparsity_lambda * sparsity_loss

        loss.backward()

        # Dormant gradient measure karo
        mask = dm.dormant_mask()
        dgrad, tgrad = dormant_gradient_norm(model, mask)

        # Gradient clipping: bade gradients se model explode na ho
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        opt.step()

        # Activity EMA update
        dm.update(avg_rate)

        loss_value = float(loss.item())

        if running_loss is None:
            running_loss = loss_value
        else:
            running_loss = 0.9 * running_loss + 0.1 * loss_value

        prev_loss = loss_value

        active_frac = float((avg_rate > 0.5).float().mean().item())
        dormant_frac = float(mask.float().mean().item())

        history.append({
            "step": step,
            "task": task,
            "loss": loss_value,
            "active_frac": active_frac,
            "dormant_frac": dormant_frac,
            "dormant_grad": dgrad,
            "total_grad": tgrad,
        })

    # ------------------------
    # Aggregate training metrics
    # ------------------------
    pre_slice = history[:switch]
    shock_slice = history[switch:switch + 10]
    post_slice = history[-10:]
    loss_a_after_slice = history[max(0, switch - 10):switch]

    pre_loss = float(np.mean([h["loss"] for h in pre_slice])) if pre_slice else float("nan")
    shock_loss = float(np.mean([h["loss"] for h in shock_slice])) if shock_slice else float("nan")
    post_loss = float(np.mean([h["loss"] for h in post_slice])) if post_slice else float("nan")
    loss_a_after = float(np.mean([h["loss"] for h in loss_a_after_slice])) if loss_a_after_slice else float("nan")

    active_a = float(np.mean([h["active_frac"] for h in pre_slice])) if pre_slice else float("nan")
    active_b = float(np.mean([h["active_frac"] for h in history[switch:]])) if switch < cfg.steps else float("nan")

    dormant_grad_avg = float(np.mean([h["dormant_grad"] for h in history]))
    total_grad_avg = float(np.mean([h["total_grad"] for h in history]))
    dormant_frac_avg = float(np.mean([h["dormant_frac"] for h in history]))

    # ------------------------
    # Held-out evaluation
    # ------------------------
    eval_a = evaluate_task(model, cfg, "A", seed_base=900000 + seed * 1000)
    eval_b = evaluate_task(model, cfg, "B", seed_base=950000 + seed * 1000)

    forgetting_a = eval_a - loss_a_after

    metrics = {
        "mode": mode,
        "seed": seed,
        "pre_loss": pre_loss,
        "shock_loss": shock_loss,
        "post_loss": post_loss,
        "loss_a_after": loss_a_after,
        "eval_a": eval_a,
        "eval_b": eval_b,
        "forgetting_a": forgetting_a,
        "active_a": active_a,
        "active_b": active_b,
        "active_delta": active_b - active_a,
        "dormant_grad_avg": dormant_grad_avg,
        "total_grad_avg": total_grad_avg,
        "dormant_frac_avg": dormant_frac_avg,
    }

    return history, metrics, model


# ----------------------------------------------------------------------------
# 10. LESION SCAN / DISINHIBITION CHECK
# ----------------------------------------------------------------------------
# Brain lesion study:
#   Important hub nodes ko hata do.
#   Agar activity kam ho -> hubs synchronizer the.
#   Agar activity badh jaye -> disinhibition / seizure-like state.
#
# Real brain analogy:
#   Inhibitory hubs damage hone par brain hyperactive ho sakta hai.
# ----------------------------------------------------------------------------
@torch.no_grad()
def lesion_scan(model: SpikingGraph3DNet, cfg: TeachConfig, seed: int = 777):
    was_training = model.training
    model.eval()

    x, ei, ea, _ = make_3d_graph(cfg, seed=seed, task="A")

    x = x.to(DEVICE)
    ei = ei.to(DEVICE)
    ea = ea.to(DEVICE)

    _, avg_healthy = model(x, ei, ea)
    healthy_activity = float(avg_healthy.abs().sum().item())

    n = x.size(0)

    # Degree = kitne edges node se judi hain
    src_np = ei[0].cpu().numpy()
    dst_np = ei[1].cpu().numpy()

    degree = (
        np.bincount(src_np, minlength=n).astype(np.float32)
        + np.bincount(dst_np, minlength=n).astype(np.float32)
    )

    # Top hubs remove karo
    k = max(1, n // 4)
    top = np.argsort(degree)[::-1][:k]
    top_set = set(int(i) for i in top)

    # Edges filter karo: agar source ya target hub hai, edge hata do
    keep = np.array([
        (int(s) not in top_set) and (int(d) not in top_set)
        for s, d in zip(src_np, dst_np)
    ], dtype=bool)

    if keep.any():
        src_l = src_np[keep]
        dst_l = dst_np[keep]
        ei_l = torch.tensor(np.stack([src_l, dst_l], axis=0), dtype=torch.long, device=DEVICE)
        mask_t = torch.tensor(keep, dtype=torch.bool, device=DEVICE)
        ea_l = ea[mask_t]
    else:
        # Koi edge nahi bachi: empty graph
        ei_l = torch.empty((2, 0), dtype=torch.long, device=DEVICE)
        ea_l = torch.empty((0, cfg.edge_dim), dtype=torch.float32, device=DEVICE)

    _, avg_lesioned = model(x, ei_l, ea_l)
    lesioned_activity = float(avg_lesioned.abs().sum().item())

    if healthy_activity > 1e-8:
        impact_pct = (1.0 - (lesioned_activity / healthy_activity)) * 100.0
    else:
        impact_pct = 0.0

    if was_training:
        model.train()

    return impact_pct, healthy_activity, lesioned_activity, int(k)


# ----------------------------------------------------------------------------
# 11. AGGREGATE RESULTS
# ----------------------------------------------------------------------------
def aggregate_metrics(metric_lists: dict):
    agg = {}

    for mode, lst in metric_lists.items():
        if not lst:
            continue

        # Sirf numeric fields ka average lo
        numeric_keys = [
            k for k, v in lst[0].items()
            if isinstance(v, (int, float, np.floating, np.integer))
        ]

        agg[mode] = {
            k: float(np.mean([m[k] for m in lst]))
            for k in numeric_keys
        }

    return agg


# ----------------------------------------------------------------------------
# 12. MAIN RUN
# ----------------------------------------------------------------------------
def main():
    cfg = TeachConfig()

    print("\n" + "=" * 90)
    print("  EXPERIMENT CONFIG")
    print("=" * 90)
    print(f"hidden_dim       : {cfg.hidden_dim}")
    print(f"num_layers       : {cfg.num_layers}")
    print(f"num_heads        : {cfg.num_heads}")
    print(f"timesteps        : {cfg.timesteps}")
    print(f"graph_nodes      : {cfg.graph_nodes}")
    print(f"graph_edges      : {cfg.graph_edges}")
    print(f"steps            : {cfg.steps}")
    print(f"task_switch      : {cfg.task_switch}")
    print(f"modes            : {cfg.modes}")
    print(f"seeds            : {cfg.seeds}")
    print("=" * 90)

    all_metrics = {m: [] for m in cfg.modes}
    models = {}

    total_t0 = time.time()

    for mode in cfg.modes:
        for seed in cfg.seeds:
            t0 = time.time()

            history, metrics, model = train_one(mode, seed, cfg)

            all_metrics[mode].append(metrics)

            if mode == "smart" and seed == cfg.seeds[0]:
                models["smart"] = model

            elapsed = time.time() - t0

            print(
                f"[{mode:<8}] seed={seed} "
                f"post={metrics['post_loss']:.4f} "
                f"forget={metrics['forgetting_a']:+.4f} "
                f"dormGrad={metrics['dormant_grad_avg']:.5f} "
                f"time={elapsed:.1f}s"
            )

    agg = aggregate_metrics(all_metrics)

    # ------------------------
    # Results table
    # ------------------------
    print("\n" + "=" * 115)
    print("  AGGREGATE RESULTS")
    print("=" * 115)
    print(
        f"{'Mode':<10}"
        f"{'PreLoss':<10}"
        f"{'Shock':<10}"
        f"{'PostLoss':<10}"
        f"{'EvalA':<10}"
        f"{'ForgetA':<10}"
        f"{'DormGrad':<12}"
        f"{'ActDelta':<10}"
        f"{'DormFrac':<10}"
    )
    print("-" * 115)

    for mode in cfg.modes:
        a = agg[mode]
        print(
            f"{mode:<10}"
            f"{a['pre_loss']:<10.4f}"
            f"{a['shock_loss']:<10.4f}"
            f"{a['post_loss']:<10.4f}"
            f"{a['eval_a']:<10.4f}"
            f"{a['forgetting_a']:<+10.4f}"
            f"{a['dormant_grad_avg']:<12.6f}"
            f"{a['active_delta']:<+10.4f}"
            f"{a['dormant_frac_avg']:<10.4f}"
        )

    print("=" * 115)

    # ------------------------
    # Honest verdict
    # ------------------------
    smart = agg["smart"]
    none = agg["none"]
    rand = agg["random"]

    post_smart_less_none = smart["post_loss"] < none["post_loss"]
    post_smart_less_random = smart["post_loss"] < rand["post_loss"]
    dorm_grad_smart_more = smart["dormant_grad_avg"] > none["dormant_grad_avg"]
    forget_not_worse = smart["forgetting_a"] <= none["forgetting_a"] + 0.05

    print("\n" + "=" * 90)
    print("  SCIENTIFIC VERDICT")
    print("=" * 90)

    if post_smart_less_none and post_smart_less_random and dorm_grad_smart_more:
        print("✅ STRONG SIGNAL:")
        print("   Smart dormant recruitment ne post-task-switch loss kam kiya")
        print("   aur dormant neurons ko zyada gradient diya.")
    elif dorm_grad_smart_more:
        print("🟡 PARTIAL SIGNAL:")
        print("   Dormant neurons ko zyada gradient mila,")
        print("   lekin loss improvement strong nahi hai.")
        print("   Steps/seeds/hidden_dim badha kar test karo.")
    else:
        print("🔴 WEAK SIGNAL:")
        print("   Is chhote teaching setup mein smart recruitment strong nahi aaya.")
        print("   Concept valid hai, hyperparameters tune karo.")

    print("-" * 90)
    print("Checks:")
    print(f"  smart post_loss < none post_loss   : {post_smart_less_none}")
    print(f"  smart post_loss < random post_loss : {post_smart_less_random}")
    print(f"  smart dormant_grad > none          : {dorm_grad_smart_more}")
    print(f"  smart forgetting not much worse    : {forget_not_worse}")
    print("=" * 90)

    # ------------------------
    # Lesion scan on smart model
    # ------------------------
    if "smart" in models:
        impact, healthy, lesioned, removed = lesion_scan(models["smart"], cfg)

        print("\n" + "=" * 90)
        print("  LESION SCAN / DISINHIBITION CHECK")
        print("=" * 90)
        print(f"Removed hub nodes : {removed}")
        print(f"Healthy activity  : {healthy:.4f}")
        print(f"Lesioned activity : {lesioned:.4f}")
        print(f"Impact            : {impact:+.2f}%")

        if impact < 0:
            print("⚠️ DISINHIBITION DETECTED:")
            print("   Hubs hatane ke baad activity badh gayi.")
            print("   Ye seizure-like / runaway excitation jaisa behavior hai.")
        else:
            print("✅ Activity decreased:")
            print("   Hubs network ko synchronize kar rahe the.")

        print("=" * 90)

    total_time = time.time() - total_t0

    print(f"\nTotal experiment time: {total_time:.2f}s")

    # ------------------------
    # Save teaching model
    # ------------------------
    if "smart" in models:
        save_path = "teaching_spiking_graph_3d_smart.pt"
        torch.save({
            "model_state_dict": models["smart"].state_dict(),
            "config": {
                "hidden_dim": cfg.hidden_dim,
                "num_layers": cfg.num_layers,
                "num_heads": cfg.num_heads,
                "timesteps": cfg.timesteps,
                "node_dim": cfg.node_dim,
                "edge_dim": cfg.edge_dim,
                "out_dim": cfg.out_dim,
            },
        }, save_path)
        print(f"Saved teaching model: {save_path}")

    # ------------------------
    # Final plain-Hindi summary
    # ------------------------
    print("\n" + "=" * 90)
    print("  PLAIN HINDI SUMMARY")
    print("=" * 90)
    print("""
1. Node = neuron/student.
   Uske paas 3D position, degree, aur average connection distance hoti hai.

2. Edge = synapse/friendship.
   Uske paas distance aur chemical/mood features hote hain.

3. Spiking neuron bucket jaisa hai.
   Signal accumulate hota hai, threshold cross hone par spike hota hai.

4. Sparse firing matlab sab neurons ek saath nahi bolte.
   Sirf zaroori neurons fire karte hain, baaki silent rehte hain.

5. Silent/dormant neurons useless nahi hote.
   Wo future learning ke liye reserve capacity hain.

6. Smart recruitment matlab:
   - sirf dormant neurons ko chance do
   - jab error badhe tab zyada chance do
   - time ke saath noise kam karo
   - active neurons ko bekar disturb mat karo

7. Task switch test:
   Model pehle Task A seekhta hai, phir Task B.
   Dekhte hain Task B seekhte waqt Task A bhoolta hai ya nahi.

8. Lesion scan:
   Important hubs hatao.
   Agar activity kam ho -> hubs controller the.
   Agar activity badh jaye -> disinhibition/seizure-like state.

9. Bade model ke liye:
   - hidden_dim badhao
   - layers badhao
   - heads badhao
   - gradient checkpointing on karo
   - 8-bit optimizer use karo
   - GPU chahiye hoga
""")
    print("=" * 90)


# ----------------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()