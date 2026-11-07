# ============================================================
# SYNTHETIC 3D GRAPH BUILDER (shared utility)
# Extracted from the original 500M training cell (section 6).
# Node features: x, y, z, degree_norm, avg_dist_norm  -> 5 dims
# Edge features: dx, dy, dz, dist, 6 chemical features -> 10 dims
# k-NN local connections + random long-range connections.
# ============================================================

import numpy as np
import torch


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
