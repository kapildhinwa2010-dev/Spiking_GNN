# ============================================================
# EXPERIMENTAL: Build a real connectome graph from Flywire
# ============================================================
# Requirements:
#   pip install caveclient numpy torch
# Setup:
#   1. Get an auth token at https://globalv1.flywire.ai/auth
#   2. export FLYWIRE_TOKEN="your_token"
#   3. python build_flywire_graph.py
# Output:
#   flywire_graph.pt -> node features (3D position + degree),
#   edge features (displacement + distance + 6 neurotransmitter scores)
# NOTE: Flywire tables evolve over time. This script resolves column
# names dynamically and prints what it finds. Treat it as scaffolding
# for real connectome training.
# ============================================================

import os

import numpy as np
import torch

NT_TYPES = ["acetylcholine", "gaba", "glutamate", "octopamine", "serotonin", "dopamine"]


def main():
    token = os.environ.get("FLYWIRE_TOKEN")
    if not token:
        print("FLYWIRE_TOKEN is not set.")
        print("Get a token at https://globalv1.flywire.ai/auth and export FLYWIRE_TOKEN.")
        return

    try:
        from caveclient import CAVEclient
    except ImportError:
        print("caveclient is not installed. Run: pip install caveclient")
        return

    client = CAVEclient()
    client.auth.save_token(token, overwrite=True)
    client = CAVEclient(datastack_name="flywire_publicdata")

    tables = client.materialize.get_tables()
    print("Available materialize tables: " + str(tables))

    conn_table = next((t for t in tables if "connection" in t.lower()), None)
    if conn_table is None:
        print("No connection-like table found. Exiting.")
        return

    print("Querying table: " + conn_table)
    df = client.materialize.query_table(conn_table)

    cols = list(df.columns)
    print("Columns: " + str(cols))

    def find_col(*patterns):
        for c in cols:
            lc = c.lower()
            if all(p in lc for p in patterns):
                return c
        return None

    pre_col = find_col("pre", "root") or find_col("pre")
    post_col = find_col("post", "root") or find_col("post")
    nt_col = find_col("nt") or find_col("neuro")
    x_col, y_col, z_col = find_col("x"), find_col("y"), find_col("z")

    if pre_col is None or post_col is None:
        print("Could not find pre/post root id columns. Exiting.")
        return

    print("Using pre=" + str(pre_col) + " post=" + str(post_col) + " nt=" + str(nt_col) + " pos=(" + str(x_col) + "," + str(y_col) + "," + str(z_col) + ")")

    df = df.dropna(subset=[pre_col, post_col])
    df = df[df[pre_col] != df[post_col]]
    print(str(len(df)) + " connections")

    src = df[pre_col].values.astype(np.int64)
    dst = df[post_col].values.astype(np.int64)
    root_ids = np.unique(np.concatenate([src, dst]))
    id_to_idx = {int(r): i for i, r in enumerate(root_ids)}
    n = len(root_ids)
    print(str(n) + " unique neurons")

    ei = np.stack([
        np.array([id_to_idx[int(s)] for s in src]),
        np.array([id_to_idx[int(d)] for d in dst]),
    ])

    if x_col and y_col and z_col:
        pos = df[[x_col, y_col, z_col]].values.astype(np.float32)
        node_pos = np.zeros((n, 3), dtype=np.float32)
        for s, p in zip(src, pos):
            node_pos[id_to_idx[int(s)]] = p
    else:
        print("No position columns found - using random normalized 3D positions.")
        rng = np.random.default_rng(42)
        node_pos = rng.standard_normal((n, 3)).astype(np.float32)
        node_pos = node_pos / (np.linalg.norm(node_pos, axis=1, keepdims=True).max() + 1e-6)

    degree = np.bincount(ei[0], minlength=n) + np.bincount(ei[1], minlength=n)
    degree_norm = (degree / (degree.max() + 1e-6)).reshape(-1, 1)
    x = np.concatenate([node_pos, degree_norm, np.ones((n, 1))], axis=1).astype(np.float32)

    disp = node_pos[ei[0]] - node_pos[ei[1]]
    dist = np.linalg.norm(disp, axis=1, keepdims=True)
    chem = np.zeros((ei.shape[1], 6), dtype=np.float32)
    if nt_col:
        nt_values = df[nt_col].astype(str).str.lower().values
        for j, nt in enumerate(NT_TYPES):
            chem[:, j] = (nt_values == nt).astype(np.float32)
        covered = (chem.sum(axis=1) > 0).mean()
        print("Neurotransmitter labels found for " + str(covered * 100) + "% of edges")
    else:
        print("No neurotransmitter column found - chemical features are zeros.")

    ea = np.concatenate([disp, dist, chem], axis=1).astype(np.float32)

    torch.save({
        "x": torch.from_numpy(x),
        "edge_index": torch.from_numpy(ei.astype(np.int64)),
        "edge_attr": torch.from_numpy(ea),
        "nt_types": NT_TYPES,
        "num_neurons": n,
        "num_edges": int(ei.shape[1]),
    }, "flywire_graph.pt")
    print("Saved flywire_graph.pt (" + str(n) + " neurons, " + str(ei.shape[1]) + " edges)")


if __name__ == "__main__":
    main()
