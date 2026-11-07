import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from train_50m import build_corpus, build_graph, node_features, three_factor_update


def test_corpus_deterministic_and_labeled():
    c1, l1, names = build_corpus(5)
    c2, l2, _ = build_corpus(5)
    assert c1 == c2 and l1 == l2          # deterministic
    assert len(names) == 10
    assert set(l1) == set(range(10))      # all 10 topics present


def test_graph_shapes():
    corpus, _, _ = build_corpus(5)
    tok, edge_index, edge_attr, n_pairs = build_graph(corpus, max_pairs=10)
    assert edge_index.shape[0] == 2
    assert edge_attr.shape[1] == 10      # model edge_dim
    assert edge_index.shape[1] == edge_attr.shape[0]


def test_node_features_shape():
    corpus, _, _ = build_corpus(5)
    tok, _, _, _ = build_graph(corpus, max_pairs=10)
    x = node_features(tok, corpus[0], len(tok))
    assert x.shape == (len(tok), 5)      # model node_dim
    assert torch.isfinite(x).all()


def test_three_factor_rule_learns():
    torch.manual_seed(0)
    FD, C = 32, 10
    anchors = torch.randn(C, FD)
    W = torch.zeros(C, FD)
    b = torch.zeros(C)
    data = [(k % C, anchors[k % C] + 0.1 * torch.randn(FD)) for k in range(60)]
    acc0 = sum(int((F.linear(f, W, b).argmax()) == y) for y, f in data)
    for _ in range(15):
        for y, f in data:
            three_factor_update(W, b, f, y, lr=0.5, reward=1.0)
    acc1 = sum(int((F.linear(f, W, b).argmax()) == y) for y, f in data)
    assert acc1 > acc0
    assert acc1 / len(data) > 0.9        # validated rule converges
