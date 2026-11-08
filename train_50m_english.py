# ============================================================
# NEUROSPIKING-50M: ASLI ENGLISH PAR DOPAMINE TRAINING
# ============================================================
# SAWAL: "50M model ko asli English par 10 min deke kya hota hai?"
# DATA   : 10 asli English novels (Project Gutenberg, public domain)
#          - pride and prejudice, alice, moby dick, sherlock holmes,
#            frankenstein, dracula, treasure island, jane eyre,
#            time machine, tom sawyer
# TASK   : har sentence kaunsi book se aaya hai (10-class)
# MODEL  : wahi 50M WholeBrainSpikingNet trunk (frozen, no BPTT) +
#          dopamine-gated 3-factor readout (train_50m.py jaisa hi)
#
# STABILITY FIXES (pichhle run me loss 2.3 -> 10.4 diverge hua tha):
#   1. feat L2-normalize (pehle magnitude explode hoti thi)
#   2. lr 0.5 -> 0.05 (readout explode rokne ke liye)
#   3. W clamp +-3 (bounded weights)
# Baaki algorithm 1:1 wahi biological 3-factor rule hai.
# ============================================================

import argparse
import json
import math
import re
import time
import urllib.request
from collections import Counter

import torch
import torch.nn.functional as F

from model import WholeBrainSpikingNet
from memory import MembraneCache, forward_cached
from tokenizer import WordTokenizer
from homeostasis import HomeostaticRegulator
from stdp import STDPLearner

BOOKS = [
    ("pride_prejudice", 1342),
    ("alice_wonderland", 11),
    ("moby_dick", 2701),
    ("sherlock_holmes", 1661),
    ("frankenstein", 84),
    ("dracula", 345),
    ("treasure_island", 120),
    ("jane_eyre", 174),
    ("time_machine", 35),
    ("tom_sawyer", 74),
]

# common function words - inhe vocab se nikalte hain taaki
# book-signature CONTENT words (whale, vampire, detective...) bache
STOP = set("""the of and to a in that it is was for on with as his he be at by i this had not
are but or have from you they which we her she him them all were been has when who will more
there one would what so no if my me said up out about into then them its""".split())


def fetch_book(gid, timeout=30):
    urls = [
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
    ]
    for u in urls:
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "SpikingGNN-CI/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"  [warn] {u} -> {str(e)[:70]}")
    return None


def sentences_from(text, limit):
    text = text.lower()
    raw = re.split(r"[.!?]+", text)
    sents = [s.strip() for s in raw if 4 <= len(s.split()) <= 30]
    return sents[:limit]


def edge_features(es, ee, cnt):
    rows = []
    for a, b in zip(es, ee):
        c = cnt.get((min(a, b), max(a, b)), 1)
        rows.append([
            math.sin(a / 9.0), math.cos(a / 9.0),
            math.sin(b / 9.0), math.cos(b / 9.0),
            math.sin((a + b) / 17.0), math.cos((a + b) / 17.0),
            math.log1p(c) / 4.0, a / 400.0, b / 400.0, 1.0,
        ])
    return torch.tensor(rows, dtype=torch.float32)


def node_features(tok, sentence, num_nodes):
    ids = tok.encode(sentence)
    x = torch.zeros(num_nodes, 5)
    for pos, wid in enumerate(ids):
        if wid >= num_nodes:
            continue
        x[wid, 0] += 1.0 / 3.0
        x[wid, 1] = math.log1p(x[wid, 0].item() * 3.0)
        x[wid, 2] = 1.0
        x[wid, 3] = max(x[wid, 3].item(), (pos + 1) / max(len(ids), 1))
        x[wid, 4] = 1.0
    return x


def three_factor_update(W, b, feat, y, lr, reward):
    """3-factor dopamine rule + stability fixes:
    dw = lr * reward * outer(onehot - softmax, feat_normalized)"""
    feat = feat / (feat.norm() + 1e-6)          # FIX 1: normalize
    logits = F.linear(feat, W, b)
    p = torch.softmax(logits, dim=-1)
    err = torch.zeros_like(p)
    err[y] = 1.0
    err = err - p
    correct = int(p.argmax()) == y
    r = reward if correct else 0.25 * reward   # dopamine gate (zero nahi)
    W.data += lr * r * torch.outer(err, feat)
    W.data.clamp_(-3.0, 3.0)                    # FIX 3: bounded weights
    b.data += lr * r * err * 0.01
    return logits, r, correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-min", type=float, default=10.0)
    ap.add_argument("--hidden", type=int, default=1152)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--timesteps", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=300)
    ap.add_argument("--edges", type=int, default=200)
    ap.add_argument("--sentences-per-book", type=int, default=50)
    ap.add_argument("--test-per-book", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.05)   # FIX 2: 0.5 -> 0.05
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    budget_s = args.budget_min * 60.0
    t0 = time.time()

    # ---------------- 1. ASLI ENGLISH DATA ----------------
    corpus, labels, names = [], [], []
    for name, gid in BOOKS:
        raw = fetch_book(gid)
        if not raw:
            continue
        sents = sentences_from(raw, args.sentences_per_book)
        if len(sents) < args.test_per_book + 5:
            continue
        corpus += sents
        labels += [len(names)] * len(sents)
        names.append(name)
        print(f"[data] {name}: {len(sents)} sentences downloaded")

    if len(names) < 4:
        raise SystemExit(f"sirf {len(names)} books mili - network/Gutenberg issue, retry karo")

    # stratified split: har book ke last N sentences = hold-out test
    test_idx, train_idx = [], []
    counts = Counter(labels)
    seen = Counter()
    for i, y in enumerate(labels):
        seen[y] += 1
        if counts[y] - seen[y] < args.test_per_book:
            test_idx.append(i)
        else:
            train_idx.append(i)

    # vocab: top content words (stopwords hata kar) - book signatures
    text = " ".join(corpus)
    tokens = WordTokenizer._tokenize(text)
    vocab = [w for w, _ in Counter(tokens).most_common(args.vocab * 4) if w not in STOP]
    vocab = vocab[:args.vocab]
    tok = WordTokenizer(vocab)
    N = len(tok)

    # co-occurrence graph
    src, dst = tok.build_text_graph(text, window=5, max_edges=args.edges * 24)
    cnt = Counter()
    for a, b in zip(src, dst):
        if a != b:
            cnt[(min(a, b), max(a, b))] += 1
    top = [p for p, _ in cnt.most_common(args.edges)]
    es, ee = [], []
    for a, b in top:
        es += [a, b]
        ee += [b, a]
    edge_index = torch.tensor([es, ee], dtype=torch.long)
    edge_attr = edge_features(es, ee, cnt)
    n_classes = len(names)
    labels_t = torch.tensor(labels)

    print(f"[data] {n_classes} books | {len(corpus)} sentences | {N} vocab nodes | {len(top)} undirected edges")

    # ---------------- 2. 50M MODEL (frozen trunk) ----------------
    model = WholeBrainSpikingNet(node_dim=5, edge_dim=10, hidden_dim=args.hidden,
                                 num_layers=args.layers, num_heads=args.heads,
                                 timesteps=args.timesteps, num_classes=n_classes, dropout=0.0)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params / 1e6:.1f}M params")
    assert 45e6 < n_params < 56e6, f"50M target miss: {n_params}"

    W = torch.zeros(n_classes, args.hidden)
    b = torch.zeros(n_classes)
    cache = MembraneCache(N, args.hidden, args.layers)
    reg = HomeostaticRegulator(args.hidden, target_rate=0.3, lr=0.02)
    stdp = STDPLearner(N, N)

    def run_set(indices, update):
        correct, loss, seen = 0, 0.0, 0
        for i in indices:
            x = node_features(tok, corpus[i], N)
            with torch.no_grad():
                rates = forward_cached(model, x, edge_index, edge_attr, cache)
                feat = rates.mean(dim=0)
            y = labels_t[i].item()
            if update:
                logits, r, ok = three_factor_update(W, b, feat, y, args.lr, 1.0)
            else:
                feat_n = feat / (feat.norm() + 1e-6)
                logits = F.linear(feat_n, W, b)
                ok = int(logits.argmax()) == y
            correct += int(ok)
            loss += F.cross_entropy(logits.unsqueeze(0), torch.tensor([y])).item()
            seen += 1
            if update and seen % 8 == 0:
                thr_delta = reg.step(rates.mean(dim=0))
                with torch.no_grad():
                    for layer in model.layers:
                        layer.neuron.threshold.data += 0.5 * thr_delta
            if update and time.time() - t0 > budget_s:
                break
        return correct / max(seen, 1), loss / max(seen, 1), seen

    start_acc, start_loss, _ = run_set(test_idx, update=False)
    print(f"[init] hold-out test acc {start_acc:.2%} (chance = {1 / n_classes:.0%})")

    history, epoch, trained = [], 0, 0
    while time.time() - t0 < budget_s:
        g = torch.Generator().manual_seed(args.seed + epoch)
        order = torch.randperm(len(train_idx), generator=g).tolist()
        acc, loss, seen = run_set([train_idx[k] for k in order], update=True)
        trained += seen
        epoch += 1
        history.append({"epoch": epoch, "train_acc": round(acc, 4), "train_loss": round(loss, 4)})
        print(f"[epoch {epoch}] train acc {acc:.2%} loss {loss:.4f} samples {trained} elapsed {(time.time() - t0) / 60:.1f}m")
        if seen < len(train_idx):
            break

    test_acc, test_loss, _ = run_set(test_idx, update=False)

    # STDP plasticity analysis
    x = node_features(tok, corpus[test_idx[0]], N)
    with torch.no_grad():
        rates = forward_cached(model, x, edge_index, edge_attr, cache)
    s = (rates.mean(dim=1) > 0.5).float()
    stdp.init_traces()
    for _ in range(3):
        stdp.step_traces(s, s)
    sparsity = float((rates.mean(dim=1) < 0.1).float().mean())

    results = {
        "experiment": "asli English par dopamine-gated 50M training (10 min budget)",
        "books": names,
        "sentences": len(corpus),
        "vocab_nodes": N,
        "model": f"NeuroSpiking-50M (WholeBrainSpikingNet hidden {args.hidden} x {args.layers} layers x {args.heads} heads)",
        "parameters": n_params,
        "training_algorithm": "Dopamine-gated 3-factor local learning (trunk frozen, no BPTT) + stability fixes: feat-normalize, lr 0.05, W-clamp",
        "test_start": {"accuracy": round(start_acc, 4), "loss": round(start_loss, 4)},
        "test_final": {"accuracy": round(test_acc, 4), "loss": round(test_loss, 4)},
        "chance_accuracy": round(1 / n_classes, 4),
        "epochs_completed": epoch,
        "samples_trained": trained,
        "spike_sparsity": round(sparsity, 4),
        "stdp_plasticity_index": round(float(stdp.w.mean()), 6),
        "history": history[-8:],
        "runtime_min": round((time.time() - t0) / 60.0, 2),
        "budget_min": args.budget_min,
    }
    print(json.dumps(results, indent=2))
    with open("train_50m_english_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved train_50m_english_results.json")


if __name__ == "__main__":
    main()
