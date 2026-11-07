import os
import urllib.request
import zipfile
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
import gc
import json
from model import WholeBrainSpikingNet

def download_text8():
    url = "http://mattmahoney.net/dc/text8.zip"
    if not os.path.exists("text8"):
        print("Downloading text8 corpus (10M+ tokens)...")
        urllib.request.urlretrieve(url, "text8.zip")
        with zipfile.ZipFile("text8.zip", 'r') as zip_ref: zip_ref.extractall(".")
    with open("text8", 'r', encoding='utf-8') as f: 
        return f.read().split()[:10_000_000]

def build_graph(tokens, vocab_size=3000):
    print(f"Building Language Graph (Top {vocab_size} words)...")
    vocab = [word for word, count in Counter(tokens).most_common(vocab_size)]
    word_to_id = {w: i for i, w in enumerate(vocab)}
    
    edges_src, edges_dst = [], []
    for i in range(len(tokens) - 5):
        if tokens[i] in word_to_id:
            src = word_to_id[tokens[i]]
            for j in range(1, 6):
                if tokens[i+j] in word_to_id:
                    dst = word_to_id[tokens[i+j]]
                    if src != dst: edges_src.append(src); edges_dst.append(dst)
        if len(edges_src) >= 5_000_000: break
        
    edges_src_t = torch.tensor(edges_src, dtype=torch.long)
    edges_dst_t = torch.tensor(edges_dst, dtype=torch.long)
    
    N = len(vocab)
    rng = np.random.default_rng(42)
    pos = rng.standard_normal((N, 3)).astype(np.float32)
    pos = pos - pos.mean(axis=0, keepdims=True)
    pos = pos / (np.linalg.norm(pos, axis=1, keepdims=True).max() + 1e-6)
    degree = np.bincount(edges_src_t.numpy(), minlength=N).astype(np.float32)
    degree_norm = (degree / (degree.max() + 1e-6)).reshape(-1, 1)
    
    x_full_t = torch.tensor(np.concatenate([pos, degree_norm, np.ones((N, 1))], axis=1), dtype=torch.float32)
    
    return x_full_t, edges_src_t, edges_dst_t, vocab, word_to_id

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokens = download_text8()
    x_full_t, edges_src_t, edges_dst_t, vocab, word_to_id = build_graph(tokens)
    
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    print("Saved vocab.json")
    
    print("Initializing 241M God-Tier Model...")
    model = WholeBrainSpikingNet(
        node_dim=5, edge_dim=10, hidden_dim=3072, 
        num_layers=8, num_heads=16, timesteps=4
    ).to(device)
    
    config = {
        "node_dim": 5, "edge_dim": 10, "hidden_dim": 3072,
        "num_layers": 8, "num_heads": 16, "timesteps": 4,
        "num_classes": 10, "weights_file": "god_tier_241m.pth",
    }
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print("Saved config.json")
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    
    print("Starting Training (30 Steps)...")
    model.train()
    for step in range(30):
        idx = torch.randperm(len(edges_src_t))[:5000]
        src, dst = edges_src_t[idx], edges_dst_t[idx]
        edge_index = torch.stack([src, dst], dim=0).to(device)
        
        x = x_full_t.to(device)
        pos_t = x[:, :3]
        disp = pos_t[src] - pos_t[dst]
        dist = disp.norm(dim=1, keepdim=True).clamp(min=1e-6)
        edge_attr = torch.cat([disp, dist, torch.rand(len(src), 6, device=device)], dim=1)
        
        opt.zero_grad(set_to_none=True)
        logits, avg_rate = model(x, edge_index, edge_attr)
        loss = F.mse_loss(avg_rate.mean(dim=1, keepdim=True), x[:, 3:4]) + 0.01 * avg_rate.abs().mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        
        if (step + 1) % 5 == 0:
            print(f"Step {step+1}/30 | Loss: {loss.item():.4f}")
            
        del x, edge_index, edge_attr, logits, avg_rate, loss
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
    torch.save(model.state_dict(), "god_tier_241m.pth")
    print("Training Complete! Saved to god_tier_241m.pth")

if __name__ == "__main__":
    train()