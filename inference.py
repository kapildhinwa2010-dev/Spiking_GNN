import argparse
import json
import os

import numpy as np
import torch

from model import WholeBrainSpikingNet

# Candidate checkpoint names (README Model Zoo names first, then legacy default)
WEIGHT_CANDIDATES = [
    "flywire_bio_brain_241m.pth",
    "language_cognitive_brain_241m.pth",
    "god_tier_241m.pth",
]


def find_weights(config):
    """Return the first checkpoint that exists, else None."""
    cands = [config.get("weights_file")] + WEIGHT_CANDIDATES
    for w in cands:
        if w and os.path.exists(w):
            return w
    return None


def load_brain(weights=None, device=None):
    print("🧠 Loading 241M God-Tier Brain...")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("config.json", "r") as f:
        config = json.load(f)

    model = WholeBrainSpikingNet(
        node_dim=config.get("node_dim", 5),
        edge_dim=config.get("edge_dim", 10),
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        timesteps=config["timesteps"],
    ).to(device)

    weights = weights or find_weights(config)
    if weights:
        try:
            model.load_state_dict(torch.load(weights, map_location=device))
            print("✅ Weights loaded successfully! (" + weights + ")")
        except Exception as exc:
            print("⚠️ Could not load weights (" + str(exc) + "). Using random brain.")
    else:
        print("⚠️ No trained weights found (tried: " + ", ".join(WEIGHT_CANDIDATES) + ").")
        print("   Download one from the README Model Zoo, or run: make weights")

    model.eval()
    return model, device


def think(model, device, word_id, vocab_size=1000):
    """Simulates a thought process for a given word node."""
    x = torch.randn(vocab_size, 5, device=device)

    src = torch.full((500,), word_id, dtype=torch.long, device=device)
    dst = torch.randint(0, vocab_size, (500,), dtype=torch.long, device=device)
    edge_index = torch.stack([src, dst], dim=0)

    pos_t = x[:, :3]
    disp = pos_t[edge_index[0]] - pos_t[edge_index[1]]
    dist = disp.norm(dim=1, keepdim=True).clamp(min=1e-6)
    edge_attr = torch.cat([disp, dist, torch.rand(500, 6, device=device)], dim=1)

    with torch.no_grad():
        logits, avg_rate = model(x, edge_index, edge_attr)

    return avg_rate.mean(dim=1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="NeuroSpiking interactive inference engine")
    ap.add_argument("--weights", type=str, default=None,
                    help="path to a .pth checkpoint (default: auto-detect from README Model Zoo names)")
    args = ap.parse_args()

    model, device = load_brain(weights=args.weights)

    vocab = None
    try:
        with open("vocab.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)
        print("✅ Loaded vocab.json (" + str(len(vocab)) + " words)")
    except (FileNotFoundError, json.JSONDecodeError):
        print("⚠️ vocab.json not found - showing node IDs only.")

    print("\n" + "=" * 50)
    print("🧠 NEUROSPIKING INFERENCE ENGINE")
    print("=" * 50)
    print("Type a concept (number 0-999) to see how the brain fires.")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("Enter Node ID (0-999): ").strip()
        if user_input.lower() == "quit":
            break

        try:
            node_id = int(user_input)
            if 0 <= node_id <= 999:
                print(f"⚡ Stimulating Node {node_id}...")
                firing_rates = think(model, device, node_id)

                top_5 = np.argsort(firing_rates)[::-1][:5]
                print("🔥 Top 5 Activated Neurons (Thought Chain):")
                for rank, idx in enumerate(top_5, 1):
                    word = f" ({vocab[idx]})" if vocab and idx < len(vocab) else ""
                    print(f"   {rank}. Node {idx}{word} (Firing Rate: {firing_rates[idx]:.4f})")
                print("-" * 50)
            else:
                print("⚠️ Please enter a number between 0 and 999.")
        except ValueError:
            print("⚠️ Invalid input. Please enter a number or 'quit'.")


if __name__ == "__main__":
    main()
