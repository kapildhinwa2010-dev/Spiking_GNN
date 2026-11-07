import torch
import numpy as np
import json
from model import WholeBrainSpikingNet

def load_brain():
    print("🧠 Loading 241M God-Tier Brain...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Config
    with open("config.json", "r") as f:
        config = json.load(f)
        
    # Initialize Model
    model = WholeBrainSpikingNet(
        node_dim=config.get("node_dim", 5),
        edge_dim=config.get("edge_dim", 10),
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        timesteps=config["timesteps"]
    ).to(device)
    
    # Load Weights
    try:
        model.load_state_dict(torch.load(config.get("weights_file", "god_tier_241m.pth"), map_location=device))
        print("✅ Weights loaded successfully!")
    except FileNotFoundError:
        print("⚠️ No trained weights found. Using randomly initialized brain.")
        
    model.eval()
    return model, device

def think(model, device, word_id, vocab_size=1000):
    """Simulates a thought process for a given word node."""
    # Create a dummy graph centered around this word
    x = torch.randn(vocab_size, 5, device=device)
    
    # Connect the target word to many others (high degree hub for this thought)
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
    model, device = load_brain()
    
    # Optional: word vocabulary (saved automatically by train.py)
    vocab = None
    try:
        with open("vocab.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)
        print("✅ Loaded vocab.json (" + str(len(vocab)) + " words)")
    except (FileNotFoundError, json.JSONDecodeError):
        print("⚠️ vocab.json not found - showing node IDs only.")
    
    print("\n" + "="*50)
    print("🧠 NEUROSPIKING INFERENCE ENGINE")
    print("="*50)
    print("Type a concept (number 0-999) to see how the brain fires.")
    print("Type 'quit' to exit.\n")
    
    while True:
        user_input = input("Enter Node ID (0-999): ").strip()
        if user_input.lower() == 'quit':
            break
            
        try:
            node_id = int(user_input)
            if 0 <= node_id <= 999:
                print(f"⚡ Stimulating Node {node_id}...")
                firing_rates = think(model, device, node_id)
                
                top_5 = np.argsort(firing_rates)[::-1][:5]
                print(f"🔥 Top 5 Activated Neurons (Thought Chain):")
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