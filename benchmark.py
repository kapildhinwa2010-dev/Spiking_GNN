import time
import numpy as np
import torch
from model import WholeBrainSpikingNet

def run_benchmarks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading 241M Model...")
    model = WholeBrainSpikingNet(
        node_dim=5, edge_dim=10, hidden_dim=3072, 
        num_layers=8, num_heads=16, timesteps=4
    ).to(device)
    
    # In production, you would load weights: model.load_state_dict(torch.load("god_tier_241m.pth"))
    model.eval()
    
    test_nodes = 1000
    x_bench = torch.randn(test_nodes, 5, device=device)
    ei_bench = torch.stack([
        torch.randint(0, test_nodes, (3000,)), 
        torch.randint(0, test_nodes, (3000,))
    ], dim=0).to(device)
    
    pos_t = x_bench[:, :3]
    disp = pos_t[ei_bench[0]] - pos_t[ei_bench[1]]
    dist = disp.norm(dim=1, keepdim=True).clamp(min=1e-6)
    ea_bench = torch.cat([disp, dist, torch.rand(3000, 6, device=device)], dim=1)

    print("\n⚡ Running Speed Benchmark...")
    with torch.no_grad():
        for _ in range(10): model(x_bench, ei_bench, ea_bench) # Warmup
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100): model(x_bench, ei_bench, ea_bench)
        torch.cuda.synchronize()
        end = time.time()
        
    latency_ms = ((end - start) / 100) * 1000
    tps = (test_nodes * 100) / (end - start)
    
    print(f"Latency: {latency_ms:.2f} ms")
    print(f"Throughput: {tps:,.0f} TPS")
    print("\n✅ Benchmark Complete!")

if __name__ == "__main__":
    run_benchmarks()