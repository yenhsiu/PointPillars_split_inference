import time
import torch
import numpy as np
import argparse
from pointpillars.model import PointPillars
from pointpillars.model.split_nets import split_pointpillars

def main():
    parser = argparse.ArgumentParser(description='Measure Cloud Latency (TailNet)')
    parser.add_argument('--ckpt', default='pretrained/epoch_160.pth', help='Checkpoint path')
    parser.add_argument('--nclasses', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--device', default='cuda', help='Device to run on')
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Load Model & Split
    print("Loading model...")
    full_model = PointPillars(nclasses=args.nclasses)
    try:
        full_model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
        print("Loaded checkpoint.")
    except:
        print("Checkpoint not found or failed to load, using random weights.")
    
    full_model.to(device)
    full_model.eval()
    
    _, tailnet = split_pointpillars(full_model)
    tailnet.eval()

    # 2. Dummy Input
    # Shape: (B, 64, 496, 432)
    # Note: PointPillars features are (B, C, H, W)
    dummy_features = torch.randn(args.batch_size, 64, 496, 432, device=device)

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = tailnet(dummy_features)
    
    torch.cuda.synchronize()
    
    # Measure TailNet
    print(f"Measuring TailNet (Cloud) latency over {args.iterations} iterations...")
    tailnet_times = []
    with torch.no_grad():
        for _ in range(args.iterations):
            torch.cuda.synchronize()
            t0 = time.time()
            _ = tailnet(dummy_features)
            torch.cuda.synchronize()
            t1 = time.time()
            tailnet_times.append((t1 - t0) * 1000) # ms
    
    avg_tail = np.mean(tailnet_times)
    print(f"TailNet Latency: {avg_tail:.2f} ms")

if __name__ == '__main__':
    main()
