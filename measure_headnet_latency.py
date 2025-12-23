import time
import torch
import numpy as np
import argparse
from pointpillars.model import PointPillars
from pointpillars.model.split_nets import split_pointpillars

def main():
    parser = argparse.ArgumentParser(description='Measure HeadNet Latency')
    parser.add_argument('--ckpt', default='pretrained/epoch_160.pth', help='Checkpoint path')
    parser.add_argument('--nclasses', type=int, default=3)
    parser.add_argument('--num_points', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--device', default='cuda', help='Device to run on')
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Load Model & Split
    print("Loading model...")
    full_model = PointPillars(nclasses=args.nclasses)
    # Load weights if available, otherwise random init is fine for latency
    try:
        full_model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
        print("Loaded checkpoint.")
    except:
        print("Checkpoint not found or failed to load, using random weights.")
    
    full_model.to(device)
    full_model.eval()
    
    headnet, _ = split_pointpillars(full_model)
    headnet.eval()

    # 2. Dummy Input
    # Point Cloud Range: [0, -39.68, -3, 69.12, 39.68, 1]
    print(f"Generating dummy input with {args.num_points} points (Uniform Distribution)...")
    dummy_pts = []
    for _ in range(args.batch_size):
        # Random points within range
        x = torch.rand(args.num_points, 1, device=device) * 69.12
        y = torch.rand(args.num_points, 1, device=device) * (39.68 * 2) - 39.68
        z = torch.rand(args.num_points, 1, device=device) * 4 - 3
        i = torch.rand(args.num_points, 1, device=device)
        pts = torch.cat([x, y, z, i], dim=1)
        dummy_pts.append(pts)

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = headnet(dummy_pts)
    
    torch.cuda.synchronize()
    
    # Measure HeadNet
    print(f"Measuring HeadNet latency over {args.iterations} iterations...")
    headnet_times = []
    with torch.no_grad():
        for _ in range(args.iterations):
            torch.cuda.synchronize()
            t0 = time.time()
            _ = headnet(dummy_pts)
            torch.cuda.synchronize()
            t1 = time.time()
            headnet_times.append((t1 - t0) * 1000) # ms
    
    headnet_times = np.array(headnet_times)
    print(f"HeadNet Latency (ms): Mean={headnet_times.mean():.2f}, Std={headnet_times.std():.2f}")

    # Save results
    import csv
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"latency_results/headnet_latency_{timestamp}.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteration', 'Latency_ms'])
        for i, lat in enumerate(headnet_times):
            writer.writerow([i, lat])
    
    print(f"Results saved to {filename}")

if __name__ == '__main__':
    main()
