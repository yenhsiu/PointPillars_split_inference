import time
import torch
import numpy as np
import argparse
from pointpillars.model import PointPillars

def main():
    parser = argparse.ArgumentParser(description='Measure Full Model Latency (Cloud)')
    parser.add_argument('--ckpt', default='pretrained/epoch_160.pth', help='Checkpoint path')
    parser.add_argument('--nclasses', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--device', default='cuda', help='Device to run on')
    parser.add_argument('--num_points', type=int, default=20000, help='Number of points per point cloud')
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Load Model
    print("Loading model...")
    model = PointPillars(nclasses=args.nclasses)
    try:
        model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
        print("Loaded checkpoint.")
    except:
        print("Checkpoint not found or failed to load, using random weights.")
    
    model.to(device)
    model.eval()

    # 2. Dummy Input
    # Point Cloud Range: [0, -39.68, -3, 69.12, 39.68, 1]
    # x: 0 to 69.12
    # y: -39.68 to 39.68
    # z: -3 to 1
    # intensity: 0 to 1 (random)
    
    print(f"Generating dummy input with {args.num_points} points...")
    batched_pts = []
    for _ in range(args.batch_size):
        # Random points within range
        x = torch.rand(args.num_points, 1, device=device) * 69.12
        y = torch.rand(args.num_points, 1, device=device) * (39.68 * 2) - 39.68
        z = torch.rand(args.num_points, 1, device=device) * 4 - 3
        i = torch.rand(args.num_points, 1, device=device)
        pts = torch.cat([x, y, z, i], dim=1)
        batched_pts.append(pts)

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(batched_pts, mode='test')
    
    torch.cuda.synchronize()
    
    # Measure Latency
    print(f"Measuring Full Model latency over {args.iterations} iterations...")
    latencies = []
    with torch.no_grad():
        for _ in range(args.iterations):
            torch.cuda.synchronize()
            t_start = time.time()
            
            _ = model(batched_pts, mode='test')
            
            torch.cuda.synchronize()
            t_end = time.time()
            latencies.append((t_end - t_start) * 1000) # ms

    latencies = np.array(latencies)
    print(f"Full Model Latency (ms): Mean={latencies.mean():.2f}, Std={latencies.std():.2f}, Min={latencies.min():.2f}, Max={latencies.max():.2f}")

    # Save results
    import csv
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"latency_results/full_model_latency_{timestamp}.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteration', 'Latency_ms'])
        for i, lat in enumerate(latencies):
            writer.writerow([i, lat])
    
    print(f"Results saved to {filename}")

if __name__ == '__main__':
    main()
