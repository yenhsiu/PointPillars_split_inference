import argparse
import numpy as np
import os
import torch
import time
from tqdm import tqdm
import csv
from datetime import datetime

from pointpillars.utils import setup_seed
from pointpillars.dataset import Kitti, get_dataloader
from pointpillars.model import PointPillars


class LatencyMeasurement:
    """測量 PointPillars 各個模組的延遲時間"""
    
    def __init__(self, model, device='cuda', warmup_iterations=10):
        self.model = model
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.results = {
            'pillar_layer': [],
            'pillar_encoder': [],
            'backbone': [],
            'neck': [],
            'head': [],
            'post_process': [],
            'total': []
        }
    
    def measure_inference(self, batched_pts, batched_gt_bboxes=None, batched_gt_labels=None, mode='val'):
        """測量單次推理的延遲時間"""
        timings = {}
        
        # Move data to device
        if isinstance(self.device, str) and self.device.startswith('cuda'):
            batched_pts = [pts.to(self.device) if torch.is_tensor(pts) else pts for pts in batched_pts]
            if batched_gt_bboxes is not None:
                batched_gt_bboxes = [bboxes.to(self.device) if torch.is_tensor(bboxes) else bboxes for bboxes in batched_gt_bboxes]
            if batched_gt_labels is not None:
                batched_gt_labels = [labels.to(self.device) if torch.is_tensor(labels) else labels for labels in batched_gt_labels]
        
        # Synchronize CUDA before starting (when using GPU)
        if isinstance(self.device, str) and self.device.startswith('cuda'):
            torch.cuda.synchronize()
        
        # Total inference time
        total_start = time.time()
        
        # 1. Pillar Layer
        start = time.time()
        pillars, coors_batch, npoints_per_pillar = self.model.pillar_layer(batched_pts)
        if isinstance(self.device, str) and self.device.startswith('cuda'):
            torch.cuda.synchronize()
        timings['pillar_layer'] = (time.time() - start) * 1000  # ms
        
        # 2. Pillar Encoder
        start = time.time()
        features = self.model.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
        if isinstance(self.device, str) and self.device.startswith('cuda'):
            torch.cuda.synchronize()
        timings['pillar_encoder'] = (time.time() - start) * 1000
        
        # 3. Backbone
        start = time.time()
        x = self.model.backbone(features)
        if isinstance(self.device, str) and self.device.startswith('cuda'):
            torch.cuda.synchronize()
        timings['backbone'] = (time.time() - start) * 1000
        
        # 4. Neck
        start = time.time()
        x = self.model.neck(x)
        if isinstance(self.device, str) and self.device.startswith('cuda'):
            torch.cuda.synchronize()
        timings['neck'] = (time.time() - start) * 1000
        
        # 5. Head
        start = time.time()
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.model.head(x)
        if isinstance(self.device, str) and self.device.startswith('cuda'):
            torch.cuda.synchronize()
        timings['head'] = (time.time() - start) * 1000
        
        # 6. Post-processing (NMS and conversion)
        # 只在 GPU 模式下測量後處理，因為 NMS 需要 CUDA
        if isinstance(self.device, str) and self.device.startswith('cuda'):
            start = time.time()
            if mode == 'val':
                # Get anchors for post-processing
                device = bbox_cls_pred.device
                feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
                anchors = self.model.anchors_generator.get_multi_anchors(feature_map_size)
                batched_anchors = [anchors for _ in range(bbox_cls_pred.size(0))]
                batch_results = self.model.get_predicted_bboxes(
                    bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors
                )
            torch.cuda.synchronize()
            timings['post_process'] = (time.time() - start) * 1000
        else:
            # CPU 模式下跳過後處理測量
            timings['post_process'] = 0.0
        
        # Total time
        if isinstance(self.device, str) and self.device.startswith('cuda'):
            torch.cuda.synchronize()
        timings['total'] = (time.time() - total_start) * 1000
        
        return timings
    
    def warmup(self, dataloader):
        """預熱模型以獲得穩定的測量結果"""
        print(f"Warming up for {self.warmup_iterations} iterations...")
        self.model.eval()
        
        with torch.no_grad():
            for i, data_dict in enumerate(dataloader):
                if i >= self.warmup_iterations:
                    break
                
                batched_pts = data_dict['batched_pts']
                batched_gt_bboxes = data_dict['batched_gt_bboxes']
                batched_labels = data_dict['batched_labels']
                
                # Move data to device
                if isinstance(self.device, str) and self.device.startswith('cuda'):
                    batched_pts = [pts.to(self.device) if torch.is_tensor(pts) else pts for pts in batched_pts]
                    batched_gt_bboxes = [bboxes.to(self.device) if torch.is_tensor(bboxes) else bboxes for bboxes in batched_gt_bboxes]
                    batched_labels = [labels.to(self.device) if torch.is_tensor(labels) else labels for labels in batched_labels]
                
                # Simple forward pass (只測量前向傳播，不做 NMS)
                pillars, coors_batch, npoints_per_pillar = self.model.pillar_layer(batched_pts)
                features = self.model.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
                x = self.model.backbone(features)
                x = self.model.neck(x)
                bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.model.head(x)
        
        print("Warmup completed.")
    
    def run_benchmark(self, dataloader, num_samples=None):
        """運行基準測試"""
        print("Starting latency measurement...")
        self.model.eval()
        
        with torch.no_grad():
            for i, data_dict in enumerate(tqdm(dataloader)):
                if num_samples is not None and i >= num_samples:
                    break
                
                batched_pts = data_dict['batched_pts']
                batched_gt_bboxes = data_dict['batched_gt_bboxes']
                batched_labels = data_dict['batched_labels']
                
                timings = self.measure_inference(batched_pts, batched_gt_bboxes, batched_labels)
                
                # Store results
                for key, value in timings.items():
                    self.results[key].append(value)
        
        print("Benchmark completed.")
    
    def compute_statistics(self):
        """計算統計數據"""
        stats = {}
        for key, values in self.results.items():
            if len(values) > 0:
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99)
                }
        return stats
    
    def print_results(self):
        """打印結果"""
        stats = self.compute_statistics()
        
        print("\n" + "="*60)
        print("PointPillars Latency Measurement Results")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Number of samples: {len(self.results['total'])}")
        print("-"*60)
        print(f"{'Module':<20} {'Mean (ms)':<15} {'Percentage':<15}")
        print("-"*60)
        
        module_order = ['pillar_layer', 'pillar_encoder', 'backbone', 'neck', 'head', 'post_process', 'total']
        total_mean = stats['total']['mean']
        
        for module in module_order:
            if module in stats:
                s = stats[module]
                if module == 'total':
                    print(f"{module:<20} {s['mean']:<15.2f} {'100.00%':<15}")
                else:
                    percentage = (s['mean'] / total_mean) * 100
                    print(f"{module:<20} {s['mean']:<15.2f} {percentage:<14.2f}%")
        
        print("="*60)
    
    def save_results(self, save_dir='latency_results'):
        """保存結果到 CSV 文件"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw data
        raw_data_path = os.path.join(save_dir, f'latency_raw_{timestamp}.csv')
        with open(raw_data_path, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = ['sample_id'] + list(self.results.keys())
            writer.writerow(headers)
            
            num_samples = len(self.results['total'])
            for i in range(num_samples):
                row = [i] + [self.results[key][i] for key in self.results.keys()]
                writer.writerow(row)
        
        print(f"\nRaw data saved to: {raw_data_path}")
        
        # Save statistics
        stats = self.compute_statistics()
        stats_path = os.path.join(save_dir, f'latency_stats_{timestamp}.csv')
        with open(stats_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Module', 'Mean (ms)', 'Std (ms)', 'Min (ms)', 'Max (ms)', 'Median (ms)', 'P95 (ms)', 'P99 (ms)'])
            
            for module, stat in stats.items():
                writer.writerow([
                    module,
                    f"{stat['mean']:.2f}",
                    f"{stat['std']:.2f}",
                    f"{stat['min']:.2f}",
                    f"{stat['max']:.2f}",
                    f"{stat['median']:.2f}",
                    f"{stat['p95']:.2f}",
                    f"{stat['p99']:.2f}"
                ])
        
        print(f"Statistics saved to: {stats_path}")


def main(args):
    # Set random seed
    setup_seed(args.seed)
    
    # Setup device (allow selecting GPU id)
    if args.no_cuda or not torch.cuda.is_available():
        device = 'cpu'
        print("Using CPU for inference")
    else:
        gpu_id = int(args.gpu_id) if hasattr(args, 'gpu_id') else 0
        # set the active CUDA device
        torch.cuda.set_device(gpu_id)
        device = f'cuda:{gpu_id}'
        try:
            dev_name = torch.cuda.get_device_name(gpu_id)
        except Exception:
            dev_name = f'cuda:{gpu_id}'
        print(f"Using GPU for inference: {dev_name} (id={gpu_id})")
    
    # Load dataset
    print(f"Loading dataset from {args.data_root}...")
    val_dataset = Kitti(data_root=args.data_root, split='val')
    val_dataloader = get_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )
    print(f"Dataset loaded: {len(val_dataset)} samples")
    
    # Load model
    print(f"Loading model from {args.ckpt}...")
    model = PointPillars(nclasses=args.nclasses)
    
    if isinstance(device, str) and device.startswith('cuda'):
        model = model.to(device)
        checkpoint = torch.load(args.ckpt)
    else:
        checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
    
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully")
    
    # Create latency measurement
    latency_measure = LatencyMeasurement(
        model=model,
        device=device,
        warmup_iterations=args.warmup_iterations
    )
    
    # Warmup
    if args.warmup_iterations > 0:
        latency_measure.warmup(val_dataloader)
    
    # Run benchmark
    latency_measure.run_benchmark(val_dataloader, num_samples=args.num_samples)
    
    # Print and save results
    latency_measure.print_results()
    latency_measure.save_results(save_dir=args.save_dir)
    
    print("\nLatency measurement completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measure PointPillars Inference Latency')
    parser.add_argument('--data_root', default='/home/yenhsiu/datasets',
                        help='your data root for kitti')
    parser.add_argument('--ckpt', default='pretrained/epoch_160.pth',
                        help='your checkpoint for kitti')
    parser.add_argument('--save_dir', default='latency_results',
                        help='directory to save results')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size (recommend 1 for accurate latency measurement)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for data loading')
    parser.add_argument('--nclasses', type=int, default=3,
                        help='number of classes')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='number of samples to measure (None = all samples)')
    parser.add_argument('--warmup_iterations', type=int, default=10,
                        help='number of warmup iterations')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id to use when using CUDA')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.batch_size > 1:
        print("Warning: batch_size > 1 may not give accurate per-sample latency measurements")
    
    main(args)
