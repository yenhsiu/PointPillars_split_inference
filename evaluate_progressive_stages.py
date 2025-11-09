"""
Progressive RQ Stages Evaluation Script
Evaluate all trained stage weights with full KITTI metrics
"""

import argparse
import numpy as np
import os
import torch
import yaml
import logging
from tqdm import tqdm
from pathlib import Path

from pointpillars.utils import setup_seed, keep_bbox_from_image_range, \
    keep_bbox_from_lidar_range, write_pickle, write_label
from pointpillars.dataset import Kitti, get_dataloader
from utils.model_utils import config_to_args, setup_model_and_data, setup_progressive_stage

# Import evaluation functions from evaluate.py
from evaluate import do_eval


def evaluate_stage(headnet, tailnet, rq_bottleneck, val_dataloader, 
                   num_codebooks, embed_size, args, saved_path, pcd_limit_range):
    """
    Evaluate a specific stage configuration.
    """
    CLASSES = Kitti.CLASSES
    LABEL2CLASSES = {v:k for k, v in CLASSES.items()}
    
    # Set evaluation stage
    rq_bottleneck.set_evaluation_stage(num_codebooks, embed_size)
    
    # Set models to eval mode
    headnet.eval()
    tailnet.eval()
    rq_bottleneck.eval()
    
    format_results = {}
    logging.info(f'Predicting for {num_codebooks} codebook(s), embedding size {embed_size}')
    
    with torch.no_grad():
        for i, data_dict in enumerate(tqdm(val_dataloader, desc='Inference')):
            if torch.cuda.is_available():
                # move the tensors to the cuda
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()
            
            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']
            
            # Forward pass through split model
            pillar_features = headnet(batched_pts)
            pillar_features_hwc = pillar_features.permute(0, 2, 3, 1)
            quantized_features, _, _, _ = rq_bottleneck(pillar_features_hwc)
            quantized_features = quantized_features.permute(0, 3, 1, 2)
            
            # Get predictions from tailnet
            bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors = tailnet(
                quantized_features,
                mode='val',
                batched_gt_bboxes=batched_gt_bboxes,
                batched_gt_labels=batched_labels,
                batch_size=len(batched_pts)
            )
            
            # Convert predictions to final results
            batch_results = tailnet.get_predicted_bboxes(
                bbox_cls_pred=bbox_cls_pred,
                bbox_pred=bbox_pred,
                bbox_dir_cls_pred=bbox_dir_cls_pred,
                batched_anchors=batched_anchors
            )
            
            # Format results
            for j, result in enumerate(batch_results):
                format_result = {
                    'name': [],
                    'truncated': [],
                    'occluded': [],
                    'alpha': [],
                    'bbox': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': []
                }
                
                calib_info = data_dict['batched_calib_info'][j]
                tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
                r0_rect = calib_info['R0_rect'].astype(np.float32)
                P2 = calib_info['P2'].astype(np.float32)
                image_shape = data_dict['batched_img_info'][j]['image_shape']
                idx = data_dict['batched_img_info'][j]['image_idx']
                
                result_filter = keep_bbox_from_image_range(result, tr_velo_to_cam, r0_rect, P2, image_shape)
                result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)

                lidar_bboxes = result_filter['lidar_bboxes']
                labels, scores = result_filter['labels'], result_filter['scores']
                bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes']
                
                for lidar_bbox, label, score, bbox2d, camera_bbox in \
                    zip(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes):
                    format_result['name'].append(LABEL2CLASSES[label])
                    format_result['truncated'].append(0.0)
                    format_result['occluded'].append(0)
                    alpha = camera_bbox[6] - np.arctan2(camera_bbox[0], camera_bbox[2])
                    format_result['alpha'].append(alpha)
                    format_result['bbox'].append(bbox2d)
                    format_result['dimensions'].append(camera_bbox[3:6])
                    format_result['location'].append(camera_bbox[:3])
                    format_result['rotation_y'].append(camera_bbox[6])
                    format_result['score'].append(score)
                
                # Save submission format
                submit_path = os.path.join(saved_path, 'submit')
                os.makedirs(submit_path, exist_ok=True)
                write_label(format_result, os.path.join(submit_path, f'{idx:06d}.txt'))
                
                format_results[idx] = {k:np.array(v) for k, v in format_result.items()}
    
    # Save results
    write_pickle(format_results, os.path.join(saved_path, 'results.pkl'))
    
    return format_results


def main(args):
    """Main evaluation function."""
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s', 
        datefmt='%H:%M:%S'
    )
    
    # Load configuration
    config_path = os.path.join("./exp", args.config + ".yaml")
    with open(config_path, "r") as f:
        exp_config = yaml.safe_load(f)
    
    # Override with evaluation settings
    exp_config['dataset']['dir'] = args.data_root
    exp_config['hardware']['gpu'] = args.gpu  # Set GPU from command line
    
    # Setup seed
    setup_seed(exp_config.get("seed", 42))
    
    # Setup device - must be set BEFORE any model loading
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    
    logging.info(f"Using device: {device}")
    
    # Setup dataset
    logging.info("Loading validation dataset...")
    val_dataset = Kitti(data_root=args.data_root, split='val')
    val_dataloader = get_dataloader(
        dataset=val_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        shuffle=False
    )
    # Keep original CLASSES format for evaluation
    CLASSES_FOR_EVAL = Kitti.CLASSES  # {'Pedestrian': 0, 'Cyclist': 1, 'Car': 2}
    LABEL2CLASSES = {v:k for k, v in CLASSES_FOR_EVAL.items()}  # {0: 'Pedestrian', 1: 'Cyclist', 2: 'Car'}
    
    # Setup models
    logging.info("Setting up models...")
    model_args = config_to_args(exp_config)
    val_loader_from_setup, headnet, tailnet, device_from_setup, _, _, pcd_limit_range = setup_model_and_data(
        model_args, mode='val'
    )
    
    # Use our device, not the one from setup
    logging.info(f"Moving models to {device}")
    
    # Create RQ bottleneck
    embedding_schedule = exp_config['progressive_learning']['embedding_schedule']
    n_codebooks = exp_config['model']['n_codebook']
    max_embed_size = max(embedding_schedule)
    
    rq_bottleneck = setup_progressive_stage(
        model_args, device, n_codebooks-1, max_embed_size, embedding_schedule, 
        ema=True, skip_stage_setup=True
    )
    
    headnet.to(device).eval()
    tailnet.to(device).eval()
    rq_bottleneck.to(device).eval()
    
    # Load weights
    weights_path = args.weights_path
    if not os.path.exists(weights_path):
        logging.error(f"Weights file not found: {weights_path}")
        return
    
    logging.info(f"Loading weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location='cpu')
    rq_bottleneck.load_state_dict(state_dict, strict=False)
    logging.info("Weights loaded successfully")
    
    # Prepare evaluation configurations
    if args.eval_stages:
        # Evaluate specific stages from checkpoint names
        eval_configs = []
        weights_dir = Path(args.weights_dir)
        
        # Parse all stage weights
        for weight_file in sorted(weights_dir.glob("weights_cb*_embed*.pth")):
            filename = weight_file.stem
            # Extract codebook number and embedding size
            # Format: weights_cb{num}_embed{size}
            parts = filename.split('_')
            cb_num = int(parts[1].replace('cb', ''))
            embed_size = int(parts[2].replace('embed', ''))
            eval_configs.append({
                'num_codebooks': cb_num,
                'embedding_size': embed_size,
                'weight_file': str(weight_file)
            })
        
        logging.info(f"Found {len(eval_configs)} stage weights to evaluate")
    else:
        # Evaluate all combinations
        eval_configs = []
        for num_cb in range(1, n_codebooks + 1):
            for embed_size in embedding_schedule:
                eval_configs.append({
                    'num_codebooks': num_cb,
                    'embedding_size': embed_size,
                    'weight_file': args.weights_path
                })
    
    # Create base results directory
    base_results_dir = args.saved_path
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Summary file
    summary_file = os.path.join(base_results_dir, 'evaluation_summary.txt')
    summary_f = open(summary_file, 'w')
    
    logging.info(f"\n{'='*80}")
    logging.info(f"Starting evaluation of {len(eval_configs)} configurations")
    logging.info(f"{'='*80}\n")
    
    # Evaluate each configuration
    for idx, config in enumerate(eval_configs):
        num_cb = config['num_codebooks']
        embed_size = config['embedding_size']
        
        logging.info(f"\n{'='*80}")
        logging.info(f"[{idx+1}/{len(eval_configs)}] Evaluating: {num_cb} codebook(s), embedding size {embed_size}")
        logging.info(f"{'='*80}\n")
        
        # Create stage-specific directory
        stage_name = f"cb{num_cb}_embed{embed_size}"
        stage_results_dir = os.path.join(base_results_dir, stage_name)
        os.makedirs(stage_results_dir, exist_ok=True)
        
        # If evaluating specific stage weights, load them
        if args.eval_stages and 'weight_file' in config:
            stage_weights = torch.load(config['weight_file'], map_location='cpu')
            rq_bottleneck.load_state_dict(stage_weights, strict=False)
        
        # Run evaluation
        format_results = evaluate_stage(
            headnet, tailnet, rq_bottleneck, val_dataloader,
            num_cb, embed_size, model_args, stage_results_dir, pcd_limit_range
        )
        
        # Perform KITTI evaluation
        logging.info("Running KITTI evaluation...")
        do_eval(format_results, val_dataset.data_infos, CLASSES_FOR_EVAL, stage_results_dir)
        
        # Copy results to summary
        eval_results_path = os.path.join(stage_results_dir, 'eval_results.txt')
        if os.path.exists(eval_results_path):
            with open(eval_results_path, 'r') as f:
                eval_content = f.read()
            
            summary_f.write(f"\n{'='*80}\n")
            summary_f.write(f"Stage: {num_cb} codebook(s), embedding size {embed_size}\n")
            summary_f.write(f"{'='*80}\n")
            summary_f.write(eval_content)
            summary_f.write("\n")
        
        logging.info(f"✓ Stage evaluation completed: {stage_name}")
    
    summary_f.close()
    
    logging.info(f"\n{'='*80}")
    logging.info(f"All evaluations completed!")
    logging.info(f"Results saved to: {base_results_dir}")
    logging.info(f"Summary saved to: {summary_file}")
    logging.info(f"{'='*80}\n")
    
    # Print summary
    logging.info("\nDisplaying summary:")
    with open(summary_file, 'r') as f:
        print(f.read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Progressive RQ Stages')
    parser.add_argument('--config', type=str, default='split1_ALL',
                        help='Config file name (without .yaml)')
    parser.add_argument('--data_root', type=str, default='/home/yenhsiu/datasets',
                        help='Path to KITTI dataset')
    parser.add_argument('--weights_dir', type=str, 
                        default='/home/yenhsiu/PointPillars_split_inference/results/weights_LOSS_20251027_202323',
                        help='Directory containing stage weights')
    parser.add_argument('--weights_path', type=str,
                        default='/home/yenhsiu/PointPillars_split_inference/results/weights_LOSS_20251027_202323/final_weights_all_codebooks.pth',
                        help='Path to final weights (used when eval_stages=False)')
    parser.add_argument('--saved_path', type=str, 
                        default='/home/yenhsiu/PointPillars_split_inference/results/eval_LOSS_20251027_202323',
                        help='Directory to save evaluation results')
    parser.add_argument('--eval_stages', action='store_true',
                        help='Evaluate individual stage weights (weights_cb*_embed*.pth)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    
    args = parser.parse_args()
    main(args)
