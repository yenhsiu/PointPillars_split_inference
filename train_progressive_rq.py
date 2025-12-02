"""
Progressive RQ Training Script for PointPillars
Self-contained version with all utility functions integrated
"""

import os
import time
import datetime
import random
import warnings
import yaml
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from pointpillars.loss import Loss
from pointpillars.utils import keep_bbox_from_image_range, keep_bbox_from_lidar_range, write_label, setup_seed
from pointpillars.dataset import Kitti, get_dataloader
from pointpillars.model import PointPillars
from pointpillars.model.split_nets import split_pointpillars
from pointpillars.model.quantizations import RQBottleneck
from evaluate import do_eval

warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")

class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after a given patience.
    Now based on mAP (higher is better) instead of loss.
    """
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print, mode='max'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_metric = -np.Inf if mode == 'max' else np.Inf
        self.delta = delta
        self.trace_func = trace_func
        self.mode = mode  # 'max' for mAP (higher is better), 'min' for loss

    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_metric = -np.Inf if self.mode == 'max' else np.Inf

    def __call__(self, val_metric, model):
        # For mAP: higher is better, so score = val_metric
        # For loss: lower is better, so score = -val_metric
        score = val_metric if self.mode == 'max' else -val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                metric_name = 'mAP' if self.mode == 'max' else 'loss'
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} ({metric_name} did not improve)')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        """Saves model when validation metric improves."""
        if self.verbose:
            if self.mode == 'max':
                self.trace_func(f'mAP improved ({self.best_metric:.4f}% --> {val_metric:.4f}%). Saving model ...')
            else:
                self.trace_func(f'Validation loss decreased ({self.best_metric:.6f} --> {val_metric:.6f}). Saving model ...')
        self.best_metric = val_metric


# ============================================================================
# >> Model Setup Functions (from model_utils.py)
# ============================================================================

def config_to_args(config):
    """Convert YAML config to argparse Namespace for compatibility"""
    args = argparse.Namespace()
    args.data_root = config['dataset']['dir']
    args.pretrained_ckpt = config['model']['pretrained_weight']
    args.batch_size = config['training']['batch_size']
    args.num_workers = config['training']['num_workers']
    args.nclasses = config['dataset']['num_classes']
    args.latent_shape = config['rq_model']['latent_shape']
    args.code_shape = config['rq_model']['code_shape']
    args.decay = config['rq_model']['decay']
    args.gpu = config['hardware']['gpu']
    
    # Training specific parameters
    if 'training' in config:
        args.init_lr = config['training']['init_lr']
        args.max_epoch = config['training']['max_epoch']
        args.ckpt_freq_epoch = config['training']['ckpt_freq_epoch']
    
    # Loss weights
    if 'loss_weights' in config:
        args.vq_weight = config['loss_weights']['vq_weight']
        args.codebook_weight = config['loss_weights']['codebook_weight']
        args.det_weight = config['loss_weights']['det_weight']
    
    # Point cloud limit range (optional)
    if 'pcd_limit_range' in config:
        args.pcd_limit_range = config['pcd_limit_range']
    
    return args


def setup_model_and_data(args, mode='train'):
    """Common setup for model and data loading"""
    setup_seed()
    
    # Load dataset
    if mode == 'train':
        train_dataset = Kitti(data_root=args.data_root, split='train')
        val_dataset = Kitti(data_root=args.data_root, split='val')
        
        train_dataloader = get_dataloader(dataset=train_dataset, 
                                          batch_size=args.batch_size, 
                                          num_workers=args.num_workers,
                                          shuffle=True)
        val_dataloader = get_dataloader(dataset=val_dataset, 
                                        batch_size=args.batch_size, 
                                        num_workers=args.num_workers,
                                        shuffle=False)
        dataloaders = (train_dataloader, val_dataloader)
    else:
        val_dataset = Kitti(data_root=args.data_root, split='val')
        val_dataloader = get_dataloader(dataset=val_dataset, 
                                        batch_size=args.batch_size, 
                                        num_workers=args.num_workers,
                                        shuffle=False)
        dataloaders = val_dataloader

    # Setup device and load pretrained model
    device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    full_model = PointPillars(nclasses=args.nclasses).to(device)
    
    # Load pretrained weights
    state = torch.load(args.pretrained_ckpt, map_location=device)
    full_model.load_state_dict(state, strict=False)
    print(f"Loaded pretrained weights from {args.pretrained_ckpt}")
    
    # Split the model into head and tail
    headnet, tailnet = split_pointpillars(full_model)
    
    # Freeze headnet and tailnet parameters for RQ training
    for param in headnet.parameters():
        param.requires_grad = False
    for param in tailnet.parameters():
        param.requires_grad = False
    print("Frozen headnet and tailnet parameters")
    
    # Define evaluation constants
    CLASSES = Kitti.CLASSES  # {'Pedestrian': 0, 'Cyclist': 1, 'Car': 2}
    LABEL2CLASSES = {v: k for k, v in CLASSES.items()}  # {0: 'Pedestrian', 1: 'Cyclist', 2: 'Car'}
    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 1], dtype=np.float32)
    
    return dataloaders, headnet, tailnet, device, CLASSES, LABEL2CLASSES, pcd_limit_range


def create_rq_bottleneck(args, device, ema=True):
    """Create RQ bottleneck with given configuration"""
    rq_bottleneck = RQBottleneck(
        latent_shape=args.latent_shape,
        code_shape=args.code_shape,
        n_embed=args.codebook_size,
        decay=args.decay,
        ema=ema,
        shared_codebook=False,
        restart_unused_codes=True,
        commitment_loss='cumsum'
    )
    rq_bottleneck = rq_bottleneck.to(device)
    
    # Initialize codebooks with proper values to avoid CUDA errors
    with torch.no_grad():
        for i, codebook in enumerate(rq_bottleneck.codebooks):
            # Initialize with small random values using normal distribution
            torch.nn.init.normal_(codebook.weight, mean=0.0, std=0.02)
            
            if hasattr(codebook, 'embed_ema') and codebook.embed_ema is not None:
                # Initialize EMA embeddings with the same values
                codebook.embed_ema.copy_(codebook.weight[:-1, :])
                
            if hasattr(codebook, 'cluster_size_ema') and codebook.cluster_size_ema is not None:
                # Initialize cluster sizes to small positive values
                codebook.cluster_size_ema.fill_(0.1)
                
            print(f"Initialized codebook {i}: {codebook.n_embed} codes, {codebook.embedding_dim}D")
    
    print(f"Created RQ bottleneck with {sum(p.numel() for p in rq_bottleneck.parameters())} parameters")
    return rq_bottleneck


def setup_progressive_stage(args, device, codebook_idx, embed_size, 
                           embedding_schedule, ema=True, skip_stage_setup: bool = False):
    """Setup RQ bottleneck for specific progressive training stage (tail-freeze only).
    If skip_stage_setup is True, do not call set_training_stage or print training range.
    """
    stage_args = argparse.Namespace(**vars(args))
    stage_args.codebook_size = embed_size
    stage_args.code_shape = args.code_shape.copy()
    stage_args.code_shape[-1] = codebook_idx + 1  # Use codebook_idx + 1 codebooks
    
    rq_bottleneck = create_rq_bottleneck(stage_args, device, ema=ema)
    
    if not skip_stage_setup:
        # Tail-freeze semantics via RQBottleneck.set_training_stage
        frozen_embed_size = embed_size  # freeze [K:max) will be enforced via codebook logic during training
        full_embed_size = embedding_schedule[-1]
        
        if hasattr(rq_bottleneck, 'set_training_stage'):
            rq_bottleneck.set_training_stage(codebook_idx, embed_size, 
                                            full_embed_size, 0)
        
        print(f"Progressive stage setup (tail-freeze):")
        print(f"  Codebook {codebook_idx}: active_n_embed={embed_size}, frozen_tail=[{embed_size}:{full_embed_size})")
    else:
        # Caller will configure and log the freeze/active ranges
        print("Progressive stage setup skipped (will be configured by caller)")
    
    return rq_bottleneck


# ============================================================================
# >> Validation Function (使用官方 do_eval)
# ============================================================================

def run_validation(rq_bottleneck, headnet, tailnet, val_dataloader, val_dataset, 
                   args, device, CLASSES, pcd_limit_range, eval_save_path=None):
    """
    Run validation using official KITTI do_eval
    
    Args:
        rq_bottleneck: RQ bottleneck module
        headnet: Head network 
        tailnet: Tail network
        val_dataloader: Validation dataloader
        val_dataset: Validation dataset (for data_infos)
        args: Training arguments
        device: Device for computation
        CLASSES: Class information
        pcd_limit_range: Point cloud range limit
        eval_save_path: File path to save evaluation results (e.g., 'result_evaluate_cb1_em128.txt')
        
    Returns:
        dict: Evaluation results from do_eval (contains bbox_3d AP metrics)
    """
    print(f"\n{'='*80}")
    print(f"Running KITTI Evaluation")
    print(f"{'='*80}")
    
    # Set models to eval mode
    rq_bottleneck.eval()
    headnet.eval() 
    tailnet.eval()
    
    # Get LABEL2CLASSES mapping
    if hasattr(val_dataset, 'CLASSES'):
        CLASSES_DICT = val_dataset.CLASSES
        if isinstance(CLASSES_DICT, dict):
            first_key = next(iter(CLASSES_DICT.keys()))
            if isinstance(first_key, int):
                LABEL2CLASSES = CLASSES_DICT
            else:
                LABEL2CLASSES = {v: k for k, v in CLASSES_DICT.items()}
        else:
            LABEL2CLASSES = {i: name for i, name in enumerate(CLASSES_DICT)}
    else:
        LABEL2CLASSES = {0: 'Pedestrian', 1: 'Cyclist', 2: 'Car'}
    
    format_results = {}
    
    # Generate predictions
    with torch.no_grad():
        for batch_idx, data_dict in enumerate(tqdm(val_dataloader, desc="Generating predictions")):
            try:
                # Move data to device
                if torch.cuda.is_available():
                    for key in data_dict:
                        for j, item in enumerate(data_dict[key]):
                            if torch.is_tensor(item):
                                data_dict[key][j] = data_dict[key][j].to(device)
                
                batched_pts = data_dict['batched_pts']
                batched_gt_bboxes = data_dict['batched_gt_bboxes']
                batched_labels = data_dict['batched_labels']
                
                # Forward pass through split model
                features = headnet(batched_pts)
                features_hwc = features.permute(0, 2, 3, 1)
                quantized_features, _, _, _ = rq_bottleneck(features_hwc)
                quantized_features = quantized_features.permute(0, 3, 1, 2)
                
                # Generate predictions
                bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors = tailnet(
                    quantized_features,
                    mode='val',
                    batched_gt_bboxes=batched_gt_bboxes,
                    batched_gt_labels=batched_labels
                )
                
                # Convert to final results
                batch_results = tailnet.get_predicted_bboxes(
                    bbox_cls_pred=bbox_cls_pred,
                    bbox_pred=bbox_pred,
                    bbox_dir_cls_pred=bbox_dir_cls_pred,
                    batched_anchors=batched_anchors
                )
                
                # Process results for evaluation
                batched_calib_info = data_dict.get('batched_calib_info', [])
                batched_img_info = data_dict.get('batched_img_info', [])
                
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
                    
                    calib_info = batched_calib_info[j]
                    tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
                    r0_rect = calib_info['R0_rect'].astype(np.float32)
                    P2 = calib_info['P2'].astype(np.float32)
                    image_shape = batched_img_info[j]['image_shape']
                    idx = batched_img_info[j]['image_idx']
                    
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
                    
                    format_results[idx] = {k: np.array(v) for k, v in format_result.items()}
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
    
    print(f"Collected {len(format_results)} results")
    
    # Prepare save directory for evaluation results
    if eval_save_path is None:
        save_dir = ""
    else:
        save_dir = os.path.dirname(eval_save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = "."
    
    # Call official do_eval
    eval_results = do_eval(format_results, val_dataset.data_infos, CLASSES, save_dir)
    
    # Rename eval_results.txt to the specified filename if provided
    if eval_save_path:
        default_result_file = os.path.join(save_dir, 'eval_results.txt')
        if os.path.exists(default_result_file) and default_result_file != eval_save_path:
            os.rename(default_result_file, eval_save_path)
            print(f"✅ Evaluation results saved to: {eval_save_path}")
    
    print(f"{'='*80}\n")
    return eval_results


def validate_epoch(rq_bottleneck, headnet, tailnet, val_dataloader, args, device, CLASSES, 
                  epoch, warmup_epochs, config=None, stage_info=None, eval_save_dir=None):
    """
    Run full KITTI evaluation using official do_eval.
    """
    print(f"\n{'='*80}")
    print(f"Running Full KITTI Evaluation - Epoch {epoch+1}")
    if stage_info:
        print(f"Codebook {stage_info.get('codebook_idx', '?')+1}, Embedding {stage_info.get('embed_size', '?')}")
    print(f"{'='*80}\n")
    if stage_info and 'embed_size' in stage_info:
        embed_size = stage_info['embed_size']
        num_codebooks = stage_info.get('codebook_idx', 0) + 1
        
        print(f"⚙️  Configuring evaluation: {num_codebooks} codebook(s), {embed_size} embeddings each")
        
        for cb_i, cb in enumerate(rq_bottleneck.codebooks[:num_codebooks]):
            cb.set_active_n_embed(embed_size)
        
        for cb_i in range(num_codebooks, len(rq_bottleneck.codebooks)):
            rq_bottleneck.codebooks[cb_i].set_active_n_embed(0)
    
    # Get validation dataset and pcd_limit_range
    val_dataset = val_dataloader.dataset
    
    if hasattr(args, 'pcd_limit_range'):
        pcd_limit_range = args.pcd_limit_range
    elif config and 'pcd_limit_range' in config:
        pcd_limit_range = config['pcd_limit_range']
    else:
        pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)
    
    # Run full KITTI evaluation using official do_eval
    eval_results = run_validation(
        rq_bottleneck, headnet, tailnet, val_dataloader, val_dataset,
        args, device, CLASSES, pcd_limit_range, eval_save_dir
    )
    
    # Print evaluation results to console (read from saved file if available)
    if eval_save_dir and os.path.exists(eval_save_dir):
        print(f"\n{'='*80}")
        print(f"Evaluation Results:")
        print(f"{'='*80}")
        with open(eval_save_dir, 'r') as f:
            print(f.read())
        print(f"{'='*80}\n")
    
    # Extract mAP from evaluation results
    bbox_3d_ap = None
    
    if eval_results and isinstance(eval_results, dict):
        if 'bbox_3d' in eval_results:
            bbox_3d_data = eval_results['bbox_3d']
            if isinstance(bbox_3d_data, (list, tuple, np.ndarray)):
                bbox_3d_ap = bbox_3d_data

    # Log validation metrics to WandB if available
    try:
        import wandb
        log_dict = {
            'val/epoch': int(epoch)
        }
        
        # Log individual difficulty levels if available
        if bbox_3d_ap is not None and len(bbox_3d_ap) >= 3:
            log_dict.update({
                'val/bbox_3d_easy': float(bbox_3d_ap[0]),
                'val/bbox_3d_moderate': float(bbox_3d_ap[1]),
                'val/bbox_3d_hard': float(bbox_3d_ap[2])
            })
        
        wandb.log(log_dict)
    except Exception:
        pass
    
    # Return metrics
    val_metrics = {
        'bbox_3d_ap': bbox_3d_ap,
        'eval_results': eval_results
    }
    
    return val_metrics


# ============================================================================
# >> Training Functions
# ============================================================================

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(
    train_loader, headnet, tailnet, rq_bottleneck, criterion, optimizer, epoch, args):
    """Execute one training epoch with adaptive loss weighting."""
    # Simple accumulators (AverageMeter removed)
    sum_loss = 0.0
    sum_det = 0.0
    sum_vq = 0.0
    sum_cb = 0.0
    total_samples = 0
    
    # Track loss scales for adaptive weighting
    det_loss_scale = []
    vq_loss_scale = []
    cb_loss_scale = []
    
    rq_bottleneck.train()
    headnet.eval()
    tailnet.eval()
    
    prefix = f"Epoch: [{epoch}]"
    end = time.time()
    for i, data_dict in enumerate(train_loader):
        # Validate batch data
        if not data_dict or 'batched_pts' not in data_dict:
            logging.error(f"Invalid batch data at iteration {i}")
            raise RuntimeError(f"Invalid batch data at iteration {i}: missing required keys")
        
        if torch.cuda.is_available():
            for key in data_dict:
                for j, item in enumerate(data_dict[key]):
                    if torch.is_tensor(item):
                        data_dict[key][j] = data_dict[key][j].cuda()

        batched_pts = data_dict['batched_pts']
        batched_gt_bboxes = data_dict['batched_gt_bboxes']
        batched_labels = data_dict['batched_labels']
        
        # Check for empty batch
        if len(batched_pts) == 0 or len(batched_gt_bboxes) == 0:
            logging.warning(f"Empty batch at iteration {i}, skipping...")
            continue

        # Forward pass
        try:
            with torch.no_grad():
                pillar_features = headnet(batched_pts)
            
            if pillar_features is None or pillar_features.numel() == 0:
                logging.error(f"Headnet produced invalid features at iteration {i}")
                raise RuntimeError(f"Headnet produced invalid output at iteration {i}")
            
            pillar_features_hwc = pillar_features.permute(0, 2, 3, 1)
            quantized_features, vq_loss, codebook_loss, codes = rq_bottleneck(pillar_features_hwc)
            
            if quantized_features is None or quantized_features.numel() == 0:
                logging.error(f"RQ bottleneck produced invalid features at iteration {i}")
                raise RuntimeError(f"RQ bottleneck produced invalid output at iteration {i}")
            
            quantized_features = quantized_features.permute(0, 3, 1, 2)
            
            # Calculate detection loss
            det_loss = calculate_detection_loss(
                tailnet, quantized_features, batched_gt_bboxes, 
                batched_labels, batched_pts, criterion, args
            )
            
            if torch.isnan(det_loss) or torch.isinf(det_loss):
                logging.error(f"Invalid detection loss at iteration {i}: {det_loss.item()}")
                raise RuntimeError(f"Invalid detection loss at iteration {i}")
            
            # Normalize detection loss to similar scale as VQ loss
            # Typical ranges: det_loss~100-1000, vq_loss~0.1-10
            det_loss_normalized = det_loss / 100.0
            
            # Combined loss: Detection Loss + VQ Loss
            # Note: Codebook loss is kept for monitoring but has no gradient effect
            loss = (args.det_weight * det_loss_normalized + 
                    args.vq_weight * vq_loss)
            
            # Track individual loss scales for monitoring
            det_loss_scale.append(det_loss.item())
            vq_loss_scale.append(vq_loss.item())
            cb_loss_scale.append(codebook_loss.item())
            
            if torch.isnan(loss) or torch.isinf(loss):
                logging.error(f"Invalid total loss at iteration {i}: {loss.item()}")
                raise RuntimeError(f"Invalid total loss at iteration {i}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rq_bottleneck.parameters(), max_norm=1.0)
            optimizer.step()
            
            # ✅ Restore frozen embeddings (protection against weight decay)
            for cb in rq_bottleneck.codebooks:
                if cb.frozen_n_embed > 0:
                    cb.restore_frozen_weights()

            n = len(batched_pts)
            sum_loss += loss.item() * n
            sum_det += det_loss.item() * n
            sum_vq += vq_loss.item() * n
            sum_cb += codebook_loss.item() * n
            total_samples += n
            end = time.time()
            
        except Exception as e:
            logging.error(f"Error in training iteration {i}: {str(e)}")
            raise RuntimeError(f"Training failed at iteration {i}: {str(e)}")
    # Compute averages
    if total_samples > 0:
        avg_loss = sum_loss / total_samples
        avg_det = sum_det / total_samples
        avg_vq = sum_vq / total_samples
        avg_cb = sum_cb / total_samples
    else:
        avg_loss = avg_det = avg_vq = avg_cb = 0.0

    
    # Log loss scales for monitoring
    if det_loss_scale:
        avg_det_scale = np.mean(det_loss_scale)
        avg_vq_scale = np.mean(vq_loss_scale)
        avg_cb_scale = np.mean(cb_loss_scale)
        
        # Calculate effective contribution percentages
        det_weighted = avg_det_scale / 100 * args.det_weight
        vq_weighted = avg_vq_scale * args.vq_weight
        total_weighted = det_weighted + vq_weighted
        
        if total_weighted > 0:
            det_pct = (det_weighted / total_weighted) * 100
            vq_pct = (vq_weighted / total_weighted) * 100
            logging.info(f"Loss Scales - Det: {avg_det_scale:.2f}, VQ: {avg_vq_scale:.4f}, CB: {avg_cb_scale:.4f} (monitor)")
            logging.info(f"Training Ratio - Det: {det_pct:.1f}%, VQ: {vq_pct:.1f}%")
        else:
            det_pct = 0
            vq_pct = 0
    else:
        det_pct = 0
        vq_pct = 0
    
    # Log to WandB if available
    try:
        import wandb
        log_dict = {
            'train/loss': float(avg_loss),
            'train/det_loss': float(avg_det),
            'train/vq_loss': float(avg_vq),
            'train/codebook_loss_monitor': float(avg_cb),
            'train/samples': int(total_samples),
            'train/epoch': int(epoch),
            'train/det_loss_scale': float(np.mean(det_loss_scale)) if det_loss_scale else 0.0,
            'train/vq_loss_scale': float(np.mean(vq_loss_scale)) if det_loss_scale else 0.0,
            'train/cb_loss_scale': float(np.mean(cb_loss_scale)) if det_loss_scale else 0.0,
            'train/det_contribution_pct': float(det_pct),
            'train/vq_contribution_pct': float(vq_pct)
        }
        wandb.log(log_dict)
    except Exception:
        pass

    return avg_loss, avg_det, avg_vq, avg_cb

def full_kitti_evaluation(val_dataset, val_loader, headnet, tailnet, rq_bottleneck, 
                          args, device, pcd_limit_range, stage_name, eval_dir):
    """
    Execute full KITTI evaluation and save results to specific directory.
    Similar to evaluate.py but with split model and saves to stage-specific directory.
    
    Args:
        val_dataset: Validation dataset
        val_loader: Validation dataloader
        headnet: Head network
        tailnet: Tail network
        rq_bottleneck: RQ bottleneck layer
        args: Arguments
        device: Device
        pcd_limit_range: Point cloud range limit
        stage_name: Stage name like "cb2_embed64"
        eval_dir: Evaluation results directory
    
    Returns:
        format_results: Formatted evaluation results
    """
    from pointpillars.dataset import Kitti
    
    CLASSES = Kitti.CLASSES
    LABEL2CLASSES = {v:k for k, v in CLASSES.items()}
    
    rq_bottleneck.eval()
    headnet.eval()
    tailnet.eval()
    
    # Create stage-specific directory
    stage_dir = os.path.join(eval_dir, stage_name)
    os.makedirs(stage_dir, exist_ok=True)
    submit_dir = os.path.join(stage_dir, 'submit')
    os.makedirs(submit_dir, exist_ok=True)
    
    format_results = {}
    
    logging.info(f"Running full KITTI evaluation for {stage_name}...")
    
    with torch.no_grad():
        for i, data_dict in enumerate(tqdm(val_loader, desc=f'Evaluating {stage_name}')):
            try:
                # Validate batch data
                if not data_dict or 'batched_pts' not in data_dict:
                    logging.warning(f"Invalid batch data at iteration {i}, skipping...")
                    continue
                
                if torch.cuda.is_available():
                    for key in data_dict:
                        for j, item in enumerate(data_dict[key]):
                            if torch.is_tensor(item):
                                data_dict[key][j] = data_dict[key][j].cuda()
                
                batched_pts = data_dict['batched_pts']
                batched_gt_bboxes = data_dict['batched_gt_bboxes']
                batched_labels = data_dict['batched_labels']
                
                # Check for empty batch
                if len(batched_pts) == 0:
                    logging.warning(f"Empty batch at iteration {i}, skipping...")
                    continue
                
                # Forward pass through split model
                pillar_features = headnet(batched_pts)
                
                if pillar_features is None or pillar_features.numel() == 0:
                    logging.warning(f"Invalid pillar features at iteration {i}, skipping...")
                    continue
                
                pillar_features_hwc = pillar_features.permute(0, 2, 3, 1)
                quantized_features, _, _, _ = rq_bottleneck(pillar_features_hwc)
                
                if quantized_features is None or quantized_features.numel() == 0:
                    logging.warning(f"Invalid quantized features at iteration {i}, skipping...")
                    continue
                
                quantized_features = quantized_features.permute(0, 3, 1, 2)
                
                # Get predictions (match eval_utils.py - tailnet returns raw predictions)
                bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors = tailnet(
                    quantized_features,
                    mode='val',
                    batched_gt_bboxes=batched_gt_bboxes,
                    batched_gt_labels=batched_labels
                )
                
                # Convert to final results (match eval_utils.py)
                batch_results = tailnet.get_predicted_bboxes(
                    bbox_cls_pred=bbox_cls_pred,
                    bbox_pred=bbox_pred,
                    bbox_dir_cls_pred=bbox_dir_cls_pred,
                    batched_anchors=batched_anchors
                )
            except Exception as e:
                logging.error(f"Error in evaluation iteration {i}: {str(e)}")
                continue
            
            # Format results (match evaluate.py)
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
                write_label(format_result, os.path.join(submit_dir, f'{idx:06d}.txt'))
                
                format_results[idx] = {k:np.array(v) for k, v in format_result.items()}
            
    
    # Run KITTI evaluation (match evaluate.py)
    logging.info(f"Computing KITTI metrics for {stage_name}...")
    do_eval(format_results, val_dataset.data_infos, CLASSES, stage_dir)
    
    # Read and return the evaluation results
    eval_results_path = os.path.join(stage_dir, 'eval_results.txt')
    if os.path.exists(eval_results_path):
        with open(eval_results_path, 'r') as f:
            eval_text = f.read()
        logging.info(f"\n{eval_text}")
    
    return format_results

def run_ema_warmup(loader, headnet, rq_bottleneck, epoch, ema_limit=None):
    """Execute one epoch of EMA warmup with forward passes only.
    
    Args:
        ema_limit: Limit from config (pretrain_ema_limit)
    """
    rq_bottleneck.train()
    headnet.eval()
    
    # Convert ema_limit to int if it's a string
    if ema_limit is not None:
        try:
            ema_limit = int(ema_limit)
        except (ValueError, TypeError):
            logging.warning(f"Invalid ema_limit value: {ema_limit}, ignoring it")
            ema_limit = None
    
    effective_limit = ema_limit
    
    batch_count = 0
    with torch.no_grad():
        for i, data_dict in enumerate(tqdm(loader, desc=f"EMA Warmup Epoch {epoch+1}")):
            if torch.cuda.is_available():
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()
            
            batched_pts = data_dict['batched_pts']
            pillar_features = headnet(batched_pts)
            pillar_features_hwc = pillar_features.permute(0, 2, 3, 1)
            rq_bottleneck(pillar_features_hwc)
            
            batch_count += 1
            if effective_limit is not None and batch_count >= effective_limit:
                logging.info(f"EMA warmup early stop at batch {batch_count}")
                break
    
    logging.info(f"EMA Warmup Epoch {epoch+1} completed ({batch_count} batches)")

def calculate_detection_loss(tailnet, quantized_features, batched_gt_bboxes, 
                           batched_labels, batched_pts, loss_func, args):
    """Calculate detection loss following original PointPillars logic."""
    bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = tailnet(
        quantized_features,
        mode='train',
        batched_gt_bboxes=batched_gt_bboxes,
        batched_gt_labels=batched_labels,
        batch_size=len(batched_pts)
    )
    
    # Prepare targets
    batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
    batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
    batched_bbox_target = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
    batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)

    bbox_cls_pred_flat = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
    bbox_pred_flat = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
    bbox_dir_cls_pred_flat = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

    pos_mask = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)
    bbox_pred_pos = bbox_pred_flat[pos_mask]
    batched_bbox_reg_pos = batched_bbox_target[pos_mask]

    if bbox_pred_pos.numel() > 0:
        heading_pred = bbox_pred_pos[:, -1].clone()
        heading_target = batched_bbox_reg_pos[:, -1].clone()
        bbox_pred_pos[:, -1] = torch.sin(heading_pred) * torch.cos(heading_target)
        batched_bbox_reg_pos[:, -1] = torch.cos(heading_pred) * torch.sin(heading_target)

    bbox_dir_cls_pred_pos = bbox_dir_cls_pred_flat[pos_mask]
    batched_dir_labels_pos = batched_dir_labels[pos_mask]

    num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
    bbox_cls_pred_valid = bbox_cls_pred_flat[batched_label_weights > 0]
    cls_labels = batched_bbox_labels.clone()
    cls_labels[cls_labels < 0] = args.nclasses
    cls_labels = cls_labels[batched_label_weights > 0]

    det_loss_dict = loss_func(
        bbox_cls_pred=bbox_cls_pred_valid,
        bbox_pred=bbox_pred_pos,
        bbox_dir_cls_pred=bbox_dir_cls_pred_pos,
        batched_labels=cls_labels,
        num_cls_pos=num_cls_pos,
        batched_bbox_reg=batched_bbox_reg_pos,
        batched_dir_labels=batched_dir_labels_pos,
    )

    return det_loss_dict['total_loss']

def setup_environment(exp_config, data_time_str):
    """Setup experiment environment including paths and random seed."""
    logging.info("Setting up environment")
    set_seed(exp_config.get("seed", 42))
    
    training_mode = exp_config.get("training", {}).get("mode", "train")
    codebook_update_mode = exp_config.get("rq_model", {}).get("codebook_update_mode", "LOSS")

    weights_dir = os.path.join("results", f"weights_{codebook_update_mode}_{data_time_str}")
    os.makedirs(weights_dir, exist_ok=True)
    
    logging.info(f"Weights directory: {weights_dir}")
    
    # Initialize WandB if requested in config
    use_wandb = exp_config.get('logging', {}).get('use_wandb', False)
    if use_wandb:
        try:
            import wandb
            wb_project = exp_config.get('logging', {}).get('wandb_project', 'pointpillars-rq-progressive')
            wb_name = exp_config.get('logging', {}).get('wandb_name', None)
            # Initialize or resume a run
            wandb.init(project=wb_project, name=wb_name, config=exp_config)
            logging.info(f"WandB initialized (project={wb_project}, name={wb_name})")
        except Exception as e:
            logging.warning(f"WandB requested but failed to initialize: {e}")

    return weights_dir

def setup_models_and_data(exp_config):
    """Setup models, datasets, and loss functions."""
    logging.info("Setting up models and data")
    
    # Convert config and setup model/data
    args = config_to_args(exp_config)
    
    (train_loader, val_loader), headnet, tailnet, device, CLASSES, LABEL2CLASSES, pcd_limit_range = setup_model_and_data(args, mode='train')
    
    # Also get validation dataset for full evaluation
    from pointpillars.dataset import Kitti
    val_dataset = Kitti(data_root=args.data_root, split='val')
    
    # Get progressive learning configuration
    embedding_schedule = exp_config['progressive_learning']['embedding_schedule']
    n_codebooks = exp_config['model']['n_codebook']
    max_embed_size = max(embedding_schedule)
    
    # Create RQ bottleneck with maximum size
    logging.info(f"Creating RQ bottleneck with maximum embedding size: {max_embed_size}")
    rq_bottleneck = setup_progressive_stage(
        args, device, n_codebooks-1, max_embed_size, embedding_schedule, 
        ema=True, skip_stage_setup=True
    )
    
    headnet.to(device).eval()
    tailnet.to(device).eval()
    rq_bottleneck.to(device)
    for param in headnet.parameters(): param.requires_grad = False
    for param in tailnet.parameters(): param.requires_grad = False
    logging.info("Models created - head/tail frozen")

    criterion = Loss().to(device)
    
    return headnet, tailnet, rq_bottleneck, train_loader, val_loader, val_dataset, criterion, args, device, CLASSES, pcd_limit_range

def run_evaluation_mode(exp_config, rq_bottleneck, headnet, tailnet, val_loader, val_dataset, 
                       args, device, CLASSES, pcd_limit_range):
    """Run evaluation mode to test trained models with full KITTI evaluation."""
    logging.info("Running Evaluation Mode")
    eval_cfg = exp_config.get('evaluation', {})
    weights_path = eval_cfg.get('rq_ckpt')
    
    if not weights_path or not os.path.exists(weights_path):
        logging.error(f"Weights file not found: {weights_path}")
        return

    logging.info(f"Loading weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location='cpu')
    
    # Log checkpoint info for debugging
    logging.info(f"Checkpoint contains {len(state_dict)} keys")
    logging.info(f"Checkpoint keys (first 5): {list(state_dict.keys())[:5]}")
    
    # Load and verify
    rq_bottleneck.load_state_dict(state_dict, strict=False)
    logging.info("Weights loaded successfully")
    
    # Verify some weights were actually loaded
    loaded_weight = rq_bottleneck.codebooks[0].weight.data.clone()
    logging.info(f"Codebook[0] weight sample: min={loaded_weight.min().item():.6f}, max={loaded_weight.max().item():.6f}, mean={loaded_weight.mean().item():.6f}")
    
    # Create evaluation results directory
    data_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    codebook_mode = exp_config.get("rq_model", {}).get("codebook_update_mode", "LOSS")
    eval_results_dir = os.path.join("results", f"eval_{codebook_mode}_{data_time_str}")
    os.makedirs(eval_results_dir, exist_ok=True)
    logging.info(f"Evaluation results directory: {eval_results_dir}")
    
    # Get evaluation parameters
    use_full_kitti_eval = eval_cfg.get('use_full_kitti_eval', True)  # 默认使用完整评估
    
    eval_combinations = eval_cfg.get('test_combinations', [])
    if not eval_combinations:
        logging.warning("No 'test_combinations' found. Auto-generating combinations.")
        embedding_schedule = exp_config.get("progressive_learning", {}).get("embedding_schedule", [256])
        eval_combinations = [
            {"num_codebooks": num_cb, "embedding_size": embed_size}
            for num_cb in range(1, len(rq_bottleneck.codebooks) + 1)
            for embed_size in embedding_schedule
        ]
        
    eval_results = []
    for combo in eval_combinations:
        num_cb, embed_size = combo['num_codebooks'], combo['embedding_size']
        if num_cb > len(rq_bottleneck.codebooks):
            logging.warning(f"Skipping combo with {num_cb} codebooks (model only has {len(rq_bottleneck.codebooks)}).")
            continue

        logging.info(f"\n{'='*80}")
        logging.info(f"Testing: {num_cb} codebook(s), embedding size {embed_size}")
        logging.info(f"{'='*80}")
        
        # Set evaluation stage
        rq_bottleneck.set_evaluation_stage(num_cb, embed_size)
        
        # Decide evaluation method
        if use_full_kitti_eval:
            # Full KITTI evaluation (same as training mode)
            stage_name = f"result_evaluate_cb{num_cb}_em{embed_size}.txt"
            stage_eval_path = os.path.join(eval_results_dir, stage_name)
            logging.info(f"Running full KITTI evaluation for {stage_name}...")
            
            try:
                # Execute full evaluation and save to stage-specific file
                val_metrics = validate_epoch(
                    rq_bottleneck, headnet, tailnet, val_loader, args, device, CLASSES,
                    epoch=0, warmup_epochs=0, config=exp_config, stage_info={},
                    eval_save_dir=stage_eval_path  # Save to result_evaluate_cb{X}_em{Y}.txt
                )
                overall_mAP = calculate_overall_map(val_metrics)
                bbox_3d_ap = val_metrics.get('bbox_3d_ap')
                logging.info(f"✅ Evaluation completed. mAP = {overall_mAP:.4f}%")
                logging.info(f"✅ Results saved to: {stage_eval_path}")
            except Exception as e:
                logging.error(f"Full KITTI evaluation failed for {stage_name}: {str(e)}")
                raise RuntimeError(f"Evaluation failed for {stage_name}") from e
            
            eval_results.append({
                "num_codebooks": num_cb, 
                "embedding_size": embed_size,
                "mAP": overall_mAP,
                "bbox_3d_ap": bbox_3d_ap,
                "eval_results_path": stage_eval_path
            })
        else:
            # Quick validation (original method)
            val_metrics = validate_epoch(
                rq_bottleneck, headnet, tailnet, val_loader, args, device, CLASSES,
                epoch=0, warmup_epochs=0, config=exp_config, stage_info={}
            )
            overall_mAP = calculate_overall_map(val_metrics)
            bbox_3d_ap = val_metrics.get('bbox_3d_ap')
            logging.info(f"Result: mAP: {overall_mAP:.2f}%")
            
            eval_results.append({
                "num_codebooks": num_cb, 
                "embedding_size": embed_size,
                "mAP": overall_mAP,
                "bbox_3d_ap": bbox_3d_ap,
                "det_loss": val_metrics.get('det_loss', 0.0),
                "vq_loss": val_metrics.get('vq_loss', 0.0),
                "cb_loss": val_metrics.get('codebook_loss', 0.0)
            })
    
    # Print summary table
    df = pd.DataFrame(eval_results)
    logging.info("\n" + "="*80)
    logging.info("Evaluation Results Summary")
    logging.info("="*80)
    logging.info("\n" + df.to_string(index=False))
    logging.info("")
    
    if use_full_kitti_eval:
        logging.info(f"\nFull KITTI evaluation results saved to: {eval_results_dir}")
        logging.info("Each configuration has its own subdirectory with:")
        logging.info("  - eval_results.txt (full KITTI metrics)")
        logging.info("  - submit/*.txt (prediction files)")
        logging.info("  - plot/*.png (visualization plots)")

def calculate_overall_map(val_metrics):
    """
    Calculate overall mAP from bbox_3d metrics.
    
    Args:
        val_metrics: Dictionary containing validation metrics
        
    Returns:
        float: Overall mAP value (average of easy, moderate, hard)
        
    Raises:
        ValueError: If no valid mAP metrics are found
    """
    # Priority 1: Check for overall_mAP directly
    if 'overall_mAP' in val_metrics and val_metrics['overall_mAP'] is not None:
        map_value = float(val_metrics['overall_mAP'])
        logging.debug(f"Using overall_mAP: {map_value:.4f}")
        return map_value
    
    # Priority 2: Check for bbox_3d_ap (numpy array or list of [easy, moderate, hard])
    if 'bbox_3d_ap' in val_metrics and val_metrics['bbox_3d_ap'] is not None:
        bbox_3d_ap = val_metrics['bbox_3d_ap']
        
        # Handle numpy array or list/tuple
        if isinstance(bbox_3d_ap, np.ndarray):
            if bbox_3d_ap.size >= 3:
                # Average of easy, moderate, hard
                map_value = float(np.mean(bbox_3d_ap[:3]))
                logging.info(f"Calculated mAP from bbox_3d_ap: {map_value:.4f} "
                            f"(Easy: {bbox_3d_ap[0]:.4f}, Moderate: {bbox_3d_ap[1]:.4f}, "
                            f"Hard: {bbox_3d_ap[2]:.4f})")
                return map_value
            elif bbox_3d_ap.size == 1:
                map_value = float(bbox_3d_ap.item())
                logging.debug(f"Using single bbox_3d_ap value: {map_value:.4f}")
                return map_value
        elif isinstance(bbox_3d_ap, (list, tuple)) and len(bbox_3d_ap) >= 3:
            # Average of easy, moderate, hard
            map_value = float(sum(bbox_3d_ap[:3]) / 3)
            logging.info(f"Calculated mAP from bbox_3d_ap: {map_value:.4f} "
                        f"(Easy: {bbox_3d_ap[0]:.4f}, Moderate: {bbox_3d_ap[1]:.4f}, "
                        f"Hard: {bbox_3d_ap[2]:.4f})")
            return map_value
        elif isinstance(bbox_3d_ap, (int, float)):
            map_value = float(bbox_3d_ap)
            logging.debug(f"Using single bbox_3d_ap value: {map_value:.4f}")
            return map_value
    
    # Priority 3: Raise error instead of returning 0.0
    error_msg = (f"No valid mAP metrics found in val_metrics. "
                f"Available keys: {list(val_metrics.keys())}. "
                f"Cannot continue training without valid evaluation metrics.")
    logging.error(error_msg)
    raise ValueError(error_msg)

def run_training_mode(exp_config, rq_bottleneck, headnet, tailnet, criterion, 
                     train_loader, val_loader, val_dataset, weights_dir, 
                     args, device, CLASSES, pcd_limit_range):
    """Run progressive training mode."""
    logging.info("Starting Training")
    progressive_cfg = exp_config["progressive_learning"]
    training_cfg = exp_config["training"]
    
    # Early stopping based on mAP (mode='max' means higher is better)
    earlystopping = EarlyStopping(
        patience=training_cfg.get("early_stopping_patience", 5), 
        verbose=True,
        mode='max'  # mAP: higher is better
    )
    
    n_codebook = exp_config['model']['n_codebook']
    embedding_stage_epochs = progressive_cfg.get("embedding_stage_epochs", 10)
    warmup_epochs = progressive_cfg.get("warmup_epochs", 2)
    pretrain_ema_limit = progressive_cfg.get("pretrain_ema_limit", None)
    lr = training_cfg.get("init_lr", 0.001)
    training_mode = exp_config.get("rq_model", {}).get("codebook_update_mode", "LOSS")
    save_stage_weights = exp_config.get("save_stage_weights", True)
    
    # 创建评估结果目录
    eval_results_dir = weights_dir.replace('weights_', 'eval_')
    os.makedirs(eval_results_dir, exist_ok=True)
    logging.info(f"Evaluation results directory: {eval_results_dir}")
    
    results = []
    
    def create_stage_optimizer(model, lr):
        """Create optimizer for current training stage."""
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        if not trainable_params:
            logging.warning("No trainable parameters found for optimizer")
            return None
        return torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    
    # Progressive training loop
    total_stages = n_codebook * len(progressive_cfg["embedding_schedule"])
    current_stage = 0
    
    for codebook_idx in range(n_codebook):
        logging.info(f"\n{'='*80}")
        logging.info(f"Training Codebook {codebook_idx + 1}/{n_codebook}")
        logging.info(f"{'='*80}")
        
        for embed_idx, embed_size in enumerate(progressive_cfg["embedding_schedule"]):
            current_stage += 1
            logging.info(f"\n{'*'*80}")
            logging.info(f"Stage {current_stage}/{total_stages}: Codebook {codebook_idx + 1}, Embedding size {embed_size}")
            logging.info(f"{'*'*80}")
            
            prev_embed_size = progressive_cfg["embedding_schedule"][embed_idx-1] if embed_idx > 0 else 0
            
            logging.info(f"Setting stage: Codebook {codebook_idx}, Embeddings [{prev_embed_size}:{embed_size}]")

            # Set training stage
            rq_bottleneck.set_training_stage(
                active_codebook_idx=codebook_idx, 
                active_embed_size=embed_size,
                full_embed_size=progressive_cfg["embedding_schedule"][-1],
                prev_embed_size=prev_embed_size
            )
            
            optimizer, scheduler = None, None

            # Phase 1: EMA Warmup
            if training_mode == 'LOSS' or training_mode == 'EMA':
                logging.info(f"Phase 1: EMA Warmup ({warmup_epochs} epochs)")
                rq_bottleneck.set_ema_mode(True)
                
                for cb_i, cb in enumerate(rq_bottleneck.codebooks):
                    if cb_i == codebook_idx:
                        # 正在訓練的 codebook：凍結前面訓練過的，只訓練新增的
                        cb.trainable = True
                        cb.set_frozen_n_embed(prev_embed_size)  # 凍結 [0:prev_embed_size]
                        cb.set_active_n_embed(embed_size)       # 使用 [0:embed_size]
                        logging.info(f"  Codebook {cb_i}: trainable=True, frozen=[0:{prev_embed_size}], active=[0:{embed_size}], training=[{prev_embed_size}:{embed_size}]")
                    elif cb_i < codebook_idx:
                        # 已訓練完成的 codebook：使用當前階段的 embedding 數量，全部凍結
                        cb.trainable = False
                        cb.set_frozen_n_embed(embed_size)  # ✅ 凍結當前使用的 embeddings
                        cb.set_active_n_embed(embed_size)  # ✅ 所有 codebook 使用相同的 embedding 數量
                        logging.info(f"  Codebook {cb_i}: trainable=False (fully frozen), frozen=[0:{embed_size}], active=[0:{embed_size}]")
                    else:
                        # 未開始訓練的 codebook：不使用
                        cb.trainable = False
                        cb.set_frozen_n_embed(0)
                        cb.set_active_n_embed(0)
                        logging.info(f"  Codebook {cb_i}: trainable=False (inactive)")
                
                for warmup_epoch in range(warmup_epochs):
                    run_ema_warmup(train_loader, headnet, rq_bottleneck, warmup_epoch, pretrain_ema_limit)
                
                try:
                    val_metrics = validate_epoch(
                        rq_bottleneck, headnet, tailnet, val_loader, args, device, CLASSES,
                        epoch=warmup_epochs-1, warmup_epochs=warmup_epochs, config=exp_config, 
                        stage_info={'codebook_idx': codebook_idx, 'embed_size': embed_size}
                    )
                except Exception as e:
                    logging.error(f"Warmup validation failed: {str(e)}")
                    raise RuntimeError(f"Warmup stopped due to validation failure") from e
                overall_mAP = calculate_overall_map(val_metrics)
                logging.info(f"Warmup completed - mAP: {overall_mAP:.4f}%")            # Phase 2: Gradient Refinement
            if training_mode == 'LOSS':
                logging.info("Phase 2: Gradient Refinement")
                
                rq_bottleneck.set_training_stage(
                    active_codebook_idx=codebook_idx,
                    active_embed_size=embed_size,
                    full_embed_size=progressive_cfg["embedding_schedule"][-1],
                    prev_embed_size=prev_embed_size
                )
                
                # 然後禁用 EMA 模式，使梯度可用
                rq_bottleneck.set_ema_mode(False)
                
                optimizer = create_stage_optimizer(rq_bottleneck, lr)
                if optimizer is None:
                    logging.error("Failed to create optimizer. Exiting.")
                    return
                
                # ✅ 清除 frozen embeddings 的 optimizer state (momentum)
                # Adam/AdamW optimizer 維護 exp_avg 和 exp_avg_sq，即使梯度為 0，
                # 這些 momentum 仍會導致權重更新。因此必須清除 frozen 部分的 state。
                logging.info("Clearing optimizer state for frozen embeddings...")
                for cb_i, cb in enumerate(rq_bottleneck.codebooks):
                    if cb.frozen_n_embed > 0 and cb.weight in optimizer.state:
                        state = optimizer.state[cb.weight]
                        if 'exp_avg' in state:
                            state['exp_avg'][:cb.frozen_n_embed].zero_()
                        if 'exp_avg_sq' in state:
                            state['exp_avg_sq'][:cb.frozen_n_embed].zero_()
                        logging.info(f"  Codebook {cb_i}: cleared optimizer state for frozen embeddings [0:{cb.frozen_n_embed}]")
                
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=embedding_stage_epochs
                )

            # Main training loop
            earlystopping.reset()
            final_val_mAP = 0.0
            final_val_metrics = None
            
            for epoch in range(embedding_stage_epochs):
                if training_mode == 'EMA':
                    run_ema_warmup(train_loader, headnet, rq_bottleneck, epoch, None)
                    train_loss, train_det, train_vq, train_cb = 0.0, 0.0, 0.0, 0.0
                elif training_mode == 'LOSS':
                    if optimizer:
                        train_loss, train_det, train_vq, train_cb = train(
                            train_loader, headnet, tailnet, rq_bottleneck, criterion, optimizer, epoch, args)
                    else:
                        logging.error("Optimizer is None in LOSS mode")
                        raise RuntimeError("Failed to create optimizer in LOSS mode")
                else:
                    train_loss, train_det, train_vq, train_cb = 0, 0, 0, 0
                
                # Validation with error handling
                try:
                    val_metrics = validate_epoch(
                        rq_bottleneck, headnet, tailnet, val_loader, args, device, CLASSES,
                        epoch=epoch, warmup_epochs=warmup_epochs, config=exp_config, 
                        stage_info={'codebook_idx': codebook_idx, 'embed_size': embed_size}
                    )
                    
                    # Use moderate difficulty mAP for early stopping (KITTI standard)
                    bbox_3d_ap = val_metrics.get('bbox_3d_ap')
                    if bbox_3d_ap is not None and len(bbox_3d_ap) >= 3:
                        moderate_mAP = float(bbox_3d_ap[1])  # Index 1 = moderate
                        overall_mAP = calculate_overall_map(val_metrics)  # For logging/summary
                    else:
                        logging.warning("No bbox_3d_ap found in validation metrics, using 0.0")
                        moderate_mAP = 0.0
                        overall_mAP = 0.0
                    
                    final_val_metrics = val_metrics
                    final_val_mAP = overall_mAP
                    
                except ValueError as e:
                    logging.error(f"Validation failed at epoch {epoch+1}: {str(e)}")
                    raise RuntimeError(f"Training stopped at epoch {epoch+1} due to validation failure") from e
                except Exception as e:
                    logging.error(f"Unexpected error during validation at epoch {epoch+1}: {str(e)}")
                    raise RuntimeError(f"Training stopped at epoch {epoch+1} due to unexpected error") from e
                
                logging.info(f"Epoch {epoch+1}/{embedding_stage_epochs}: Train Loss={train_loss:.4f}, mAP(overall)={overall_mAP:.4f}%, mAP(moderate)={moderate_mAP:.4f}%")
                
                # Log stage info to WandB
                try:
                    import wandb
                    log_dict = {
                        'stage/codebook_idx': codebook_idx,
                        'stage/embed_size': embed_size,
                        'stage/current_stage': current_stage,
                        'stage/total_stages': total_stages
                    }
                    if scheduler:
                        log_dict['learning_rate'] = scheduler.get_last_lr()[0]
                    wandb.log(log_dict)
                except Exception:
                    pass
                
                if scheduler: scheduler.step()
                
                # Early stopping based on moderate mAP (KITTI standard, higher is better)
                earlystopping(moderate_mAP, rq_bottleneck)
                if earlystopping.early_stop:
                    logging.info(f"🛑 Early stopping triggered at epoch {epoch+1} (moderate mAP: {moderate_mAP:.4f}%)")
                    break
            
            logging.info(f"\n{'='*80}")
            logging.info(f"✅ Stage {current_stage}/{total_stages} Training Completed")
            logging.info(f"Codebook {codebook_idx+1}, Embedding [0:{embed_size}]")
            logging.info(f"{'='*80}\n")
            
            if final_val_metrics is None:
                logging.warning("⚠️  No validation was performed during training!")
                final_val_mAP = 0.0
            else:
                logging.info(f"📊 Using last epoch's validation result: mAP = {final_val_mAP:.4f}%")
            
            stage_name = f"result_evaluate_cb{codebook_idx+1}_em{embed_size}.txt"
            stage_eval_path = os.path.join(eval_results_dir, stage_name)
            
            if final_val_metrics:
                try:
                    logging.info(f"💾 Saving detailed evaluation to: {stage_name}")
                    detailed_val_metrics = validate_epoch(
                        rq_bottleneck, headnet, tailnet, val_loader, args, device, CLASSES,
                        epoch=0, warmup_epochs=0, config=exp_config, stage_info={},
                        eval_save_dir=stage_eval_path
                    )
                    logging.info(f"✅ Detailed results saved")
                except Exception as e:
                    logging.warning(f"⚠️  Could not save detailed evaluation: {str(e)}")
            results.append({
                "codebook_num": codebook_idx + 1, 
                "embedding_dim": embed_size, 
                "final_val_mAP": float(final_val_mAP)
            })
            
            if save_stage_weights:
                weights_path = os.path.join(weights_dir, f'weights_cb{codebook_idx+1}_embed{embed_size}.pth')
                torch.save(rq_bottleneck.state_dict(), weights_path)
                logging.info(f"💾 Saved stage weights: weights_cb{codebook_idx+1}_embed{embed_size}.pth")
            
            logging.info(f"Stage {current_stage}/{total_stages} completed\n")
    
    logging.info("Training finished.")
    final_df = pd.DataFrame(results)
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    logging.info("\n" + "="*80)
    logging.info("Final Training Results Summary")
    logging.info("="*80)
    logging.info("\n" + final_df.to_string(index=False))
    logging.info("")
    
    logging.info("="*80)
    logging.info("Statistics:")
    logging.info(f"  Total stages trained: {len(final_df)}")
    logging.info(f"  Best mAP: {final_df['final_val_mAP'].max():.4f}% (Codebook {final_df.loc[final_df['final_val_mAP'].idxmax(), 'codebook_num']}, Embedding {final_df.loc[final_df['final_val_mAP'].idxmax(), 'embedding_dim']})")
    logging.info(f"  Average mAP: {final_df['final_val_mAP'].mean():.4f}%")
    logging.info("="*80)

    final_weights_path = os.path.join(weights_dir, f'final_weights_all_codebooks.pth')
    torch.save(rq_bottleneck.state_dict(), final_weights_path)
    logging.info(f"Final weights saved: {final_weights_path}")


def Exp(exp_config, evaluate, data_time_str):
    """Main experiment workflow coordinating training or evaluation."""
    results_tuple = setup_models_and_data(exp_config)
    headnet, tailnet, rq_bottleneck, train_loader, val_loader, val_dataset, criterion, args, device, CLASSES, pcd_limit_range = results_tuple

    if evaluate:
        set_seed(exp_config.get("seed", 42))
        run_evaluation_mode(exp_config, rq_bottleneck, headnet, tailnet, val_loader, val_dataset,
                          args, device, CLASSES, pcd_limit_range)
    else:
        weights_dir = setup_environment(exp_config, data_time_str)
        run_training_mode(exp_config, rq_bottleneck, headnet, tailnet, criterion,
                          train_loader, val_loader, val_dataset, weights_dir,
                          args, device, CLASSES, pcd_limit_range)# ==============================================================================
# >> Main Entry Point
# ==============================================================================

def main(config_path_or_name):
    """Main function to load config from a YAML file or an experiment name under ./exp.

    Args:
        config_path_or_name: Either a path to a .yaml file, or the name of a yaml under ./exp
                             (without the .yaml extension).
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

    # Determine config path: if the provided value is an existing file, use it directly.
    if os.path.exists(config_path_or_name) and config_path_or_name.endswith('.yaml'):
        config_path = config_path_or_name
    else:
        # Fallback to ./exp/<name>.yaml (maintains backward compatibility with previous usage)
        config_path = os.path.join("./exp", config_path_or_name + ".yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        exp_config = yaml.safe_load(f)

    evaluate = exp_config.get('evaluation', {}).get('enabled', False)
    data_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set CUDA device if provided in the config
    try:
        if 'hardware' in exp_config and 'gpu' in exp_config['hardware']:
            torch.cuda.set_device(exp_config['hardware']['gpu'])
    except Exception:
        # If device selection fails, continue and let PyTorch pick the device
        logging.warning("Could not set CUDA device from config; continuing with default device.")

    logging.info(f"Starting experiment using config: {config_path}")
    logging.info(f"Mode: {'Evaluation' if evaluate else 'Training'}")

    Exp(exp_config, evaluate, data_time_str)

    logging.info("Experiment completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Progressive Training with RQ-Codebooks for PointPillars')
    parser.add_argument('config', nargs='?', type=str, default="split1_ALL",
                        help='Path to the YAML config file, or the name of a config under ./exp (without .yaml). Defaults to "split1_ALL"')
    args = parser.parse_args()
    main(args.config)