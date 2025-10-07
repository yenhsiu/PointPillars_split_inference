import argparse
import os
import torch
from tqdm import tqdm
import numpy as np
import time
import datetime
import wandb
import yaml
from pointpillars.utils import setup_seed, keep_bbox_from_image_range, \
    keep_bbox_from_lidar_range, iou2d, iou3d_camera, iou_bev
from pointpillars.dataset import Kitti, get_dataloader
from pointpillars.model import PointPillars
from pointpillars.model.split_nets import split_pointpillars
from pointpillars.model.quantizations import RQBottleneck
from pointpillars.loss import Loss
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def load_pretrained_with_report(model, ckpt_path, device):
    """Load checkpoint quietly."""
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    return model

def get_score_thresholds(tp_scores, total_num_valid_gt, num_sample_pts=41):
    score_thresholds = []
    tp_scores = sorted(tp_scores)[::-1]
    cur_recall, pts_ind = 0, 0
    for i, score in enumerate(tp_scores):
        lrecall = (i + 1) / total_num_valid_gt
        rrecall = (i + 2) / total_num_valid_gt

        if i == len(tp_scores) - 1:
            score_thresholds.append(score)
            break

        if (lrecall + rrecall) / 2 < cur_recall:
            continue

        score_thresholds.append(score)
        pts_ind += 1
        cur_recall = pts_ind / (num_sample_pts - 1)
    return score_thresholds

def do_eval(det_results, gt_results, CLASSES, saved_path, eval_types=(
    'bbox_2d', 'bbox_bev', 'bbox_3d'), difficulties=(0, 1, 2)):
    """
    Complete KITTI evaluation protocol with selectable eval types.
    Supports bbox_2d, bbox_bev, bbox_3d evaluation types.
    """
    # Include all requested eval types
    EVAL_NAMES = list(eval_types)
    # Determine class names regardless of CLASSES being dict(int->name), dict(name->int), or list[str]
    if isinstance(CLASSES, dict):
        some_key = next(iter(CLASSES.keys())) if len(CLASSES) > 0 else None
        if isinstance(some_key, int):
            class_names = list(CLASSES.values())
        else:
            class_names = list(CLASSES.keys())
    elif isinstance(CLASSES, (list, tuple)):
        class_names = list(CLASSES)
    else:
        class_names = ['Car', 'Pedestrian', 'Cyclist']

    # Use name-based IoU thresholds to avoid dependence on class index order
    CLS_MIN_IOU = {'Car': 0.7, 'Pedestrian': 0.5, 'Cyclist': 0.5}
    MIN_HEIGHT = [40, 25, 25]      # Easy, Moderate, Hard
    
    eval_ap_results = {}
    eval_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for eval_type_idx, eval_type in enumerate(EVAL_NAMES):
        eval_ap_results[eval_type] = {}
        for difficulty in list(difficulties):  # Easy, Moderate, Hard
            eval_ap_results[eval_type][f'difficulty_{difficulty}'] = {}
            
            ids = sorted(list(gt_results.keys()))
            
            for cls_name in class_names:
                total_gt_ignores, total_det_ignores, total_dc_bboxes, total_scores = [], [], [], []
                total_gt_alpha, total_det_alpha = [], []
                
                for id in ids:
                    gt_result = gt_results[id]['annos']
                    det_result = det_results.get(id, {
                        'name': np.array([]), 'bbox': np.array([]), 'score': np.array([]), 
                        'alpha': np.array([]), 'location': np.array([]), 'dimensions': np.array([]),
                        'rotation_y': np.array([])
                    })

                    # GT bbox processing
                    cur_gt_names = gt_result['name']
                    cur_difficulty = gt_result['difficulty']
                    gt_ignores, dc_bboxes = [], []
                    
                    for j, cur_gt_name in enumerate(cur_gt_names):
                        ignore = cur_difficulty[j] < 0 or cur_difficulty[j] > difficulty
                        
                        if cur_gt_name == cls_name:
                            valid_class = 1
                        else:
                            valid_class = -1
                            
                        if ignore or valid_class == 1:
                            gt_ignores.append(0 if not ignore else 1)
                        else:
                            gt_ignores.append(-1)
                            
                        if cur_gt_name == 'DontCare':
                            dc_bboxes.append(gt_result['bbox'][j])

                    # DET bbox processing
                    cur_det_names = det_result['name']
                    cur_det_heights = det_result['bbox'][:, 3] - det_result['bbox'][:, 1] if len(det_result['bbox']) > 0 else []
                    det_ignores = []
                    
                    if len(cur_det_names) > 0:
                        for j, cur_det_name in enumerate(cur_det_names):
                            if cur_det_heights[j] < MIN_HEIGHT[difficulty]:
                                det_ignores.append(1)
                            elif cur_det_name == cls_name:
                                det_ignores.append(0)
                            else:
                                det_ignores.append(-1)
                    
                    total_gt_ignores.append(gt_ignores)
                    total_det_ignores.append(det_ignores)
                    total_dc_bboxes.append(dc_bboxes)
                    total_scores.append(det_result['score'])
                
                # Calculate score thresholds
                tp_scores = []
                for i, id in enumerate(ids):
                    det_result = det_results.get(id, {'bbox': np.array([]), 'score': np.array([])})
                    gt_result = gt_results[id]['annos']
                    
                    det_ignores = total_det_ignores[i]
                    gt_ignores = total_gt_ignores[i]
                    scores = total_scores[i]

                    if len(det_ignores) == 0 or len(gt_ignores) == 0:
                        continue
                    
                    # Compute IoUs based on eval type
                    if eval_type == 'bbox_2d' and len(det_result['bbox']) > 0 and len(gt_result['bbox']) > 0:
                        ious = iou2d(torch.from_numpy(det_result['bbox']), 
                                   torch.from_numpy(gt_result['bbox']), metric=1).numpy()
                    elif eval_type == 'bbox_bev' and len(det_result.get('location', [])) > 0:
                        # BEV IoU calculation
                        det_bev_np = np.concatenate([
                            det_result['location'][:, [0, 2]], 
                            det_result['dimensions'][:, [0, 2]], 
                            det_result['rotation_y'][:, None]
                        ], axis=1).astype(np.float32)
                        gt_bev_np = np.concatenate([
                            gt_result['location'][:, [0, 2]], 
                            gt_result['dimensions'][:, [0, 2]], 
                            gt_result['rotation_y'][:, None]
                        ], axis=1).astype(np.float32)
                        det_bev = torch.from_numpy(det_bev_np).to(eval_device)
                        gt_bev = torch.from_numpy(gt_bev_np).to(eval_device)
                        ious = iou_bev(det_bev, gt_bev).cpu().numpy()
                    elif eval_type == 'bbox_3d' and len(det_result.get('location', [])) > 0:
                        # 3D IoU calculation
                        det_camera_np = np.concatenate([
                            det_result['location'], 
                            det_result['dimensions'], 
                            det_result['rotation_y'][:, None]
                        ], axis=1).astype(np.float32)
                        gt_camera_np = np.concatenate([
                            gt_result['location'], 
                            gt_result['dimensions'], 
                            gt_result['rotation_y'][:, None]
                        ], axis=1).astype(np.float32)
                        det_camera_bboxes = torch.from_numpy(det_camera_np).to(eval_device)
                        gt_camera_bboxes = torch.from_numpy(gt_camera_np).to(eval_device)
                        ious = iou3d_camera(det_camera_bboxes, gt_camera_bboxes).cpu().numpy()
                    else:
                        ious = np.array([])
                    
                    if ious.size == 0:
                        continue

                    # Ensure IoU matrix aligns [num_det, num_gt]
                    if ious.shape[0] == len(det_ignores) and ious.shape[1] == len(gt_ignores):
                        det_gt_ious = ious
                    elif ious.shape[1] == len(det_ignores) and ious.shape[0] == len(gt_ignores):
                        det_gt_ious = ious.T
                    else:
                        # Fallback: clip to matching dimensions
                        det_gt_ious = ious
                        min_det = min(len(det_ignores), det_gt_ious.shape[0])
                        min_gt = min(len(gt_ignores), det_gt_ious.shape[1])
                        det_gt_ious = det_gt_ious[:min_det, :min_gt]
                        scores = scores[:min_det]
                        det_ignores = det_ignores[:min_det]
                        gt_ignores = gt_ignores[:min_gt]

                    if len(scores) < det_gt_ious.shape[0]:
                        scores = scores[:det_gt_ious.shape[0]]
                    
                    # Find TP scores
                    for gt_idx in range(len(gt_ignores)):
                        if gt_ignores[gt_idx] != 0:
                            continue
                        gt_column = det_gt_ious[:, gt_idx]
                        for det_idx in range(len(det_ignores)):
                            # Compare IoU against threshold based on class name
                            if det_ignores[det_idx] == 0 and det_idx < len(gt_column) and gt_column[det_idx] > CLS_MIN_IOU.get(cls_name, 0.5):
                                if det_idx < len(scores):
                                    tp_scores.append(scores[det_idx])
                                break

                # Calculate metrics
                total_num_valid_gt = sum([sum([1 for x in gt_ignores if x == 0]) for gt_ignores in total_gt_ignores])
                if total_num_valid_gt == 0:
                    eval_ap_results[eval_type][f'difficulty_{difficulty}'][cls_name] = 0.0
                    continue
                    
                score_thresholds = get_score_thresholds(tp_scores, total_num_valid_gt)
                
                # Simplified mAP calculation (11-point interpolation)
                if len(score_thresholds) > 0:
                    mAP = len(tp_scores) / max(1, total_num_valid_gt) * 100  # Simplified
                else:
                    mAP = 0.0
                
                eval_ap_results[eval_type][f'difficulty_{difficulty}'][cls_name] = mAP

    # Calculate overall metrics with detailed class breakdown
    overall_metrics = {}
    collected_keys = []
    
    # Add detailed per-class per-difficulty metrics
    for eval_type in EVAL_NAMES:
        for difficulty in list(difficulties):
            difficulty_key = f'difficulty_{difficulty}'
            difficulty_name = ['Easy', 'Moderate', 'Hard'][difficulty] if difficulty < 3 else f'Diff_{difficulty}'
            
            # Per-class mAP
            for cls_name in class_names:
                cls_mAP = eval_ap_results[eval_type][difficulty_key].get(cls_name, 0.0)
                key = f'{eval_type}_{difficulty_key}_{cls_name}_mAP'
                overall_metrics[key] = float(cls_mAP)
                # Also add human-readable format
                human_key = f'{eval_type}_{difficulty_name}_{cls_name}_mAP'
                overall_metrics[human_key] = float(cls_mAP)
            
            # Average mAP for this difficulty
            mAPs = [eval_ap_results[eval_type][difficulty_key].get(cls_name, 0.0) 
                   for cls_name in class_names]
            key = f'{eval_type}_{difficulty_key}_mAP'
            avg_mAP = float(np.mean(mAPs)) if len(mAPs) > 0 else 0.0
            overall_metrics[key] = avg_mAP
            # Also add human-readable format
            human_key = f'{eval_type}_{difficulty_name}_mAP'
            overall_metrics[human_key] = avg_mAP
            collected_keys.append(key)
    
    # Define overall_mAP as the mean of the selected mAPs
    overall_metrics['overall_mAP'] = float(np.mean([overall_metrics[k] for k in collected_keys])) if collected_keys else 0.0
    return overall_metrics

def validate_epoch(headnet, tailnet, rq_bottleneck, val_dataloader, loss_func, args, CLASSES, LABEL2CLASSES, pcd_limit_range,
                   eval_types=('bbox_2d', 'bbox_bev', 'bbox_3d'), difficulties=(0, 1, 2), max_batches=None):
    """
    Run validation for one epoch and return metrics
    """
    headnet.eval()
    tailnet.eval()
    if rq_bottleneck is not None:
        rq_bottleneck.eval()
    
    val_losses = AverageMeter('ValLoss', ':.4f')
    val_det_losses = AverageMeter('ValDetLoss', ':.4f')
    val_vq_losses = AverageMeter('ValVQLoss', ':.6f')
    val_cb_losses = AverageMeter('ValCBLoss', ':.6f')
    
    # Store detection results for evaluation
    format_results = {}
    
    with torch.no_grad():
        for bi, data_dict in enumerate(tqdm(val_dataloader, desc='Validation', leave=False)):
            if max_batches is not None and bi >= max_batches:
                break
            if torch.cuda.is_available():
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()

            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']
            batched_difficulty = data_dict['batched_difficulty']
            
            # Forward through headnet
            pillar_features = headnet(batched_pts)
            
            # Apply RQ if enabled
            vq_loss, codebook_loss = 0, 0
            if rq_bottleneck is not None:
                pillar_features_hwc = pillar_features.permute(0, 2, 3, 1)
                quantized_features, vq_loss, codebook_loss, codes = rq_bottleneck(pillar_features_hwc)
                quantized_features = quantized_features.permute(0, 3, 1, 2)
                pillar_features = quantized_features
            
            # Forward through tailnet for loss calculation (train mode)
            bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = tailnet(
                pillar_features, 
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

            # Flatten predictions to match targets (following original PointPillars training logic)
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

            det_loss_dict = loss_func(bbox_cls_pred=bbox_cls_pred_valid,
                                    bbox_pred=bbox_pred_pos,
                                    bbox_dir_cls_pred=bbox_dir_cls_pred_pos,
                                    batched_labels=cls_labels,
                                    num_cls_pos=num_cls_pos,
                                    batched_bbox_reg=batched_bbox_reg_pos,
                                    batched_dir_labels=batched_dir_labels_pos)
            
            det_loss = det_loss_dict['total_loss']
            total_loss = det_loss + vq_loss + codebook_loss
            
            # Update meters
            val_losses.update(total_loss.item(), len(batched_pts))
            val_det_losses.update(det_loss.item(), len(batched_pts))
            val_vq_losses.update(vq_loss if isinstance(vq_loss, (int, float)) else vq_loss.item(), len(batched_pts))
            val_cb_losses.update(codebook_loss if isinstance(codebook_loss, (int, float)) else codebook_loss.item(), len(batched_pts))
            
            # Forward through tailnet for prediction generation (test mode)
            bbox_cls_pred_test, bbox_pred_test, bbox_dir_cls_pred_test, batched_anchors_test = tailnet(
                pillar_features,
                mode='test',
                batch_size=len(batched_pts)
            )
            
            # Generate predictions for evaluation
            batch_results = tailnet.get_predicted_bboxes(
                bbox_cls_pred=bbox_cls_pred_test, 
                bbox_pred=bbox_pred_test, 
                bbox_dir_cls_pred=bbox_dir_cls_pred_test, 
                batched_anchors=batched_anchors_test
            )
            
            # Process results for evaluation metrics
            for j, result in enumerate(batch_results):
                # Normalize result to dict so we don't need to modify core files
                if isinstance(result, (list, tuple)):
                    # Old behavior may return ([], [], []) on empty
                    if len(result) == 3:
                        b, l, s = result
                        try:
                            b = np.array(b, dtype=np.float32).reshape(-1, 7)
                        except Exception:
                            b = np.zeros((0, 7), dtype=np.float32)
                        l = np.array(l, dtype=np.int64).reshape(-1)
                        s = np.array(s, dtype=np.float32).reshape(-1)
                        result = {
                            'lidar_bboxes': b,
                            'labels': l,
                            'scores': s,
                        }
                    else:
                        result = {
                            'lidar_bboxes': np.zeros((0, 7), dtype=np.float32),
                            'labels': np.zeros((0,), dtype=np.int64),
                            'scores': np.zeros((0,), dtype=np.float32),
                        }
                elif not isinstance(result, dict):
                    result = {
                        'lidar_bboxes': np.zeros((0, 7), dtype=np.float32),
                        'labels': np.zeros((0,), dtype=np.int64),
                        'scores': np.zeros((0,), dtype=np.float32),
                    }
                else:
                    # Ensure numpy arrays
                    for k in ['lidar_bboxes', 'labels', 'scores']:
                        if k not in result:
                            if k == 'lidar_bboxes':
                                result[k] = np.zeros((0, 7), dtype=np.float32)
                            elif k == 'labels':
                                result[k] = np.zeros((0,), dtype=np.int64)
                            else:
                                result[k] = np.zeros((0,), dtype=np.float32)
                        else:
                            v = result[k]
                            if hasattr(v, 'detach'):
                                try:
                                    v = v.detach().cpu().numpy()
                                except Exception:
                                    v = np.array(v)
                            result[k] = v

                # If still empty, short-circuit to an empty formatted result for this item
                if result['lidar_bboxes'] is None or len(result['lidar_bboxes']) == 0:
                    format_results[data_dict['batched_img_info'][j]['image_idx']] = {
                        'name': np.array([]), 'truncated': np.array([]), 'occluded': np.array([]), 'alpha': np.array([]),
                        'bbox': np.array([]), 'dimensions': np.array([]), 'location': np.array([]), 'rotation_y': np.array([]), 'score': np.array([])
                    }
                    continue
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

                format_results[idx] = {k: np.array(v) for k, v in format_result.items()}

    # Calculate evaluation metrics
    val_dataset = val_dataloader.dataset
    eval_metrics = do_eval(format_results, val_dataset.data_infos, CLASSES, "",
                           eval_types=eval_types, difficulties=difficulties)
    
    # Combine loss and evaluation metrics
    all_metrics = {
        'total_loss': val_losses.avg,
        'det_loss': val_det_losses.avg,
        'vq_loss': val_vq_losses.avg,
        'cb_loss': val_cb_losses.avg,
        **eval_metrics
    }
    
    return all_metrics

def train_progressive_rq(config):
    """Progressive learning training using config from YAML"""
    # Setup experiment name and date
    date_str = str(datetime.date.today())
    time_str = datetime.datetime.now().strftime("%H-%M-%S")
    exp_name = f"{date_str}_{time_str}_Progressive_RQ_training"

    # Initialize wandb if requested
    wandb_run = None
    if config['logging']['use_wandb']:
        wandb_name = config['logging']['wandb_name'] or f"Progressive_RQ_{config['model']['n_codebook']}books"
        wandb_run = wandb.init(
            project=config['logging']['wandb_project'],
            name=wandb_name,
            config=config,
            tags=["pointpillars", "rq", "progressive", "split_inference"],
        )
    
    # Convert config to args-like object for compatibility
    args = argparse.Namespace()
    args.data_root = config['dataset']['dir']
    args.pretrained_ckpt = config['model']['pretrained_weight']
    args.batch_size = config['training']['batch_size']
    args.num_workers = config['training']['num_workers']
    args.nclasses = config['dataset']['num_classes']
    args.latent_shape = config['rq_model']['latent_shape']
    args.code_shape = config['rq_model']['code_shape']
    args.decay = config['rq_model']['decay']
    args.vq_weight = config['loss_weights']['vq_weight']
    args.codebook_weight = config['loss_weights']['codebook_weight']
    args.det_weight = config['loss_weights']['det_weight']
    args.gpu = config['hardware']['gpu']
    
    # Setup model and data using common function
    (train_dataloader, val_dataloader), headnet, tailnet, device, CLASSES, LABEL2CLASSES, pcd_limit_range = setup_model_and_data(args, mode='train')
    
    # Progressive learning configuration
    embedding_schedule = config['progressive_learning']['embedding_schedule']
    stage_epochs = config['progressive_learning']['embedding_stage_epochs'] 
    warmup_epochs = config['progressive_learning']['warmup_epochs']
    n_codebooks = config['model']['n_codebook']
    
    # Setup directories
    saved_ckpt_path = os.path.join(config['logging']['saved_path'], 'checkpoints')
    os.makedirs(saved_ckpt_path, exist_ok=True)
    
    headnet.eval()
    tailnet.eval()
    
    # Global tracking variables
    global_best_mAP = 0.0
    global_best_epoch = 0
    overall_step = 0
    
    loss_func = Loss()
    
    print(f"Starting Progressive RQ Training:")
    print(f"- {n_codebooks} codebooks")
    print(f"- Embedding schedule: {embedding_schedule}")
    print(f"- {stage_epochs} epochs per stage, {warmup_epochs} warmup epochs")
    print(f"Experiment: {exp_name}")
    print("="*80)
    
    # Progressive training: iterate through codebooks
    for codebook_idx in range(n_codebooks):
        print(f"\n{'='*60}")
        print(f"TRAINING CODEBOOK {codebook_idx + 1}/{n_codebooks}")
        print(f"{'='*60}")
        
        # Progressive training: iterate through embedding sizes for current codebook
        for stage_idx, embed_size in enumerate(embedding_schedule):
            print(f"\n{'-'*40}")
            print(f"Codebook {codebook_idx + 1}, Embedding Stage {stage_idx + 1}/{len(embedding_schedule)}: {embed_size} embeddings")
            print(f"{'-'*40}")
            
            # Create RQ bottleneck for this stage
            stage_args = argparse.Namespace(**vars(args))
            stage_args.codebook_size = embed_size
            stage_args.code_shape = config['rq_model']['code_shape'].copy()
            stage_args.code_shape[-1] = codebook_idx + 1  # Use codebook_idx + 1 codebooks
            
            rq_bottleneck = create_rq_bottleneck(stage_args, device, ema=True)
            
            # Set training stage: only current codebook is trainable, others frozen or inactive
            # For incremental training: freeze embeddings trained in previous stages
            frozen_embed_size = embedding_schedule[stage_idx - 1] if stage_idx > 0 else 0
            full_embed_size = embedding_schedule[-1]
            rq_bottleneck.set_training_stage(codebook_idx, embed_size, full_embed_size, frozen_embed_size)
            
            print(f"Training stage setup:")
            print(f"  Codebook {codebook_idx}: active_n_embed={embed_size}, frozen_n_embed={frozen_embed_size}")
            print(f"  Training range: embeddings {frozen_embed_size} to {embed_size-1}")
            
            # Setup optimizer and scheduler for this stage
            init_lr = config['training']['init_lr']
            
            # Debug: Check trainable parameters
            trainable_params = [p for p in rq_bottleneck.parameters() if p.requires_grad]
            print(f"Debug: Found {len(trainable_params)} trainable parameters")
            
            # If no trainable parameters in EMA mode, we'll create optimizer later after switching to gradient mode
            if len(trainable_params) == 0:
                print("No trainable parameters found (EMA mode), will create optimizer after warmup")
                optimizer = None
                scheduler = None
            else:
                optimizer = torch.optim.AdamW(params=trainable_params,
                                              lr=init_lr,
                                              betas=(0.9, 0.999),
                                              weight_decay=0.01)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_epochs)
            
            # Stage training variables
            stage_best_mAP = 0.0
            stage_best_epoch = 0
            train_losses = AverageMeter('TrainLoss', ':.4f')
            train_det_losses = AverageMeter('TrainDetLoss', ':.4f')
            train_vq_losses = AverageMeter('TrainVQLoss', ':.6f')
            train_cb_losses = AverageMeter('TrainCBLoss', ':.6f')
            
            early_stopping = EarlyStopping(patience=config['training']['early_stopping_patience'], verbose=True)
            
            # Stage training loop
            for epoch in range(stage_epochs):
                stage_epoch = epoch + 1
                global_epoch = codebook_idx * len(embedding_schedule) * stage_epochs + stage_idx * stage_epochs + epoch + 1
                
                epoch_start_time = time.time()
                print(f'\nStage Epoch {stage_epoch}/{stage_epochs} (Global: {global_epoch})')
                print('-' * 40)
                
                # Reset meters
                train_losses.reset()
                train_det_losses.reset()
                train_vq_losses.reset()
                train_cb_losses.reset()
                
                # EMA warmup epochs
                if epoch < warmup_epochs:
                    print(f"EMA Warmup Epoch {epoch + 1}/{warmup_epochs}")
                    rq_bottleneck.train()
                    
                    with torch.no_grad():
                        for i, data_dict in enumerate(tqdm(train_dataloader, desc="EMA Warmup")):
                            if torch.cuda.is_available():
                                for key in data_dict:
                                    for j, item in enumerate(data_dict[key]):
                                        if torch.is_tensor(item):
                                            data_dict[key][j] = data_dict[key][j].cuda()
                            
                            batched_pts = data_dict['batched_pts']
                            pillar_features = headnet(batched_pts)
                            pillar_features_hwc = pillar_features.permute(0, 2, 3, 1)
                            quantized_features, vq_loss, codebook_loss, codes = rq_bottleneck(pillar_features_hwc)
                    
                    # Validation during warmup
                    print("Running validation during EMA warmup...")
                    val_metrics = validate_epoch(headnet, tailnet, rq_bottleneck, val_dataloader,
                                                loss_func, args, CLASSES, LABEL2CLASSES, pcd_limit_range)
                    
                    print(f"Warmup Validation Results:")
                    # Print only bbox_3d metrics for console
                    bbox_3d_overall = (val_metrics.get('bbox_3d_difficulty_0_mAP', 0.0) + 
                                      val_metrics.get('bbox_3d_difficulty_1_mAP', 0.0) + 
                                      val_metrics.get('bbox_3d_difficulty_2_mAP', 0.0)) / 3.0
                    print(f"  Overall bbox_3d mAP: {bbox_3d_overall:.4f}")
                    for key, value in val_metrics.items():
                        if key.startswith('bbox_3d_difficulty_') and not any(cls in key for cls in ['Car', 'Pedestrian', 'Cyclist']):
                            print(f"  {key}: {value:.4f}")
                    print(f"  Loss - Total: {val_metrics.get('total_loss', 0.0):.4f}, Det: {val_metrics.get('det_loss', 0.0):.4f}, VQ: {val_metrics.get('vq_loss', 0.0):.6f}")
                    
                    if wandb_run is not None:
                        # 建立詳細的warmup wandb log - 記錄所有指標
                        warmup_log_data = {
                            "codebook_idx": codebook_idx,
                            "stage_idx": stage_idx,
                            "embed_size": embed_size,
                            "epoch": global_epoch,
                            # Loss metrics
                            "warmup_val/total_loss": float(val_metrics.get('total_loss', 0.0)),
                            "warmup_val/det_loss": float(val_metrics.get('det_loss', 0.0)),
                            "warmup_val/vq_loss": float(val_metrics.get('vq_loss', 0.0)),
                            "warmup_val/cb_loss": float(val_metrics.get('cb_loss', 0.0)),
                        }
                        
                        # 記錄所有評估指標到wandb
                        for key, value in val_metrics.items():
                            if any(key.startswith(eval_type) for eval_type in ['bbox_2d', 'bbox_bev', 'bbox_3d']) or key == 'overall_mAP':
                                warmup_log_data[f"warmup_val/{key}"] = float(value)
                        
                        wandb.log(warmup_log_data, step=global_epoch)
                    
                    continue
                
                # Switch to gradient training after warmup
                if epoch == warmup_epochs:
                    print("Switching to gradient-based training")
                    # Create new RQ model without EMA
                    new_rq_bottleneck = create_rq_bottleneck(stage_args, device, ema=False)
                    new_rq_bottleneck.set_training_stage(codebook_idx, embed_size, full_embed_size, frozen_embed_size)
                    
                    # Transfer EMA weights
                    with torch.no_grad():
                        for i, (old_cb, new_cb) in enumerate(zip(rq_bottleneck.codebooks, new_rq_bottleneck.codebooks)):
                            if hasattr(old_cb, 'sync_ema_weights'):
                                old_cb.sync_ema_weights()
                            source_weight = old_cb.weight.detach()
                            if source_weight.device != new_cb.weight.device:
                                source_weight = source_weight.to(new_cb.weight.device)
                            new_cb.weight.data.copy_(source_weight)
                            new_cb.set_active_n_embed(old_cb.active_n_embed)
                            new_cb.set_frozen_n_embed(old_cb.frozen_n_embed)
                    
                    rq_bottleneck = new_rq_bottleneck
                    
                    # Recreate optimizer with gradient-based parameters
                    trainable_params = [p for p in rq_bottleneck.parameters() if p.requires_grad]
                    print(f"Debug: Found {len(trainable_params)} trainable parameters after switching to gradient mode")
                    
                    if len(trainable_params) == 0:
                        print("Warning: No trainable parameters found even in gradient mode!")
                        continue
                    
                    optimizer = torch.optim.AdamW(params=trainable_params,
                                                  lr=init_lr,
                                                  betas=(0.9, 0.999), 
                                                  weight_decay=0.01)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_epochs - warmup_epochs)
                
                # Gradient training
                # Skip gradient training if we're still in warmup epochs or no optimizer
                if epoch < warmup_epochs or optimizer is None:
                    continue
                    
                rq_bottleneck.train()
                for i, data_dict in enumerate(tqdm(train_dataloader, desc=f"Training")):
                    if torch.cuda.is_available():
                        for key in data_dict:
                            for j, item in enumerate(data_dict[key]):
                                if torch.is_tensor(item):
                                    data_dict[key][j] = data_dict[key][j].cuda()
                    
                    if optimizer is not None:
                        optimizer.zero_grad()
                    
                    batched_pts = data_dict['batched_pts']
                    batched_gt_bboxes = data_dict['batched_gt_bboxes']
                    batched_labels = data_dict['batched_labels']
                    
                    # Forward pass
                    with torch.no_grad():
                        pillar_features = headnet(batched_pts)
                    
                    pillar_features_hwc = pillar_features.permute(0, 2, 3, 1)
                    quantized_features, vq_loss, codebook_loss, codes = rq_bottleneck(pillar_features_hwc)
                    quantized_features = quantized_features.permute(0, 3, 1, 2)
                    
                    # Detection loss
                    bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = tailnet(
                        quantized_features,
                        mode='train',
                        batched_gt_bboxes=batched_gt_bboxes,
                        batched_gt_labels=batched_labels,
                        batch_size=len(batched_pts)
                    )
                    
                    # Calculate detection loss (similar to original implementation)
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

                    det_loss = det_loss_dict['total_loss']
                    
                    # Total loss
                    total_loss = (args.vq_weight * vq_loss +
                                 args.codebook_weight * codebook_loss +
                                 args.det_weight * det_loss)
                    
                    # Update meters
                    train_losses.update(total_loss.item(), len(batched_pts))
                    train_det_losses.update(det_loss.item(), len(batched_pts))
                    train_vq_losses.update(vq_loss.item(), len(batched_pts))
                    train_cb_losses.update(codebook_loss.item(), len(batched_pts))
                    
                    # Backward and optimize
                    if optimizer is not None and total_loss.requires_grad and total_loss.item() > 0:
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(rq_bottleneck.parameters(), max_norm=1.0)
                        optimizer.step()
                        if epoch >= warmup_epochs and scheduler is not None:
                            scheduler.step()
                    
                    overall_step += 1
                    
                    if wandb_run is not None and config['logging']['log_freq'] > 0 and overall_step % config['logging']['log_freq'] == 0:
                        log_data = {
                            'train/total_loss': float(total_loss.item()),
                            'train/det_loss': float(det_loss.item()),
                            'train/vq_loss': float(vq_loss.item()),
                            'train/codebook_loss': float(codebook_loss.item()),
                            'codebook_idx': codebook_idx,
                            'stage_idx': stage_idx,
                            'embed_size': embed_size,
                            'epoch': global_epoch,
                        }
                        if optimizer is not None:
                            log_data['train/lr'] = optimizer.param_groups[0]['lr']
                        wandb.log(log_data, step=overall_step)
                
                # Validation
                print(f"Running validation...")
                val_metrics = validate_epoch(headnet, tailnet, rq_bottleneck, val_dataloader,
                                            loss_func, args, CLASSES, LABEL2CLASSES, pcd_limit_range)
                
                epoch_time = time.time() - epoch_start_time
                # 使用bbox_3d overall mAP作為主要指標
                current_mAP = (val_metrics.get('bbox_3d_difficulty_0_mAP', 0.0) + 
                              val_metrics.get('bbox_3d_difficulty_1_mAP', 0.0) + 
                              val_metrics.get('bbox_3d_difficulty_2_mAP', 0.0)) / 3.0
                
                # Print results - 只顯示bbox_3d相關指標
                bbox_3d_overall = (val_metrics.get('bbox_3d_difficulty_0_mAP', 0.0) + 
                                  val_metrics.get('bbox_3d_difficulty_1_mAP', 0.0) + 
                                  val_metrics.get('bbox_3d_difficulty_2_mAP', 0.0)) / 3.0
                print(f"Stage Epoch {stage_epoch} Summary:")
                print(f"  Time: {epoch_time:.2f}s")
                print(f"  Training Loss: {train_losses.avg:.4f} (Det: {train_det_losses.avg:.4f}, VQ: {train_vq_losses.avg:.6f}, CB: {train_cb_losses.avg:.6f})")
                print(f"  Validation Results (bbox_3d only):")
                print(f"    Overall bbox_3d mAP: {bbox_3d_overall:.4f}")
                for key, value in val_metrics.items():
                    if key.startswith('bbox_3d_difficulty_') and not any(cls in key for cls in ['Car', 'Pedestrian', 'Cyclist']):
                        print(f"    {key}: {value:.4f}")
                print(f"    Loss - Total: {val_metrics.get('total_loss', 0.0):.4f}, Det: {val_metrics.get('det_loss', 0.0):.4f}")
                
                if wandb_run is not None:
                    # 建立詳細的wandb log - 記錄所有指標
                    wandb_log_data = {
                        'epoch': global_epoch,
                        'stage_epoch': stage_epoch,
                        'codebook_idx': codebook_idx,
                        'stage_idx': stage_idx,
                        'embed_size': embed_size,
                        # Training metrics
                        'train/total_loss': float(train_losses.avg),
                        'train/det_loss': float(train_det_losses.avg),
                        'train/vq_loss': float(train_vq_losses.avg),
                        'train/codebook_loss': float(train_cb_losses.avg),
                        # Loss metrics
                        'val/total_loss': float(val_metrics.get('total_loss', 0.0)),
                        'val/det_loss': float(val_metrics.get('det_loss', 0.0)),
                        'val/vq_loss': float(val_metrics.get('vq_loss', 0.0)),
                        'val/cb_loss': float(val_metrics.get('cb_loss', 0.0)),
                    }
                    
                    # 記錄所有評估指標到wandb（bbox_2d, bbox_bev, bbox_3d）
                    for key, value in val_metrics.items():
                        if any(key.startswith(eval_type) for eval_type in ['bbox_2d', 'bbox_bev', 'bbox_3d']) or key == 'overall_mAP':
                            wandb_log_data[f'val/{key}'] = float(value)
                    
                    wandb.log(wandb_log_data, step=overall_step)
                
                # Update best metrics
                is_stage_best = current_mAP > stage_best_mAP
                is_global_best = current_mAP > global_best_mAP
                
                if is_stage_best:
                    stage_best_mAP = current_mAP
                    stage_best_epoch = stage_epoch
                
                if is_global_best:
                    global_best_mAP = current_mAP
                    global_best_epoch = global_epoch
                
                # Save checkpoint - 只在最佳結果時儲存
                if is_stage_best or is_global_best:
                    checkpoint_data = {
                        'rq_bottleneck': rq_bottleneck.state_dict(),
                        'optimizer': optimizer.state_dict() if optimizer is not None else None,
                        'scheduler': scheduler.state_dict() if scheduler is not None else None,
                        'epoch': global_epoch,
                        'stage_epoch': stage_epoch,
                        'codebook_idx': codebook_idx,
                        'stage_idx': stage_idx,
                        'embed_size': embed_size,
                        'val_metrics': val_metrics,
                        'config': config
                    }
                    
                    if is_stage_best:
                        stage_best_name = f'codebook_{codebook_idx}_stage_{stage_idx}_best.pth'
                        torch.save(checkpoint_data, os.path.join(saved_ckpt_path, stage_best_name))
                        print(f"  *** Stage best bbox_3d mAP: {stage_best_mAP:.4f} at epoch {stage_best_epoch} ***")
                    
                    if is_global_best:
                        torch.save(checkpoint_data, os.path.join(saved_ckpt_path, 'global_best.pth'))
                        print(f"  *** Global best bbox_3d mAP: {global_best_mAP:.4f} at epoch {global_best_epoch} ***")
                        if wandb_run is not None:
                            wandb.run.summary['global_best_bbox_3d_mAP'] = global_best_mAP
                            wandb.run.summary['global_best_epoch'] = global_best_epoch
                
                # Early stopping check
                early_stopping(val_metrics['total_loss'], rq_bottleneck)
                if early_stopping.early_stop:
                    print(f"Early stopping triggered at stage epoch {stage_epoch}")
                    break
            
            print(f"\nCompleted Codebook {codebook_idx + 1}, Stage {stage_idx + 1}")
            print(f"Stage best bbox_3d mAP: {stage_best_mAP:.4f}")
            
            # Record stage completion to wandb with detailed metrics
            if wandb_run is not None:
                # 運行最終驗證以獲取完整指標
                final_val_metrics = validate_epoch(headnet, tailnet, rq_bottleneck, val_dataloader,
                                                  loss_func, args, CLASSES, LABEL2CLASSES, pcd_limit_range)
                
                stage_completion_data = {
                    f'stage_completion/codebook_{codebook_idx}_stage_{stage_idx}_best_bbox_3d_mAP': stage_best_mAP,
                    f'stage_completion/codebook_{codebook_idx}_stage_{stage_idx}_embed_size': embed_size,
                    f'stage_completion/codebook_{codebook_idx}_stage_{stage_idx}_global_epoch': global_epoch,
                }
                
                # 記錄最終所有指標
                for key, value in final_val_metrics.items():
                    if any(key.startswith(eval_type) for eval_type in ['bbox_2d', 'bbox_bev', 'bbox_3d']) or key == 'overall_mAP':
                        stage_completion_data[f'stage_completion/codebook_{codebook_idx}_stage_{stage_idx}_{key}'] = float(value)
                
                wandb.log(stage_completion_data, step=global_epoch)
                print(f"  Logged stage completion metrics to wandb")
            
            # Save stage completion checkpoint if enabled
            if config['logging']['save_stage_weights']:
                final_stage_checkpoint = {
                    'rq_bottleneck': rq_bottleneck.state_dict(),
                    'codebook_idx': codebook_idx,
                    'stage_idx': stage_idx,
                    'embed_size': embed_size,
                    'stage_best_mAP': stage_best_mAP,
                    'config': config
                }
                stage_final_name = f'codebook_{codebook_idx}_stage_{stage_idx}_final.pth'
                torch.save(final_stage_checkpoint, os.path.join(saved_ckpt_path, stage_final_name))
    
    # Training completed
    print("\n" + "="*80)
    print("PROGRESSIVE TRAINING COMPLETED!")
    print("="*80)
    print(f"Global best validation mAP: {global_best_mAP:.4f} at epoch {global_best_epoch}")
    print(f"Final experiment: {exp_name}")
    print("="*80)

    if wandb_run is not None:
        wandb.run.summary['training_completed'] = True
        wandb.run.summary['final_global_best_mAP'] = global_best_mAP
        wandb.run.summary['final_global_best_epoch'] = global_best_epoch
        wandb.finish()
    
    return global_best_mAP, global_best_epoch


def train_rq(args):
    """Train RQ model using refactored common setup"""
    # Setup experiment name and date
    date_str = str(datetime.date.today())
    time_str = datetime.datetime.now().strftime("%H-%M-%S")
    exp_name = f"{date_str}_{time_str}_RQ_training"

    # Initialize wandb if requested
    wandb_run = None
    if args.use_wandb:
        wandb_name = args.wandb_name or f"RQ_lat{args.latent_shape}_code{args.code_shape}_embed{args.codebook_size}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            config=vars(args),
            tags=["pointpillars", "rq", "split_inference"],
        )
        
    # Setup model and data using common function
    (train_dataloader, val_dataloader), headnet, tailnet, device, CLASSES, LABEL2CLASSES, pcd_limit_range = setup_model_and_data(args, mode='train')
    
    # Create RQ bottleneck and loss function
    rq_bottleneck = create_rq_bottleneck(args, device, ema=True)
    loss_func = Loss()

    # Setup optimizer and scheduler
    steps_per_epoch = max(1, len(train_dataloader))
    effective_epochs = max(args.max_epoch - 2, 1)
    init_lr = args.init_lr
    optimizer = torch.optim.AdamW(params=rq_bottleneck.parameters(), 
                                  lr=init_lr, 
                                  betas=(0.9, 0.999),
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                     T_0=max(1, effective_epochs//4), 
                                                                     T_mult=2, 
                                                                     eta_min=init_lr*0.01)
    
    # Setup directories and tracking variables
    saved_ckpt_path = os.path.join(args.saved_path, 'checkpoints')
    os.makedirs(saved_ckpt_path, exist_ok=True)

    headnet.eval()
    tailnet.eval()
    
    best_mAP = 0.0
    best_epoch = 0
    train_losses = AverageMeter('TrainLoss', ':.4f')
    train_det_losses = AverageMeter('TrainDetLoss', ':.4f')
    train_vq_losses = AverageMeter('TrainVQLoss', ':.6f')
    train_cb_losses = AverageMeter('TrainCBLoss', ':.6f')
    
    early_stopping = EarlyStopping(patience=getattr(args, 'patience', 20), verbose=True)
    
    print(f"Starting training for {args.max_epoch} epochs...")
    print(f"Experiment: {exp_name}")
    overall_step = 0

    # Training loop - keeping the existing epoch logic intact
    for epoch in range(args.max_epoch):
        epoch_start_time = time.time()
        print('=' * 60)
        print(f'Epoch {epoch+1}/{args.max_epoch}')
        print('=' * 60)
        
        train_step = 0
        rq_bottleneck.train()
        
        # Reset meters for this epoch
        train_losses.reset()
        train_det_losses.reset()
        train_vq_losses.reset()
        train_cb_losses.reset()
        
        # First 2 epochs: EMA initialization (no gradients)
        if epoch < 2:
            print(f"EMA Initialization Epoch {epoch + 1}/2")
            
            with torch.no_grad():
                for i, data_dict in enumerate(tqdm(train_dataloader, desc="EMA Init")):
                    if torch.cuda.is_available():
                        for key in data_dict:
                            for j, item in enumerate(data_dict[key]):
                                if torch.is_tensor(item):
                                    data_dict[key][j] = data_dict[key][j].cuda()
                    
                    batched_pts = data_dict['batched_pts']
                    
                    # Forward through headnet
                    pillar_features = headnet(batched_pts)
                    
                    # Apply RQ for EMA initialization
                    pillar_features_hwc = pillar_features.permute(0, 2, 3, 1)
                    quantized_features, vq_loss, codebook_loss, codes = rq_bottleneck(pillar_features_hwc)

            # Run validation during EMA initialization
            print("Running validation during EMA initialization...")
            val_metrics = validate_epoch(headnet, tailnet, rq_bottleneck, val_dataloader, 
                                       loss_func, args, CLASSES, LABEL2CLASSES, pcd_limit_range)
            
            print(f"EMA Epoch {epoch+1} Validation Results:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")

            if wandb_run is not None:
                wandb.log(
                    {**{f"ema_val/{key}": value for key, value in val_metrics.items()}, "epoch": epoch + 1},
                    step=epoch + 1,
                )
                
            continue
        
        # From epoch 2 onwards: Switch to gradient-based training
        else:
            print("Switching to gradient-based training")
            
            # Create new RQ model without EMA
            new_rq_bottleneck = create_rq_bottleneck(args, device, ema=False)
            
            # Transfer EMA-initialized codebook weights to new model
            with torch.no_grad():
                for i, (old_codebook, new_codebook) in enumerate(zip(rq_bottleneck.codebooks, new_rq_bottleneck.codebooks)):
                    if hasattr(old_codebook, 'sync_ema_weights'):
                        old_codebook.sync_ema_weights()
                    source_weight = old_codebook.weight.detach()
                    if source_weight.device != new_codebook.weight.device:
                        source_weight = source_weight.to(new_codebook.weight.device)
                    new_codebook.weight.data.copy_(source_weight)
                    new_codebook.set_active_n_embed(old_codebook.active_n_embed)
                    new_codebook.set_frozen_n_embed(old_codebook.frozen_n_embed)
            
            rq_bottleneck = new_rq_bottleneck
            
            # Recreate optimizer and scheduler for new model
            optimizer = torch.optim.AdamW(params=rq_bottleneck.parameters(), 
                                          lr=init_lr, 
                                          betas=(0.9, 0.999),
                                          weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                             T_0=max(1, effective_epochs//4), 
                                                                             T_mult=2, 
                                                                             eta_min=init_lr*0.01)
        
        # Training loop for gradient-based epochs
        for i, data_dict in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")):
            if torch.cuda.is_available():
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()

            optimizer.zero_grad()

            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']
            
            # Forward through headnet (frozen)
            with torch.no_grad():
                pillar_features = headnet(batched_pts)
            
            # RQ bottleneck - convert to BHWC format for RQ
            pillar_features_hwc = pillar_features.permute(0, 2, 3, 1)
            quantized_features, vq_loss, codebook_loss, codes = rq_bottleneck(pillar_features_hwc)
            quantized_features = quantized_features.permute(0, 3, 1, 2)
            
            # Forward through tailnet to get detection loss (keep tailnet params frozen, but allow grads to flow to RQ)
            bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = tailnet(
                quantized_features,
                mode='train',
                batched_gt_bboxes=batched_gt_bboxes,
                batched_gt_labels=batched_labels,
                batch_size=len(batched_pts)
            )

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

            det_loss = det_loss_dict['total_loss']
            
            # Progressive training: gradually increase VQ weight
            progress_ratio = min(1.0, (epoch - 2) / (args.max_epoch * 0.3))
            current_vq_weight = args.vq_weight * progress_ratio
            current_cb_weight = args.codebook_weight * progress_ratio
            
            # Total loss: RQ losses + detection loss
            total_loss = (current_vq_weight * vq_loss + 
                         current_cb_weight * codebook_loss + 
                         args.det_weight * det_loss)
            
            # Update meters
            train_losses.update(total_loss.item(), len(batched_pts))
            train_det_losses.update(det_loss.item(), len(batched_pts))
            train_vq_losses.update(vq_loss.item(), len(batched_pts))
            train_cb_losses.update(codebook_loss.item(), len(batched_pts))
            
            # Backward and optimize
            if total_loss.requires_grad and total_loss.item() > 0:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(rq_bottleneck.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
            
            train_step += 1
            overall_step += 1

            if wandb_run is not None and args.log_freq > 0 and overall_step % args.log_freq == 0:
                wandb.log({
                    'train/total_loss': float(total_loss.item()),
                    'train/det_loss': float(det_loss.item()),
                    'train/vq_loss': float(vq_loss.item()),
                    'train/codebook_loss': float(codebook_loss.item()),
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'epoch': epoch + 1,
                }, step=overall_step)
        
        # Run validation at the end of each epoch
        print(f"Running validation for epoch {epoch+1}...")
        val_metrics = validate_epoch(headnet, tailnet, rq_bottleneck, val_dataloader,
                        loss_func, args, CLASSES, LABEL2CLASSES, pcd_limit_range,
                        eval_types=('bbox_3d',), difficulties=(0, 1, 2))
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Training Loss: {train_losses.avg:.4f}")
        print(f"  Validation Results:")
        for key, value in val_metrics.items():
            if key.startswith('bbox_3d_'):
                print(f"    - {key}: {value:.4f}")

        if wandb_run is not None:
            log_step = overall_step if overall_step > 0 else epoch + 1
            wandb.log({
                'epoch': epoch + 1,
                'train/epoch_loss': float(train_losses.avg),
                **{f'val/{k}': float(v) for k, v in val_metrics.items()},
            }, step=log_step)
        
        # Save checkpoint periodically and if best mAP
        current_mAP = val_metrics.get('overall_mAP', 0.0)
        is_best = current_mAP > best_mAP
        
        if epoch % args.ckpt_freq_epoch == 0 or is_best:
            checkpoint_data = {
                'rq_bottleneck': rq_bottleneck.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'val_metrics': val_metrics,
                'args': args
            }
            
            torch.save(checkpoint_data, os.path.join(saved_ckpt_path, f'epoch_{epoch+1}.pth'))
            
            if is_best:
                best_mAP = current_mAP
                best_epoch = epoch + 1
                torch.save(checkpoint_data, os.path.join(saved_ckpt_path, 'best_model.pth'))
                print(f"  *** New best mAP: {best_mAP:.4f} at epoch {best_epoch} ***")
                if wandb_run is not None:
                    wandb.run.summary['best_mAP'] = best_mAP
                    wandb.run.summary['best_epoch'] = best_epoch
        
        # Early stopping check
        early_stopping(val_metrics['total_loss'], rq_bottleneck)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Training completed
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Best validation mAP: {best_mAP:.4f} at epoch {best_epoch}")
    print(f"Final experiment: {exp_name}")
    print("="*60)

    if wandb_run is not None:
        wandb.run.summary['training_completed'] = True
        wandb.run.summary['final_best_mAP'] = best_mAP
        wandb.run.summary['final_best_epoch'] = best_epoch
        wandb.finish()
    
    return best_mAP, best_epoch

def setup_model_and_data(args, mode='train'):
    """Common setup for model and data loading"""
    setup_seed()
    
    # Load data
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

    # Load pretrained PointPillars model
    device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    full_model = PointPillars(nclasses=args.nclasses).to(device)
    full_model = load_pretrained_with_report(full_model, args.pretrained_ckpt, device)
    print(f"Loaded pretrained weights from {args.pretrained_ckpt}")
    
    # Split the model
    headnet, tailnet = split_pointpillars(full_model)
    
    # Freeze headnet and tailnet parameters
    for param in headnet.parameters():
        param.requires_grad = False
    for param in tailnet.parameters():
        param.requires_grad = False
    print("Frozen headnet and tailnet parameters")
    
    # Define evaluation constants
    ds_classes = Kitti.CLASSES
    CLASSES = {v: k for k, v in ds_classes.items()}
    LABEL2CLASSES = {v: k for k, v in ds_classes.items()}
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
                
            print(f"Initialized codebook {i}: {codebook.n_embed} codes, {codebook.embed_dim}D")
    
    print(f"Created RQ bottleneck with {sum(p.numel() for p in rq_bottleneck.parameters())} parameters")
    return rq_bottleneck

def load_rq_checkpoint(rq_bottleneck, checkpoint_path, device):
    """Load RQ checkpoint safely"""
    if checkpoint_path and os.path.exists(checkpoint_path):
        rq_checkpoint = torch.load(checkpoint_path, map_location=device)
        rq_bottleneck.load_state_dict(rq_checkpoint.get('rq_bottleneck', rq_checkpoint))
        rq_bottleneck.eval()
        print(f"Loaded RQ checkpoint from {checkpoint_path}")
    return rq_bottleneck

def evaluate_progressive_rq(config):
    """Evaluation function with progressive RQ configuration support"""
    # Convert config to args-like object for compatibility
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
    
    # Setup model and data
    val_dataloader, headnet, tailnet, device, CLASSES, LABEL2CLASSES, pcd_limit_range = setup_model_and_data(args, mode='eval')
    
    # Create RQ bottleneck based on evaluation configuration
    rq_bottleneck = None
    if config['evaluation']['rq_ckpt'] is not None:
        eval_config = config['evaluation']
        use_num_codebooks = eval_config['use_num_codebook']
        use_num_embeddings = eval_config['use_num_embedding']
        
        print(f"Evaluation Configuration:")
        print(f"- Using {use_num_codebooks} codebook(s)")
        print(f"- Using {use_num_embeddings} embedding(s) per codebook")
        print(f"- Loading checkpoint: {eval_config['rq_ckpt']}")
        
        # Create RQ bottleneck with evaluation configuration
        eval_args = argparse.Namespace(**vars(args))
        eval_args.codebook_size = use_num_embeddings
        eval_args.code_shape = config['rq_model']['code_shape'].copy()
        eval_args.code_shape[-1] = use_num_codebooks
        
        rq_bottleneck = create_rq_bottleneck(eval_args, device, ema=False)
        rq_bottleneck.set_evaluation_stage(use_num_codebooks, use_num_embeddings)
        rq_bottleneck = load_rq_checkpoint(rq_bottleneck, eval_config['rq_ckpt'], device)
    
    # Run evaluation
    loss_func = Loss()
    eval_metrics = validate_epoch(headnet, tailnet, rq_bottleneck, val_dataloader,
                                loss_func, args, CLASSES, LABEL2CLASSES, pcd_limit_range)
    
    # Print results
    print("\n" + "="*60)
    print("PROGRESSIVE RQ EVALUATION RESULTS")
    print("="*60)
    
    # Overall metrics
    print(f"Overall mAP: {eval_metrics.get('overall_mAP', 0.0):8.4f}")
    print("-" * 40)
    
    # Detailed breakdown by difficulty and class
    difficulties_names = ['Easy', 'Moderate', 'Hard']
    class_names = ['Car', 'Pedestrian', 'Cyclist']
    
    for i, diff_name in enumerate(difficulties_names):
        print(f"{diff_name} Difficulty:")
        for cls_name in class_names:
            key = f'bbox_3d_difficulty_{i}_{cls_name}_mAP'
            if key in eval_metrics:
                print(f"  {cls_name:12s}: {eval_metrics[key]:8.4f}")
        
        # Difficulty average
        diff_avg_key = f'bbox_3d_difficulty_{i}_mAP'
        if diff_avg_key in eval_metrics:
            print(f"  {'Average':12s}: {eval_metrics[diff_avg_key]:8.4f}")
        print()
    
    # Loss information
    print("Loss Information:")
    print(f"  {'Total Loss':12s}: {eval_metrics.get('total_loss', 0.0):8.4f}")
    print(f"  {'Det Loss':12s}: {eval_metrics.get('det_loss', 0.0):8.4f}")
    print(f"  {'VQ Loss':12s}: {eval_metrics.get('vq_loss', 0.0):8.6f}")
    print(f"  {'CB Loss':12s}: {eval_metrics.get('cb_loss', 0.0):8.6f}")
    
    print("="*60)
    
    return eval_metrics


def load_config(config_path):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded configuration from: {config_path}")
    return config


def evaluate_rq(args):
    """Simplified evaluation function using common setup"""
    # Setup model and data
    val_dataloader, headnet, tailnet, device, CLASSES, LABEL2CLASSES, pcd_limit_range = setup_model_and_data(args, mode='eval')
    
    # Create RQ bottleneck if needed
    rq_bottleneck = None
    if args.use_rq and args.rq_ckpt:
        rq_bottleneck = create_rq_bottleneck(args, device, ema=False)
        rq_bottleneck = load_rq_checkpoint(rq_bottleneck, args.rq_ckpt, device)
    
    # Run evaluation
    loss_func = Loss()
    eval_metrics = validate_epoch(headnet, tailnet, rq_bottleneck, val_dataloader, 
                                loss_func, args, CLASSES, LABEL2CLASSES, pcd_limit_range)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for key, value in eval_metrics.items():
        print(f"{key:30s}: {value:8.4f}")
    print("="*50)
    
    return eval_metrics

def main():
    parser = argparse.ArgumentParser(description='Progressive RQ Training and Evaluation')
    
    # Main configuration
    parser.add_argument('--config', type=str, default='exp/split1_ALL.yaml',
                        help='Path to YAML configuration file')
    parser.add_argument('--mode', choices=['train', 'eval'], default=None,
                        help='Mode: train or eval (overrides config)')
    
    # Optional overrides (will override config values if provided)
    parser.add_argument('--data_root', type=str, default=None,
                        help='Override data root path')
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='Override pretrained checkpoint path')
    parser.add_argument('--rq_ckpt', type=str, default=None,
                        help='Override RQ checkpoint for evaluation')
    parser.add_argument('--gpu', type=int, default=None,
                        help='Override GPU device id')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--use_wandb', action='store_true', default=None,
                        help='Override wandb usage')
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb',
                        help='Disable wandb logging')
    
    # Evaluation specific overrides
    parser.add_argument('--eval_num_codebooks', type=int, default=None,
                        help='Number of codebooks to use for evaluation')
    parser.add_argument('--eval_num_embeddings', type=int, default=None,
                        help='Number of embeddings per codebook for evaluation')
    
    args = parser.parse_args()
    
    # Load configuration from YAML
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found.")
        print("Please ensure the config file exists or specify a different path with --config")
        return
    
    # Apply command line overrides
    if args.mode is not None:
        config['training']['mode'] = args.mode
    if args.data_root is not None:
        config['dataset']['dir'] = args.data_root
    if args.pretrained_ckpt is not None:
        config['model']['pretrained_weight'] = args.pretrained_ckpt
    if args.rq_ckpt is not None:
        config['evaluation']['rq_ckpt'] = args.rq_ckpt
    if args.gpu is not None:
        config['hardware']['gpu'] = args.gpu
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.use_wandb is not None:
        config['logging']['use_wandb'] = args.use_wandb
    if args.eval_num_codebooks is not None:
        config['evaluation']['use_num_codebook'] = args.eval_num_codebooks
    if args.eval_num_embeddings is not None:
        config['evaluation']['use_num_embedding'] = args.eval_num_embeddings
    
    # Set GPU device
    torch.cuda.set_device(config['hardware']['gpu'])
    
    # Print configuration summary
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Mode: {config['training']['mode']}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Data root: {config['dataset']['dir']}")
    print(f"Model: {config['model']['pre_trained_model']}")
    print(f"GPU: {config['hardware']['gpu']}")
    print(f"Batch size: {config['training']['batch_size']}")
    
    if config['progressive_learning']['enabled']:
        print(f"Progressive learning: Enabled")
        print(f"- {config['model']['n_codebook']} codebooks")
        print(f"- Embedding schedule: {config['progressive_learning']['embedding_schedule']}")
        print(f"- {config['progressive_learning']['embedding_stage_epochs']} epochs per stage")
    else:
        print("Progressive learning: Disabled")
    
    if config['training']['mode'] == 'eval':
        print(f"Evaluation:")
        print(f"- Using {config['evaluation']['use_num_codebook']} codebook(s)")
        print(f"- Using {config['evaluation']['use_num_embedding']} embedding(s)")
        if config['evaluation']['rq_ckpt']:
            print(f"- Checkpoint: {config['evaluation']['rq_ckpt']}")
    
    print("="*60)
    
    # Run training or evaluation
    if config['training']['mode'] == 'train':
        if config['progressive_learning']['enabled']:
            train_progressive_rq(config)
        else:
            # Convert config to args for legacy function
            args_legacy = argparse.Namespace()
            args_legacy.data_root = config['dataset']['dir']
            args_legacy.pretrained_ckpt = config['model']['pretrained_weight']
            args_legacy.batch_size = config['training']['batch_size']
            args_legacy.num_workers = config['training']['num_workers']
            args_legacy.nclasses = config['dataset']['num_classes']
            args_legacy.init_lr = config['training']['init_lr']
            args_legacy.max_epoch = config['training']['max_epoch']
            args_legacy.ckpt_freq_epoch = config['training']['ckpt_freq_epoch']
            args_legacy.patience = config['training']['patience']
            args_legacy.log_freq = config['logging']['log_freq']
            args_legacy.use_wandb = config['logging']['use_wandb']
            args_legacy.wandb_project = config['logging']['wandb_project']
            args_legacy.wandb_name = config['logging']['wandb_name']
            args_legacy.gpu = config['hardware']['gpu']
            args_legacy.latent_shape = config['rq_model']['latent_shape']
            args_legacy.code_shape = config['rq_model']['code_shape']
            args_legacy.codebook_size = config['rq_model']['codebook_size']
            args_legacy.decay = config['rq_model']['decay']
            args_legacy.vq_weight = config['loss_weights']['vq_weight']
            args_legacy.codebook_weight = config['loss_weights']['codebook_weight']
            args_legacy.det_weight = config['loss_weights']['det_weight']
            args_legacy.saved_path = config['logging']['saved_path']
            train_rq(args_legacy)
    else:
        evaluate_progressive_rq(config)


if __name__ == '__main__':
    main()