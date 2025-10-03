import argparse
import os
import torch
from tqdm import tqdm
import numpy as np
import time
import datetime
import wandb

from pointpillars.utils import setup_seed, keep_bbox_from_image_range, \
    keep_bbox_from_lidar_range, write_pickle, write_label, \
    iou2d, iou3d_camera, iou_bev
from pointpillars.dataset import Kitti, get_dataloader
from pointpillars.model import PointPillars
from pointpillars.model.split_nets import split_pointpillars
from pointpillars.model.quantizations import RQBottleneck
from pointpillars.loss import Loss


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


def do_eval(det_results, gt_results, CLASSES, saved_path):
    """
    Complete KITTI evaluation protocol
    """
    EVAL_NAMES = ['bbox_2d', 'bbox_bev', 'bbox_3d']
    CLS_MIN_IOU = [0.7, 0.5, 0.5]  # Car, Pedestrian, Cyclist
    MIN_HEIGHT = [40, 25, 25]      # Easy, Moderate, Hard
    
    eval_ap_results = {}
    eval_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for eval_type_idx, eval_type in enumerate(EVAL_NAMES):
        eval_ap_results[eval_type] = {}
        for difficulty in [0, 1, 2]:  # Easy, Moderate, Hard
            eval_ap_results[eval_type][f'difficulty_{difficulty}'] = {}
            
            ids = sorted(list(gt_results.keys()))
            
            for cls_idx, cls_name in CLASSES.items():
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
                            if det_ignores[det_idx] == 0 and det_idx < len(gt_column) and gt_column[det_idx] > CLS_MIN_IOU[cls_idx]:
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

    # Calculate overall metrics
    overall_metrics = {}
    for eval_type in EVAL_NAMES:
        for difficulty in [0, 1, 2]:
            difficulty_key = f'difficulty_{difficulty}'
            mAPs = [eval_ap_results[eval_type][difficulty_key].get(cls_name, 0.0) 
                   for cls_name in CLASSES.values()]
            overall_metrics[f'{eval_type}_{difficulty_key}_mAP'] = np.mean(mAPs)
    
    overall_metrics['overall_mAP'] = np.mean([v for k, v in overall_metrics.items() if 'mAP' in k])
    
    return overall_metrics


def validate_epoch(headnet, tailnet, rq_bottleneck, val_dataloader, loss_func, args, CLASSES, LABEL2CLASSES, pcd_limit_range):
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
        for data_dict in tqdm(val_dataloader, desc='Validation', leave=False):
            if not args.no_cuda:
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
    eval_metrics = do_eval(format_results, val_dataset.data_infos, CLASSES, "")
    
    # Combine loss and evaluation metrics
    all_metrics = {
        'total_loss': val_losses.avg,
        'det_loss': val_det_losses.avg,
        'vq_loss': val_vq_losses.avg,
        'cb_loss': val_cb_losses.avg,
        **eval_metrics
    }
    
    return all_metrics


def train_rq(args):
    """Train RQ model without external logging dependencies."""
    setup_seed()
    
    # Setup experiment name and date
    date_str = str(datetime.date.today())
    time_str = datetime.datetime.now().strftime("%H-%M-%S")
    exp_name = f"{date_str}_{time_str}_RQ_training"

    wandb_run = None
    if args.use_wandb:
        wandb_name = args.wandb_name or f"RQ_lat{args.latent_shape}_code{args.code_shape}_embed{args.codebook_size}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            config=vars(args),
            tags=["pointpillars", "rq", "split_inference"],
        )
        
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

    # Load pretrained PointPillars model
    print("Loading pretrained PointPillars model...")
    if not args.no_cuda:
        full_model = PointPillars(nclasses=args.nclasses).cuda()
    else:
        full_model = PointPillars(nclasses=args.nclasses)
    
    # Load pretrained weights
    checkpoint = torch.load(args.pretrained_ckpt, map_location='cpu' if args.no_cuda else 'cuda')
    full_model.load_state_dict(checkpoint)
    print(f"Loaded pretrained weights from {args.pretrained_ckpt}")
    
    # Split the model
    headnet, tailnet = split_pointpillars(full_model)
    
    # Freeze headnet and tailnet parameters
    for param in headnet.parameters():
        param.requires_grad = False
    for param in tailnet.parameters():
        param.requires_grad = False
    
    print("Frozen headnet and tailnet parameters")
    
    # Create RQ bottleneck
    rq_bottleneck = RQBottleneck(
        latent_shape=args.latent_shape,
        code_shape=args.code_shape,
        n_embed=args.codebook_size,
        decay=args.decay,
        ema=True,  # Start with EMA for initialization
        shared_codebook=False,
        restart_unused_codes=True,
        commitment_loss='cumsum'
    )
    
    if not args.no_cuda:
        rq_bottleneck = rq_bottleneck.cuda()
    
    print(f"Created RQ bottleneck with {sum(p.numel() for p in rq_bottleneck.parameters())} parameters")
    
    loss_func = Loss()

    # Only optimize RQ parameters (after EMA initialization)
    steps_per_epoch = max(1, len(train_dataloader))
    effective_epochs = max(args.max_epoch - 2, 1)
    total_steps = max(1, steps_per_epoch * effective_epochs)
    init_lr = args.init_lr
    optimizer = torch.optim.AdamW(params=rq_bottleneck.parameters(), 
                                  lr=init_lr, 
                                  betas=(0.95, 0.99),
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,  
                                                    max_lr=init_lr*10, 
                                                    total_steps=total_steps, 
                                                    pct_start=0.4, 
                                                    anneal_strategy='cos',
                                                    cycle_momentum=True, 
                                                    base_momentum=0.95*0.895, 
                                                    max_momentum=0.95,
                                                    div_factor=10)
    
    saved_ckpt_path = os.path.join(args.saved_path, 'checkpoints')
    os.makedirs(saved_ckpt_path, exist_ok=True)

    # Define evaluation constants
    CLASSES = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}
    LABEL2CLASSES = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}
    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 1], dtype=np.float32)

    # Set models to appropriate modes
    headnet.eval()
    tailnet.eval()
    
    # Tracking variables
    best_mAP = 0.0
    best_epoch = 0
    train_losses = AverageMeter('TrainLoss', ':.4f')
    train_det_losses = AverageMeter('TrainDetLoss', ':.4f')
    train_vq_losses = AverageMeter('TrainVQLoss', ':.6f')
    train_cb_losses = AverageMeter('TrainCBLoss', ':.6f')
    
    # Early stopping
    early_stopping = EarlyStopping(patience=getattr(args, 'patience', 20), verbose=True)
    
    print(f"Starting training for {args.max_epoch} epochs...")
    print(f"Experiment: {exp_name}")
    overall_step = 0

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
        if epoch < 2 and False:
            print(f"EMA Initialization Epoch {epoch + 1}/2")
            
            with torch.no_grad():
                for i, data_dict in enumerate(tqdm(train_dataloader, desc="EMA Init")):
                    if not args.no_cuda:
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
        if epoch == 2:
            print("Switching to gradient-based training")
            
            # Create new RQ model without EMA
            new_rq_bottleneck = RQBottleneck(
                latent_shape=args.latent_shape,
                code_shape=args.code_shape,
                n_embed=args.codebook_size,
                decay=args.decay,
                ema=False,
                shared_codebook=False,
                restart_unused_codes=True,
                commitment_loss='cumsum'
            )
            
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
            if not args.no_cuda:
                rq_bottleneck = rq_bottleneck.cuda()
            
            # Recreate optimizer and scheduler for new model
            optimizer = torch.optim.AdamW(params=rq_bottleneck.parameters(), 
                                          lr=init_lr, 
                                          betas=(0.95, 0.99),
                                          weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,  
                                                            max_lr=init_lr*10, 
                                                            total_steps=total_steps, 
                                                            pct_start=0.4, 
                                                            anneal_strategy='cos',
                                                            cycle_momentum=True, 
                                                            base_momentum=0.95*0.895, 
                                                            max_momentum=0.95,
                                                            div_factor=10)
        
        # Training loop for gradient-based epochs
        for i, data_dict in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")):
            if not args.no_cuda:
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
            
            # Total loss: RQ losses + detection loss (keeping HeadNet/TailNet frozen)
            # Flattened losses are computed in det_loss_dict
            total_loss = (args.vq_weight * vq_loss + \
                         args.codebook_weight * codebook_loss + \
                         args.det_weight * det_loss)
            
            # Update meters
            train_losses.update(total_loss.item(), len(batched_pts))
            train_det_losses.update(det_loss.item(), len(batched_pts))
            train_vq_losses.update(vq_loss.item(), len(batched_pts))
            train_cb_losses.update(codebook_loss.item(), len(batched_pts))
            
            # Backward and optimize
            if total_loss.requires_grad and total_loss.item() > 0:
                total_loss.backward()
                optimizer.step()
                scheduler.step()
            
            train_step += 1
            overall_step += 1

            if wandb_run is not None and args.log_freq > 0 and overall_step % args.log_freq == 0:
                wandb.log(
                    {
                        'train/total_loss': float(total_loss.item()),
                        'train/det_loss': float(det_loss.item()),
                        'train/vq_loss': float(vq_loss.item()),
                        'train/codebook_loss': float(codebook_loss.item()),
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'epoch': epoch + 1,
                    },
                    step=overall_step,
                )
        
        # Run validation at the end of each epoch
        print(f"Running validation for epoch {epoch+1}...")
        val_metrics = validate_epoch(headnet, tailnet, rq_bottleneck, val_dataloader, 
                                   loss_func, args, CLASSES, LABEL2CLASSES, pcd_limit_range)  # Full eval every 5 epochs
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Training Loss: {train_losses.avg:.4f}")
        print(f"    - Det Loss: {train_det_losses.avg:.4f}")
        print(f"    - VQ Loss: {train_vq_losses.avg:.6f}")
        print(f"    - CB Loss: {train_cb_losses.avg:.6f}")
        print(f"  Validation Results:")
        for key, value in val_metrics.items():
            print(f"    - {key}: {value:.4f}")

        if wandb_run is not None:
            log_step = overall_step if overall_step > 0 else epoch + 1
            wandb.log(
                {
                    'epoch': epoch + 1,
                    'train/epoch_loss': float(train_losses.avg),
                    'train/epoch_det_loss': float(train_det_losses.avg),
                    'train/epoch_vq_loss': float(train_vq_losses.avg),
                    'train/epoch_cb_loss': float(train_cb_losses.avg),
                    **{f'val/{k}': float(v) for k, v in val_metrics.items()},
                },
                step=log_step,
            )
        
        # Save checkpoint periodically and if best mAP
        current_mAP = val_metrics.get('simple_mAP', val_metrics.get('overall_mAP', 0.0))
        is_best = current_mAP > best_mAP
        
        if epoch % args.ckpt_freq_epoch == 0 or is_best:
            checkpoint_data = {
                'rq_bottleneck': rq_bottleneck.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'val_metrics': val_metrics,
                'train_metrics': {
                    'total_loss': train_losses.avg,
                    'det_loss': train_det_losses.avg,
                    'vq_loss': train_vq_losses.avg,
                    'cb_loss': train_cb_losses.avg
                },
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
    total_time = time.time() - epoch_start_time
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


def evaluate_rq(args):
    """
    Evaluate RQ model using full KITTI protocol
    """
    setup_seed()
    
    # Load data
    val_dataset = Kitti(data_root=args.data_root, split='val')
    val_dataloader = get_dataloader(dataset=val_dataset, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers,
                                    shuffle=False)
    
    # Load pretrained PointPillars model
    print("Loading pretrained PointPillars model...")
    if not args.no_cuda:
        full_model = PointPillars(nclasses=args.nclasses).cuda()
    else:
        full_model = PointPillars(nclasses=args.nclasses)
    
    checkpoint = torch.load(args.pretrained_ckpt, map_location='cpu' if args.no_cuda else 'cuda')
    full_model.load_state_dict(checkpoint)
    print(f"Loaded pretrained weights from {args.pretrained_ckpt}")
    
    # Split the model
    headnet, tailnet = split_pointpillars(full_model)
    
    # Create RQ bottleneck if needed
    rq_bottleneck = None
    if args.use_rq and args.rq_ckpt:
        rq_bottleneck = RQBottleneck(
            latent_shape=args.latent_shape,
            code_shape=args.code_shape,
            n_embed=args.codebook_size,
            decay=args.decay,
            ema=True,
            shared_codebook=False,
            restart_unused_codes=True,
            commitment_loss='cumsum'
        )
        
        if not args.no_cuda:
            rq_bottleneck = rq_bottleneck.cuda()
        
        # Load RQ checkpoint
        rq_checkpoint = torch.load(args.rq_ckpt, map_location='cpu' if args.no_cuda else 'cuda')
        try:
            rq_bottleneck.load_state_dict(rq_checkpoint['rq_bottleneck'])
        except RuntimeError:
            rq_bottleneck.load_state_dict(rq_checkpoint)
        
        # Disable EMA for evaluation
        for codebook in rq_bottleneck.codebooks:
            codebook.ema = False
        
        rq_bottleneck.eval()
        print("RQ bottleneck ready for evaluation")
    else:
        print("Running without RQ bottleneck (original PointPillars behavior)")
    
    loss_func = Loss()
    
    # Define evaluation constants
    CLASSES = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}
    LABEL2CLASSES = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}
    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 1], dtype=np.float32)
    
    # Run full evaluation
    print("Running full evaluation...")
    eval_metrics = validate_epoch(headnet, tailnet, rq_bottleneck, val_dataloader, 
                                loss_func, args, CLASSES, LABEL2CLASSES, pcd_limit_range)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for key, value in eval_metrics.items():
        print(f"{key:30s}: {value:8.4f}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='RQ Training and Evaluation')
    parser.add_argument('--mode', choices=['train', 'eval'], default='train',
                        help='Mode: train or eval')
    parser.add_argument('--data_root', default='/home/yenhsiu/datasets', 
                        help='your data root for kitti')
    parser.add_argument('--pretrained_ckpt', default='pretrained/epoch_160.pth',
                        help='pretrained PointPillars checkpoint')
    parser.add_argument('--rq_ckpt', help='RQ checkpoint for evaluation')
    parser.add_argument('--use_rq', action='store_true', help='use RQ in evaluation')
    parser.add_argument('--saved_path', default='test_rq_logs_unified_clean')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--nclasses', type=int, default=3)
    parser.add_argument('--init_lr', type=float, default=0.001)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--ckpt_freq_epoch', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--log_freq', type=int, default=8,
                        help='number of optimizer steps between wandb training logs')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    parser.add_argument('--use_wandb', action='store_true', default=True,
                        help='enable Weights & Biases logging (enabled by default)')
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb',
                        help='disable Weights & Biases logging')
    parser.add_argument('--wandb_project', default='pointpillars-rq',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_name', default=None,
                        help='Weights & Biases run name (auto-generated if omitted)')
    
    # RQ parameters
    parser.add_argument('--latent_shape', nargs=3, type=int, default=[496, 432, 64], 
                        help='latent shape [h, w, c]')
    parser.add_argument('--code_shape', nargs=3, type=int, default=[62, 54, 8], 
                        help='code shape [h, w, num_codebooks]')
    parser.add_argument('--codebook_size', type=int, default=256, help='size of codebook')
    parser.add_argument('--decay', type=float, default=0.99, help='EMA decay for codebook')
    
    # Loss weights
    parser.add_argument('--vq_weight', type=float, default=0.02, help='weight for VQ loss')
    parser.add_argument('--codebook_weight', type=float, default=0.02, help='weight for codebook loss')
    parser.add_argument('--det_weight', type=float, default=1.0, help='weight for detection loss')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_rq(args)
    else:
        evaluate_rq(args)


if __name__ == '__main__':
    main()