"""
Progressive RQ Training Script for PointPillars
Clean, modular implementation with proper logging and checkpoint management
"""

import os
import time
import torch
import wandb
import argparse
from tqdm import tqdm

from pointpillars.loss import Loss
from utils.config_utils import load_config, print_config_summary
from utils.model_utils import config_to_args, setup_model_and_data, setup_progressive_stage
from fix_frozen_embeddings import add_frozen_embedding_protection
from utils.eval_utils import validate_epoch
from utils.training_utils import (
    AverageMeter, EarlyStopping, get_experiment_name, 
    setup_directories, print_epoch_summary, save_stage_checkpoint
)


def train_progressive_rq(config):
    """Progressive RQ training with clean modular structure"""
    
    # Setup experiment
    exp_name = get_experiment_name("Progressive_RQ")
    checkpoint_dir = setup_directories(config['logging']['saved_path'])
    
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
    
    # Convert config and setup model/data
    args = config_to_args(config)
    
    # Apply frozen embedding protection before creating model
    add_frozen_embedding_protection()
    print("✓ Applied enhanced frozen embedding protection")
    
    (train_dataloader, val_dataloader), headnet, tailnet, device, CLASSES, LABEL2CLASSES, pcd_limit_range = setup_model_and_data(args, mode='train')
    
    # Progressive learning configuration
    embedding_schedule = config['progressive_learning']['embedding_schedule']
    stage_epochs = config['progressive_learning']['embedding_stage_epochs'] 
    warmup_epochs = config['progressive_learning']['warmup_epochs']
    n_codebooks = config['model']['n_codebook']
    max_embed_size = max(embedding_schedule)  # Get maximum embedding size
    
    # Freeze head and tail networks
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
    print(f"- Maximum embedding size: {max_embed_size}")
    print(f"- {stage_epochs} epochs per stage, {warmup_epochs} warmup epochs")
    print(f"Experiment: {exp_name}")
    print("="*80)
    
    # Create RQ bottleneck with maximum size once
    print(f"\nCreating RQ bottleneck with maximum embedding size: {max_embed_size}")
    rq_bottleneck = setup_progressive_stage(
        args, device, n_codebooks-1, max_embed_size, embedding_schedule, ema=True,
        skip_stage_setup=True
    )
    
    # EMA initialization phase
    print(f"\nEMA Initialization Phase ({warmup_epochs} epochs)")
    print("="*60)
    for warmup_epoch in range(warmup_epochs):
        print(f"EMA Warmup Epoch {warmup_epoch + 1}/{warmup_epochs}")
        with torch.no_grad():
            for i, data_dict in enumerate(tqdm(train_dataloader, desc="EMA Init")):
                if torch.cuda.is_available():
                    for key in data_dict:
                        for j, item in enumerate(data_dict[key]):
                            if torch.is_tensor(item):
                                data_dict[key][j] = data_dict[key][j].cuda()
                
                batched_pts = data_dict['batched_pts']
                pillar_features = headnet(batched_pts)
                pillar_features_hwc = pillar_features.permute(0, 2, 3, 1)
                quantized_features, vq_loss, codebook_loss, codes = rq_bottleneck(pillar_features_hwc)
                
    
    # Switch to gradient mode after EMA initialization
    print(f"\nSwitching to gradient-based training mode")
    for codebook in rq_bottleneck.codebooks:
        if hasattr(codebook, 'use_ema'):
            codebook.use_ema = False
        if hasattr(codebook, 'trainable'):
            codebook.trainable = True
        # Enable gradients for parameters
        for param in codebook.parameters():
            param.requires_grad_(True)
    
    print("RQ bottleneck switched to gradient mode")
    
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
            
            # Configure progressive stage (adjust active embeddings) - tail-freeze only
            configure_progressive_stage(rq_bottleneck, codebook_idx, embed_size, max_embed_size)
            
            # Setup optimizer and scheduler for current stage
            optimizer, scheduler = setup_stage_optimizer(rq_bottleneck, config, 0, stage_epochs)  # No warmup needed
            
            # Stage training variables
            stage_best_mAP = 0.0
            stage_best_epoch = 0
            early_stopping = EarlyStopping(patience=config['training']['early_stopping_patience'], verbose=True)
            
            # Stage training loop
            for epoch in range(stage_epochs):
                stage_epoch = epoch + 1
                global_epoch = codebook_idx * len(embedding_schedule) * stage_epochs + stage_idx * stage_epochs + epoch + 1
                
                epoch_start_time = time.time()
                print(f'\nStage Epoch {stage_epoch}/{stage_epochs} (Global: {global_epoch})')
                print('-' * 40)
                
                # Train one epoch (simplified - no EMA warmup needed)
                train_metrics, overall_step = train_one_epoch_simplified(
                    rq_bottleneck, headnet, tailnet, train_dataloader, loss_func,
                    optimizer, scheduler, args, overall_step,
                    wandb_run, config, codebook_idx, stage_idx, embed_size, global_epoch
                )
                
                # Validation
                print("Running validation...")
                val_metrics = validate_epoch(
                    rq_bottleneck, headnet, tailnet, val_dataloader, args, device, CLASSES,
                    stage_epoch, 0, config, {'codebook_idx': codebook_idx, 'stage_idx': stage_idx, 'embed_size': embed_size}
                )
                
                epoch_time = time.time() - epoch_start_time
                current_mAP = calculate_overall_map(val_metrics)
                
                # Print epoch summary
                print_epoch_summary(global_epoch, stage_epoch, train_metrics, val_metrics, epoch_time)
                
                # Log to wandb
                if wandb_run is not None:
                    log_to_wandb(wandb_run, train_metrics, val_metrics, global_epoch, stage_epoch,
                               codebook_idx, stage_idx, embed_size, overall_step)
                
                # Update best metrics
                is_stage_best = current_mAP > stage_best_mAP
                is_global_best = current_mAP > global_best_mAP
                
                if is_stage_best:
                    stage_best_mAP = current_mAP
                    stage_best_epoch = stage_epoch
                    print(f"  *** Stage best bbox_3d mAP: {stage_best_mAP:.4f} at epoch {stage_best_epoch} ***")
                
                if is_global_best:
                    global_best_mAP = current_mAP
                    global_best_epoch = global_epoch
                    print(f"  *** Global best bbox_3d mAP: {global_best_mAP:.4f} at epoch {global_best_epoch} ***")
                    if wandb_run is not None:
                        wandb.run.summary['global_best_bbox_3d_mAP'] = global_best_mAP
                        wandb.run.summary['global_best_epoch'] = global_best_epoch
                
                # Early stopping check
                early_stopping(val_metrics['total_loss'], rq_bottleneck)
                if early_stopping.early_stop:
                    print(f"Early stopping triggered at stage epoch {stage_epoch}")
                    break
            
            # Save final checkpoint for completed stage
            if config['logging']['save_stage_weights']:
                save_stage_checkpoint(
                    rq_bottleneck, codebook_idx, stage_idx, embed_size,
                    stage_best_mAP, config, checkpoint_dir
                )
            
            print(f"\nCompleted Codebook {codebook_idx + 1}, Stage {stage_idx + 1}")
            print(f"Stage best bbox_3d mAP: {stage_best_mAP:.4f}")
            
            # Log stage completion to wandb
            if wandb_run is not None:
                log_stage_completion(wandb_run, codebook_idx, stage_idx, embed_size, 
                                   stage_best_mAP, global_epoch, val_metrics)
    
    # Training completed
    print("\n" + "="*80)
    print("PROGRESSIVE TRAINING COMPLETED!")
    print("="*80)
    print(f"Global best validation mAP: {global_best_mAP:.4f} at epoch {global_best_epoch}")
    print(f"Final experiment: {exp_name}")
    print("="*80)

    # Save final global checkpoint
    save_global_final_checkpoint(checkpoint_dir, global_best_mAP, global_best_epoch, 
                                n_codebooks, embedding_schedule, config)

    if wandb_run is not None:
        wandb.run.summary['training_completed'] = True
        wandb.run.summary['final_global_best_mAP'] = global_best_mAP
        wandb.run.summary['final_global_best_epoch'] = global_best_epoch
        wandb.finish()
    
    return global_best_mAP, global_best_epoch


def configure_progressive_stage(rq_bottleneck, codebook_idx, embed_size, max_embed_size):
    """Configure RQ bottleneck for progressive training stage (tail-freeze only).
    Tail-freeze: freeze [K:max), train [0:K).
    """
    print(f"Configuring progressive stage: codebook {codebook_idx}, embed_size {embed_size}, freeze_mode=tail")
    
    # Set active and frozen embeddings for all codebooks
    for cb_idx, codebook in enumerate(rq_bottleneck.codebooks):
        if cb_idx <= codebook_idx:
            # This codebook is active (tail-freeze): train [0:K), freeze [K:max)
            if hasattr(codebook, 'set_active_n_embed'):
                codebook.set_active_n_embed(embed_size)
            elif hasattr(codebook, 'active_n_embed'):
                codebook.active_n_embed = embed_size
            # Ensure head freeze is off
            if hasattr(codebook, 'set_frozen_n_embed'):
                codebook.set_frozen_n_embed(0)
            elif hasattr(codebook, 'frozen_n_embed'):
                codebook.frozen_n_embed = 0
            
            # Enable gradients for active embeddings
            if hasattr(codebook, 'weight') and codebook.weight is not None:
                codebook.weight.requires_grad_(True)
                # Remove previous hook (K can change stage-by-stage)
                if hasattr(codebook, '_gradient_hook_handle'):
                    try:
                        codebook._gradient_hook_handle.remove()
                    except Exception:
                        pass
                    delattr(codebook, '_gradient_hook_handle')

                # Register gradient hook for frozen part after enabling gradients
                def gradient_hook(grad, cb=codebook, K=embed_size, M=max_embed_size):
                    # Tail-freeze: zero grads for [K:M)
                    if K < M:
                        grad[K:M] = 0.0
                    return grad
                codebook._gradient_hook_handle = codebook.weight.register_hook(gradient_hook)

                # Optionally zero out inactive region weights (only for clarity, not necessary)
                with torch.no_grad():
                    # For tail-freeze: weights in [K:M) remain but won't get gradients
                    pass

            print(f"  Codebook {cb_idx}: active_n_embed={embed_size}, frozen_tail=[{embed_size}:{max_embed_size})")
        else:
            # This codebook is not active yet
            if hasattr(codebook, 'active_n_embed'):
                codebook.active_n_embed = 0
            if hasattr(codebook, 'frozen_n_embed'):
                codebook.frozen_n_embed = 0
            
            # Disable gradients for inactive codebooks
            if hasattr(codebook, 'weight') and codebook.weight is not None:
                codebook.weight.requires_grad_(False)
            
            print(f"  Codebook {cb_idx}: inactive")


def train_one_epoch_simplified(rq_bottleneck, headnet, tailnet, train_dataloader, loss_func,
                              optimizer, scheduler, args, overall_step,
                              wandb_run, config, codebook_idx, stage_idx, embed_size, global_epoch):
    """Simplified training function for one epoch (no EMA warmup)"""
    
    # Initialize meters
    train_losses = AverageMeter('TrainLoss', ':.4f')
    train_det_losses = AverageMeter('TrainDetLoss', ':.4f')
    train_vq_losses = AverageMeter('TrainVQLoss', ':.6f')
    train_cb_losses = AverageMeter('TrainCBLoss', ':.6f')
    
    rq_bottleneck.train()
    
    for i, data_dict in enumerate(tqdm(train_dataloader, desc=f"Training")):
        if torch.cuda.is_available():
            for key in data_dict:
                for j, item in enumerate(data_dict[key]):
                    if torch.is_tensor(item):
                        data_dict[key][j] = data_dict[key][j].cuda()
        
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
        
        # Calculate detection loss
        det_loss = calculate_detection_loss(
            tailnet, quantized_features, batched_gt_bboxes, 
            batched_labels, batched_pts, loss_func, args
        )
        
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
        if total_loss.requires_grad and total_loss.item() > 0:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(rq_bottleneck.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        
        overall_step += 1
        
        # Log to wandb
        if (wandb_run is not None and config['logging']['log_freq'] > 0 and 
            overall_step % config['logging']['log_freq'] == 0):
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

    # Return training metrics
    return {
        'total_loss': train_losses.avg,
        'det_loss': train_det_losses.avg,
        'vq_loss': train_vq_losses.avg,
        'cb_loss': train_cb_losses.avg,
    }, overall_step


def setup_stage_optimizer(rq_bottleneck, config, warmup_epochs, stage_epochs):
    """Setup optimizer and scheduler for current stage"""
    init_lr = config['training']['init_lr']
    
    # Check trainable parameters
    trainable_params = [p for p in rq_bottleneck.parameters() if p.requires_grad]
    print(f"Found {len(trainable_params)} trainable parameters")
    
    # Debug: show details of trainable parameters
    for i, p in enumerate(trainable_params):
        print(f"  Param {i}: shape={p.shape}, numel={p.numel()}")
    
    if len(trainable_params) == 0:
        print("No trainable parameters found (EMA mode), will create optimizer after warmup")
        return None, None
    else:
        optimizer = torch.optim.AdamW(
            params=trainable_params,
            lr=init_lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage_epochs - warmup_epochs
        )
        return optimizer, scheduler


def train_one_epoch(rq_bottleneck, headnet, tailnet, train_dataloader, loss_func,
                   optimizer, scheduler, args, epoch, warmup_epochs, overall_step,
                   wandb_run, config, codebook_idx, stage_idx, embed_size, global_epoch):
    """Train one epoch with proper EMA handling"""
    
    # Initialize meters
    train_losses = AverageMeter('TrainLoss', ':.4f')
    train_det_losses = AverageMeter('TrainDetLoss', ':.4f')
    train_vq_losses = AverageMeter('TrainVQLoss', ':.6f')
    train_cb_losses = AverageMeter('TrainCBLoss', ':.6f')
    
    # EMA warmup epochs
    if epoch < warmup_epochs:
        print(f"EMA Warmup Epoch {epoch + 1}/{warmup_epochs}")
        rq_bottleneck.train()
        
        with torch.no_grad():
            for i, data_dict in enumerate(tqdm(train_dataloader, desc="EMA Init")):
                if torch.cuda.is_available():
                    for key in data_dict:
                        for j, item in enumerate(data_dict[key]):
                            if torch.is_tensor(item):
                                data_dict[key][j] = data_dict[key][j].cuda()
                
                batched_pts = data_dict['batched_pts']
                pillar_features = headnet(batched_pts)
                pillar_features_hwc = pillar_features.permute(0, 2, 3, 1)
                quantized_features, vq_loss, codebook_loss, codes = rq_bottleneck(pillar_features_hwc)
    
    # Switch to gradient training after warmup
    if epoch == warmup_epochs and optimizer is None:
        print("Switching to gradient-based training")
        
        # Create new RQ model without EMA
        from utils.model_utils import setup_progressive_stage
        new_rq_bottleneck = setup_progressive_stage(
            args, torch.cuda.current_device(), codebook_idx, embed_size, 
            config['progressive_learning']['embedding_schedule'], ema=False
        )
        
        # Transfer EMA weights
        transfer_ema_weights(rq_bottleneck, new_rq_bottleneck)
        rq_bottleneck = new_rq_bottleneck
        
        # Ensure codebooks are set to trainable mode
        for i, codebook in enumerate(rq_bottleneck.codebooks):
            codebook.use_ema = False
            codebook.trainable = True
            print(f"Codebook {i}: use_ema={codebook.use_ema}, trainable={codebook.trainable}, weight.requires_grad={codebook.weight.requires_grad}")
        
        # Create optimizer
        trainable_params = [p for p in rq_bottleneck.parameters() if p.requires_grad]
        print(f"Found {len(trainable_params)} trainable parameters after switching to gradient mode")
        
        if len(trainable_params) > 0:
            optimizer = torch.optim.AdamW(
                params=trainable_params,
                lr=config['training']['init_lr'],
                betas=(0.9, 0.999), 
                weight_decay=0.01
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config['progressive_learning']['embedding_stage_epochs'] - warmup_epochs
            )
        else:
            print("ERROR: No trainable parameters found even in gradient mode!")

    # Gradient training
    if epoch >= warmup_epochs and optimizer is not None:
        rq_bottleneck.train()
        
        for i, data_dict in enumerate(tqdm(train_dataloader, desc=f"Training")):
            if torch.cuda.is_available():
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()
            
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
            
            # Calculate detection loss
            det_loss = calculate_detection_loss(
                tailnet, quantized_features, batched_gt_bboxes, 
                batched_labels, batched_pts, loss_func, args
            )
            
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
            if total_loss.requires_grad and total_loss.item() > 0:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(rq_bottleneck.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            
            overall_step += 1
            
            # Log to wandb
            if (wandb_run is not None and config['logging']['log_freq'] > 0 and 
                overall_step % config['logging']['log_freq'] == 0):
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

    # Handle case where we're past warmup but don't have optimizer
    elif epoch >= warmup_epochs and optimizer is None:
        print(f"WARNING: Past warmup epoch ({epoch} >= {warmup_epochs}) but no optimizer available!")
        print("This suggests the gradient training setup failed.")

    # Return training metrics, and potentially updated models/optimizers
    return {
        'total_loss': train_losses.avg,
        'det_loss': train_det_losses.avg,
        'vq_loss': train_vq_losses.avg,
        'cb_loss': train_cb_losses.avg,
    }, overall_step, rq_bottleneck, optimizer, scheduler


def calculate_detection_loss(tailnet, quantized_features, batched_gt_bboxes, 
                           batched_labels, batched_pts, loss_func, args):
    """Calculate detection loss following original PointPillars logic"""
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


def transfer_ema_weights(ema_model, gradient_model):
    """Transfer EMA weights to gradient-based model"""
    with torch.no_grad():
        for i, (old_codebook, new_codebook) in enumerate(zip(ema_model.codebooks, gradient_model.codebooks)):
            if hasattr(old_codebook, 'sync_ema_weights'):
                old_codebook.sync_ema_weights()
            source_weight = old_codebook.weight.detach()
            if source_weight.device != new_codebook.weight.device:
                source_weight = source_weight.to(new_codebook.weight.device)
            new_codebook.weight.data.copy_(source_weight)
            new_codebook.set_active_n_embed(old_codebook.active_n_embed)
            new_codebook.set_frozen_n_embed(old_codebook.frozen_n_embed)


def calculate_overall_map(val_metrics):
    """Calculate overall mAP from bbox_3d metrics"""
    # Handle different possible formats
    if 'overall_map' in val_metrics:
        return val_metrics['overall_map']
    elif 'bbox_3d_ap' in val_metrics:
        bbox_3d_ap = val_metrics['bbox_3d_ap']
        if isinstance(bbox_3d_ap, (list, tuple)) and len(bbox_3d_ap) >= 3:
            return sum(bbox_3d_ap) / len(bbox_3d_ap)
        else:
            return float(bbox_3d_ap) if bbox_3d_ap else 0.0
    else:
        # Fallback: try to find individual difficulty metrics
        return (val_metrics.get('bbox_3d_difficulty_0_mAP', 0.0) + 
                val_metrics.get('bbox_3d_difficulty_1_mAP', 0.0) + 
                val_metrics.get('bbox_3d_difficulty_2_mAP', 0.0)) / 3.0


def log_to_wandb(wandb_run, train_metrics, val_metrics, global_epoch, stage_epoch,
                codebook_idx, stage_idx, embed_size, overall_step):
    """Log metrics to wandb"""
    wandb_log_data = {
        'epoch': global_epoch,
        'stage_epoch': stage_epoch,
        'codebook_idx': codebook_idx,
        'stage_idx': stage_idx,
        'embed_size': embed_size,
        # Training metrics
        'train/total_loss': float(train_metrics['total_loss']),
        'train/det_loss': float(train_metrics['det_loss']),
        'train/vq_loss': float(train_metrics['vq_loss']),
        'train/codebook_loss': float(train_metrics['cb_loss']),
        # Loss metrics
        'val/total_loss': float(val_metrics.get('total_loss', 0.0)),
        'val/det_loss': float(val_metrics.get('det_loss', 0.0)),
        'val/vq_loss': float(val_metrics.get('vq_loss', 0.0)),
        'val/cb_loss': float(val_metrics.get('cb_loss', 0.0)),
    }
    
    # Log all evaluation metrics
    for key, value in val_metrics.items():
        if any(key.startswith(eval_type) for eval_type in ['bbox_2d', 'bbox_bev', 'bbox_3d']) or key == 'overall_mAP':
            # Handle list values (like bbox_3d_ap which might be [easy, mod, hard])
            if isinstance(value, (list, tuple)):
                if len(value) == 3:  # bbox_3d_ap format [easy, moderate, hard]
                    wandb_log_data[f'val/{key}_easy'] = float(value[0])
                    wandb_log_data[f'val/{key}_moderate'] = float(value[1])
                    wandb_log_data[f'val/{key}_hard'] = float(value[2])
                    wandb_log_data[f'val/{key}_mean'] = float(sum(value) / len(value))
                else:
                    # For other list formats, just take the mean
                    wandb_log_data[f'val/{key}'] = float(sum(value) / len(value))
            else:
                wandb_log_data[f'val/{key}'] = float(value)
    
    wandb.log(wandb_log_data, step=overall_step)


def log_stage_completion(wandb_run, codebook_idx, stage_idx, embed_size, 
                        stage_best_mAP, global_epoch, final_val_metrics):
    """Log stage completion to wandb"""
    stage_completion_data = {
        f'stage_completion/codebook_{codebook_idx}_stage_{stage_idx}_best_bbox_3d_mAP': stage_best_mAP,
        f'stage_completion/codebook_{codebook_idx}_stage_{stage_idx}_embed_size': embed_size,
        f'stage_completion/codebook_{codebook_idx}_stage_{stage_idx}_global_epoch': global_epoch,
    }
    
    # Log final metrics for this stage
    for key, value in final_val_metrics.items():
        if any(key.startswith(eval_type) for eval_type in ['bbox_2d', 'bbox_bev', 'bbox_3d']) or key == 'overall_mAP':
            stage_completion_data[f'stage_completion/codebook_{codebook_idx}_stage_{stage_idx}_{key}'] = float(value)
    
    wandb.log(stage_completion_data, step=global_epoch)
    print(f"  Logged stage completion metrics to wandb")


def save_global_final_checkpoint(checkpoint_dir, global_best_mAP, global_best_epoch, 
                                n_codebooks, embedding_schedule, config):
    """Save final global checkpoint"""
    final_checkpoint = {
        'global_best_mAP': global_best_mAP,
        'global_best_epoch': global_best_epoch,
        'config': config
    }
    global_final_name = f'global_cb{n_codebooks}_{embedding_schedule[-1]}embeds_final.pth'
    torch.save(final_checkpoint, os.path.join(checkpoint_dir, global_final_name))
    print(f"Saved global final checkpoint: {global_final_name}")


def main():
    """Main entry point for progressive RQ training"""
    parser = argparse.ArgumentParser(description='Progressive RQ Training for PointPillars')
    
    parser.add_argument('--config', type=str, default='exp/split1_ALL.yaml',
                        help='Path to YAML configuration file')
    parser.add_argument('--mode', choices=['train', 'eval'], default=None,
                        help='Mode: train or eval (overrides config)')
    
    # Optional overrides
    parser.add_argument('--data_root', type=str, default=None,
                        help='Override data root path')
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='Override pretrained checkpoint path')
    parser.add_argument('--gpu', type=int, default=None,
                        help='Override GPU device id')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--use_wandb', action='store_true', default=None,
                        help='Override wandb usage')
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb',
                        help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # Load and apply overrides to config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found.")
        return
    
    # Apply command line overrides
    if args.mode is not None:
        config['training']['mode'] = args.mode
    if args.data_root is not None:
        config['dataset']['dir'] = args.data_root
    if args.pretrained_ckpt is not None:
        config['model']['pretrained_weight'] = args.pretrained_ckpt
    if args.gpu is not None:
        config['hardware']['gpu'] = args.gpu
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.use_wandb is not None:
        config['logging']['use_wandb'] = args.use_wandb
    
    # Set GPU device
    torch.cuda.set_device(config['hardware']['gpu'])
    
    # Print configuration summary
    print_config_summary(config)
    
    # Run training
    if config['training']['mode'] == 'train' and config['progressive_learning']['enabled']:
        train_progressive_rq(config)
    else:
        print("Error: This script only supports progressive RQ training mode.")
        print("Please set mode='train' and progressive_learning.enabled=True in your config.")


if __name__ == '__main__':
    main()