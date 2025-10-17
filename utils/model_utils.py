"""
Model setup and data loading utilities for PointPillars RQ training
"""

import torch
import numpy as np
import argparse
from pointpillars.dataset import Kitti, get_dataloader
from pointpillars.model import PointPillars
from pointpillars.model.split_nets import split_pointpillars
from pointpillars.model.quantizations import RQBottleneck
from pointpillars.utils import setup_seed


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
    
    return args


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