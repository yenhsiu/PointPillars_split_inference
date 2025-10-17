"""
Progressive RQ Evaluation Script for PointPillars
Evaluates models trained with progressive residual quantization
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.config_utils import load_config, print_config_summary
from utils.model_utils import config_to_args, setup_model_and_data
from utils.eval_utils import validate_epoch, analyze_checkpoint
from fix_frozen_embeddings import add_frozen_embedding_protection


def safe_load_state_dict(model, state_dict):
    """Load state dict safely, handling dimension mismatches by reshaping or ignoring"""
    model_dict = model.state_dict()
    
    # Check for shape mismatches and handle them appropriately
    compatible_dict = {}
    shape_mismatch = False
    missing_keys = []
    
    for k, v in state_dict.items():
        if k in model_dict:
            # If shapes don't match, we need to handle specially
            if v.shape != model_dict[k].shape:
                shape_mismatch = True
                print(f"Shape mismatch for {k}: checkpoint={v.shape}, model={model_dict[k].shape}")
                
                # For codebook weights, try to adapt if possible
                if 'codebooks.' in k and '.weight' in k:
                    # Extract codebook number and current dimensions
                    ckpt_embed_size, ckpt_dim = v.shape
                    model_embed_size, model_dim = model_dict[k].shape
                    
                    # If checkpoint has more embeddings, truncate
                    if ckpt_embed_size > model_embed_size:
                        compatible_dict[k] = v[:model_embed_size, :model_dim]
                        print(f"  Truncating codebook to first {model_embed_size} embeddings")
                    # If model has more embeddings, use checkpoint embeddings and initialize rest
                    else:
                        new_weight = torch.zeros_like(model_dict[k])
                        # Copy available embeddings from checkpoint
                        new_weight[:ckpt_embed_size, :min(ckpt_dim, model_dim)] = v[:, :min(ckpt_dim, model_dim)]
                        # Initialize remaining embeddings if any
                        if ckpt_embed_size < model_embed_size:
                            torch.nn.init.normal_(new_weight[ckpt_embed_size:, :], mean=0.0, std=0.02)
                            print(f"  Initializing {model_embed_size - ckpt_embed_size} new embeddings")
                        compatible_dict[k] = new_weight
            else:
                compatible_dict[k] = v
        else:
            missing_keys.append(k)
    
    # Load the compatible weights
    model.load_state_dict(compatible_dict, strict=False)
    
    # Report on loading results
    if missing_keys:
        print(f"Ignored {len(missing_keys)} keys from checkpoint that don't match model")
    if shape_mismatch:
        print("Shape mismatches were handled by adapting weights")
    return model


def load_rq_checkpoint(rq_bottleneck, checkpoint_path, device):
    """Load RQ checkpoint safely with comprehensive error handling"""
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            rq_checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = rq_checkpoint.get('rq_bottleneck', rq_checkpoint)
            
            # Extract checkpoint information for better logging
            n_codebooks = sum(1 for k in state_dict.keys() if k.startswith('codebooks.') and k.endswith('.weight'))
            if n_codebooks > 0:
                first_codebook_key = f'codebooks.0.weight'
                if first_codebook_key in state_dict:
                    codebook_shape = state_dict[first_codebook_key].shape
                    print(f"Checkpoint has {n_codebooks} codebook(s), first codebook shape: {codebook_shape}")
            
            # Use safe loading to handle mismatches
            safe_load_state_dict(rq_bottleneck, state_dict)
            rq_bottleneck.eval()
            print(f"Successfully loaded RQ checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Using initialized model instead")
    else:
        print(f"Warning: Checkpoint path {checkpoint_path} does not exist")
    
    return rq_bottleneck


def analyze_checkpoint(checkpoint_path, device):
    """Analyze checkpoint to extract configuration information"""
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
        ckpt_state = ckpt.get('rq_bottleneck', ckpt)
        
        # Analyze checkpoint structure to determine actual codebook sizes
        codebook_keys = [k for k in ckpt_state.keys() if k.startswith('codebooks.') and k.endswith('.weight')]
        if not codebook_keys:
            return None, None
            
        max_codebook_idx = max([int(k.split('.')[1]) for k in codebook_keys])
        ckpt_num_codebooks = max_codebook_idx + 1
        
        # Get embedding size from first codebook
        if 'codebooks.0.weight' in ckpt_state:
            first_codebook = ckpt_state['codebooks.0.weight']
            ckpt_num_embeddings = first_codebook.shape[0]
            return ckpt_num_codebooks, ckpt_num_embeddings
            
    except Exception as e:
        print(f"Error analyzing checkpoint: {e}")
        return None, None


def evaluate_progressive_rq(config):
    """Evaluation function with progressive RQ configuration support"""
    # Apply frozen embedding protection before creating model
    add_frozen_embedding_protection()
    print("✓ Applied enhanced frozen embedding protection")
    
    # Convert config to args-like object for compatibility
    args = config_to_args(config)
    
    # Setup model and data
    val_dataloader, headnet, tailnet, device, CLASSES, LABEL2CLASSES, pcd_limit_range = setup_model_and_data(args, mode='eval')
    
    # Create RQ bottleneck based on evaluation configuration
    rq_bottleneck = None
    if config['evaluation']['rq_ckpt'] is not None:
        eval_config = config['evaluation']
        checkpoint_path = eval_config['rq_ckpt']
        
        # Analyze checkpoint to get actual structure
        ckpt_num_codebooks, ckpt_num_embeddings = analyze_checkpoint(checkpoint_path, device)
        
        if ckpt_num_codebooks is not None and ckpt_num_embeddings is not None:
            print(f"Checkpoint analysis: found {ckpt_num_codebooks} codebooks with {ckpt_num_embeddings} embeddings")
            
            # Use checkpoint values if user specified different/incompatible values
            use_num_codebooks = eval_config['use_num_codebook']
            use_num_embeddings = eval_config['use_num_embedding']
            
            if use_num_codebooks > ckpt_num_codebooks:
                print(f"Warning: Requested {use_num_codebooks} codebooks but checkpoint only has {ckpt_num_codebooks}")
                print(f"Setting codebook count to match checkpoint: {ckpt_num_codebooks}")
                use_num_codebooks = ckpt_num_codebooks
        else:
            # Fallback if unable to analyze checkpoint
            use_num_codebooks = eval_config['use_num_codebook']
            use_num_embeddings = eval_config['use_num_embedding']
            print(f"Unable to analyze checkpoint structure, using specified values")
        
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
        
        # Set evaluation stage if method exists
        if hasattr(rq_bottleneck, 'set_evaluation_stage'):
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


def main():
    """Main entry point for progressive RQ evaluation"""
    parser = argparse.ArgumentParser(description='Progressive RQ Evaluation for PointPillars')
    
    parser.add_argument('--config', type=str, default='exp/split1_EVAL.yaml',
                        help='Path to YAML configuration file')
    
    # Optional overrides
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
    parser.add_argument('--eval_num_codebooks', type=int, default=None,
                        help='Number of codebooks to use for evaluation')
    parser.add_argument('--eval_num_embeddings', type=int, default=None,
                        help='Number of embeddings per codebook for evaluation')
    
    args = parser.parse_args()
    
    # Load and apply overrides to config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found.")
        return
    
    # Apply command line overrides
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
    if args.eval_num_codebooks is not None:
        config['evaluation']['use_num_codebook'] = args.eval_num_codebooks
    if args.eval_num_embeddings is not None:
        config['evaluation']['use_num_embedding'] = args.eval_num_embeddings
    
    # Set GPU device
    torch.cuda.set_device(config['hardware']['gpu'])
    
    # Print configuration summary
    print_config_summary(config)
    
    # Run evaluation
    evaluate_progressive_rq(config)


if __name__ == '__main__':
    main()