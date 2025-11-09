"""
Checkpoint verification tool for Progressive RQ training
Helps verify that frozen embeddings are preserved correctly across stages
"""

import torch
import numpy as np
import argparse
import os


def load_checkpoint_weights(checkpoint_path):
    """Load RQ bottleneck weights from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'rq_bottleneck' in checkpoint:
            return checkpoint['rq_bottleneck'], checkpoint
        else:
            return checkpoint, checkpoint
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None, None


def extract_codebook_weights(state_dict):
    """Extract codebook weights from state dict"""
    codebook_weights = {}
    for key, value in state_dict.items():
        if key.startswith('codebooks.') and key.endswith('.weight'):
            codebook_idx = int(key.split('.')[1])
            codebook_weights[codebook_idx] = value.detach().cpu().numpy()
    return codebook_weights


def compare_frozen_embeddings(ckpt1_path, ckpt2_path, frozen_size=16):
    """Compare frozen embeddings between two checkpoints"""
    print(f"\nComparing frozen embeddings between:")
    print(f"  Checkpoint 1: {ckpt1_path}")
    print(f"  Checkpoint 2: {ckpt2_path}")
    print(f"  Expected frozen size: {frozen_size}")
    print("-" * 60)
    
    # Load checkpoints
    weights1, info1 = load_checkpoint_weights(ckpt1_path)
    weights2, info2 = load_checkpoint_weights(ckpt2_path)
    
    if weights1 is None or weights2 is None:
        print("Failed to load one or both checkpoints")
        return False
    
    # Extract codebook weights
    codebooks1 = extract_codebook_weights(weights1)
    codebooks2 = extract_codebook_weights(weights2)
    
    print(f"Checkpoint 1 has {len(codebooks1)} codebook(s)")
    print(f"Checkpoint 2 has {len(codebooks2)} codebook(s)")
    
    # Print checkpoint info if available
    if 'codebook_idx' in info1:
        print(f"Checkpoint 1: Codebook {info1['codebook_idx']}, Stage {info1.get('stage_idx', 'N/A')}, Embed size {info1.get('embed_size', 'N/A')}")
    if 'codebook_idx' in info2:
        print(f"Checkpoint 2: Codebook {info2['codebook_idx']}, Stage {info2.get('stage_idx', 'N/A')}, Embed size {info2.get('embed_size', 'N/A')}")
    
    print()
    
    # Compare common codebooks
    all_match = True
    common_codebooks = set(codebooks1.keys()) & set(codebooks2.keys())
    
    if not common_codebooks:
        print("No common codebooks found between checkpoints")
        return False
    
    for codebook_idx in sorted(common_codebooks):
        weights_1 = codebooks1[codebook_idx]
        weights_2 = codebooks2[codebook_idx]
        
        print(f"Codebook {codebook_idx}:")
        print(f"  Shape in ckpt1: {weights_1.shape}")
        print(f"  Shape in ckpt2: {weights_2.shape}")
        
        # Compare dimensions
        min_embed_size = min(weights_1.shape[0], weights_2.shape[0])
        min_dim = min(weights_1.shape[1], weights_2.shape[1])
        
        if frozen_size > min_embed_size:
            print(f"  Warning: frozen_size ({frozen_size}) > min_embed_size ({min_embed_size})")
            compare_size = min_embed_size
        else:
            compare_size = frozen_size
        
        # Compare first 'frozen_size' embeddings (excluding padding if exists)
        # Note: Some codebooks might have padding embedding at the end
        if compare_size > 0:
            frozen_1 = weights_1[:compare_size, :min_dim]
            frozen_2 = weights_2[:compare_size, :min_dim]
            
            # Calculate differences
            diff = np.abs(frozen_1 - frozen_2)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"  Comparing first {compare_size} embeddings:")
            print(f"    Max difference: {max_diff:.6f}")
            print(f"    Mean difference: {mean_diff:.6f}")
            
            # Check if frozen (threshold for floating point comparison)
            threshold = 1e-6
            is_frozen = max_diff < threshold
            
            if is_frozen:
                print(f"    ✓ FROZEN: Embeddings match (diff < {threshold})")
            else:
                print(f"    ✗ NOT FROZEN: Embeddings differ significantly")
                all_match = False
                
                # Show some sample differences for debugging
                print(f"    Sample differences (first 3 embeddings, first 5 dimensions):")
                for i in range(min(3, compare_size)):
                    for j in range(min(5, min_dim)):
                        print(f"      [{i},{j}]: {frozen_1[i,j]:.6f} vs {frozen_2[i,j]:.6f} (diff: {diff[i,j]:.6f})")
        else:
            print(f"  No embeddings to compare (compare_size = {compare_size})")
        
        print()
    
    return all_match


def verify_progressive_checkpoints(checkpoint_dir, embedding_schedule=[16, 32], n_codebooks=2):
    """Verify that progressive checkpoints maintain frozen embeddings correctly"""
    print("="*80)
    print("PROGRESSIVE RQ CHECKPOINT VERIFICATION")
    print("="*80)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Embedding schedule: {embedding_schedule}")
    print(f"Number of codebooks: {n_codebooks}")
    print("="*80)
    
    all_good = True
    
    # Check within each codebook progression
    for codebook_idx in range(n_codebooks):
        print(f"\n{'='*60}")
        print(f"VERIFYING CODEBOOK {codebook_idx + 1} PROGRESSION")
        print(f"{'='*60}")
        
        prev_checkpoint = None
        
        for stage_idx, embed_size in enumerate(embedding_schedule):
            checkpoint_name = f"codebook{codebook_idx+1}_{embed_size}embeds_stage{stage_idx}_final.pth"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            
            if not os.path.exists(checkpoint_path):
                print(f"Warning: Checkpoint {checkpoint_name} not found")
                continue
            
            print(f"\nFound checkpoint: {checkpoint_name}")
            
            # Compare with previous stage if exists
            if prev_checkpoint is not None and stage_idx > 0:
                frozen_size = embedding_schedule[stage_idx - 1]
                is_frozen = compare_frozen_embeddings(prev_checkpoint, checkpoint_path, frozen_size)
                if not is_frozen:
                    all_good = False
                    print(f"❌ ISSUE: Frozen embeddings not preserved in {checkpoint_name}")
                else:
                    print(f"✅ GOOD: Frozen embeddings preserved in {checkpoint_name}")
            
            prev_checkpoint = checkpoint_path
    
    # Check between different codebooks at same stage
    for stage_idx, embed_size in enumerate(embedding_schedule):
        if stage_idx == 0:  # Skip first stage as there's nothing to freeze
            continue
            
        print(f"\n{'='*60}")
        print(f"VERIFYING CROSS-CODEBOOK CONSISTENCY AT STAGE {stage_idx}")
        print(f"{'='*60}")
        
        checkpoints = []
        for codebook_idx in range(n_codebooks):
            checkpoint_name = f"codebook{codebook_idx+1}_{embed_size}embeds_stage{stage_idx}_final.pth"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            if os.path.exists(checkpoint_path):
                checkpoints.append((codebook_idx, checkpoint_path))
        
        # Compare codebook 0 across different codebook training stages
        if len(checkpoints) >= 2:
            frozen_size = embedding_schedule[stage_idx - 1]
            for i in range(len(checkpoints) - 1):
                codebook_idx1, path1 = checkpoints[i]
                codebook_idx2, path2 = checkpoints[i + 1]
                
                print(f"\nComparing codebook 0 between:")
                print(f"  Training codebook {codebook_idx1+1}: {os.path.basename(path1)}")
                print(f"  Training codebook {codebook_idx2+1}: {os.path.basename(path2)}")
                
                # Load and compare only codebook 0
                weights1, _ = load_checkpoint_weights(path1)
                weights2, _ = load_checkpoint_weights(path2)
                
                if weights1 and weights2:
                    codebooks1 = extract_codebook_weights(weights1)
                    codebooks2 = extract_codebook_weights(weights2)
                    
                    if 0 in codebooks1 and 0 in codebooks2:
                        cb0_1 = codebooks1[0]
                        cb0_2 = codebooks2[0]
                        
                        min_size = min(cb0_1.shape[0], cb0_2.shape[0], frozen_size)
                        min_dim = min(cb0_1.shape[1], cb0_2.shape[1])
                        
                        if min_size > 0:
                            diff = np.abs(cb0_1[:min_size, :min_dim] - cb0_2[:min_size, :min_dim])
                            max_diff = np.max(diff)
                            
                            threshold = 1e-6
                            if max_diff < threshold:
                                print(f"  ✅ GOOD: Codebook 0 consistent across different training stages")
                            else:
                                print(f"  ❌ ISSUE: Codebook 0 differs between training stages (max diff: {max_diff:.6f})")
                                all_good = False
    
    print(f"\n{'='*80}")
    if all_good:
        print("🎉 ALL CHECKS PASSED: Progressive training appears to be working correctly!")
    else:
        print("⚠️  ISSUES FOUND: Some frozen embeddings are not preserved correctly.")
        print("   This suggests the freezing mechanism may not be working as expected.")
    print("="*80)
    
    return all_good


def main():
    parser = argparse.ArgumentParser(description='Verify Progressive RQ Checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory containing checkpoints')
    parser.add_argument('--embedding_schedule', nargs='+', type=int, default=[16, 32],
                        help='Embedding schedule used during training')
    parser.add_argument('--n_codebooks', type=int, default=2,
                        help='Number of codebooks')
    parser.add_argument('--compare_two', nargs=2, type=str, default=None,
                        help='Compare two specific checkpoints')
    parser.add_argument('--frozen_size', type=int, default=16,
                        help='Expected frozen size for two-checkpoint comparison')
    
    args = parser.parse_args()
    
    if args.compare_two:
        # Compare two specific checkpoints
        compare_frozen_embeddings(args.compare_two[0], args.compare_two[1], args.frozen_size)
    else:
        # Verify entire progression
        verify_progressive_checkpoints(args.checkpoint_dir, args.embedding_schedule, args.n_codebooks)


if __name__ == '__main__':
    main()