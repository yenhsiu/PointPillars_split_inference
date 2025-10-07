# Progressive RQ Training for PointPillars

This document describes how to use the progressive RQ training system that has been implemented for PointPillars split inference.

## Features

### 1. Progressive Learning
- **Codebook-wise Training**: Train one codebook at a time from 1 to 5 codebooks
- **Embedding Size Schedule**: For each codebook, progressively increase embedding size from [16, 32, 64, 128, 256]
- **EMA Warmup**: Each stage starts with EMA warmup for stable initialization
- **Flexible Evaluation**: Choose any number of codebooks and embeddings for evaluation

### 2. YAML Configuration
- All experimental settings are now organized in `exp/split1_ALL.yaml`
- Easy to modify and reproduce experiments
- Command line overrides available for quick adjustments

## Configuration Structure

The main configuration file `exp/split1_ALL.yaml` contains:

```yaml
# Dataset configuration
dataset:
  name: KITTI
  dir: "/home/yenhsiu/datasets"
  num_classes: 3

# Model configuration  
model:
  pre_trained_model: "b0"
  n_codebook: 5                    # Maximum number of codebooks
  pretrained_weight: 'pretrained/epoch_160.pth'

# Progressive learning configuration
progressive_learning:
  enabled: True                     # Enable/disable progressive learning
  embedding_schedule: [16, 32, 64, 128, 256]  # Embedding size progression
  embedding_stage_epochs: 20        # Epochs per embedding stage
  warmup_epochs: 2                  # EMA warmup epochs per stage

# Training hyperparameters
training:
  mode: 'train'                     # 'train' or 'eval'
  batch_size: 6
  num_workers: 4
  max_epoch: 100                    # Will be divided across stages
  init_lr: 0.001
  ckpt_freq_epoch: 20
  patience: 5
  early_stopping_patience: 5

# RQ model parameters
rq_model:
  latent_shape: [496, 432, 64]
  code_shape: [496, 432, 1]
  codebook_size: 64                 # Overridden by progressive schedule
  decay: 0.99

# Loss weights
loss_weights:
  vq_weight: 0.25
  codebook_weight: 0.1
  det_weight: 0.8

# Evaluation configuration
evaluation:
  enabled: False
  use_num_codebook: 1               # Number of codebooks for evaluation
  use_num_embedding: 16             # Number of embeddings per codebook
  rq_ckpt: null                     # Path to checkpoint for evaluation

# Logging and saving
logging:
  use_wandb: True
  wandb_project: "pointpillars-rq-progressive"
  wandb_name: null
  log_freq: 8
  saved_path: "progressive_rq_logs"
  save_stage_weights: True

# Hardware
hardware:
  gpu: 1
```

## Usage Examples

### 1. Progressive Training (Default)
```bash
# Train with progressive learning (using default config)
python train_eval_rq.py --config exp/split1_ALL.yaml

# Train with different GPU
python train_eval_rq.py --config exp/split1_ALL.yaml --gpu 0

# Train without wandb logging
python train_eval_rq.py --config exp/split1_ALL.yaml --no_wandb
```

### 2. Evaluation with Different Configurations
```bash
# Evaluate using 1 codebook with 16 embeddings
python train_eval_rq.py --config exp/split1_ALL.yaml --mode eval \
    --eval_num_codebooks 1 --eval_num_embeddings 16 \
    --rq_ckpt path/to/checkpoint.pth

# Evaluate using 3 codebooks with 64 embeddings each
python train_eval_rq.py --config exp/split1_ALL.yaml --mode eval \
    --eval_num_codebooks 3 --eval_num_embeddings 64 \
    --rq_ckpt path/to/checkpoint.pth

# Evaluate using all 5 codebooks with maximum embeddings
python train_eval_rq.py --config exp/split1_ALL.yaml --mode eval \
    --eval_num_codebooks 5 --eval_num_embeddings 256 \
    --rq_ckpt path/to/checkpoint.pth
```

### 3. Legacy Mode (Non-Progressive)
```bash
# To use the original training mode, set progressive_learning.enabled: False in config
# Then run normal training
python train_eval_rq.py --config exp/split1_ALL.yaml
```

## Training Process

### Progressive Training Flow:
1. **Codebook 1**: Train with embedding sizes [16 → 32 → 64 → 128 → 256]
2. **Codebook 2**: Train with embedding sizes [16 → 32 → 64 → 128 → 256] (Codebook 1 frozen)
3. **Codebook 3**: Train with embedding sizes [16 → 32 → 64 → 128 → 256] (Codebooks 1-2 frozen)
4. **Codebook 4**: Train with embedding sizes [16 → 32 → 64 → 128 → 256] (Codebooks 1-3 frozen)
5. **Codebook 5**: Train with embedding sizes [16 → 32 → 64 → 128 → 256] (Codebooks 1-4 frozen)

### Each Stage:
1. **EMA Warmup** (2 epochs): Initialize embeddings using EMA
2. **Gradient Training** (18 epochs): Train with gradient-based optimization
3. **Validation**: Evaluate performance after each epoch
4. **Checkpointing**: Save best models per stage and globally

## Checkpoints and Logging

### Checkpoint Structure:
```
progressive_rq_logs/checkpoints/
├── codebook_0_stage_0_epoch_5.pth          # Regular checkpoints
├── codebook_0_stage_0_best.pth             # Best for this stage
├── codebook_0_stage_0_final.pth            # Stage completion
├── codebook_1_stage_2_epoch_10.pth
├── ...
└── global_best.pth                         # Overall best model
```

### WandB Logging:
- Training losses (total, detection, VQ, codebook)
- Validation metrics (mAP, loss)
- Stage tracking (codebook_idx, stage_idx, embed_size)
- Learning rate scheduling
- Best performance tracking

## Model Architecture

The progressive RQ system maintains the same architecture as the original PointPillars but with:

1. **Frozen Headnet**: Feature extraction (frozen during RQ training)
2. **Progressive RQ Bottleneck**: Quantization with progressive training
3. **Frozen Tailnet**: Detection head (frozen during RQ training)

Only the RQ bottleneck is trained, allowing focused learning of quantization representations.

## Evaluation Flexibility

The evaluation system allows testing different configurations:
- **1-5 codebooks**: Test compression levels
- **16-256 embeddings**: Test quantization granularity
- **Mixed configurations**: e.g., 3 codebooks with 128 embeddings each

This enables comprehensive analysis of the trade-off between model compression and detection performance.

## Advanced Usage

### Custom Embedding Schedules:
Modify `progressive_learning.embedding_schedule` in the config:
```yaml
progressive_learning:
  embedding_schedule: [8, 16, 32, 64]  # Smaller schedule
  # or
  embedding_schedule: [32, 64, 128, 256, 512]  # Larger schedule
```

### Different Stage Epochs:
```yaml
progressive_learning:
  embedding_stage_epochs: 30  # More epochs per stage
  warmup_epochs: 3            # Longer warmup
```

### Custom Loss Weights:
```yaml
loss_weights:
  vq_weight: 0.5      # Higher VQ loss weight
  codebook_weight: 0.2  # Higher codebook loss weight  
  det_weight: 0.6     # Lower detection weight for focus on quantization
```

This implementation provides a comprehensive framework for progressive RQ training while maintaining compatibility with the original PointPillars architecture.