"""
Training utilities for PointPillars RQ model
Includes common classes and functions used across training scripts
"""

import time
import torch
import numpy as np
import os


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
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        self.val_loss_min = val_loss


def load_pretrained_with_report(model, ckpt_path, device):
    """Load pretrained checkpoint with device mapping"""
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    return model


def setup_directories(base_path):
    """Setup directory structure for experiment"""
    import os
    checkpoint_dir = os.path.join(base_path, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def get_experiment_name(prefix="Progressive_RQ"):
    """Generate experiment name with timestamp"""
    import datetime
    date_str = str(datetime.date.today())
    time_str = datetime.datetime.now().strftime("%H-%M-%S")
    return f"{date_str}_{time_str}_{prefix}_training"


def print_training_summary(config):
    """Print training configuration summary"""
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Mode: {config['training']['mode']}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Data root: {config['dataset']['dir']}")
    print(f"GPU: {config['hardware']['gpu']}")
    print(f"Batch size: {config['training']['batch_size']}")
    
    if config['progressive_learning']['enabled']:
        print(f"Progressive learning: Enabled")
        print(f"- {config['model']['n_codebook']} codebooks")
        print(f"- Embedding schedule: {config['progressive_learning']['embedding_schedule']}")
        print(f"- {config['progressive_learning']['embedding_stage_epochs']} epochs per stage")
    else:
        print("Progressive learning: Disabled")
    
    print("="*60)


def print_epoch_summary(epoch, stage_epoch, train_metrics, val_metrics, epoch_time):
    """Print epoch training summary"""
    bbox_3d_overall = (val_metrics.get('bbox_3d_difficulty_0_mAP', 0.0) + 
                      val_metrics.get('bbox_3d_difficulty_1_mAP', 0.0) + 
                      val_metrics.get('bbox_3d_difficulty_2_mAP', 0.0)) / 3.0
    
    print(f"Epoch {epoch} Summary:")
    print(f"  Time: {epoch_time:.2f}s")
    print(f"  Training Loss: {train_metrics['total_loss']:.4f} "
          f"(Det: {train_metrics['det_loss']:.4f}, "
          f"VQ: {train_metrics['vq_loss']:.6f}, "
          f"CB: {train_metrics['cb_loss']:.6f})")
    print(f"  Validation Results (bbox_3d):")
    print(f"    Overall bbox_3d mAP: {bbox_3d_overall:.4f}")
    for key, value in val_metrics.items():
        if key.startswith('bbox_3d_difficulty_') and not any(cls in key for cls in ['Car', 'Pedestrian', 'Cyclist']):
            print(f"    {key}: {value:.4f}")
    print(f"    Loss - Total: {val_metrics.get('total_loss', 0.0):.4f}, "
          f"Det: {val_metrics.get('det_loss', 0.0):.4f}")


def create_checkpoint_filename(codebook_idx, stage_idx, embed_size, is_final=True):
    """Create standardized checkpoint filename"""
    stage_type = "final" if is_final else "best"
    return f"codebook{codebook_idx+1}_{embed_size}embeds_stage{stage_idx}_{stage_type}.pth"


def save_stage_checkpoint(rq_bottleneck, codebook_idx, stage_idx, embed_size, 
                         stage_best_mAP, config, save_path):
    """Save final checkpoint for completed training stage"""
    checkpoint_data = {
        'rq_bottleneck': rq_bottleneck.state_dict(),
        'codebook_idx': codebook_idx,
        'stage_idx': stage_idx,
        'embed_size': embed_size,
        'stage_best_mAP': stage_best_mAP,
        'config': config
    }
    
    filename = create_checkpoint_filename(codebook_idx, stage_idx, embed_size, is_final=True)
    full_path = os.path.join(save_path, filename)
    torch.save(checkpoint_data, full_path)
    print(f"  Saved stage final checkpoint: {filename}")
    return full_path