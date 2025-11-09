"""
Configuration utilities for loading and normalizing YAML configs
"""

import os
import yaml


def load_config(config_path):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config = _normalize_config(config or {})
    print(f"Loaded configuration from: {config_path}")
    return config


def _normalize_config(config):
    """Ensure required config sections exist with sensible defaults."""
    # Training defaults
    hyper = config.get('hyperparameters', {})
    training_defaults = {
        'mode': 'eval',
        'batch_size': hyper.get('batch_size', 4),
        'num_workers': hyper.get('num_workers', 4),
        'init_lr': hyper.get('init_lr', 1e-3),
        'max_epoch': hyper.get('max_epoch', 50),
        'ckpt_freq_epoch': hyper.get('ckpt_freq_epoch', 5),
        'early_stopping_patience': hyper.get('patience', 20),
    }
    training_section = config.setdefault('training', {})
    for key, value in training_defaults.items():
        training_section.setdefault(key, value)

    # Logging defaults
    logging_defaults = {
        'use_wandb': config.get('wandb', False),
        'wandb_project': None,
        'wandb_name': None,
        'log_freq': hyper.get('log_freq', 10),
        'saved_path': 'progressive_rq_logs',
        'save_stage_weights': True,
    }
    logging_section = config.setdefault('logging', {})
    for key, value in logging_defaults.items():
        logging_section.setdefault(key, value)

    # Progressive learning defaults
    progressive_defaults = {
        'enabled': False,
        'embedding_schedule': hyper.get('embedding_schedule', []),
        'embedding_stage_epochs': hyper.get('embedding_stage_epochs', 0),
        'warmup_epochs': hyper.get('warmup_epochs', 0),
    }
    progressive_section = config.setdefault('progressive_learning', {})
    for key, value in progressive_defaults.items():
        progressive_section.setdefault(key, value)

    # RQ model defaults
    rq_defaults = {
        'latent_shape': hyper.get('latent_shape', [62, 54, 64]),
        'code_shape': hyper.get('code_shape', [62, 54, config.get('model', {}).get('n_codebook', 1)]),
        'decay': hyper.get('decay', 0.99),
    }
    rq_section = config.setdefault('rq_model', {})
    for key, value in rq_defaults.items():
        rq_section.setdefault(key, value)

    # Loss weights defaults
    loss_defaults = {
        'vq_weight': hyper.get('vq_weight', 1.0),
        'codebook_weight': hyper.get('codebook_weight', 1.0),
        'det_weight': hyper.get('det_weight', 1.0),
    }
    loss_section = config.setdefault('loss_weights', {})
    for key, value in loss_defaults.items():
        loss_section.setdefault(key, value)

    # Evaluation defaults
    evaluation_section = config.setdefault('evaluation', {})
    if 'weights_path' in evaluation_section and 'rq_ckpt' not in evaluation_section:
        evaluation_section['rq_ckpt'] = evaluation_section['weights_path']
    evaluation_defaults = {
        'rq_ckpt': None,
        'use_num_codebook': config.get('model', {}).get('n_codebook', 1),
        'use_num_embedding': hyper.get('embedding_schedule', [hyper.get('codebook_size', 256)])[:1][0] if hyper.get('embedding_schedule') else hyper.get('codebook_size', 256),
        'eval_all_codebook_stages': False,
    }
    for key, value in evaluation_defaults.items():
        evaluation_section.setdefault(key, value)

    return config


def print_config_summary(config):
    """Print configuration summary"""
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Mode: {config['training']['mode']}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Data root: {config['dataset']['dir']}")
    
    model_cfg = config.get('model', {})
    pretrained_desc = model_cfg.get('pre_trained_model', model_cfg.get('pretrained_weight', 'N/A'))
    print(f"Model: {pretrained_desc}")
    
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