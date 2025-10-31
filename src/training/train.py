"""
Main training script for glacier segmentation.

Usage:
    # HKH Pretraining
    python train.py --config configs/hkh_pretrain_kaggle.yaml --experiment_name hkh_pretrain

    # Competition Fine-tuning (single fold)
    python train.py --config configs/competition_finetune_kaggle.yaml --fold 0
    
    # Resume training
    python train.py --config configs/hkh_pretrain_kaggle.yaml --resume weights/last_checkpoint.pth
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import GlacierDataset
from data.samplers import PixelBalancedSampler
from models.glacier_unet import create_model
from losses import CombinedLoss
from training.trainer import Trainer


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_data_loaders(config: dict, fold: int = None) -> tuple:
    """
    Create training and validation data loaders.
    
    Args:
        config: Configuration dict
        fold: Fold number for competition data (None for HKH)
    
    Returns:
        (train_loader, val_loader)
    """
    data_config = config['data']
    training_config = config['training']
    
    # Determine dataset type
    if 'hkh_dir' in data_config:
        # HKH dataset
        dataset = GlacierDataset(
            data_dir=data_config['hkh_dir'],
            mode='train',
            num_channels=config['model']['in_channels']
        )
        
        # Split into train/val
        train_size = int(data_config['train_split'] * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config.get('seed', 42))
        )
        
        # Update dataset mode
        train_dataset.dataset.mode = 'train'
        val_dataset.dataset.mode = 'val'
        
    elif 'comp_dir' in data_config:
        # Competition dataset with CV splits
        from sklearn.model_selection import KFold
        
        dataset = GlacierDataset(
            data_dir=data_config['comp_dir'],
            mode='train',
            num_channels=config['model']['in_channels']
        )
        
        # Create CV splits
        kfold = KFold(
            n_splits=data_config['num_folds'],
            shuffle=True,
            random_state=config.get('seed', 42)
        )
        
        all_indices = list(range(len(dataset)))
        splits = list(kfold.split(all_indices))
        
        if fold is None:
            fold = data_config.get('current_fold', 0)
        
        train_indices, val_indices = splits[fold]
        
        # Create subset datasets
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        # Update mode
        train_dataset.dataset.mode = 'train'
        val_dataset.dataset.mode = 'val'
    
    else:
        raise ValueError("Config must specify either 'hkh_dir' or 'comp_dir'")
    
    # Create samplers
    sampler_config = training_config.get('sampler', {})
    
    if sampler_config.get('type') == 'pixel_balanced':
        # Use pixel-balanced sampler for training
        train_sampler = PixelBalancedSampler(
            dataset=train_dataset,
            target_distribution=sampler_config.get('target_distribution'),
            oversample_rare=sampler_config.get('oversample_rare', True)
        )
        shuffle_train = False
    else:
        train_sampler = None
        shuffle_train = True
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        sampler=train_sampler,
        shuffle=shuffle_train if train_sampler is None else False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        persistent_workers=data_config.get('persistent_workers', True)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        persistent_workers=data_config.get('persistent_workers', True)
    )
    
    return train_loader, val_loader


def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    opt_config = config['training']['optimizer']
    
    if opt_config['type'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 0.0),
            betas=opt_config.get('betas', (0.9, 0.999))
        )
    elif opt_config['type'] == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 0.0)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_config['type']}")
    
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler from config."""
    sched_config = config['training']['scheduler']
    
    if sched_config['type'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_config.get('mode', 'max'),
            factor=sched_config.get('factor', 0.5),
            patience=sched_config.get('patience', 5),
            min_lr=sched_config.get('min_lr', 1e-7),
            verbose=sched_config.get('verbose', True)
        )
    elif sched_config['type'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=sched_config.get('min_lr', 1e-7)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {sched_config['type']}")
    
    return scheduler


def create_criterion(config: dict) -> nn.Module:
    """Create loss criterion from config."""
    loss_config = config['training']['loss']
    
    criterion = CombinedLoss(
        focal_weight=loss_config.get('focal_weight', 0.25),
        dice_weight=loss_config.get('dice_weight', 0.25),
        mcc_weight=loss_config.get('mcc_weight', 0.35),
        boundary_weight=loss_config.get('boundary_weight', 0.15),
        boundary_ramp=loss_config.get('boundary_ramp', False),
        boundary_ramp_start=loss_config.get('boundary_ramp_start', 0.05),
        boundary_ramp_epochs=loss_config.get('boundary_ramp_epochs', 30),
        focal_alpha=loss_config.get('focal_alpha'),
        focal_gamma=loss_config.get('focal_gamma', 2.0),
        dice_smooth=loss_config.get('dice_smooth', 1.0),
        mcc_phi=loss_config.get('mcc_phi', 0.7)
    )
    
    return criterion


def main(args):
    """Main training function."""
    # Load config
    config = load_config(args.config)
    
    # Set random seed
    seed = config.get('seed', 42)
    if args.seed:
        seed = args.seed
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Print configuration
    print(f"\n{'='*80}")
    print(f"Training Configuration")
    print(f"{'='*80}")
    print(f"Experiment: {args.experiment_name or 'unnamed'}")
    print(f"Config: {args.config}")
    if args.fold is not None:
        print(f"Fold: {args.fold}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")
    
    # Create model
    print("Creating model...")
    model = create_model(config['model'])
    
    # Load pretrained weights if specified
    if 'pretrained_path' in config['model'] and config['model']['pretrained_path']:
        print(f"Loading pretrained weights from {config['model']['pretrained_path']}")
        checkpoint = torch.load(config['model']['pretrained_path'])
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights (may need to adjust for different input channels)
        try:
            model.load_state_dict(state_dict, strict=False)
            print("✓ Loaded pretrained weights")
        except Exception as e:
            print(f"⚠ Warning: Could not load some weights: {e}")
            print("  Continuing with partially loaded weights...")
    
    # Multi-GPU setup
    device_config = config.get('device', {})
    if device_config.get('use_parallel', False) and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model, device_ids=device_config.get('gpu_ids'))
        device = 'cuda'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.1f}M)\n")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config, fold=args.fold)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}\n")
    
    # Create optimizer, scheduler, criterion
    print("Creating optimizer, scheduler, and loss...")
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    criterion = create_criterion(config)
    print(f"Optimizer: {config['training']['optimizer']['type']}")
    print(f"Scheduler: {config['training']['scheduler']['type']}")
    print(f"Loss: Combined (Focal+Dice+MCC+Boundary)\n")
    
    # Create trainer
    checkpoint_dir = config['checkpoints']['save_dir']
    if args.experiment_name:
        checkpoint_dir = Path(checkpoint_dir) / args.experiment_name
    
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=config['training'].get('use_amp', True),
        gradient_clip_val=config['training'].get('gradient_clip_val', 1.0),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        checkpoint_dir=str(checkpoint_dir),
        log_every=config['logging'].get('log_every', 10)
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}...")
        checkpoint = trainer.load_checkpoint(args.resume, resume=True)
        start_epoch = checkpoint['epoch'] + 1
        print()
    
    # Train
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        start_epoch=start_epoch
    )
    
    print(f"\n✓ Training complete!")
    print(f"Best checkpoint: {checkpoint_dir}/best_checkpoint.pth")
    print(f"Best MCC: {trainer.best_metric:.4f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train glacier segmentation model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name for checkpoint directory')
    parser.add_argument('--fold', type=int, default=None,
                        help='Fold number for competition CV (0-4)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    main(args)
