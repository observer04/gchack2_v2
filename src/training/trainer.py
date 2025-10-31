"""
Trainer for glacier segmentation models.

Handles:
- Training loop with AMP (mixed precision)
- Gradient accumulation
- Learning rate scheduling
- Checkpointing
- Metrics tracking
- TensorBoard logging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm
import numpy as np

from .metrics import MetricTracker, format_metrics, format_class_metrics


class Trainer:
    """
    Trainer for segmentation models.
    
    Args:
        model: Segmentation model
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        use_amp: Whether to use automatic mixed precision
        gradient_clip_val: Max gradient norm for clipping
        gradient_accumulation_steps: Number of steps to accumulate gradients
        checkpoint_dir: Directory to save checkpoints
        log_every: Log every N batches
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        use_amp: bool = True,
        gradient_clip_val: float = 1.0,
        gradient_accumulation_steps: int = 1,
        checkpoint_dir: str = 'weights',
        log_every: int = 10
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp
        self.gradient_clip_val = gradient_clip_val
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_every = log_every
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Initialize AMP scaler
        self.scaler = GradScaler() if use_amp else None
        
        # Tracking
        self.current_epoch = 0
        self.best_metric = 0.0
        self.history = {
            'train': [],
            'val': []
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
        
        Returns:
            dict of training metrics
        """
        self.model.train()
        self.current_epoch = epoch
        
        # Update criterion epoch (for boundary ramp)
        if hasattr(self.criterion, 'set_epoch'):
            self.criterion.set_epoch(epoch)
        
        # Metric tracker
        tracker = MetricTracker(num_classes=4)
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # Get data
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass with AMP
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    
                    # Handle combined loss (returns dict)
                    if isinstance(self.criterion(outputs, masks), dict):
                        loss_dict = self.criterion(outputs, masks)
                        loss = loss_dict['loss']
                    else:
                        loss = self.criterion(outputs, masks)
                        loss_dict = {'loss': loss}
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(images)
                
                if isinstance(self.criterion(outputs, masks), dict):
                    loss_dict = self.criterion(outputs, masks)
                    loss = loss_dict['loss']
                else:
                    loss = self.criterion(outputs, masks)
                    loss_dict = {'loss': loss}
                
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    # Gradient clipping
                    if self.gradient_clip_val > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip_val
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip_val
                        )
                    
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Update metrics
            tracker.update(
                outputs.detach(),
                masks,
                loss=loss.item() * self.gradient_accumulation_steps
            )
            
            # Update progress bar
            if batch_idx % self.log_every == 0:
                current_metrics = tracker.compute()
                pbar.set_postfix({
                    'loss': f"{current_metrics.get('loss', 0):.4f}",
                    'mcc': f"{current_metrics.get('mcc', 0):.4f}"
                })
        
        # Compute epoch metrics
        epoch_metrics = tracker.compute()
        
        return epoch_metrics
    
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
        
        Returns:
            dict of validation metrics
        """
        self.model.eval()
        
        # Metric tracker
        tracker = MetricTracker(num_classes=4)
        
        # Progress bar
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        
        with torch.no_grad():
            for batch in pbar:
                # Get data
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        
                        if isinstance(self.criterion(outputs, masks), dict):
                            loss_dict = self.criterion(outputs, masks)
                            loss = loss_dict['loss']
                        else:
                            loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    
                    if isinstance(self.criterion(outputs, masks), dict):
                        loss_dict = self.criterion(outputs, masks)
                        loss = loss_dict['loss']
                    else:
                        loss = self.criterion(outputs, masks)
                
                # Update metrics
                tracker.update(outputs, masks, loss=loss.item())
                
                # Update progress bar
                current_metrics = tracker.compute()
                pbar.set_postfix({
                    'loss': f"{current_metrics.get('loss', 0):.4f}",
                    'mcc': f"{current_metrics.get('mcc', 0):.4f}"
                })
        
        # Compute epoch metrics
        epoch_metrics = tracker.compute()
        
        return epoch_metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        start_epoch: int = 0
    ):
        """
        Train model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            start_epoch: Starting epoch (for resume)
        """
        print(f"\n{'='*80}")
        print(f"Starting training for {epochs} epochs")
        print(f"{'='*80}\n")
        
        for epoch in range(start_epoch, epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['mcc'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            print(f"\nEpoch {epoch}/{epochs-1}")
            print(f"Train: {format_metrics(train_metrics, 'train_')}")
            print(f"Val:   {format_metrics(val_metrics, 'val_')}")
            
            # Print per-class metrics every 10 epochs
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(format_class_metrics(val_metrics))
            
            # Save history
            self.history['train'].append(train_metrics)
            self.history['val'].append(val_metrics)
            
            # Save checkpoints
            is_best = val_metrics['mcc'] > self.best_metric
            if is_best:
                self.best_metric = val_metrics['mcc']
            
            self.save_checkpoint(
                epoch=epoch,
                metrics=val_metrics,
                is_best=is_best
            )
            
            # Print separator
            print(f"{'-'*80}")
        
        print(f"\n{'='*80}")
        print(f"Training complete!")
        print(f"Best validation MCC: {self.best_metric:.4f}")
        print(f"{'='*80}\n")
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """
        Save checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'best_metric': self.best_metric,
            'history': self.history
        }
        
        # Save last checkpoint
        last_path = self.checkpoint_dir / 'last_checkpoint.pth'
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best checkpoint (MCC: {metrics['mcc']:.4f})")
        
        # Save periodic checkpoint (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            periodic_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, periodic_path)
    
    def load_checkpoint(self, checkpoint_path: str, resume: bool = False):
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            resume: Whether to resume training (load optimizer, scheduler)
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if resume:
            # Load optimizer
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            self.current_epoch = checkpoint['epoch'] + 1
            self.best_metric = checkpoint.get('best_metric', 0.0)
            self.history = checkpoint.get('history', {'train': [], 'val': []})
            
            print(f"✓ Resumed from epoch {self.current_epoch}")
        else:
            print(f"✓ Loaded model weights")
        
        return checkpoint


if __name__ == '__main__':
    print("Trainer module - use from train.py script")
