"""
Loss functions for glacier segmentation.

Implements:
- FocalLoss: Handles class imbalance by down-weighting easy examples
- DiceLoss: Optimizes for overlap (IoU-based)
- BoundaryLoss: Penalizes errors near class boundaries
- MCCLoss: Focal-Phi MCC Loss for direct MCC optimization
- CombinedLoss: Weighted combination with progressive ramp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Class weights [C]. Higher values for harder classes.
        gamma: Focusing parameter. Higher = more focus on hard examples.
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        alpha: Optional[List[float]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha))
        else:
            self.alpha = None
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C, H, W) - logits
            targets: (B, H, W) - class indices
        
        Returns:
            loss: scalar
        """
        # Get probabilities
        probs = F.softmax(inputs, dim=1)  # (B, C, H, W)
        
        # Get log probabilities
        log_probs = F.log_softmax(inputs, dim=1)  # (B, C, H, W)
        
        # Create mask for valid pixels
        valid_mask = (targets != self.ignore_index)
        
        # Get per-pixel class probabilities
        B, C, H, W = inputs.shape
        targets_one_hot = F.one_hot(
            targets.clamp(0, C - 1),  # Clamp to valid range
            num_classes=C
        ).permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # p_t: probability of correct class
        p_t = (probs * targets_one_hot).sum(dim=1)  # (B, H, W)
        
        # Focal term: (1 - p_t)^γ
        focal_weight = (1 - p_t) ** self.gamma
        
        # Class weights
        if self.alpha is not None:
            alpha_t = (self.alpha.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * targets_one_hot).sum(dim=1)
        else:
            alpha_t = 1.0
        
        # Focal loss: -α_t * (1 - p_t)^γ * log(p_t)
        focal_loss = -alpha_t * focal_weight * (log_probs * targets_one_hot).sum(dim=1)
        
        # Apply mask
        focal_loss = focal_loss * valid_mask
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.sum() / (valid_mask.sum() + 1e-6)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    DiceLoss = 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
    
    Args:
        smooth: Smoothing factor to avoid division by zero
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        smooth: float = 1.0,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C, H, W) - logits
            targets: (B, H, W) - class indices
        
        Returns:
            loss: scalar
        """
        # Get probabilities
        probs = F.softmax(inputs, dim=1)  # (B, C, H, W)
        
        # One-hot encode targets
        B, C, H, W = inputs.shape
        targets_one_hot = F.one_hot(
            targets.clamp(0, C - 1),
            num_classes=C
        ).permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Create mask for valid pixels
        valid_mask = (targets != self.ignore_index).float()
        valid_mask = valid_mask.unsqueeze(1)  # (B, 1, H, W)
        
        # Apply mask
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask
        
        # Compute Dice coefficient per class
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))  # (B, C)
        cardinality = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))  # (B, C)
        
        dice_coeff = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_coeff  # (B, C)
        
        # Reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class BoundaryLoss(nn.Module):
    """
    Boundary Loss - penalizes errors near class boundaries.
    
    Computes distance transform of boundaries and weights errors by distance.
    Helps produce sharper segmentation boundaries.
    
    Args:
        kernel_size: Size for boundary detection (Sobel filter)
    """
    
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
    
    def _compute_boundaries(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Detect boundaries using Sobel-like gradient.
        
        Args:
            targets: (B, H, W) - class indices
        
        Returns:
            boundaries: (B, H, W) - binary boundary map
        """
        B, H, W = targets.shape
        
        # Simple boundary detection: where neighboring pixels differ
        boundaries = torch.zeros_like(targets, dtype=torch.float)
        
        # Horizontal gradients
        boundaries[:, :, :-1] = (targets[:, :, :-1] != targets[:, :, 1:]).float()
        
        # Vertical gradients
        boundaries[:, :-1, :] = torch.max(
            boundaries[:, :-1, :],
            (targets[:, :-1, :] != targets[:, 1:, :]).float()
        )
        
        return boundaries
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C, H, W) - logits
            targets: (B, H, W) - class indices
        
        Returns:
            loss: scalar
        """
        # Detect boundaries
        boundaries = self._compute_boundaries(targets)  # (B, H, W)
        
        # Compute standard cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # (B, H, W)
        
        # Weight loss by boundary proximity
        # Higher weight near boundaries
        boundary_weight = 1.0 + 2.0 * boundaries  # 1.0 for interior, 3.0 for boundary
        
        weighted_loss = ce_loss * boundary_weight
        
        return weighted_loss.mean()


class MCCLoss(nn.Module):
    """
    Focal-Phi MCC Loss for direct MCC optimization.
    
    Matthews Correlation Coefficient loss with focal-phi weighting
    to handle class imbalance.
    
    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    
    Args:
        phi: Focal-phi parameter (0-1). Higher = more focus on hard classes.
        smooth: Smoothing factor
    """
    
    def __init__(self, phi: float = 0.7, smooth: float = 1e-6):
        super().__init__()
        self.phi = phi
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C, H, W) - logits
            targets: (B, H, W) - class indices
        
        Returns:
            loss: 1 - MCC (to minimize)
        """
        # Get probabilities
        probs = F.softmax(inputs, dim=1)  # (B, C, H, W)
        
        # One-hot encode targets
        B, C, H, W = inputs.shape
        targets_one_hot = F.one_hot(
            targets.clamp(0, C - 1),
            num_classes=C
        ).permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Flatten spatial dimensions
        probs_flat = probs.view(B, C, -1)  # (B, C, N)
        targets_flat = targets_one_hot.view(B, C, -1)  # (B, C, N)
        
        # Compute per-class MCC
        mcc_per_class = []
        
        for c in range(C):
            # Binary classification for class c
            pred_c = probs_flat[:, c, :]  # (B, N)
            target_c = targets_flat[:, c, :]  # (B, N)
            
            # Apply focal-phi weighting
            weight = (1 - pred_c) ** self.phi
            
            # Weighted confusion matrix elements
            tp = (weight * pred_c * target_c).sum(dim=1)
            fp = (weight * pred_c * (1 - target_c)).sum(dim=1)
            fn = (weight * (1 - pred_c) * target_c).sum(dim=1)
            tn = (weight * (1 - pred_c) * (1 - target_c)).sum(dim=1)
            
            # MCC formula
            numerator = tp * tn - fp * fn
            denominator = torch.sqrt(
                (tp + fp + self.smooth) *
                (tp + fn + self.smooth) *
                (tn + fp + self.smooth) *
                (tn + fn + self.smooth)
            )
            
            mcc_c = numerator / (denominator + self.smooth)
            mcc_per_class.append(mcc_c)
        
        # Average MCC across classes
        mcc = torch.stack(mcc_per_class, dim=1).mean()
        
        # Return loss: 1 - MCC
        return 1.0 - mcc


class CombinedLoss(nn.Module):
    """
    Combined loss with progressive ramp for boundary loss.
    
    Loss = w1*Focal + w2*Dice + w3*MCC + w4(t)*Boundary
    
    where w4(t) ramps from w4_start to w4_final over ramp_epochs.
    
    Args:
        focal_weight: Weight for focal loss
        dice_weight: Weight for dice loss
        mcc_weight: Weight for MCC loss
        boundary_weight: Final weight for boundary loss
        boundary_ramp: Whether to ramp boundary weight
        boundary_ramp_start: Starting weight for boundary (fraction of final)
        boundary_ramp_epochs: Number of epochs to ramp over
        focal_alpha: Class weights for focal loss
        focal_gamma: Gamma parameter for focal loss
        dice_smooth: Smoothing for dice loss
        mcc_phi: Phi parameter for MCC loss
    """
    
    def __init__(
        self,
        focal_weight: float = 0.25,
        dice_weight: float = 0.25,
        mcc_weight: float = 0.35,
        boundary_weight: float = 0.15,
        boundary_ramp: bool = True,
        boundary_ramp_start: float = 0.05,
        boundary_ramp_epochs: int = 30,
        focal_alpha: Optional[List[float]] = None,
        focal_gamma: float = 2.0,
        dice_smooth: float = 1.0,
        mcc_phi: float = 0.7
    ):
        super().__init__()
        
        # Loss components
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss(smooth=dice_smooth)
        self.boundary = BoundaryLoss()
        self.mcc = MCCLoss(phi=mcc_phi)
        
        # Weights
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.mcc_weight = mcc_weight
        self.boundary_weight = boundary_weight
        
        # Ramp configuration
        self.boundary_ramp = boundary_ramp
        self.boundary_ramp_start = boundary_ramp_start
        self.boundary_ramp_epochs = boundary_ramp_epochs
        
        # Current epoch (updated externally)
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """Update current epoch for boundary ramp."""
        self.current_epoch = epoch
    
    def get_boundary_weight(self) -> float:
        """Compute current boundary weight with ramp."""
        if not self.boundary_ramp:
            return self.boundary_weight
        
        if self.current_epoch >= self.boundary_ramp_epochs:
            return self.boundary_weight
        
        # Linear ramp from start to final
        start_weight = self.boundary_weight * self.boundary_ramp_start
        progress = self.current_epoch / self.boundary_ramp_epochs
        current_weight = start_weight + progress * (self.boundary_weight - start_weight)
        
        return current_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Args:
            inputs: (B, C, H, W) - logits
            targets: (B, H, W) - class indices
        
        Returns:
            dict with 'loss' and individual components
        """
        # Compute individual losses
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        boundary_loss = self.boundary(inputs, targets)
        mcc_loss = self.mcc(inputs, targets)
        
        # Get current boundary weight
        current_boundary_weight = self.get_boundary_weight()
        
        # Combined loss
        total_loss = (
            self.focal_weight * focal_loss +
            self.dice_weight * dice_loss +
            self.mcc_weight * mcc_loss +
            current_boundary_weight * boundary_loss
        )
        
        return {
            'loss': total_loss,
            'focal_loss': focal_loss.item(),
            'dice_loss': dice_loss.item(),
            'boundary_loss': boundary_loss.item(),
            'mcc_loss': mcc_loss.item(),
            'boundary_weight': current_boundary_weight
        }


if __name__ == '__main__':
    # Test losses
    B, C, H, W = 2, 4, 64, 64
    
    # Dummy data
    inputs = torch.randn(B, C, H, W)
    targets = torch.randint(0, C, (B, H, W))
    
    # Test individual losses
    print("Testing losses...")
    
    focal = FocalLoss(alpha=[1.0, 2.0, 3.0, 4.0])
    print(f"Focal Loss: {focal(inputs, targets).item():.4f}")
    
    dice = DiceLoss()
    print(f"Dice Loss: {dice(inputs, targets).item():.4f}")
    
    boundary = BoundaryLoss()
    print(f"Boundary Loss: {boundary(inputs, targets).item():.4f}")
    
    mcc = MCCLoss()
    print(f"MCC Loss: {mcc(inputs, targets).item():.4f}")
    
    # Test combined loss
    combined = CombinedLoss(
        focal_weight=0.25,
        dice_weight=0.25,
        mcc_weight=0.35,
        boundary_weight=0.15,
        focal_alpha=[1.0, 2.0, 3.0, 4.0]
    )
    
    for epoch in [0, 15, 30]:
        combined.set_epoch(epoch)
        result = combined(inputs, targets)
        print(f"\nEpoch {epoch}:")
        print(f"  Total Loss: {result['loss'].item():.4f}")
        print(f"  Focal: {result['focal_loss']:.4f}")
        print(f"  Dice: {result['dice_loss']:.4f}")
        print(f"  MCC: {result['mcc_loss']:.4f}")
        print(f"  Boundary: {result['boundary_loss']:.4f}")
        print(f"  Boundary Weight: {result['boundary_weight']:.4f}")
    
    print("\n✓ All losses working!")
