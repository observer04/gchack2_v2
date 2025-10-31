"""
Metrics for glacier segmentation evaluation.

Implements:
- Matthews Correlation Coefficient (MCC) - primary metric
- Per-class IoU (Intersection over Union)
- Per-class Precision, Recall, F1
- Confusion Matrix
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
from sklearn.metrics import matthews_corrcoef, confusion_matrix


def compute_confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 4
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        pred: (B, H, W) - predicted class indices
        target: (B, H, W) - ground truth class indices
        num_classes: Number of classes
    
    Returns:
        cm: (num_classes, num_classes) confusion matrix
    """
    pred = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    
    cm = confusion_matrix(target, pred, labels=list(range(num_classes)))
    return cm


def compute_mcc(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 4
) -> float:
    """
    Compute Matthews Correlation Coefficient.
    
    MCC is the primary evaluation metric for this competition.
    
    Args:
        pred: (B, H, W) - predicted class indices
        target: (B, H, W) - ground truth class indices
        num_classes: Number of classes
    
    Returns:
        mcc: Matthews Correlation Coefficient
    """
    pred = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    
    try:
        mcc = matthews_corrcoef(target, pred)
    except:
        mcc = 0.0
    
    return mcc


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 4
) -> Dict[str, float]:
    """
    Compute per-class Intersection over Union (IoU).
    
    Args:
        pred: (B, H, W) - predicted class indices
        target: (B, H, W) - ground truth class indices
        num_classes: Number of classes
    
    Returns:
        dict with 'class_0', 'class_1', etc. and 'mean_iou'
    """
    ious = {}
    
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        ious[f'class_{c}'] = iou
    
    ious['mean_iou'] = np.mean([ious[f'class_{c}'] for c in range(num_classes)])
    
    return ious


def compute_precision_recall_f1(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 4
) -> Dict[str, float]:
    """
    Compute per-class precision, recall, F1.
    
    Args:
        pred: (B, H, W) - predicted class indices
        target: (B, H, W) - ground truth class indices
        num_classes: Number of classes
    
    Returns:
        dict with precision, recall, f1 for each class
    """
    metrics = {}
    
    cm = compute_confusion_matrix(pred, target, num_classes)
    
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        metrics[f'class_{c}_precision'] = precision
        metrics[f'class_{c}_recall'] = recall
        metrics[f'class_{c}_f1'] = f1
    
    # Macro averages
    metrics['macro_precision'] = np.mean([metrics[f'class_{c}_precision'] for c in range(num_classes)])
    metrics['macro_recall'] = np.mean([metrics[f'class_{c}_recall'] for c in range(num_classes)])
    metrics['macro_f1'] = np.mean([metrics[f'class_{c}_f1'] for c in range(num_classes)])
    
    return metrics


class MetricTracker:
    """
    Tracks metrics over an epoch.
    
    Accumulates predictions and targets, then computes
    metrics at the end of the epoch.
    """
    
    def __init__(self, num_classes: int = 4):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset accumulators."""
        self.all_preds = []
        self.all_targets = []
        self.loss_sum = 0.0
        self.count = 0
    
    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        loss: float = None
    ):
        """
        Update with batch predictions.
        
        Args:
            preds: (B, C, H, W) logits or (B, H, W) class indices
            targets: (B, H, W) class indices
            loss: Optional loss value
        """
        # Convert logits to class indices if needed
        if preds.dim() == 4:
            preds = torch.argmax(preds, dim=1)
        
        self.all_preds.append(preds.cpu())
        self.all_targets.append(targets.cpu())
        
        if loss is not None:
            self.loss_sum += loss * targets.size(0)
            self.count += targets.size(0)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            dict with all metrics
        """
        # Concatenate all predictions and targets
        preds = torch.cat(self.all_preds, dim=0)
        targets = torch.cat(self.all_targets, dim=0)
        
        # Compute metrics
        metrics = {}
        
        # MCC (primary metric)
        metrics['mcc'] = compute_mcc(preds, targets, self.num_classes)
        
        # IoU
        iou_metrics = compute_iou(preds, targets, self.num_classes)
        metrics.update(iou_metrics)
        
        # Precision, Recall, F1
        prf_metrics = compute_precision_recall_f1(preds, targets, self.num_classes)
        metrics.update(prf_metrics)
        
        # Average loss
        if self.count > 0:
            metrics['loss'] = self.loss_sum / self.count
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        preds = torch.cat(self.all_preds, dim=0)
        targets = torch.cat(self.all_targets, dim=0)
        return compute_confusion_matrix(preds, targets, self.num_classes)


def format_metrics(metrics: Dict[str, float], prefix: str = '') -> str:
    """
    Format metrics for pretty printing.
    
    Args:
        metrics: dict of metric values
        prefix: Optional prefix (e.g., 'train_', 'val_')
    
    Returns:
        formatted string
    """
    # Primary metrics
    main_metrics = ['loss', 'mcc', 'mean_iou', 'macro_f1']
    
    parts = []
    for key in main_metrics:
        if key in metrics:
            parts.append(f"{prefix}{key}: {metrics[key]:.4f}")
    
    return " | ".join(parts)


def format_class_metrics(metrics: Dict[str, float], class_names: list = None) -> str:
    """
    Format per-class metrics.
    
    Args:
        metrics: dict of metric values
        class_names: Optional class names (default: ['BG', 'Glacier', 'Debris', 'Lake'])
    
    Returns:
        formatted string
    """
    if class_names is None:
        class_names = ['BG', 'Glacier', 'Debris', 'Lake']
    
    lines = []
    lines.append("\nPer-class metrics:")
    lines.append("-" * 80)
    lines.append(f"{'Class':<12} {'IoU':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    lines.append("-" * 80)
    
    for c, name in enumerate(class_names):
        iou = metrics.get(f'class_{c}', 0)
        prec = metrics.get(f'class_{c}_precision', 0)
        rec = metrics.get(f'class_{c}_recall', 0)
        f1 = metrics.get(f'class_{c}_f1', 0)
        
        lines.append(f"{name:<12} {iou:>8.4f} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f}")
    
    lines.append("-" * 80)
    lines.append(f"{'Mean':<12} {metrics.get('mean_iou', 0):>8.4f} "
                 f"{metrics.get('macro_precision', 0):>8.4f} "
                 f"{metrics.get('macro_recall', 0):>8.4f} "
                 f"{metrics.get('macro_f1', 0):>8.4f}")
    lines.append("-" * 80)
    
    return "\n".join(lines)


if __name__ == '__main__':
    # Test metrics
    print("Testing metrics...")
    
    # Dummy data
    B, H, W = 4, 128, 128
    num_classes = 4
    
    # Simulated predictions (logits)
    logits = torch.randn(B, num_classes, H, W)
    preds = torch.argmax(logits, dim=1)
    
    # Simulated targets
    targets = torch.randint(0, num_classes, (B, H, W))
    
    # Test individual metrics
    mcc = compute_mcc(preds, targets)
    print(f"MCC: {mcc:.4f}")
    
    iou_metrics = compute_iou(preds, targets)
    print(f"Mean IoU: {iou_metrics['mean_iou']:.4f}")
    
    prf_metrics = compute_precision_recall_f1(preds, targets)
    print(f"Macro F1: {prf_metrics['macro_f1']:.4f}")
    
    # Test MetricTracker
    tracker = MetricTracker(num_classes=num_classes)
    
    # Simulate epoch
    for i in range(4):
        batch_logits = torch.randn(2, num_classes, H, W)
        batch_targets = torch.randint(0, num_classes, (2, H, W))
        tracker.update(batch_logits, batch_targets, loss=0.5)
    
    # Compute epoch metrics
    epoch_metrics = tracker.compute()
    print(f"\n{format_metrics(epoch_metrics, 'val_')}")
    print(format_class_metrics(epoch_metrics))
    
    # Confusion matrix
    cm = tracker.get_confusion_matrix()
    print(f"\nConfusion Matrix:\n{cm}")
    
    print("\n✓ Metrics working!")
