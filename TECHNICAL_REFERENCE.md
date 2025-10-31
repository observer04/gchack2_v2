# Technical Reference - Architecture & Implementation

Complete technical documentation for the glacier segmentation pipeline.

---

## 📐 Model Architecture

### Boundary-Aware U-Net

**Design Philosophy:**
- Encoder-decoder architecture for dense prediction
- Skip connections for preserving spatial details
- Attention mechanisms for boundary refinement
- NO ImageNet pretraining (multispectral data)

### Encoder: ResNet34

```python
from segmentation_models_pytorch.encoders import get_encoder

encoder = get_encoder(
    'resnet34',
    in_channels=5,      # or 7 for HKH
    weights=None,       # CRITICAL: No ImageNet
    depth=5
)

# Output channels: [3, 64, 64, 128, 256, 512]
```

**Why ResNet34?**
- Good balance: 11.2M parameters (not too large)
- Proven performance on segmentation tasks
- Fast training on dual T4 (~90 sec/epoch for HKH)

**Why `weights=None`?**
- ImageNet trained on RGB (3 channels)
- Our data: 5-7 multispectral bands
- Different spectral characteristics
- V1 showed: ImageNet → MCC 0.04, From scratch → MCC 0.75+

### Decoder: U-Net with cSE Attention

**Architecture:**
```
Encoder Output (512 channels, H/32 × W/32)
    ↓
DecoderBlock 1: Upsample → Concat(skip4) → Conv → cSE → 256 channels
    ↓
DecoderBlock 2: Upsample → Concat(skip3) → Conv → cSE → 128 channels
    ↓
DecoderBlock 3: Upsample → Concat(skip2) → Conv → cSE → 64 channels
    ↓
DecoderBlock 4: Upsample → Concat(skip1) → Conv → cSE → 32 channels
    ↓
DecoderBlock 5: Upsample → Conv → cSE → 16 channels
    ↓
Segmentation Head: Conv1x1 → 4 classes
```

**Channel-Spatial Squeeze & Excitation (cSE):**

```python
class ChannelSpatialSELayer(nn.Module):
    def __init__(self, num_channels, reduction=2):
        # Channel SE: Global pool → FC → FC → Sigmoid
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_channels, num_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(num_channels // reduction, num_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial SE: Conv → Sigmoid
        self.sSE = nn.Sequential(
            nn.Conv2d(num_channels, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.cSE(x) * x + self.sSE(x) * x
```

**Why cSE?**
- Recalibrates features both channel-wise and spatially
- Helps focus on boundaries (critical for MCC)
- Adds minimal parameters (~1% increase)
- Empirically: +0.03-0.05 MCC improvement

### Model Summary

```
Total parameters: 11,271,874 (11.3M)
Trainable parameters: 11,271,874
FP32 size: 43.0 MB
FP16 size (with AMP): 21.5 MB
```

**Memory usage (batch_size=48):**
- Model: ~2 GB per GPU
- Activations: ~10 GB per GPU
- Optimizer state: ~2 GB per GPU
- **Total: ~14 GB per GPU** (safe for 15GB T4)

---

## 🎯 Loss Functions

### Combined Loss with Progressive Ramp

```python
Loss(t) = w1*Focal + w2*Dice + w3*MCC + w4(t)*Boundary

where w4(t) = ramp from w4_start to w4_final over T epochs
```

### 1. Focal Loss

**Purpose:** Handle extreme class imbalance (background:lake = 1468:1)

**Formula:**
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

**Parameters:**
- `alpha = [1.0, 2.0, 3.0, 4.0]` (BG, Glacier, Debris, Lake)
- `gamma = 2.0` (focusing parameter)

**Implementation highlights:**
```python
class FocalLoss(nn.Module):
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, C).permute(0, 3, 1, 2)
        
        # Get probability of correct class
        p_t = (probs * targets_one_hot).sum(dim=1)
        
        # Focal weight
        focal_weight = (1 - p_t) ** gamma
        
        # Class weight
        alpha_t = (alpha * targets_one_hot).sum(dim=1)
        
        # Loss
        loss = -alpha_t * focal_weight * (log_probs * targets_one_hot).sum(dim=1)
        return loss.mean()
```

**Why it works:**
- Down-weights easy examples (high p_t)
- Up-weights hard examples (low p_t)
- Class-specific weights for rare classes

### 2. Dice Loss

**Purpose:** Optimize for overlap (IoU-like)

**Formula:**
```
DiceLoss = 1 - (2*|X ∩ Y| + smooth) / (|X| + |Y| + smooth)
```

**Parameters:**
- `smooth = 1.0` (Laplace smoothing)

**Implementation:**
```python
class DiceLoss(nn.Module):
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, C).permute(0, 3, 1, 2)
        
        # Per-class Dice
        intersection = (probs * targets_one_hot).sum(dim=(2,3))
        cardinality = probs.sum(dim=(2,3)) + targets_one_hot.sum(dim=(2,3))
        
        dice = (2 * intersection + smooth) / (cardinality + smooth)
        return (1 - dice).mean()
```

**Why it works:**
- Differentiable approximation of IoU
- Handles class imbalance better than CE
- Smooth gradients even for small objects

### 3. Boundary Loss

**Purpose:** Sharpen class boundaries

**Method:**
1. Detect boundaries (where neighboring pixels differ)
2. Weight loss higher near boundaries
3. Encourages model to focus on decision boundaries

**Implementation:**
```python
class BoundaryLoss(nn.Module):
    def _compute_boundaries(self, targets):
        # Horizontal and vertical gradients
        h_diff = (targets[:, :, :-1] != targets[:, :, 1:])
        v_diff = (targets[:, :-1, :] != targets[:, 1:, :])
        # ... combine and pad
        return boundaries
    
    def forward(self, logits, targets):
        boundaries = self._compute_boundaries(targets)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Weight: 1.0 interior, 3.0 boundary
        weight = 1.0 + 2.0 * boundaries
        return (ce_loss * weight).mean()
```

**Why it works:**
- Focuses learning on hard-to-segment boundaries
- Glacier-debris boundaries are critical for MCC
- Lake boundaries are often ambiguous

### 4. MCC Loss (Focal-Phi)

**Purpose:** Direct optimization of competition metric

**Formula:**
```
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

MCCLoss = 1 - MCC_weighted

where MCC_weighted uses (1-p)^φ weighting
```

**Parameters:**
- `phi = 0.7` (focal-phi parameter)

**Implementation:**
```python
class MCCLoss(nn.Module):
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        
        # Per-class MCC
        for c in range(C):
            pred_c = probs[:, c]
            target_c = (targets == c).float()
            
            # Focal-phi weighting
            weight = (1 - pred_c) ** phi
            
            # Weighted confusion matrix
            tp = (weight * pred_c * target_c).sum()
            fp = (weight * pred_c * (1 - target_c)).sum()
            fn = (weight * (1 - pred_c) * target_c).sum()
            tn = (weight * (1 - pred_c) * (1 - target_c)).sum()
            
            mcc_c = (tp*tn - fp*fn) / sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn))
        
        return 1 - mcc_c.mean()
```

**Why it works:**
- Directly optimizes evaluation metric
- Focal-phi handles imbalance
- Stable gradients with smoothing

### 5. Combined Loss with Ramp

**Phase A (HKH Pretraining):**
```python
Loss = 0.60*Focal + 0.40*Dice + 0.00*MCC + 0.00*Boundary
```

**Phase B (Competition Fine-Tuning):**
```python
Loss = 0.25*Focal + 0.25*Dice + 0.35*MCC + w_boundary(t)*Boundary

where w_boundary(t) ramps from 0.0075 to 0.15 over 30 epochs
```

**Ramp schedule:**
```python
def get_boundary_weight(epoch, max_epochs=30):
    start_weight = 0.15 * 0.05  # 5% of final
    final_weight = 0.15
    
    if epoch >= max_epochs:
        return final_weight
    
    progress = epoch / max_epochs
    return start_weight + progress * (final_weight - start_weight)
```

**Why progressive ramp?**
- MCC loss unstable in early training (predictions random)
- Boundary loss requires decent predictions first
- Gradual introduction = stable convergence
- V1 failed with full MCC from epoch 0

---

## 📊 Data Pipeline

### Dataset: GlacierDataset

**Multi-band loading:**
```python
class GlacierDataset:
    def __getitem__(self, idx):
        # Load 5 or 7 band image
        image = np.stack([
            rasterio.open(band1_path).read(1),
            rasterio.open(band2_path).read(1),
            # ...
        ], axis=0)  # Shape: (C, H, W)
        
        # Load mask
        mask = rasterio.open(mask_path).read(1)  # Shape: (H, W)
        
        # Normalize (per-band statistics)
        image = (image - mean) / std
        
        # GLCM texture features (optional)
        if use_glcm:
            glcm_features = compute_glcm(image)
            image = np.concatenate([image, glcm_features], axis=0)
        
        # Augmentation
        if mode == 'train':
            augmented = self.transform(image=image.transpose(1,2,0), 
                                       mask=mask)
            image = augmented['image'].transpose(2,0,1)
            mask = augmented['mask']
        
        return {
            'image': torch.from_numpy(image).float(),
            'mask': torch.from_numpy(mask).long()
        }
```

**Normalization statistics:**
```python
# Computed from HKH dataset (NOT ImageNet!)
# Used for both HKH and competition
mean = [b1_mean, b2_mean, b3_mean, b4_mean, b5_mean]
std = [b1_std, b2_std, b3_std, b4_std, b5_std]
```

### Sampler: PixelBalancedSampler

**Problem:** Extreme class imbalance
```
Background: 73.4%  (1468x more than lake)
Glacier:    15.2%
Debris:     11.3%
Lake:        0.05%  (rarest class)
```

**Image-level sampling fails:**
```python
# Even with weighted sampling
weighted_sampler = WeightedRandomSampler(weights=class_weights)
# Result: 62% background in batches (not 10% target)
```

**Solution: Pixel-level balanced sampling**

```python
class PixelBalancedSampler:
    def __init__(self, dataset, target_distribution):
        # Precompute pixel counts per image
        self.class_counts = []
        for idx in range(len(dataset)):
            mask = dataset[idx]['mask']
            counts = {c: (mask == c).sum() for c in range(4)}
            self.class_counts.append(counts)
        
        # Compute sampling weights
        self.weights = self._compute_weights(target_distribution)
    
    def _compute_weights(self, target_dist):
        weights = []
        for counts in self.class_counts:
            # Weight by deviation from target distribution
            current_dist = normalize(counts)
            deviation = sum(abs(current_dist[c] - target_dist[c]) 
                           for c in range(4))
            weights.append(1.0 / (deviation + 0.01))
        return weights
    
    def __iter__(self):
        # Sample with replacement using weights
        indices = torch.multinomial(
            self.weights, 
            num_samples=len(self.dataset),
            replacement=True
        )
        return iter(indices.tolist())
```

**Result:**
```
Target:  BG 10%, Glacier 35%, Debris 40%, Lake 15%
Actual:  BG 12%, Glacier 34%, Debris 38%, Lake 16%
```

**Impact:** +0.15 MCC improvement over image-level sampling

---

## 🏋️ Training Pipeline

### Trainer Class

**Key features:**
- Automatic Mixed Precision (AMP)
- Gradient accumulation
- Gradient clipping
- ReduceLROnPlateau scheduler
- Checkpointing
- Metrics tracking

**Training loop:**
```python
def train_epoch(self, train_loader, epoch):
    self.model.train()
    self.criterion.set_epoch(epoch)  # Update boundary ramp
    
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward with AMP
        with autocast():
            outputs = self.model(images)
            loss_dict = self.criterion(outputs, masks)
            loss = loss_dict['loss'] / grad_accum_steps
        
        # Backward
        self.scaler.scale(loss).backward()
        
        # Optimizer step (every N batches)
        if (batch_idx + 1) % grad_accum_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
    
    # Update scheduler
    self.scheduler.step(val_metrics['mcc'])
```

### Optimization

**AdamW optimizer:**
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0005,        # HKH
    # lr=0.0001,      # Competition (lower for fine-tuning)
    weight_decay=0.0001,
    betas=(0.9, 0.999)
)
```

**Why AdamW?**
- Better generalization than Adam
- Decoupled weight decay
- Standard for vision tasks

**ReduceLROnPlateau scheduler:**
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',      # Maximize MCC
    factor=0.5,      # Reduce by 50%
    patience=5,      # Wait 5 epochs
    min_lr=1e-7
)
```

**Why not CosineAnnealingWarmRestarts?**
- V1 showed: crashes minority class gradients
- Sudden LR jumps destabilize MCC loss
- ReduceLROnPlateau: smooth, adaptive

**Expected LR schedule:**
```
Epoch  0-15:  lr = 0.0005
Epoch 16-30:  lr = 0.00025  (1st reduction)
Epoch 31-45:  lr = 0.000125 (2nd reduction)
Epoch 46-59:  lr = 0.0000625 (3rd reduction)
```

### Gradient Management

**Gradient clipping:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
- Prevents exploding gradients
- Critical for MCC loss stability

**Gradient accumulation:**
```python
# Competition: small batch_size=32, accumulate 2 steps
# Effective batch size = 32 * 2 = 64
```
- Simulates larger batches
- Better gradient estimates
- More stable training

---

## 📈 Metrics

### Matthews Correlation Coefficient (MCC)

**Formula:**
```
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

**Range:** -1 to +1
- +1: Perfect prediction
- 0: Random prediction  
- -1: Perfect inverse prediction

**Why MCC?**
- Handles class imbalance better than accuracy
- Single metric for multi-class
- Competition evaluation metric

**Implementation:**
```python
from sklearn.metrics import matthews_corrcoef

def compute_mcc(preds, targets):
    preds_flat = preds.cpu().numpy().flatten()
    targets_flat = targets.cpu().numpy().flatten()
    return matthews_corrcoef(targets_flat, preds_flat)
```

### Per-Class Metrics

**Intersection over Union (IoU):**
```
IoU_c = TP_c / (TP_c + FP_c + FN_c)
```

**Precision, Recall, F1:**
```
Precision_c = TP_c / (TP_c + FP_c)
Recall_c = TP_c / (TP_c + FN_c)
F1_c = 2 * Precision_c * Recall_c / (Precision_c + Recall_c)
```

### Metric Tracker

```python
class MetricTracker:
    def update(self, preds, targets, loss):
        self.all_preds.append(preds.cpu())
        self.all_targets.append(targets.cpu())
    
    def compute(self):
        preds = torch.cat(self.all_preds)
        targets = torch.cat(self.all_targets)
        
        return {
            'mcc': compute_mcc(preds, targets),
            'mean_iou': compute_iou(preds, targets),
            'macro_f1': compute_f1(preds, targets),
            # ... per-class metrics
        }
```

---

## 🎛️ Hyperparameters

### HKH Pretraining

```yaml
epochs: 60
batch_size: 48
lr: 0.0005
optimizer: AdamW
scheduler: ReduceLROnPlateau
  patience: 5
  factor: 0.5
loss:
  focal: 0.60
  dice: 0.40
augmentation:
  flip_h: 0.5
  flip_v: 0.5
  rotate: 45°
  scale: ±20%
```

### Competition Fine-Tuning

```yaml
epochs: 150
batch_size: 32
gradient_accumulation: 2
lr: 0.0001  # Lower for fine-tuning
optimizer: AdamW
scheduler: ReduceLROnPlateau
  patience: 8  # More patience
  factor: 0.5
loss:
  focal: 0.25
  dice: 0.25
  mcc: 0.35
  boundary: 0.15 (ramped)
sampler: pixel_balanced
augmentation:
  flip_h: 0.5
  flip_v: 0.5
  rotate: 90°
  scale: ±30%
  elastic: true
  grid_distortion: true
```

---

## 🔬 Design Decisions

### Why These Choices?

**encoder_weights=None:**
- ImageNet: RGB, natural images
- Our data: 5-7 multispectral bands, satellite imagery
- Spectral characteristics completely different
- V1 proof: ImageNet → MCC 0.04, Scratch → MCC 0.75+

**HKH Pretraining:**
- Competition: only 25 images
- HKH: 7,229 tiles
- Domain-specific features (glaciers, satellite data)
- Expected: +0.50 MCC over no pretraining

**Pixel-Balanced Sampling:**
- Class imbalance: 1468:1 (BG:Lake)
- Image-level sampling: still 62% BG in batches
- Pixel-level: true control over batch composition
- Expected: +0.15 MCC over image-level

**Progressive Loss Ramp:**
- MCC loss unstable early (random predictions)
- Boundary loss needs decent features first
- Gradual introduction = stable convergence
- V1 proof: Full MCC from start → training collapse

**ReduceLROnPlateau:**
- Adaptive to validation performance
- No manual schedule tuning
- Smooth LR reduction
- V1 proof: CosineAnnealing → minority class crash

---

**For implementation details, see source code in `src/`**
