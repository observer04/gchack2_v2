# V2 Notebook Blueprint: Evidence-Based Improvements

## Overview

Based on critical analysis of Perplexity recommendations + current implementation + actual training results, this blueprint defines **exactly what to change** for V2.

**Philosophy:** Keep what works, fix what's broken, add only proven techniques.

---

## CHANGES SUMMARY

### ✅ **IMPLEMENT (High Priority)**

1. **Replace custom ratios with NDSI/NDWI** - Literature-standard glacier indices
2. **Fix loss curriculum** - Reduce MCC dominance (currently causing loss increase)
3. **Simplify loss function** - Remove sklearn MCC from training, use only for validation
4. **Add TTA wrapper** - 4-fold test-time augmentation for submission

### 🔧 **TUNE (Medium Priority)**

5. **Test higher learning rate** - 1e-4 vs current 8e-5
6. **Adjust class weights** - Test [1, 5, 20, 80] if training stable

### ❌ **DO NOT CHANGE**

7. Keep U-Net architecture (NOT DeepLabV3+)
8. Keep EfficientNet-B3 encoder (NOT B4)
9. Keep 384 patch size (NOT 512)
10. Keep tile-level sampling (NOT pixel-level)
11. NO CRF post-processing
12. NO pseudo-labeling (test set hidden)

---

## DETAILED IMPLEMENTATION PLAN

### CHANGE 1: Feature Engineering - NDSI/NDWI Indices ✅

**Current (WRONG):**
```python
green_swir = np.divide(green, swir + eps)  # Unbounded, unstable
swir_tir = np.divide(swir, tir + eps)
swir_tir_log = np.log1p(swir_tir)  # log(unbounded) = still unstable
```

**V2 (CORRECT):**
```python
# NDSI = Normalized Difference Snow Index (standard glacier index)
ndsi = (green - swir) / (green + swir + eps)  # Bounded [-1, 1]

# NDWI = Normalized Difference Water Index (standard lake/water index)
ndwi = (green - tir) / (green + tir + eps)  # Bounded [-1, 1]
```

**Rationale:**
- ✅ NDSI/NDWI are **literature-standard** for glacier/water detection
- ✅ Normalized differencing creates **bounded features** [-1, 1] → better numerical stability
- ✅ Standard normalization → consistent across different images
- 📊 **Expected gain:** +0.02-0.04 MCC from stability + domain knowledge

**Code change:**
```python
def stack_bands_with_indices(band1, band2, band3, band4, band5, *, eps: float = 1e-8):
    """
    Stack 7 channels with standard remote sensing indices:
    - Channels 0-2: RGB (pretrained)
    - Channels 3-4: SWIR, TIR (zero-init)
    - Channel 5: NDSI = (Green - SWIR)/(Green + SWIR) - glacier detection
    - Channel 6: NDWI = (Green - TIR)/(Green + TIR) - water/lake detection
    """
    blue = band1.astype(np.float32)
    green = band2.astype(np.float32)
    red = band3.astype(np.float32)
    swir = band4.astype(np.float32)
    tir = band5.astype(np.float32)
    
    # Standard glacier indices (literature-based)
    ndsi = (green - swir) / (green + swir + eps)  # [-1, 1]
    ndwi = (green - tir) / (green + tir + eps)    # [-1, 1]
    
    return np.stack([blue, green, red, swir, tir, ndsi, ndwi], axis=-1)
```

---

### CHANGE 2: Simplified Loss Function ✅

**Current Problem:**
```python
# sklearn MCC has NO GRADIENTS (discrete metric)
mcc = matthews_corrcoef(target_flat, pred_classes)  # Non-differentiable!
# Using this as 70% of loss → model random walks
```

**Root Cause:** MCC loss with 70% weight caused:
```
Epoch 1: Loss=0.7682
Epoch 10: Loss=0.7834
Epoch 11: Loss=0.8427 (JUMPED!)  ← MCC weight went 50%→70%
```

**V2 Solution: REMOVE MCC FROM TRAINING LOSS**

```python
class SimplifiedLoss(nn.Module):
    """
    Proven 3-component loss for extreme imbalance.
    NO MCC in training (it's non-differentiable).
    """
    def __init__(self, num_classes=4, class_weights=None):
        super().__init__()
        self.focal = smp.losses.FocalLoss(mode='multiclass', gamma=3.0, alpha=None)
        self.dice = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
        self.lovasz = smp.losses.LovaszLoss(mode='multiclass')
        
        # Class weights for Focal loss
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def forward(self, logits, target):
        """Fixed weights - NO curriculum complexity"""
        focal_loss = self.focal(logits, target)
        dice_loss = self.dice(logits, target)
        lovasz_loss = self.lovasz(logits, target)
        
        # Fixed combination (no epoch-dependent curriculum)
        total = 0.5 * focal_loss + 0.3 * dice_loss + 0.2 * lovasz_loss
        
        return total, {
            'focal': focal_loss.item(),
            'dice': dice_loss.item(),
            'lovasz': lovasz_loss.item()
        }
```

**Validation MCC (separate):**
```python
def validate(model, val_loader, config):
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)
            logits = model(images)
            preds = logits.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
    
    # Compute MCC ONLY for validation (not training)
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    mcc = matthews_corrcoef(all_targets, all_preds)
    
    return mcc
```

**Rationale:**
- ✅ Focal (50%) - Handles extreme imbalance (1468:1 ratio)
- ✅ Dice (30%) - Handles small objects (lakes)
- ✅ Lovász (20%) - Handles boundary precision
- ✅ NO MCC in training - Use only for validation/selection
- ✅ NO curriculum - Simpler, more stable
- 📊 **Expected:** Stable training loss, no random jumps

---

### CHANGE 3: Moderate Class Weights ✅

**Current:**
```python
CLASS_WEIGHTS = [1.0, 2.5, 10.0, 40.0]
class_weights = priors^-0.5  # Aggressive: [0.5, 2.0, 10.0, 40.0] approx
```

**V2: Test Two Configurations**

**Config A (Conservative - Start Here):**
```python
CLASS_WEIGHTS = [1.0, 3.0, 12.0, 50.0]  # Moderate increase
```

**Config B (Aggressive - If A Stable):**
```python
CLASS_WEIGHTS = [1.0, 5.0, 20.0, 80.0]  # 2x increase for minorities
```

**Rationale:**
- Lake is 0.05% = 1/2000 pixels
- Current 40x might be too low
- BUT 100x (Perplexity) risks gradient explosion
- **Strategy:** Start moderate, increase if stable

---

### CHANGE 4: Test-Time Augmentation (TTA) ✅

**Add for Final Submission:**

```python
def tta_predict(model, image, device):
    """
    4-fold TTA: Original + Horizontal Flip + Vertical Flip + Both
    Returns averaged probabilities.
    """
    model.eval()
    predictions = []
    
    # Original
    with torch.no_grad():
        out = model(image.to(device))
        predictions.append(torch.softmax(out, dim=1))
    
    # Horizontal flip
    with torch.no_grad():
        flipped = torch.flip(image, dims=[3])  # Width dimension
        out = model(flipped.to(device))
        out = torch.flip(out, dims=[3])
        predictions.append(torch.softmax(out, dim=1))
    
    # Vertical flip
    with torch.no_grad():
        flipped = torch.flip(image, dims=[2])  # Height dimension
        out = model(flipped.to(device))
        out = torch.flip(out, dims=[2])
        predictions.append(torch.softmax(out, dim=1))
    
    # Both flips
    with torch.no_grad():
        flipped = torch.flip(image, dims=[2, 3])
        out = model(flipped.to(device))
        out = torch.flip(out, dims=[2, 3])
        predictions.append(torch.softmax(out, dim=1))
    
    # Average predictions
    avg_pred = torch.stack(predictions).mean(dim=0)
    return avg_pred.argmax(dim=1)
```

**Usage:**
```python
# For submission/inference only (not training)
final_pred = tta_predict(model, image, device)
```

**Expected gain:** +0.02-0.04 MCC

---

### CHANGE 5: Tune Learning Rate 🔧

**Current:**
```python
BASE_LR = 8e-5
ENCODER_LR_SCALE = 0.1  # Encoder: 8e-6
```

**V2: Test Higher**
```python
BASE_LR = 1e-4  # Perplexity's recommendation
ENCODER_LR_SCALE = 0.1  # Encoder: 1e-5
```

**Rationale:**
- Perplexity: 1e-4 converges faster (good for 12h Kaggle limit)
- Risk: Might be unstable with 20 images
- **Strategy:** Try 1e-4 first, if loss spikes, fallback to 8e-5

---

### CHANGE 6: Update Channel Names/Comments

**Current:**
```python
CHANNEL_NAMES = [
    'Band1_Blue', 'Band2_Green', 'Band3_Red',
    'Band4_SWIR', 'Band5_TIR',
    'Green_over_SWIR',
    'log1p_SWIR_over_TIR'
]
```

**V2:**
```python
CHANNEL_NAMES = [
    'Band1_Blue', 'Band2_Green', 'Band3_Red',
    'Band4_SWIR', 'Band5_TIR',
    'NDSI',  # Normalized Difference Snow Index
    'NDWI'   # Normalized Difference Water Index
]
```

---

## WHAT TO KEEP (DO NOT CHANGE)

### ✅ Architecture
```python
model = smp.Unet(
    encoder_name='timm-efficientnet-b3',  # NOT B4 (overkill for 20 images)
    encoder_weights='noisy-student',
    in_channels=7,
    classes=4
)
```

**Rationale:** U-Net's simple decoder prevents overfitting on small datasets better than DeepLabV3+'s complex ASPP decoder.

### ✅ Training Strategy
```python
CROP_SIZE = 384  # NOT 512 (better for tiny lake class)
CROPS_PER_TILE = 24
MINORITY_FOCUS_PROB = 0.40
LAKE_BOOST_PROB = 0.20
```

**Rationale:** Our tile-level weighted sampling + minority cropping is optimal for small datasets.

### ✅ Augmentation
```python
# Current augmentation probabilities (50-80%) are already aggressive
# DO NOT increase further (risk of unrealistic samples)
```

### ✅ Optimizer
```python
# Keep Lookahead wrapper around AdamW
# It's proven to help with small datasets
optimizer = Lookahead(AdamW(...))
```

---

## V2 CONFIGURATION

```python
class ConfigV2:
    # Data
    DATA_ROOT = ...  # Auto-resolved
    ENCODER = 'timm-efficientnet-b3'  # Keep
    ENCODER_WEIGHTS = 'noisy-student'  # Keep
    IN_CHANNELS = 7  # Keep (5 bands + NDSI + NDWI)
    NUM_CLASSES = 4
    
    # Training
    CROP_SIZE = 384  # Keep
    CROPS_PER_TILE = 24  # Keep
    MINORITY_FOCUS_PROB = 0.40  # Keep
    LAKE_BOOST_PROB = 0.20  # Keep
    
    BATCH_SIZE = 32  # Keep (single GPU)
    NUM_WORKERS = 2  # Keep (stable)
    NUM_EPOCHS = 150  # Keep
    
    # CHANGED: Higher learning rate (test)
    BASE_LR = 1e-4  # Was 8e-5
    ENCODER_LR_SCALE = 0.1
    WEIGHT_DECAY = 5e-4
    
    # CHANGED: Removed encoder freeze (already 0, keep it)
    FREEZE_ENCODER_EPOCHS = 0  # Good!
    
    # CHANGED: Test higher class weights
    CLASS_WEIGHTS = torch.tensor([1.0, 3.0, 12.0, 50.0])  # Was [1, 2.5, 10, 40]
    
    # Keep
    GRAD_CLIP = 1.0
    PATIENCE = 30
    MIXED_PRECISION = True
```

---

## EXPECTED PERFORMANCE

### Realistic Targets (V2):

**Single Model:**
- Baseline: 0.60-0.65 MCC
- With tuning: 0.65-0.70 MCC

**With TTA (4-fold):**
- +0.02-0.04 MCC
- **Final: 0.67-0.74 MCC**

**With 5-Fold CV + TTA:**
- +0.03-0.05 MCC more
- **Final: 0.70-0.78 MCC**

### Perplexity's Prediction: ❌
- 0.85-0.91 MCC **UNREALISTIC** with 20 training images
- Would require 200+ images or external pretraining data

---

## IMPLEMENTATION CHECKLIST

### Phase 1: Core Fixes (Do First)
- [ ] Replace Green/SWIR, log(SWIR/TIR) with NDSI, NDWI
- [ ] Remove sklearn MCC from training loss
- [ ] Simplify to 3-component loss (Focal + Dice + Lovász)
- [ ] Update channel names/comments
- [ ] Test higher LR (1e-4)

### Phase 2: Optimization (Do Second)
- [ ] Test higher class weights [1, 3, 12, 50]
- [ ] If stable, test [1, 5, 20, 80]
- [ ] Monitor training loss stability

### Phase 3: Inference (Do Last)
- [ ] Implement TTA wrapper
- [ ] Test TTA on validation set
- [ ] Verify MCC improvement

---

## VALIDATION STRATEGY

After each change:
1. Train for 30 epochs
2. Check if:
   - Loss is **decreasing** (not increasing!)
   - Val MCC > 0.05 (model learning minorities)
   - Debris MCC > 0.01
   - Lake MCC > 0.005
3. If ANY metric worse, **revert change**

---

## SUCCESS CRITERIA

**V2 is successful if:**
- ✅ Training loss **decreases** consistently (no jumps)
- ✅ Val MCC reaches 0.65+ (single model, no TTA)
- ✅ Debris MCC > 0.05 (was ~0.00 before)
- ✅ Lake MCC > 0.01 (was negative before)
- ✅ Training completes 150 epochs without kernel crash

**V2 failed if:**
- ❌ Loss still increases after epoch 10
- ❌ Val MCC stuck < 0.10
- ❌ Lake MCC still negative
- ❌ Kernel crashes

---

## CONCLUSION

**V2 Strategy:**
1. **Fix the broken** - Remove MCC from training loss, use NDSI/NDWI
2. **Keep what works** - U-Net, tile sampling, minority focusing
3. **Add proven techniques** - TTA for inference
4. **Set realistic goals** - 0.65-0.70 MCC, not 0.85+

**Perplexity Score: 4/10**
- Good: Feature indices, class imbalance awareness
- Bad: Architecture change, CRF, pseudo-labeling, unrealistic targets
- Missing: Understanding of extreme data scarcity (20 images)

**V2 Expected: 0.65-0.74 MCC** (realistic, achievable)
