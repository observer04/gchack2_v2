# Baseline MCC Training Notebook - Implementation Summary

## ✅ COMPLETED: Full Baseline Notebook with All EDA-Recommended Features

### Critical Fixes Applied

#### 1. **Data Structure Corrections**
- **Fixed band loading**: Changed from non-existent `Band2,3,4,5,10` to actual `Band1,2,3,4,5` (Blue, Green, Red, SWIR, TIR)
- **Added label remapping**: Implemented `_remap_labels()` to convert 0/85/170/255 → 0/1/2/3 before augmentation
- **6-channel input**: Stack 5 raw bands + SWIR/TIR ratio (key discriminator from EDA showing 4× separation)

#### 2. **Model Architecture (EfficientNet-B3 with Noisy-Student)**
- **Pretrained weights**: Using `noisy-student` (better than ImageNet for 25-image regime)
- **First conv expansion**: Loads 3-channel pretrained weights, expands `conv_stem` to 6 channels
  - Channels 0-2: Copy RGB pretrained weights
  - Channels 3-5: Initialize with mean RGB weights (SWIR, TIR, ratio channels)
- **Encoder freezing**: First 5 epochs frozen, then unfrozen with 10× reduced LR
- **Dual GPU optimization**: DataParallel enabled for 2× T4 GPUs

#### 3. **Training Configuration (Optimized for Kaggle)**
```python
IN_CHANNELS = 6              # 5 bands + SWIR/TIR ratio
BATCH_SIZE = 12              # Per GPU (24 total)
GRAD_ACCUM_STEPS = 2         # Effective batch = 48
BASE_LR = 3e-4               
ENCODER_LR_SCALE = 0.1       # Encoder at 3e-5 when unfrozen
CLASS_WEIGHTS = [1.0, 2.5, 10.0, 40.0]  # Background, Glacier, Debris, Lake
FREEZE_ENCODER_EPOCHS = 5    # Freeze encoder initially
NUM_EPOCHS = 100
PATIENCE = 15
MIXED_PRECISION = True
```

#### 4. **Augmentation Pipeline (Heavy for 25 Images)**
```python
A.ElasticTransform(alpha=150, sigma=7.5, p=0.3)
A.GridDistortion(p=0.3)
A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3)
A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.6)
A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5)
A.GaussNoise / GaussianBlur (p=0.4)
```

#### 5. **Loss Curriculum (DynamicCurriculumLoss)**
Progressive weighting schedule over 100 epochs:

| Phase | Epochs | Focal | Dice | Boundary (Lovász) | MCC | Description |
|-------|--------|-------|------|-------------------|-----|-------------|
| 1     | 1-10   | 0.65→0.50 | 0.25 | 0.10 | 0.00→0.20 | Ramp-up MCC |
| 2     | 11-60  | 0.50→0.40 | 0.20 | 0.15 | 0.20→0.50 | Balanced blend |
| 3     | 61-100 | 0.40→0.30 | 0.15 | 0.15 | 0.50→0.90 | Focus MCC |

- **Focal**: Class-weighted with `[1.0, 2.5, 10.0, 40.0]` addressing 1468:1 imbalance
- **Dice**: Multi-class soft Dice for overlap
- **Boundary (Lovász)**: Edge-aware loss for debris-glacier boundaries (critical for MCC)
- **MCC**: Direct optimization of Matthews Correlation Coefficient

#### 6. **Sampling Strategy**
- **Minority-focused cropping**: 60% of crops target debris/lake pixels
- **Stratified split**: Ensures debris and lake presence in validation set
- **Weighted sampler**: Oversamples tiles with minority classes by 2×
- **24 crops per tile**: Maximizes effective training data (25 tiles → 600 crops per epoch)

#### 7. **Optimization Stack**
- **AdamW optimizer**: `weight_decay=1e-4`, separate LR for encoder/decoder
- **Lookahead wrapper**: `alpha=0.5, k=5` for stability with small dataset
- **Warmup + Cosine schedule**: 
  - 3 epochs linear warmup from 0.2×LR
  - 97 epochs cosine annealing to 0.1×LR
- **Gradient clipping**: Max norm 1.0 to prevent divergence
- **Mixed precision**: FP16 training with automatic loss scaling

#### 8. **Training Orchestration**
- **Encoder freezing**: Epochs 1-5 train only decoder (prevent catastrophic forgetting)
- **Unfreezing at epoch 6**: Encoder LR set to `BASE_LR × 0.1 × 0.1 = 3e-6`
- **Early stopping**: Patience 15 epochs, monitors overall validation MCC
- **Best model saving**: Saves checkpoint when validation MCC improves
- **Progress tracking**: Logs loss components, MCC per class, curriculum stage

### Dataset Implementation

```python
class GlacierDataset:
    def _load_bands(img_id):
        """Load 5 Landsat bands from Band1-5 folders"""
        band1 = Image.open(f'{DATA_ROOT}/Band1/{img_id}.tif')  # Blue
        band2 = Image.open(f'{DATA_ROOT}/Band2/{img_id}.tif')  # Green
        band3 = Image.open(f'{DATA_ROOT}/Band3/{img_id}.tif')  # Red
        band4 = Image.open(f'{DATA_ROOT}/Band4/{img_id}.tif')  # SWIR
        band5 = Image.open(f'{DATA_ROOT}/Band5/{img_id}.tif')  # TIR
        mask = Image.open(f'{DATA_ROOT}/labels/{img_id}.tif')   # 0/85/170/255
        
    def _stack_channels(band1, band2, band3, band4, band5):
        """Compute SWIR/TIR ratio and stack 6 channels"""
        swir_tir_ratio = band4 / (band5 + 1e-8)
        swir_tir_ratio = np.clip(swir_tir_ratio, 0, 1e10)
        return np.stack([band1, band2, band3, band4, band5, swir_tir_ratio], axis=-1)
    
    def _remap_labels(mask):
        """Remap 0/85/170/255 to 0/1/2/3"""
        label_mapping = {0: 0, 85: 1, 170: 2, 255: 3}
        remapped = np.zeros_like(mask)
        for orig, new in label_mapping.items():
            remapped[mask == orig] = new
        return remapped
    
    def __getitem__(idx):
        ...
        band1, band2, band3, band4, band5, mask = self._load_bands(img_id)
        image = self._stack_channels(band1, band2, band3, band4, band5)
        mask = self._remap_labels(mask)  # CRITICAL: remap before augmentation
        ...
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
        return augmented['image'], augmented['mask'].long()
```

### Model Building

```python
def build_model(config):
    # Load with 3-channel pretrained
    model = smp.Unet(encoder_name='efficientnet-b3', 
                    encoder_weights='noisy-student',
                    in_channels=3, classes=4)
    
    # Expand first conv to 6 channels
    first_conv = model.encoder.conv_stem
    old_weight = first_conv.weight.data  # [40, 3, 3, 3]
    
    new_conv = nn.Conv2d(6, 40, kernel_size=3, stride=2, padding=1)
    with torch.no_grad():
        # Copy RGB weights
        new_conv.weight[:, :3, :, :] = old_weight
        # Init SWIR, TIR, ratio channels with mean RGB
        for i in range(3, 6):
            new_conv.weight[:, i:i+1, :, :] = old_weight.mean(dim=1, keepdim=True)
        new_conv.bias.copy_(first_conv.bias)
    
    model.encoder.conv_stem = new_conv
    
    # Enable DataParallel for dual GPUs
    model = model.to(config.DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model
```

### Training Loop

```python
for epoch in range(1, NUM_EPOCHS + 1):
    # Unfreeze encoder after 5 epochs
    if epoch == 6:
        unfreeze_encoder(model, optimizer, config)
    
    # Train one epoch
    train_loss, components = train_one_epoch(...)
    val_mcc, class_mcc = validate(...)
    scheduler.step()
    
    # Track metrics
    history['val_mcc'].append(val_mcc)
    history['debris_mcc'].append(class_mcc[2])
    history['lake_mcc'].append(class_mcc[3])
    
    # Early stopping
    if val_mcc > best_mcc:
        save_checkpoint('/kaggle/working/best_model.pth')
        patience = 0
    else:
        patience += 1
    
    if patience >= 15:
        break
```

### Expected Performance

Based on EDA analysis and optimization choices:

| Metric | Target | Rationale |
|--------|--------|-----------|
| Overall MCC | 0.70-0.80 | SWIR/TIR ratio provides 4× separation glacier vs debris |
| Debris MCC | 0.60-0.75 | Boundary loss + heavy augmentation + oversampling |
| Lake MCC | 0.50-0.70 | 1468:1 imbalance, but class weight 40× + MCC loss direct optimization |
| Training time | ~8-10 hrs | 100 epochs × 25 steps/epoch × 2s/step on dual T4 |

### Output Files

1. `/kaggle/working/best_model.pth` - Best checkpoint by validation MCC
2. `/kaggle/working/training_curves.png` - Loss and MCC plots
3. Training history in `history` dict for post-analysis

### Next Steps (After This Baseline)

1. **Run baseline**: Execute notebook on Kaggle to establish MCC baseline
2. **Analyze failure modes**: Visualize predictions on validation set, identify weak tiles
3. **Iterate on weak points**:
   - If debris MCC < 0.60: Increase boundary loss weight, add edge detection preprocessing
   - If lake MCC < 0.50: Try focal loss gamma=3.0, increase class weight to 80×
   - If overfitting: Add more CoarseDropout, increase weight decay
4. **Test-time augmentation**: 4× rotation + flip averaging for final submission
5. **Ensemble**: Train 3 folds with different random crops, average predictions

### Key Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| EfficientNet-B3 over ResNet34 | 10M params vs 21M - better for 25 images, noisy-student weights |
| 6 channels not 7 | No NIR band available; SWIR/TIR ratio shown by EDA to be key discriminator |
| Lovász loss as "boundary loss" | Approximates IoU, edge-aware, works well with MCC optimization |
| Freeze encoder 5 epochs | Prevents catastrophic forgetting of pretrained features on tiny dataset |
| Batch size 12×2 | Maximal for dual T4 (16GB each) with mixed precision + 512px crops |
| 24 crops per tile | Balances diversity vs convergence speed (600 crops/epoch) |
| MCC curriculum | Early epochs learn class separation, late epochs directly optimize MCC |

### Validation

All changes validated against:
- ✅ EDA findings (SWIR/TIR ratio, RGB redundancy, class imbalance)
- ✅ Perplexity recommendations (6-channel hybrid, boundary loss, heavy augmentation)
- ✅ Kaggle constraints (dual T4 GPUs, 12hr limit, disk space)
- ✅ Independent analysis (EfficientNet-B3 > ResNet34 for 25 images)

---

**Status**: Notebook ready for execution on Kaggle
**Expected Runtime**: 8-10 hours for 100 epochs
**Risk**: Early stopping may trigger before optimal MCC (monitor debris/lake MCC trends)
