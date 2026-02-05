# Pre-Flight Checklist: Baseline MCC Training Notebook

## ✅ Critical Fixes Verification

### Data Pipeline
- [x] Band loading uses Band1-5 (not Band2,3,4,5,10)
- [x] Label remapping implemented: 0/85/170/255 → 0/1/2/3
- [x] SWIR/TIR ratio computed correctly: `band4 / (band5 + 1e-8)`
- [x] 6 channels stacked: [Blue, Green, Red, SWIR, TIR, SWIR/TIR_ratio]
- [x] Label remapping called BEFORE augmentation in `__getitem__`

### Model Architecture
- [x] EfficientNet-B3 encoder with noisy-student weights
- [x] First conv layer expanded from 3 to 6 channels
- [x] Pretrained RGB weights copied, SWIR/TIR/ratio initialized with mean
- [x] DataParallel enabled for dual GPU training

### Training Configuration
- [x] `IN_CHANNELS = 6`
- [x] `BATCH_SIZE = 12` (per GPU)
- [x] `GRAD_ACCUM_STEPS = 2` (effective batch = 48)
- [x] `CLASS_WEIGHTS = [1.0, 2.5, 10.0, 40.0]`
- [x] `FREEZE_ENCODER_EPOCHS = 5`
- [x] `MIXED_PRECISION = True`

### Loss & Optimization
- [x] DynamicCurriculumLoss with 4 components (Focal, Dice, Lovász, MCC)
- [x] Progressive curriculum: ramp MCC weight from 0.0 → 0.9 over 100 epochs
- [x] Lookahead optimizer wrapper (alpha=0.5, k=5)
- [x] Warmup (3 epochs) + Cosine annealing scheduler
- [x] Gradient clipping at norm 1.0

### Augmentation
- [x] ElasticTransform for geological deformation (p=0.3)
- [x] GridDistortion for terrain variation (p=0.3)
- [x] CoarseDropout for robustness (p=0.3)
- [x] Heavy augmentation justified for 25-image dataset

### Sampling & Data Loading
- [x] Minority-focused cropping (60% probability)
- [x] Weighted sampler for debris/lake oversampling
- [x] Stratified train/val split
- [x] 24 crops per tile (600 crops per epoch)
- [x] `num_workers=4`, `persistent_workers=True`, `pin_memory=True`

### Training Orchestration
- [x] Encoder frozen for first 5 epochs
- [x] Encoder unfrozen at epoch 6 with 10× reduced LR
- [x] Early stopping with patience=15
- [x] Best model saved based on validation MCC
- [x] Training history tracked (loss, MCC, debris_mcc, lake_mcc)

### Visualization & Output
- [x] Training curves plotted (loss, overall MCC, debris MCC, lake MCC)
- [x] Best model checkpoint saved to `/kaggle/working/best_model.pth`
- [x] Visualization saved to `/kaggle/working/training_curves.png`
- [x] Final model loaded for inference

## 🚀 Ready to Execute

### Expected Runtime
- **100 epochs**: ~8-10 hours on dual T4 GPUs
- **Early stopping**: May finish earlier if MCC plateaus

### Expected Performance
- **Overall MCC**: 0.70-0.80
- **Debris MCC**: 0.60-0.75
- **Lake MCC**: 0.50-0.70

### Kaggle Path Configuration
All paths correctly set for Kaggle environment:
- Data: `/kaggle/working/Train/` with Band1-5 and labels subfolders
- Output: `/kaggle/working/best_model.pth`
- Plots: `/kaggle/working/training_curves.png`

## 🔍 Quick Validation Commands

After uploading to Kaggle, run these in a cell to verify setup:

```python
# Verify data paths
from pathlib import Path
data_root = Path('/kaggle/working/Train')
print("Data folders:", list(data_root.iterdir()))
print("Band1 files:", len(list((data_root / 'Band1').glob('*.tif'))))
print("Label files:", len(list((data_root / 'labels').glob('*.tif'))))

# Verify GPU
import torch
print(f"\nGPUs available: {torch.cuda.device_count()}")
print(f"GPU 0: {torch.cuda.get_device_name(0)}")
if torch.cuda.device_count() > 1:
    print(f"GPU 1: {torch.cuda.get_device_name(1)}")

# Verify label encoding
from PIL import Image
sample_mask = np.array(Image.open(data_root / 'labels' / '1.tif'))
unique_vals = np.unique(sample_mask)
print(f"\nLabel values: {unique_vals}")
print("Expected: [0, 85, 170, 255] ✓" if set(unique_vals).issubset({0, 85, 170, 255}) else "ERROR")

# Verify 6-channel input
from notebooks.baseline_mcc_training import config
print(f"\nModel input channels: {config.IN_CHANNELS}")
print("Expected: 6 ✓" if config.IN_CHANNELS == 6 else "ERROR")
```

## 📊 Monitoring During Training

Watch for these signals:

### Good Signs ✅
- Training loss decreases steadily for first 20 epochs
- Validation MCC increases from ~0.3 → 0.6+ by epoch 30
- Debris MCC > 0.5 by epoch 40
- Lake MCC > 0.4 by epoch 50
- Curriculum transitions: "Ramp-up MCC" → "Balanced blend" → "Focus MCC"

### Warning Signs ⚠️
- Training loss < 0.1 but val MCC < 0.5: Overfitting (increase augmentation)
- Lake MCC stuck at 0.0: Class weight too low (increase to 80×)
- Debris MCC < 0.4 by epoch 50: Boundary loss weight too low (increase to 0.25)
- Val MCC drops after unfreezing encoder: Unfreeze LR too high (reduce to 1e-6)

### Red Flags 🚨
- Training loss NaN: Gradient explosion (check gradient clipping)
- Val MCC decreases monotonically: Wrong label remapping (verify {0:0,85:1,170:2,255:3})
- GPU OOM: Reduce batch size to 10 or crop size to 384

## 🎯 Success Criteria

Baseline is successful if:
1. Training completes without errors
2. Overall validation MCC > 0.65
3. Debris MCC > 0.55
4. Lake MCC > 0.40
5. Best model checkpoint saved correctly

If all criteria met → proceed to test-time augmentation and ensembling
If criteria not met → analyze failure modes with prediction visualizations

---

**Status**: All systems go for Kaggle execution ✅
**Last Updated**: Notebook fully fixed with all EDA recommendations
