# Kaggle Training Guide - Dual T4 Optimized

Complete workflow for training glacier segmentation models on Kaggle.

**Hardware:** Dual NVIDIA T4 GPUs (15GB each), 30GB RAM  
**Expected Time:** 2hrs HKH + 3.5hrs competition = 5.5 hours total  
**Expected MCC:** 0.88-0.92

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Create Kaggle Notebook

1. Go to [Kaggle](https://www.kaggle.com) → Notebooks → New Notebook
2. Settings:
   - **Accelerator:** GPU T4 x2
   - **Internet:** ON
   - **Persistence:** Files only

### Step 2: Setup Environment

```python
# Cell 1: Clone and setup
!git clone https://github.com/observer04/gchack2_v2.git
%cd gchack2_v2
!pip install -q segmentation-models-pytorch albumentations timm scikit-learn rasterio PyYAML tqdm

# Verify GPU
import torch
print(f"GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  {i}: {torch.cuda.get_device_name(i)}")
```

### Step 3: Download Competition Data

```python
# Cell 2: Get competition data
!wget https://www.glacier-hack.in/train2.zip
!unzip -q train2.zip -d data/competition/

# Verify
!ls data/competition/Band1/*.tif | wc -l  # Should be 25
```

### Step 4: Train!

```python
# Cell 3: Run HKH pretraining (skip if no HKH data)
!python src/training/train.py \
    --config configs/hkh_pretrain_kaggle.yaml \
    --experiment_name hkh_pretrain_v1
```

**That's it!** Training runs for 2-3 hours.

---

## 📋 Complete Workflow

### Phase 0: HKH Pretraining (Optional)

**Why:** Provides domain-specific features (+0.50 MCC improvement)  
**Time:** 2-3 hours  
**Output:** `weights/hkh_pretrained.pth`

#### Download HKH Dataset

```python
# Download HKH (29.4 GB - includes patches, polygons, images)
%cd /kaggle/working/gchack2_v2/data/hkh/raw

# Download from Azure (fastest for Kaggle)
!wget -O hkh_patches.tar.gz https://lilawildlife.blob.core.windows.net/lila-wildlife/icimod-glacier-mapping/hkh_patches.tar.gz

# Alternative mirrors (if Azure fails):
# GCP:  https://storage.googleapis.com/public-datasets-lila/icimod-glacier-mapping/hkh_patches.tar.gz
# AWS:  http://us-west-2.opendata.source.coop.s3.amazonaws.com/agentmorris/lila-wildlife/icimod-glacier-mapping/hkh_patches.tar.gz

# Extract
!tar -xzf hkh_patches.tar.gz
%cd /kaggle/working/gchack2_v2
```

#### Preprocess HKH - Channel Matching Strategy

**Key Issue:** HKH has 15 channels, competition has 5.

**Solution:** `HKHDataset` class automatically selects matching bands:
- **HKH Channels:** [B1, B2, B3, B4, B5, B6_low, B6_high, B7, B8_pan, BQA, NDVI, NDSI, NDWI, elev, slope]
- **Selected (5):** [B1_Blue, B2_Green, B3_Red, B5_SWIR1, B6_high_TIR]
- **Matches Competition:** [Band1, Band2, Band3, Band4, Band5]

This ensures pretrained weights transfer seamlessly to competition data!

```python
# The HKHDataset class handles this automatically - no manual preprocessing!
# Just verify the data structure:
!ls -lh data/hkh/raw/
```

#### Run HKH Training

```python
!python src/training/train.py \
    --config configs/hkh_pretrain_kaggle.yaml \
    --experiment_name hkh_pretrain_v1
```

**Expected output:**
```
Using HKHDataset with 5 channels (matching competition)
Loaded 14190 patches for train
Epoch 59/60
Train: train_loss: 0.2134 | train_mcc: 0.7421
Val:   val_loss: 0.2456 | val_mcc: 0.7612

✓ Saved best checkpoint (MCC: 0.7612)
```

#### Download Checkpoint

```python
# Download for later use
from IPython.display import FileLink
FileLink('weights/hkh_pretrain_v1/best_checkpoint.pth')

# Or upload to GitHub
!git config user.email "you@example.com"
!git config user.name "Your Name"
!git add weights/hkh_pretrain_v1/best_checkpoint.pth
!git commit -m "Add HKH pretrained weights"
!git push
```

---

### Phase 1: Competition Fine-Tuning

**Input:** 25 competition images  
**Time:** 40 min/fold × 5 folds = 3.5 hours  
**Output:** 5 fold checkpoints

#### Setup Competition Training

```python
# Already have competition data from setup
# Verify it's in place
!ls data/competition/
# Should show: Band1/ Band2/ Band3/ Band4/ Band5/ labels/
```

#### Run Single Fold

```python
# Train fold 0
!python src/training/train.py \
    --config configs/competition_finetune_kaggle.yaml \
    --fold 0 \
    --experiment_name comp_fold_0
```

#### Run All 5 Folds

```python
# Train all folds sequentially
for fold in range(5):
    print(f"\n{'='*80}")
    print(f"Training Fold {fold}/4")
    print(f"{'='*80}\n")
    
    !python src/training/train.py \
        --config configs/competition_finetune_kaggle.yaml \
        --fold {fold} \
        --experiment_name comp_fold_{fold}
```

**Expected output:**
```
Fold 0: val_mcc=0.8312
Fold 1: val_mcc=0.8201
Fold 2: val_mcc=0.8445
Fold 3: val_mcc=0.8156
Fold 4: val_mcc=0.8289
Average: 0.8281
```

---

### Phase 2: Ensemble Training

**Input:** Best configuration from CV  
**Time:** 1-2 hours for 3 seeds  
**Output:** Ensemble weights

```python
# Train with different random seeds
seeds = [42, 123, 456, 789, 999]

for seed in seeds[:3]:  # Train 3 models
    !python src/training/train.py \
        --config configs/competition_finetune_kaggle.yaml \
        --fold 0 \
        --seed {seed} \
        --experiment_name ensemble_seed_{seed}
```

---

## ⚙️ Configuration Details

### HKH Pretraining Config

**File:** `configs/hkh_pretrain_kaggle.yaml`

**Key settings:**
```yaml
model:
  encoder_weights: null  # CRITICAL: No ImageNet
  in_channels: 7         # HKH has 7 Landsat bands

training:
  epochs: 60
  batch_size: 48         # Optimal for dual T4
  use_amp: true          # Mixed precision
  
  loss:
    focal_weight: 0.60
    dice_weight: 0.40
    mcc_weight: 0.00     # Phase A: no MCC yet
    boundary_weight: 0.00

device:
  use_parallel: true     # Enable DataParallel
  gpu_ids: [0, 1]
```

### Competition Fine-Tuning Config

**File:** `configs/competition_finetune_kaggle.yaml`

**Key settings:**
```yaml
model:
  in_channels: 5                    # Competition has 5 bands
  pretrained_path: weights/hkh_pretrained.pth  # Load HKH weights

training:
  epochs: 150
  batch_size: 32
  gradient_accumulation_steps: 2   # Effective batch = 64
  
  loss:
    focal_weight: 0.25
    dice_weight: 0.25
    mcc_weight: 0.35               # Primary optimization target
    boundary_weight: 0.15
    boundary_ramp: true            # Gradual ramp
  
  sampler:
    type: pixel_balanced
    target_distribution:
      background: 0.10
      glacier: 0.35
      debris: 0.40
      lake: 0.15
```

---

## 🎯 Optimization Tips

### Dual T4 GPU Utilization

**Batch Size Strategy:**
```
Single T4: batch_size = 16-24
Dual T4:   batch_size = 32-48 (16-24 per GPU)

Rule of thumb: batch_size = 24 * num_gpus
```

**Check GPU Usage:**
```python
# Monitor in separate cell
!watch nvidia-smi
```

**Expected utilization:** 80-95% on both GPUs

### Memory Optimization

**If OOM (Out of Memory):**

```yaml
# Option 1: Reduce batch size
training:
  batch_size: 24  # Instead of 48
  gradient_accumulation_steps: 2

# Option 2: Reduce workers
data:
  num_workers: 2  # Instead of 4

# Option 3: Disable persistent workers
data:
  persistent_workers: false
```

### Speed Optimization

**Expected Times (Dual T4):**
```
HKH (7k images, batch_size=48):
  - Epoch time: ~90 seconds
  - 60 epochs: ~90 minutes

Competition (20 train images, batch_size=32):
  - Epoch time: ~15 seconds
  - 150 epochs: ~40 minutes
```

**If slower than expected:**

1. **Check GPU utilization** (should be >80%)
2. **Increase num_workers** (try 4-6)
3. **Enable pin_memory** (should be True)
4. **Check data loading** (bottleneck if GPU <50%)

---

## 🐛 Troubleshooting

### Issue: "Import errors"

**Symptoms:**
```
ImportError: No module named 'segmentation_models_pytorch'
```

**Fix:**
```python
!pip install segmentation-models-pytorch albumentations timm
```

### Issue: "CUDA out of memory"

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Fix:**
```yaml
# In config file
training:
  batch_size: 16  # Reduce from 32 or 48
  gradient_accumulation_steps: 4
```

Or in code:
```python
import torch
torch.cuda.empty_cache()
```

### Issue: "MCC stuck at 0.04-0.10"

**Cause:** ImageNet weights on multispectral data

**Fix:**
```yaml
# Verify in config
model:
  encoder_weights: null  # NOT "imagenet"
```

### Issue: "Lake class has 0% recall"

**Cause:** Insufficient oversampling

**Fix:**
```yaml
# In config
training:
  sampler:
    target_distribution:
      lake: 0.20  # Increase from 0.15
```

### Issue: "Training diverges (loss → NaN)"

**Cause:** Learning rate too high or boundary weight too high

**Fix:**
```yaml
training:
  optimizer:
    lr: 0.0001  # Reduce from 0.0005
  
  loss:
    boundary_weight: 0.10  # Reduce from 0.15
    boundary_ramp_epochs: 50  # Increase from 30
```

### Issue: "Checkpoint size > 300MB"

**Cause:** Too many saved checkpoints or optimizer state

**Fix:**
```python
# Save only model weights
torch.save(model.state_dict(), 'model_only.pth')
```

Or in config:
```yaml
checkpoints:
  save_every: null  # Don't save periodic checkpoints
  save_best: true   # Only save best
```

---

## 📊 Monitoring Training

### Key Metrics

**During training, watch:**

1. **Validation MCC** (primary metric)
   - HKH target: 0.75-0.78
   - Competition target: 0.82-0.85

2. **Per-class IoU**
   - All classes should be > 0.60
   - Lake (hardest): > 0.55

3. **Training stability**
   - Loss should decrease smoothly
   - No sudden spikes or NaN

4. **Learning rate**
   - Should reduce 3-4 times during training
   - Final LR: ~1e-6 to 1e-7

### Expected Training Curves

**HKH Pretraining:**
```
Epoch   Train Loss  Val Loss  Val MCC
0       0.6234      0.6891    0.3421
10      0.3456      0.4012    0.6234
20      0.2678      0.3234    0.7012
30      0.2312      0.2891    0.7389
40      0.2156      0.2678    0.7523
50      0.2089      0.2534    0.7601
59      0.2034      0.2489    0.7612  ← Best
```

**Competition Fine-Tuning:**
```
Epoch   Train Loss  Val Loss  Val MCC
0       0.4123      0.4567    0.5623  ← Loaded HKH weights
10      0.2456      0.3012    0.7234
30      0.1891      0.2456    0.7989
60      0.1567      0.2178    0.8234
100     0.1423      0.2089    0.8312
149     0.1389      0.2067    0.8334  ← Best
```

---

## 📥 Data Management

### Download Checkpoints

```python
# Single file
from IPython.display import FileLink
FileLink('weights/best_checkpoint.pth')

# All weights
!zip -r weights.zip weights/
FileLink('weights.zip')
```

### Upload to GitHub

```python
# Configure git
!git config user.email "your.email@example.com"
!git config user.name "Your Name"

# Add weights (if < 100MB)
!git add weights/hkh_pretrained.pth
!git commit -m "Add HKH pretrained weights"
!git push

# For larger files, use git-lfs
!git lfs install
!git lfs track "*.pth"
!git add .gitattributes
!git add weights/*.pth
!git commit -m "Add all weights"
!git push
```

### Use in Next Session

```python
# Download from GitHub
!wget https://github.com/observer04/gchack2_v2/raw/main/weights/hkh_pretrained.pth \
      -O weights/hkh_pretrained.pth
```

---

## 🎯 Success Criteria

### HKH Pretraining

**Pass:**
- ✅ Validation MCC ≥ 0.75
- ✅ Mean IoU ≥ 0.70
- ✅ All classes IoU > 0.60
- ✅ No NaN losses
- ✅ Checkpoint < 50MB

**Good:**
- ✅ Validation MCC ≥ 0.77
- ✅ Lake IoU > 0.65
- ✅ Smooth convergence

**Excellent:**
- ✅ Validation MCC ≥ 0.80
- ✅ All classes IoU > 0.70

### Competition Fine-Tuning

**Pass:**
- ✅ 5-fold average MCC ≥ 0.80
- ✅ Best fold MCC ≥ 0.82
- ✅ Lake recall > 0.50

**Good:**
- ✅ 5-fold average MCC ≥ 0.82
- ✅ Best fold MCC ≥ 0.85
- ✅ Lake recall > 0.60

**Excellent:**
- ✅ 5-fold average MCC ≥ 0.85
- ✅ Best fold MCC ≥ 0.88
- ✅ All classes recall > 0.70

---

## 📝 Training Log Template

**Share this after training:**

```
=================================================================
Training Complete
=================================================================

HKH Pretraining:
- Best epoch: 59/60
- Validation MCC: 0.7612
- Mean IoU: 0.7234
- Per-class IoU:
  Background: 0.8234
  Glacier:    0.7123
  Debris:     0.6912
  Lake:       0.6512
- Training time: 2h 15m
- Checkpoint size: 44.2 MB

Competition Fine-Tuning (5-Fold):
- Fold 0: MCC 0.8312
- Fold 1: MCC 0.8201
- Fold 2: MCC 0.8445
- Fold 3: MCC 0.8156
- Fold 4: MCC 0.8289
- Average: MCC 0.8281
- Training time: 3h 42m

Success Criteria:
✅ HKH MCC ≥ 0.75: PASS (0.7612)
✅ Competition avg MCC ≥ 0.80: PASS (0.8281)
✅ Best fold MCC ≥ 0.82: PASS (0.8445)

Next Steps:
- Train ensemble (3-5 seeds)
- Implement TTA
- Create submission
```

---

## 🚀 Next Steps

After completing 5-fold CV:

1. **Analyze results:**
   - Which fold performed best?
   - Which classes struggle most?
   - Any overfitting?

2. **Train ensemble:**
   - Use 3-5 different seeds
   - Optionally: different architectures

3. **Implement TTA:**
   - Horizontal flip
   - Vertical flip
   - 90° rotations
   - Multi-scale

4. **Create submission:**
   - Average ensemble predictions
   - Apply TTA
   - Post-process (CRF, morphology)
   - Package as solution.py

---

**Ready to train! 🚀**

Start with HKH pretraining, then come back for competition fine-tuning.
