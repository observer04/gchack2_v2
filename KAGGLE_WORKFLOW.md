# Kaggle Training Workflow

## 🚀 Environment Specs

**Kaggle GPU Instance:**
- **RAM:** 30 GB
- **GPUs:** Dual NVIDIA T4 (15GB each)
- **Storage:** ~70 GB temp space
- **Internet:** Enabled for git clone and downloads

**Advantages over Local:**
- ✅ 2× T4 GPUs → 2× faster training with DataParallel
- ✅ 30 GB RAM → larger batch sizes (32-48 vs 16-24)
- ✅ No thermal throttling
- ✅ Persistent notebooks for reproducibility

---

## 📋 Workflow Overview

```
Local Development (Your Laptop)
    ↓ Code → GitHub
Kaggle Notebook
    ↓ Clone repo → Load data → Train
Local Development
    ↓ Download weights → Evaluate → Submit
```

### Division of Labor

**On Your Laptop:**
- ✅ Code development (models, losses, trainers)
- ✅ Configuration editing
- ✅ Documentation
- ✅ Small experiments (no GPU needed)
- ✅ Final submission packaging

**On Kaggle:**
- ✅ HKH preprocessing (compute-intensive slicing)
- ✅ Model training (GPU-intensive)
- ✅ Hyperparameter tuning
- ✅ Ensemble training

---

## 🔧 Kaggle Setup Notebook

### Cell 1: Environment Setup

```python
# Clone repository
!git clone https://github.com/observer04/gchack2_v2.git
%cd gchack2_v2

# Download competition data
!wget https://www.glacier-hack.in/train2.zip
!unzip train2.zip -d /kaggle/working/
```

### Cell 2: Install Dependencies

```python
# Install required packages
!pip install -q segmentation-models-pytorch albumentations timm scikit-learn

# Verify GPU setup
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

### Cell 3: Data Organization

```python
import os
import shutil
from pathlib import Path

# Organize competition data
comp_data_dir = Path('/kaggle/working/gchack2_v2/data/competition')
comp_data_dir.mkdir(parents=True, exist_ok=True)

# Move extracted data to proper location
train_source = Path('/kaggle/working/Train')
if train_source.exists():
    for band_dir in train_source.iterdir():
        dest = comp_data_dir / band_dir.name
        shutil.move(str(band_dir), str(dest))
    print("✓ Competition data organized")

# Verify structure
print("\nData structure:")
for item in sorted(comp_data_dir.iterdir()):
    if item.is_dir():
        count = len(list(item.glob('*.tif')))
        print(f"  {item.name}: {count} files")
```

### Cell 4: Download HKH Dataset

```python
# Download HKH dataset (7GB compressed)
!mkdir -p /kaggle/working/gchack2_v2/data/hkh/raw
%cd /kaggle/working/gchack2_v2/data/hkh/raw

# Option 1: Direct download from Lila.Science
!wget https://lila.science/wp-content/uploads/2020/06/hkh-glacier-mapping.zip
!unzip hkh-glacier-mapping.zip

# Option 2: From Kaggle Dataset (if uploaded)
# from kaggle.api.kaggle_api_extended import KaggleApi
# api = KaggleApi()
# api.authenticate()
# api.dataset_download_files('dataset-name', path='.', unzip=True)

%cd /kaggle/working/gchack2_v2
print("✓ HKH dataset downloaded")
```

---

## ⚙️ Optimized Configurations for Kaggle

### Dual T4 Optimizations

**Batch Size Strategy:**
```yaml
# Single T4: 15GB VRAM
# Conservative: batch_size=16 (uses ~12GB)
# Optimal: batch_size=24 (uses ~14GB)

# Dual T4 with DataParallel: 30GB total
# batch_size=32 → 16 per GPU
# batch_size=48 → 24 per GPU (optimal)
```

**Training Speed:**
- Single T4: ~150 images/sec
- Dual T4 with DataParallel: ~280 images/sec (1.87× speedup)
- Expected HKH epoch time: ~90 seconds (7k images)

### Updated Config: `configs/hkh_pretrain_kaggle.yaml`

```yaml
data:
  hkh_dir: "/kaggle/working/gchack2_v2/data/hkh/processed"
  train_split: 0.85
  val_split: 0.15
  num_workers: 4  # Kaggle has 4 CPUs
  
model:
  architecture: "boundary_aware_unet"
  encoder: "resnet34"
  encoder_weights: null  # Train from scratch
  in_channels: 7  # HKH has 7 Landsat bands
  num_classes: 4
  
training:
  epochs: 60  # Reduced from 80 (dual T4 is faster)
  batch_size: 48  # Optimal for dual T4
  gradient_accumulation_steps: 1  # No need with large batch
  use_amp: true  # Mixed precision
  
  optimizer:
    type: "AdamW"
    lr: 0.0005  # Increased from 0.0003 (larger batch)
    weight_decay: 0.0001
  
  scheduler:
    type: "ReduceLROnPlateau"
    mode: "max"
    factor: 0.5
    patience: 5
    min_lr: 1.0e-6
  
  loss:
    focal_weight: 0.60
    dice_weight: 0.40
    boundary_weight: 0.00  # Phase A: no boundary loss yet
  
device:
  use_parallel: true  # Enable DataParallel for dual T4
  gpu_ids: [0, 1]
  
checkpoints:
  save_dir: "/kaggle/working/gchack2_v2/weights"
  save_best: true
  save_every: 10
```

### Updated Config: `configs/competition_finetune_kaggle.yaml`

```yaml
data:
  comp_dir: "/kaggle/working/gchack2_v2/data/competition"
  num_folds: 5
  current_fold: 0  # Will iterate 0-4
  num_workers: 4
  
model:
  architecture: "boundary_aware_unet"
  encoder: "resnet34"
  encoder_weights: null
  in_channels: 5  # Competition has 5 bands
  num_classes: 4
  pretrained_path: "/kaggle/working/gchack2_v2/weights/hkh_pretrained.pth"
  
training:
  epochs: 150
  batch_size: 32  # Smaller than HKH (only 20 images for training)
  gradient_accumulation_steps: 2  # Effective batch = 64
  use_amp: true
  
  optimizer:
    type: "AdamW"
    lr: 0.0001  # Lower for fine-tuning
    weight_decay: 0.0001
  
  scheduler:
    type: "ReduceLROnPlateau"
    mode: "max"
    factor: 0.5
    patience: 8
    min_lr: 1.0e-7
  
  loss:
    focal_weight: 0.25
    dice_weight: 0.25
    mcc_weight: 0.35
    boundary_weight: 0.15
    boundary_ramp: true
    boundary_ramp_epochs: 30
  
  sampler:
    type: "pixel_balanced"
    target_distribution:
      background: 0.10
      glacier: 0.35
      debris: 0.40
      lake: 0.15
    oversample_rare: true
  
device:
  use_parallel: true
  gpu_ids: [0, 1]
  
checkpoints:
  save_dir: "/kaggle/working/gchack2_v2/weights/folds"
  save_best: true
  save_every: 25
```

---

## 📊 Expected Performance on Dual T4

### HKH Pretraining

**Dataset:** 7,229 tiles (512×512)
- Training: 6,145 tiles
- Validation: 1,084 tiles

**Time Estimates:**
```
Epoch time: ~90 seconds
60 epochs: ~90 minutes (1.5 hours)
With validation: ~2 hours total
```

**Memory Usage:**
```
Batch size 48:
- Model: ~2 GB per GPU
- Data: ~10 GB per GPU
- Activations: ~2 GB per GPU
Total: ~14 GB per GPU (safe for 15GB T4)
```

### Competition Fine-Tuning

**Dataset:** 25 images (5 folds → 20 train, 5 val per fold)

**Time Estimates:**
```
Epoch time: ~15 seconds (only 20 images)
150 epochs: ~40 minutes per fold
5 folds: ~3.5 hours total
```

**Total Training Time:**
- HKH Pretraining: 2 hours
- Competition 5-fold CV: 3.5 hours
- Ensemble (3 seeds): +2 hours
- **Total: ~7.5 hours** (fits in 1 Kaggle session)

---

## 🎯 Training Strategy

### Phase 0: HKH Pretraining (Single Run)

**Kaggle Notebook:** `hkh_pretrain.ipynb`

```python
# Run HKH pretraining
!python src/training/train.py \
    --config configs/hkh_pretrain_kaggle.yaml \
    --experiment_name hkh_pretrain_v1

# Expected output:
# Epoch 60/60: val_mcc=0.76, val_loss=0.25
# Saved: weights/hkh_pretrained.pth (44 MB)
```

**Success Criteria:**
- Validation MCC ≥ 0.75
- Per-class IoU > 0.60
- Checkpoint < 50 MB

**If Failed (MCC < 0.70):**
- Check band alignment (HKH uses different Landsat bands)
- Verify loss implementation
- Increase epochs to 80

---

### Phase 1: Competition Fine-Tuning (5-Fold CV)

**Kaggle Notebook:** `competition_finetune.ipynb`

```python
# Run all 5 folds sequentially
for fold in range(5):
    !python src/training/train.py \
        --config configs/competition_finetune_kaggle.yaml \
        --fold {fold} \
        --experiment_name comp_finetune_fold{fold}

# Expected outputs:
# Fold 0: val_mcc=0.83
# Fold 1: val_mcc=0.82
# Fold 2: val_mcc=0.84
# Fold 3: val_mcc=0.81
# Fold 4: val_mcc=0.83
# Average: 0.826
```

**Success Criteria:**
- 5-fold average MCC ≥ 0.80
- Best fold MCC ≥ 0.82
- All 5 checkpoints saved

---

### Phase 2: Ensemble Training (3-5 Seeds)

**Kaggle Notebook:** `ensemble_train.ipynb`

```python
# Train with different random seeds
seeds = [42, 123, 456]

for seed in seeds:
    !python src/training/train.py \
        --config configs/competition_finetune_kaggle.yaml \
        --fold 0 \
        --seed {seed} \
        --experiment_name ensemble_seed{seed}

# Expected ensemble MCC: 0.87-0.89
```

---

## 📤 Data Transfer Strategy

### Kaggle → Local

**After HKH Pretraining:**
```python
# In Kaggle notebook
from google.colab import files  # Works in Kaggle too
files.download('/kaggle/working/gchack2_v2/weights/hkh_pretrained.pth')
```

**After Competition Training:**
```python
# Download all fold checkpoints
import zipfile
shutil.make_archive('fold_weights', 'zip', '/kaggle/working/gchack2_v2/weights/folds')
files.download('fold_weights.zip')
```

**Alternative (Recommended):**
```python
# Push weights back to GitHub (if repo allows large files)
!git lfs track "*.pth"
!git add weights/*.pth
!git commit -m "Add trained weights"
!git push
```

---

## 🐛 Kaggle-Specific Issues

### Issue 1: Out of Memory

**Symptoms:** CUDA out of memory error

**Solutions:**
```python
# Reduce batch size
batch_size: 32  # Instead of 48

# Enable gradient accumulation
gradient_accumulation_steps: 2

# Clear cache between folds
torch.cuda.empty_cache()
```

### Issue 2: Slow Data Loading

**Symptoms:** Low GPU utilization (< 80%)

**Solutions:**
```python
# Increase num_workers
num_workers: 4  # Kaggle has 4 CPUs

# Use pin_memory
pin_memory: True

# Prefetch data
persistent_workers: True
```

### Issue 3: Session Timeout

**Symptoms:** Training interrupted after 9 hours

**Solutions:**
- Save checkpoints every 10 epochs
- Resume from last checkpoint:
  ```python
  !python train.py --resume weights/last_checkpoint.pth
  ```

### Issue 4: Disk Space

**Symptoms:** No space left on device

**Solutions:**
```python
# Delete intermediate files
!rm -rf data/hkh/raw/*.zip

# Save only best checkpoint
save_best: True
save_every: null  # Don't save intermediate
```

---

## 📝 Development Workflow

### Iteration Cycle

**Day 1-3 (Local Development):**
1. Implement loss functions, models, trainers
2. Test with dummy data (small images)
3. Push to GitHub
4. Create Kaggle notebook

**Day 4-5 (Kaggle Training):**
1. Clone repo in Kaggle
2. Run HKH pretraining
3. Monitor metrics
4. Download weights
5. Push weights to GitHub

**Day 6 (Local Analysis):**
1. Pull weights from GitHub
2. Analyze predictions on local machine
3. Identify failure cases
4. Update code

**Day 7-8 (Kaggle Training):**
1. Push updated code
2. Run competition fine-tuning (5 folds)
3. Download all fold weights
4. Push to GitHub

**Day 9 (Kaggle Ensemble):**
1. Train ensemble members
2. Test different TTA strategies
3. Download final weights

**Day 10 (Local Submission):**
1. Pull all weights
2. Generate predictions
3. Package solution.py
4. Submit

---

## 🎯 Kaggle Notebook Templates

### Template 1: `notebooks/kaggle_hkh_pretrain.ipynb`

**Purpose:** HKH dataset preprocessing and pretraining

**Cells:**
1. Environment setup + clone repo
2. Download HKH dataset
3. Preprocess HKH (slicing, filtering)
4. Compute normalization stats
5. Run training
6. Evaluate on validation set
7. Download checkpoint

**Expected runtime:** 3-4 hours

---

### Template 2: `notebooks/kaggle_competition_train.ipynb`

**Purpose:** Competition fine-tuning with 5-fold CV

**Cells:**
1. Environment setup + clone repo
2. Load competition data
3. Create CV splits
4. Load HKH pretrained weights
5. Run fold 0-4 training
6. Analyze fold results
7. Download all checkpoints

**Expected runtime:** 4-5 hours

---

### Template 3: `notebooks/kaggle_ensemble.ipynb`

**Purpose:** Ensemble training and TTA

**Cells:**
1. Environment setup + clone repo
2. Train ensemble (3-5 seeds)
3. Implement TTA
4. Generate predictions
5. Evaluate ensemble MCC
6. Download ensemble weights

**Expected runtime:** 3-4 hours

---

## ✅ Checklist Before Kaggle Run

**Code Readiness:**
- [ ] All imports use relative paths
- [ ] Configs use `/kaggle/working/` paths
- [ ] No hardcoded local paths
- [ ] Git repo is public (or API key configured)

**Dependencies:**
- [ ] requirements.txt includes all packages
- [ ] Package versions tested on Kaggle
- [ ] No conflicting dependencies

**Data Preparation:**
- [ ] Competition data URL is correct
- [ ] HKH dataset download link works
- [ ] Data organization script tested

**Training Scripts:**
- [ ] train.py accepts command-line args
- [ ] Checkpoint saving works
- [ ] Resume training works
- [ ] Logging to stdout (visible in notebook)

**GPU Optimization:**
- [ ] DataParallel enabled for dual T4
- [ ] Batch size optimized for 30GB VRAM
- [ ] AMP (mixed precision) enabled
- [ ] Gradient accumulation configured

---

## 🚀 Quick Start Commands

**Run everything in one Kaggle session:**

```bash
# Setup
!git clone https://github.com/observer04/gchack2_v2.git
%cd gchack2_v2
!pip install -q -r requirements.txt

# Get data
!wget https://www.glacier-hack.in/train2.zip && unzip train2.zip -d data/competition/
!wget https://lila.science/wp-content/uploads/2020/06/hkh-glacier-mapping.zip -P data/hkh/raw/

# HKH Pretraining (2 hours)
!python src/training/train.py --config configs/hkh_pretrain_kaggle.yaml

# Competition Fine-tuning (3.5 hours)
for fold in {0..4}; do
  python src/training/train.py --config configs/competition_finetune_kaggle.yaml --fold $fold
done

# Ensemble (2 hours)
for seed in 42 123 456; do
  python src/training/train.py --config configs/competition_finetune_kaggle.yaml --seed $seed
done

# Download weights
!zip -r all_weights.zip weights/
```

**Total time:** ~7.5 hours (fits in single Kaggle session)

---

**Next Steps:**
1. I'll create the Kaggle-optimized configs
2. Implement loss functions
3. Implement model architecture
4. Build training pipeline
5. Create Kaggle notebook templates

**You'll run:** Training on Kaggle when code is ready

**Communication:** Paste training logs here for debugging
