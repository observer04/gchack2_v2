# Implementation Action Plan - Updated with glacier_mapping Workflow

## 🎯 Overview

Based on the new information about the `glacier_mapping` repository and GlaViTU integration options, here's the updated implementation strategy.

---

## 📊 Strategy Comparison

### Original Plan (MONOLITH.md)
- Download raw HKH tiles → Build custom preprocessing
- Implement all components from scratch
- Standard U-Net with ResNet34 encoder
- Timeline: 12 days

### Updated Plan (With glacier_mapping)
- **Use glacier_mapping preprocessing tools** (proven, tested)
- Adapt their boundary-aware U-Net as baseline
- Leverage their filtering/slicing utilities
- Optional: Add GlaViTU as ensemble member
- Timeline: **10 days** (2 days saved on preprocessing)

**Verdict:** Updated plan is more efficient and proven. Adopt it.

---

## 🗓️ Revised Timeline (10 Days)

### Day 1: Setup & Preprocessing (6 hours)

**Morning:**
1. ✅ Clone `glacier_mapping` repo
2. ✅ Download HKH dataset from Lila.Science
3. ✅ Install dependencies

**Afternoon:**
4. Run preprocessing scripts:
   - Slice to 512×512 tiles
   - Generate masks from shapefiles
   - Filter background-heavy tiles
5. Compute HKH normalization statistics
6. Verify band alignment (competition ↔ Landsat 7)

**Output:** 
- ~7k processed tiles in `data/hkh/processed/`
- HKH statistics: `hkh_mean.npy`, `hkh_std.npy`

---

### Day 2: Loss Functions & Model Setup (8 hours)

**Loss Functions (src/losses/):**
1. `focal.py` - FocalLoss with per-class weights
2. `dice.py` - DiceLoss (multiclass)
3. `boundary.py` - BoundaryLoss with Sobel edge detection
4. `combined.py` - Wrapper with phase scheduling

**Model Setup (src/models/):**
5. Copy `glacier_mapping/models/unet.py` → `src/models/glacier_unet.py`
6. Modify: Set `encoder_weights=None`, add cSE attention
7. Test model builds without errors

**Output:**
- All loss functions implemented
- Model architecture ready

---

### Day 3: Training Pipeline (8 hours)

**Implementation (src/training/):**
1. `metrics.py` - MCC, IoU, Dice computation
2. `trainer.py` - Training loop with:
   - AMP (mixed precision)
   - Gradient clipping
   - Early stopping
   - ReduceLROnPlateau scheduler
3. `train.py` - Main script reading YAML configs

**Testing:**
4. Dry run on 1 HKH batch (sanity check)
5. Verify loss computation, backward pass, optimizer step

**Output:**
- Complete training pipeline functional
- Ready to start HKH pretraining

---

### Days 4-5: HKH Pretraining (12-16 hours)

**Configuration:**
```yaml
# configs/hkh_pretrain.yaml (updated)
data:
  train_path: data/hkh/processed/filtered/train
  val_path: data/hkh/processed/filtered/val
  normalization:
    mean: [from hkh_mean.npy]
    std: [from hkh_std.npy]

training:
  epochs: 50-80  # Adjust based on convergence
  batch_size: 16
```

**Execution:**
```bash
python src/training/train.py --config configs/hkh_pretrain.yaml
```

**Monitoring:**
- Track validation MCC (target: 0.75-0.78 by epoch 50)
- Learning rate decay events
- Early stopping triggers

**Output:**
- `weights/hkh_pretrained.pth` (~44 MB)
- Training logs in `reports/hkh_training.log`
- MCC ≥ 0.75 on HKH validation

**Milestone M1:** HKH model ready for transfer learning

---

### Day 6: Competition Data Prep & CV Splits (4 hours)

**Tasks:**
1. Create 5-fold stratified splits (by debris/lake presence)
2. Verify class distribution in each fold
3. Update competition dataset loader to use HKH normalization stats
4. Test pixel-balanced sampler on competition data

**Validation:**
```python
# Test sampler on competition data
sampler = PixelBalancedSampler(
    dataset,
    target_dist={'background': 0.10, 'glacier': 0.35, 'debris': 0.40, 'lake': 0.15},
    oversampling={'background': 1, 'glacier': 3, 'debris': 8, 'lake': 10}
)

# Sample 10 batches, measure class distribution
# Should match target within ±5%
```

**Output:**
- 5-fold splits saved in `data/competition_splits/`
- Sampler validated

---

### Days 7-8: Competition Fine-Tuning (16 hours)

**Configuration:**
```yaml
# configs/competition_finetune.yaml (updated)
model:
  pretrained_path: weights/hkh_pretrained.pth
  
data:
  normalization:
    mean: [same as HKH]  # Consistency!
    std: [same as HKH]

loss:
  # Phase B: Add MCC + Boundary ramp
  focal_weight: 0.25
  dice_weight: 0.25
  mcc_weight: 0.35
  boundary_weight: 0.15

training:
  epochs: 80-150
  lr: 5e-5  # Lower for fine-tuning
```

**Execution:**
```bash
# Train all 5 folds
for fold in {0..4}; do
    python src/training/train.py \
        --config configs/competition_finetune.yaml \
        --fold $fold \
        --pretrained weights/hkh_pretrained.pth
done
```

**Monitoring:**
- Per-fold MCC (target: 0.82-0.85)
- 5-fold average (target: 0.80-0.83)
- Confusion matrices (identify weak classes)

**Output:**
- `weights/best_fold{0-4}.pth` (5 files)
- Cross-validation summary in `reports/cv_summary.md`

**Milestone M2:** Competition model trained, MCC ≥ 0.80

---

### Day 9: Ensemble & TTA (8 hours)

**Option A: Standard Ensemble (Conservative)**

Train 2-3 additional seeds:
```bash
for seed in 1337 999; do
    python src/training/train.py \
        --config configs/competition_finetune.yaml \
        --fold 0 \
        --seed $seed
done
```

Ensemble: Average softmax from 3-5 models

**Expected Gain:** +0.03-0.05 MCC

---

**Option B: Add GlaViTU (Advanced)**

1. Clone GlaViTU repo
2. Train on HKH (parallel to BA-UNet)
3. Fine-tune on competition
4. Ensemble with BA-UNet

**Expected Gain:** +0.04-0.06 MCC (more than standard ensemble)

**Recommendation:** 
- If ahead of schedule → Try Option B
- If time-constrained → Use Option A

---

**TTA Implementation:**

```python
# src/inference/tta.py
def tta_predict(model, image):
    transforms = [
        lambda x: x,                    # Original
        lambda x: torch.flip(x, [2]),   # H-flip
        lambda x: torch.flip(x, [3]),   # V-flip
        lambda x: torch.rot90(x, 1, [2, 3]),  # 90°
        lambda x: torch.rot90(x, 2, [2, 3]),  # 180°
        lambda x: torch.rot90(x, 3, [2, 3]),  # 270°
    ]
    
    preds = []
    for aug_fn in transforms:
        aug_img = aug_fn(image)
        pred = model(aug_img)
        pred = inverse_aug(pred, aug_fn)
        preds.append(torch.softmax(pred, dim=1))
    
    return torch.stack(preds).mean(0)
```

**Output:**
- Ensemble weights optimized on validation
- TTA tested and validated

---

### Day 10: Solution.py & Submission (6 hours)

**Tasks:**

1. **Implement solution.py:**
   ```python
   def maskgeration(imagepath, out_dir):
       # Load ensemble (3-5 models)
       models = load_ensemble()
       
       for tile in get_tiles(imagepath['Band1']):
           # Load 5 bands
           image = load_multispectral(imagepath, tile)
           
           # Normalize (HKH stats)
           image = normalize(image, hkh_mean, hkh_std)
           
           # TTA ensemble
           probs = tta_ensemble_predict(models, image)
           
           # Post-process
           mask = postprocess(probs)
           
           # Map {0,1,2,3} → {0,85,170,255}
           output = encode_mask(mask)
           
           # Save
           save_mask(output, out_dir, tile)
   ```

2. **Validate:**
   - Test on validation fold
   - Verify MCC matches training results
   - Check model size < 300 MB
   - Ensure no hardcoded paths

3. **Final Testing:**
   ```bash
   python solution.py \
       --data Train \
       --masks Train/labels \
       --out predictions_test
   
   # Compute MCC on validation
   python scripts/evaluate.py \
       --predictions predictions_test \
       --labels Train/labels
   ```

4. **Package Submission:**
   - `solution.py`
   - `model.pth` (ensemble weights)
   - Verify total size < 300 MB

**Output:**
- Submission package ready
- Final validation MCC documented

**Milestone M3:** Submission ready, MCC ≥ 0.88

---

## 📋 Updated Deliverables

### Code Structure
```
src/
├── data/
│   ├── dataset.py           ✅ Implemented
│   ├── samplers.py          ✅ Implemented
│   └── hkh_preprocessing.py 🔲 To implement (Day 1)
├── losses/
│   ├── focal.py             🔲 Day 2
│   ├── dice.py              🔲 Day 2
│   ├── boundary.py          🔲 Day 2
│   └── combined.py          🔲 Day 2
├── models/
│   ├── glacier_unet.py      🔲 Day 2
│   └── ensemble.py          🔲 Day 9
├── training/
│   ├── metrics.py           🔲 Day 3
│   ├── trainer.py           🔲 Day 3
│   └── train.py             🔲 Day 3
└── inference/
    ├── tta.py               🔲 Day 9
    └── postprocess.py       🔲 Day 9
```

### Documentation
```
✅ MONOLITH.md               - Complete blueprint
✅ README.md                 - Quick start
✅ GETTING_STARTED.md        - Implementation checklist
✅ HKH_PREPROCESSING_GUIDE.md - glacier_mapping workflow
✅ PROJECT_SUMMARY.md        - Executive overview
🔲 reports/cv_summary.md     - Cross-validation results
🔲 reports/ablations.md      - Ablation study results
```

---

## 🎯 Decision Points

### Day 5 Decision: Continue HKH Training?
- **If MCC ≥ 0.75 by epoch 50:** Stop, proceed to competition fine-tuning
- **If MCC < 0.70 at epoch 50:** Investigate (check data, loss, sampling)
- **If MCC 0.70-0.75:** Continue to epoch 80, then proceed

### Day 8 Decision: GlaViTU Integration?
- **If 5-fold avg MCC ≥ 0.82:** Consider adding GlaViTU for extra boost
- **If 5-fold avg MCC < 0.80:** Focus on improving BA-UNet first
- **If time < 2 days remaining:** Skip GlaViTU, focus on TTA + post-processing

### Day 9 Decision: Ensemble Size?
- **3 models:** Safe, fits in 132 MB (plenty of headroom)
- **5 models:** Maximum diversity, 220 MB (still safe)
- **7 models:** Risk exceeding 300 MB limit

**Recommendation:** Start with 3, add more if size allows

---

## 🔍 Quality Gates

Each milestone must pass these gates before proceeding:

**M1 (HKH Pretraining):**
- [ ] Validation MCC ≥ 0.75
- [ ] Checkpoint size < 50 MB
- [ ] No NaN losses during training
- [ ] Per-class IoU > 0.60 for all classes

**M2 (Competition Fine-Tuning):**
- [ ] 5-fold avg MCC ≥ 0.80
- [ ] Single best fold MCC ≥ 0.82
- [ ] Lake class recall > 0.50
- [ ] Debris class recall > 0.70

**M3 (Final Submission):**
- [ ] Validation MCC ≥ 0.88
- [ ] Model package < 300 MB
- [ ] Output format correct (0/85/170/255)
- [ ] Inference time < 5 min per 25 images

---

## 🚀 Start Here

**Immediate Next Steps (Today):**

1. **Clone glacier_mapping:**
   ```bash
   cd data/hkh
   git clone https://github.com/krisrs1128/glacier_mapping.git
   ```

2. **Download HKH dataset:**
   ```bash
   wget https://lila.science/wp-content/uploads/2020/06/hkh-glacier-mapping.zip
   unzip hkh-glacier-mapping.zip
   ```

3. **Read HKH_PREPROCESSING_GUIDE.md** (detailed workflow)

4. **Run preprocessing scripts** (following guide)

5. **Compute normalization statistics**

---

**This updated plan leverages proven tools (glacier_mapping) while maintaining our evidence-based architectural decisions. Expected completion: 10 days to MCC 0.88-0.92.**
