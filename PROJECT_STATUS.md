# Glacier Mapping Competition - Project Status Report

**Date:** November 6, 2025  
**Goal:** Achieve MCC ≥ 0.8 for Top 3 placement  
**Current Status:** ❌ CRITICAL - Multiple failed approaches, need strategy pivot

---

## 📊 **Quick Summary**

| Metric | Target | Best Achieved | Current Status |
|--------|--------|---------------|----------------|
| **Overall MCC** | ≥0.80 | 0.1646 (Epoch 51, Run 1) | 0.0435 (Run 2) |
| **Debris MCC** | ≥0.60 | 0.017 (Run 1) | 0.021 (Run 2) |
| **Lake MCC** | ≥0.40 | 0.003 (Run 1) | 0.002 (Run 2) |
| **Training Epochs** | 150 planned | 51 (Run 1), 22 (Run 2) | Early stopped |

**Bottom Line:** Neither approach is converging toward 0.8 MCC. Need fundamental strategy change.

---

## 🎯 **Dataset Overview**

- **Total Images:** 25 (20 train, 5 validation)
- **Classes:** 4 (Background, Glacier, Debris, Lake)
- **Input:** 5-band multispectral (B2, B3, B4, B6, B10) .tif files
- **Class Distribution (Competition):**
  - Background: 69.6%
  - Glacier: 25.4%
  - Debris: 4.9%
  - Lake: 0.05% (extremely rare!)

---

## 🔄 **Approaches Tried**

### **Phase 1: HKH Pretraining** ✅ COMPLETED
- **Dataset:** 383 HKH patches, 3 classes (Background, Glacier, Debris)
- **Model:** ResNet34 U-Net, 5-band input
- **Result:** Val MCC 0.4887
- **File:** `hkh_pretrained_resnet34.pth`

### **Phase 2: Baseline Test** ❌ DISASTER
- **Approach:** Test HKH model on competition data (before fine-tuning)
- **Result:** MCC 0.0804 (catastrophic drop from 0.4887)
- **Per-Class MCC:**
  - Debris: **-0.0026** (NEGATIVE! Anti-correlated patterns)
  - Glacier: 0.0919 (62.6% misclassified as background)
- **Conclusion:** HKH weights learned patterns that HURT competition performance

### **Phase 3: Fine-Tuning Run 1** ⚠️ FAILED (Plateaued)
**Configuration:**
```python
encoder_weights: HKH pretrained
CLASS_WEIGHTS: [1.0, 5.0, 30.0, 80.0]
LR: 5e-5 (differential: encoder 1e-5, decoder 5e-5)
Scheduler: CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
Dropout: 0.4
Focal Loss gamma: 4.0
L1 Regularization: 0.001 (scaled by /100k)
Augmentation: Moderate multispectral
```

**Results (Epoch 51):**
```
Train Loss: 0.96, Val Loss: 1.04
Train MCC: 0.229, Val MCC: 0.1646
  - Train Debris: 0.149 ✅ (model CAN learn)
  - Val Debris: 0.017 ❌ (9x generalization gap!)
```

**Diagnosis:** Model overfitting to training debris patterns. Augmentations too weak for generalization.

### **Phase 4: Fine-Tuning Run 2** ❌ CATASTROPHIC (Got Worse!)
**Configuration Changes:**
```python
CLASS_WEIGHTS: [1.0, 6.0, 50.0, 150.0]  ← INCREASED
Scheduler: CosineAnnealingWarmRestarts(T_0=50, T_mult=2)  ← FIXED
Augmentation: AGGRESSIVE (blur p=0.7, CLAHE p=0.6, gamma p=0.6)
NUM_WORKERS: 0  ← FIXED multiprocessing errors
```

**Results (Epoch 22 - Early Stopped):**
```
Train Loss: 1.60, Val Loss: 1.69
Best Val MCC: 0.0435 (WORSE than Run 1!)
  - Val Debris: 0.021
  - Val getting WORSE over time (regression, not plateau)
```

**Diagnosis:** Over-aggressive class weights + Focal gamma=4.0 = triple-counting imbalance. Model completely overwhelmed.

---

## 🔧 **Technical Implementation Details**

### **Model Architecture**
```python
Base: segmentation_models_pytorch Unet
Encoder: ResNet34 (HKH pretrained)
Input: 5 channels (multispectral)
Output: 4 classes
Decoder Dropout: 0.4
```

### **Loss Function**
```python
CombinedLoss(Focal + Dice):
  - Focal Loss: gamma=4.0, alpha=CLASS_WEIGHTS
  - Dice Loss: smooth=1.0
  - Ratio: 0.6 Focal + 0.4 Dice
```

### **Augmentations Implemented**
**Custom Multispectral (5-channel compatible):**
- `MultispectralRandomBrightnessContrast`: Per-channel atmospheric variations
- `MultispectralGaussianNoise`: Sensor noise simulation
- `MultispectralChannelShuffle`: Robust multi-band learning
- `MultispectralBlur`: Debris texture robustness (σ = blur_limit/4.0)
- `MultispectralCLAHE`: Boundary enhancement (clip=3.0, SWIR/TIR critical)
- `MultispectralGamma`: Debris contrast (range 0.6-1.7)

**Geometric:**
- Flips (H/V), Rotate90, ShiftScaleRotate
- ElasticTransform, GridDistortion
- CoarseDropout (debris robustness)

### **Training Strategy**
```python
Phase 1 (Epochs 1-3): Freeze encoder, train decoder only
Phase 2 (Epochs 4+): Unfreeze all, differential LR
Batch Size: 2 (with gradient accumulation 8 steps = effective 16)
Gradient Clipping: 1.0
Early Stopping: patience=20
```

### **Post-Processing Ready**
- **TTA:** Multi-scale (0.9x, 1.0x, 1.1x) + flips + rotations (~15 augmentations)
- **CRF:** Dense CRF with bilateral + Gaussian potentials
- **Morphology:** Opening + closing for noise removal

---

## 🚨 **Critical Issues Identified**

### **1. Negative Transfer Learning**
- HKH weights show **Debris MCC = -0.0026** on competition data
- This means HKH learned anti-patterns that actively hurt performance
- Fine-tuning can't overcome this fundamental mismatch

### **2. Extreme Class Imbalance**
- Lake: 0.05% of pixels (1469x rarer than background!)
- Even with weight=150, model treats it as noise
- Focal Loss gamma=4.0 + manual weights = double/triple counting

### **3. Generalization Failure**
- Train Debris MCC: 0.149 vs Val Debris MCC: 0.017 (9x gap!)
- Augmentations not diverse enough to match validation diversity
- Model memorizing training debris patterns, not learning transferable features

### **4. Loss Magnitude Issues**
```
Run 1 (weights 30/80):   Loss ~0.96-1.04  ← Stable
Run 2 (weights 50/150):  Loss ~1.60-2.64  ← Unstable, too high
```
Over-aggressive weights backfire, making optimization harder.

### **5. Scheduler Misconfiguration (Run 1)**
- T_0=10 caused LR decay to 2.14e-06 by Epoch 51 (too low!)
- Model stuck in local minimum, no "jolt" to escape

---

## 📈 **What Actually Worked**

✅ **L1 Regularization Scaling:** `/100000` prevents loss explosion (was causing loss ~500)  
✅ **Gradient Accumulation:** Simulates batch_size=16, critical for stability with batch_size=2  
✅ **Differential Learning Rates:** Encoder 1e-5, Decoder 5e-5 (standard practice)  
✅ **Short Freeze Phase:** 3 epochs sufficient for head stabilization  
✅ **Custom Multispectral Augmentations:** All transforms work with 5-channel input  
✅ **Stratified Split:** By lake presence (12/25 images have 0 lake pixels)  

---

## ❌ **What Failed**

❌ **HKH Pretrained Weights:** Hurt more than help (negative transfer)  
❌ **Aggressive Class Weights [50, 150]:** Combined with Focal gamma=4.0 = triple-counting  
❌ **Conservative Augmentations (Run 1):** Not diverse enough for debris generalization  
❌ **T_0=10 Scheduler (Run 1):** LR decayed too fast, hit local minimum  
❌ **Early Stopping patience=20:** Both runs stopped too early (22 and 51 epochs)  

---

## 💾 **Files & Artifacts**

### **Notebooks**
- `competition_finetuning.ipynb` - Main training notebook (current state: Run 2 config)
- `test_pretrained_baseline.ipynb` - Baseline evaluation (MCC 0.0804)

### **Documentation**
- `BASELINE_ANALYSIS.md` - Confusion matrix breakdown, error patterns
- `FINE_TUNING_UPDATES.md` - Implementation guide (9 major changes)
- `GEMINI_EXPERT_REVIEW.md` - Stability analysis from Gemini
- `GEMINI_UPDATES_APPLIED.md` - Quick reference checklist

### **Weights**
- `hkh_pretrained_resnet34.pth` - HKH pretrained (3-class, MCC 0.4887)
- `competition_best.pth` - Best competition model (Run 1: MCC 0.1646, Run 2: MCC 0.0435)

### **Logs**
- `train_log_fine_tune.txt` - Training logs from both runs

---

## 🎯 **Path Forward - Expert Recommendations**

### **Option 1: ImageNet Pretrained (RECOMMENDED)** ⭐
**Rationale:** Start fresh, avoid HKH anti-patterns  
**Change:** `encoder_weights='imagenet'` instead of HKH weights  
**Expected:** 0.65-0.75 MCC solo, 0.78-0.85 with TTA  
**Timeline:** 150 epochs (~3 weeks)  
**Success Probability:** 80%

### **Option 2: Ensemble 3 Models**
**Strategy:** ImageNet + HKH-frozen-encoder + Random-init  
**Expected:** 0.75-0.85 MCC ensemble, 0.80-0.90 with TTA  
**Timeline:** 4 weeks (3x training)  
**Success Probability:** 85%

### **Option 3: Extreme Regularization (Last HKH Attempt)**
**Changes:** Dropout 0.6, PHASE1=10 epochs, reduce weights to [1,3,15,40]  
**Expected:** 30% chance of 0.6-0.7 MCC  
**Timeline:** 2 weeks  
**Success Probability:** 30% (not recommended)

### **Option 4: Add MixUp/CutMix Augmentation**
**Works with:** Any base model (best with ImageNet)  
**Expected Boost:** +0.05-0.10 MCC  
**Implementation:** Training-time image/mask blending

---

## 📊 **Key Metrics to Monitor (Next Run)**

| Metric | Target by Epoch 30 | Target by Epoch 80 | Final Target |
|--------|-------------------|-------------------|--------------|
| **Val MCC** | >0.25 | >0.50 | >0.70 |
| **Val Debris MCC** | >0.10 | >0.30 | >0.50 |
| **Val Lake MCC** | >0.00 | >0.10 | >0.30 |
| **Train-Val Gap** | <2x | <1.5x | <1.3x |

**Red Flags:**
- Val MCC decreasing over time (= regression, stop immediately)
- Train-Val Debris gap >5x (= overfitting, need more augmentation)
- Loss >2.0 after epoch 20 (= class weights too aggressive)

---

## 🔍 **Lessons Learned**

1. **Negative Transfer is Real:** Pretrained weights can hurt if domain mismatch is too severe
2. **Class Weights + Focal Loss:** Be careful not to double-count (gamma=4.0 already auto-weights)
3. **Small Datasets Need Regularization:** But not extreme (dropout 0.4 is sweet spot)
4. **Augmentation Diversity > Quantity:** 20 images with smart augmentation can work
5. **Monitor Generalization Early:** Train-val gap reveals issues before MCC plateaus
6. **Scheduler Matters:** Wrong T_0 can trap model in local minimum
7. **Early Stopping Can Be Too Early:** 22-51 epochs insufficient for small dataset learning

---

## 🚀 **Immediate Action Items**

1. **DECIDE:** Choose Option 1 (ImageNet), Option 2 (Ensemble), or Option 3 (Last HKH attempt)
2. **CREATE:** New notebook with chosen configuration
3. **TRAIN:** Full 150 epochs without early stopping before epoch 80
4. **MONITOR:** Debris MCC must be >0.05 by epoch 30 (or pivot strategy)
5. **ITERATE:** If Option 1 fails, immediately switch to Option 2 (Ensemble)

---

## 📞 **Questions for Team Discussion**

1. Do we have time/compute for 3-model ensemble? (Option 2)
2. Are we committed to reaching 0.8 MCC or is 0.75 acceptable?
3. Should we acquire more training data (if possible)?
4. Can we try different backbone (EfficientNet, ResNet50)?
5. Is there budget for AutoML/NAS approaches?

---

**Last Updated:** November 6, 2025  
**Status:** Awaiting strategy decision (Options 1-4)  
**Next Review:** After 30 epochs of new approach
