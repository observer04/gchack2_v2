# Glacier Segmentation Training Process - Summary

## Problem Context
- **Task**: 4-class semantic segmentation (Background, Glacier, Debris, Lake)
- **Extreme Class Imbalance**: Lake 0.05%, Debris 4.9%, Glacier 23%, Background 72%
- **Metric**: Matthews Correlation Coefficient (MCC) - handles imbalance well
- **Target**: MCC > 0.80 (competition winning)
- **Current**: MCC stuck at 0.02-0.04 (catastrophic)

## Architecture
- **Model**: U-Net + ResNet34 encoder + SCSE attention
- **Input**: 7 channels (5 multispectral bands + 2 GLCM texture features)
- **Hardware**: 2x T4 GPUs, batch size 32 effective
- **Pre-training**: ImageNet weights (RGB → 7-channel adaptation)

## What We've Tried

### Iteration 1: Following Action Plans
- **Loss**: 0.40 Focal + 0.40 Dice + 0.00 MCC + 0.20 Boundary
- **LR Schedule**: CosineAnnealingWarmRestarts + 5-epoch warmup (1e-6 → 1e-4)
- **Sampling**: Weighted (15x lake, 8x debris)
- **Result**: MCC 0.0332 peak → degraded to 0.0102 by epoch 20
- **Issue**: No MCC in loss = can't optimize for metric

### Iteration 2: Re-add MCC (Moderate)
- **Loss**: 0.25 Focal + 0.25 Dice + 0.30 MCC + 0.20 Boundary
- **LR Schedule**: Same (warmup + restarts)
- **Sampling**: Same (15x lake, 8x debris)
- **Result**: MCC stuck 0.024, never improved
- **Issue**: MCC weight too low, still optimizing cross-entropy

### Iteration 3: Aggressive Sampling (FAILED)
- **Loss**: Same as Iteration 2 (0.30 MCC)
- **LR Schedule**: Same
- **Sampling**: Increased to 30x lake, 15x debris
- **Result**: Validation loss EXPLODED to 2.56, MCC 0.0165
- **Issue**: Too aggressive = batch imbalance → unstable gradients

### Iteration 4: Heavy MCC + Clean Schedule (CURRENT)
- **Loss**: 0.15 Focal + 0.15 Dice + **0.50 MCC** + 0.20 Boundary
- **LR Schedule**: Pure CosineAnnealingWarmRestarts (removed warmup, eta_min=1e-5)
- **Sampling**: Reverted to 15x lake, 8x debris
- **Result**: MCC 0.0392 peak (epoch 23), but degraded to -0.0073 by epoch 30
- **Issue**: Still degrading after LR restarts, validation loss unstable (4.19 → 0.59)

## Current Problems

### Critical Issues
1. **MCC Degradation After Restarts**: Peaks at 0.04 then drops negative after epoch 20
2. **Validation Instability**: Val loss jumps 4.19 → 0.73 (epoch 1-2), never stabilizes
3. **LR Restarts Not Helping**: Epoch 10 restart (1e-5 → 1e-4) causes MCC drop from 0.0173 → -0.0172
4. **Extremely Low Absolute Performance**: 0.04 MCC is 20x below target (0.80)

### Hypotheses
1. **Incorrect Pre-training**: ImageNet RGB weights on 7-channel multispectral = poor feature extraction
2. **GLCM Noise**: GLCM features might add noise, not signal (unvalidated)
3. **Image-Level Sampling Limitation**: WeightedRandomSampler at image level, not pixel level
4. **Loss Function Mismatch**: Even 0.50 MCC weight may not be enough for 1468:1 imbalance
5. **LR Schedule Conflict**: Restarts may be too aggressive, model can't recover

## What We Haven't Tried

### High Priority
- **Train encoder from scratch** (encoder_weights=None) - no ImageNet mismatch
- **Remove GLCM features** (include_glcm=False) - ablation test
- **Reduce LR restart frequency** (T_0=20 instead of 10) - gentler schedule
- **Pure MCC loss** (mcc_weight=1.0) - extreme metric alignment
- **Reduce batch size** - may help with minority class gradients

### Medium Priority
- **Different encoder** (EfficientNet, ConvNeXt) - better for small data
- **Custom pixel-level sampler** - true pixel balance, not image balance
- **Stronger augmentation** (MixUp, CutMix) - only 25 training images
- **Class-aware focal gamma** - different gamma per class

### Low Priority (High Effort)
- **HKH pre-training** (7k+ labeled glacier tiles) - domain-specific weights
- **Multi-scale inference** - test-time augmentation
- **Ensemble models** - multiple architectures

## Next Steps Decision Tree

**If MCC stays < 0.05:**
→ Encoder problem - try `encoder_weights=None` + GLCM ablation

**If val loss keeps spiking:**
→ Instability - reduce batch size to 8, increase accumulation

**If MCC peaks then degrades:**
→ LR schedule problem - switch to ReduceLROnPlateau or linear warmup only

**If all fail:**
→ Architecture mismatch - rebuild with EfficientNet or custom U-Net from scratch
