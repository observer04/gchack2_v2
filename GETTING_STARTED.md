# Project Setup Complete! 🎉

## ✅ What's Been Created

### Documentation
- **MONOLITH.md** - Master blueprint with complete architecture, training strategy, and implementation details
- **README.md** - Quick start guide with key decisions and performance targets
- **This file** - Implementation checklist

### Directory Structure
```
✓ src/data/          - Dataset, samplers, transforms
✓ src/models/        - Model architectures (to be implemented)
✓ src/losses/        - Loss functions (to be implemented)
✓ src/training/      - Training loop (to be implemented)
✓ src/inference/     - TTA, post-processing (to be implemented)
✓ configs/           - YAML configurations
✓ weights/           - Model checkpoints (empty)
✓ reports/           - Experiment logs (empty)
✓ data/hkh/          - HKH dataset (to be downloaded)
```

### Configuration Files
- **configs/hkh_pretrain.yaml** - Phase 0 (HKH pretraining) configuration
- **configs/competition_finetune.yaml** - Phase 1 (competition fine-tuning) configuration

### Implemented Code
- **src/data/dataset.py** - GlacierDataset with 5-band loading, GLCM features, augmentation
- **src/data/samplers.py** - PixelBalancedSampler (critical for class imbalance)

---

## 🚀 Next Steps (In Order)

### Phase 0: Setup & Foundation (Day 1)

1. **Install Additional Dependencies**
   ```bash
   source gc/bin/activate
   pip install segmentation-models-pytorch albumentations pydensecrf scikit-image
   ```

2. **Download HKH Dataset**
   ```bash
   cd data/hkh
   # Download from: https://lila.science/datasets/hkh-glacier-mapping/
   # Extract and organize into train/val splits
   cd ../..
   ```

3. **Implement Loss Functions**
   Create the following files in `src/losses/`:
   - `focal.py` - FocalLoss with class weights
   - `dice.py` - DiceLoss (per-class)
   - `mcc.py` - FocalPhiMCCLoss (differentiable MCC)
   - `boundary.py` - BoundaryLoss with debris-glacier interface detection
   - `combined.py` - Wrapper combining all losses with phase scheduling

4. **Implement Model Architecture**
   Create in `src/models/`:
   - `unet.py` - Wrapper around smp.Unet with proper initialization
   - `attention.py` - Channel-Squeeze-Excitation (cSE) blocks
   - `ensemble.py` - Multi-model ensemble with TTA

5. **Implement Training Pipeline**
   Create in `src/training/`:
   - `metrics.py` - MCC, IoU, Dice computation
   - `trainer.py` - Training loop with AMP, gradient clipping, early stopping
   - `train.py` - Main script to run training from config

### Phase 1: HKH Pretraining (Days 2-3)

6. **Harmonize HKH Bands**
   - Map Landsat-7 bands to competition format (B2/B3/B4/SWIR/TIR)
   - Compute normalization statistics
   - Create train/val splits

7. **Run HKH Pretraining**
   ```bash
   python src/training/train.py --config configs/hkh_pretrain.yaml
   ```
   - Target: MCC 0.75-0.78 on HKH validation
   - Output: `weights/hkh_pretrained.pth` (~44 MB)
   - Time: ~6-8 hours on T4 GPU

### Phase 2: Competition Fine-Tuning (Days 4-6)

8. **Create Cross-Validation Splits**
   - 5-fold stratified by debris/lake presence
   - Image-level (no pixel leakage)

9. **Run Competition Fine-Tuning**
   ```bash
   python src/training/train.py \
       --config configs/competition_finetune.yaml \
       --pretrained weights/hkh_pretrained.pth
   ```
   - Train all 5 folds
   - Target: MCC 0.82-0.85 per fold
   - Output: `weights/best_fold{0-4}.pth`

10. **Run Ablation Studies**
    - 5ch vs 7ch (GLCM)
    - Channel attention on/off
    - Boundary weights {3×, 5×, 7×}
    - Focal gamma {2, 3, 4}
    - Document in `reports/ablations.md`

### Phase 3: Optimization (Days 7-9)

11. **Train Ensemble Members**
    - 3-5 models with different seeds/configurations
    - Implement TTA in `src/inference/predict.py`

12. **Implement Post-Processing**
    Create in `src/inference/`:
    - `postprocess.py` - Morphology, CRF, threshold optimization

13. **Complete solution.py**
    - Package best ensemble (< 300 MB)
    - Integrate TTA and post-processing
    - Test on validation set

### Phase 4: Final Validation (Days 10-12)

14. **Final Testing**
    ```bash
    python solution.py --data Train --masks Train/labels --out predictions
    ```
    - Verify MCC on validation
    - Check model size < 300 MB
    - Profile inference time

15. **Submit to Leaderboard**
    - Package `solution.py` + `model.pth`
    - Document final configuration
    - Submit!

---

## 📊 Critical Success Factors

### Must Do (Non-Negotiable)
1. ✅ Use `encoder_weights=None` (no ImageNet on multispectral)
2. ✅ Complete HKH pretraining (7k tiles → competition transfer)
3. ✅ Use ReduceLROnPlateau (not CosineAnnealingWarmRestarts)
4. ✅ Implement pixel-balanced sampling (not image-level)
5. ✅ Gradual loss complexity (Phase A → Phase B)

### Should Do (High Impact)
- Ensemble 3-5 diverse models
- Apply TTA (6 augmentations)
- Post-process with morphology + CRF
- Run ablation studies (justify all choices)

### Nice to Have (Low Priority)
- Multi-scale inference
- Knowledge distillation for smaller model
- Additional architecture variants

---

## 🎯 Performance Targets

| Milestone | MCC Target | Status |
|-----------|------------|--------|
| HKH Pretrain | 0.75-0.78 | 🔲 Pending |
| Single Model (1 fold) | 0.82-0.85 | 🔲 Pending |
| 5-Fold Average | 0.80-0.83 | 🔲 Pending |
| Ensemble + TTA | 0.87-0.89 | 🔲 Pending |
| Final (optimized) | **0.88-0.92** | 🔲 Pending |

---

## 📚 Key Resources

- **MONOLITH.md** - Complete technical blueprint
- **Perplexity Research** - https://www.perplexity.ai/search/act-as-expert-ml-dl-model-buil-7tAUdNYPQE.Es.2H79zhmw
- **HKH Dataset** - https://lila.science/datasets/hkh-glacier-mapping/
- **Boundary-Aware U-Net Paper** - https://arxiv.org/abs/2301.11454
- **Focal-Phi MCC Loss** - https://arxiv.org/abs/2010.13454

---

## 🐛 Troubleshooting Guide

**Issue: MCC stays below 0.10**
- Check: Is `encoder_weights=None`? (ImageNet will break multispectral)
- Check: Did HKH pretraining complete successfully?
- Check: Are all 5 bands loading correctly?

**Issue: Training unstable / loss explodes**
- Reduce batch size to 4, increase accumulation_steps to 8
- Check boundary loss ramp (should start at 0.05, not 0.30)
- Verify gradient clipping is enabled (clip_value=1.0)

**Issue: Lake class has 0% recall**
- Increase lake oversampling to 15× (from 10×)
- Check pixel-balanced sampler is active
- Consider auxiliary BCE loss for lake class

**Issue: MCC peaks then degrades**
- LR scheduler problem - verify ReduceLROnPlateau is used
- Increase early stopping patience to 25
- Check for data leakage in CV splits

---

## ✅ Pre-Flight Checklist

Before HKH Pretraining:
- [ ] HKH dataset downloaded and harmonized
- [ ] Loss functions implemented and tested
- [ ] Model architecture builds without errors
- [ ] Training script runs for 1 epoch (sanity check)
- [ ] GPU/CUDA available and configured

Before Competition Fine-Tuning:
- [ ] HKH pretrained weights exist (`weights/hkh_pretrained.pth`)
- [ ] HKH validation MCC ≥ 0.75
- [ ] 5-fold CV splits created
- [ ] Pixel-balanced sampler tested
- [ ] Phase B loss configuration verified

Before Submission:
- [ ] `solution.py` tested on validation data
- [ ] Model size < 300 MB verified
- [ ] No hardcoded paths in code
- [ ] Output format correct (0/85/170/255)
- [ ] Filenames match Band1 exactly
- [ ] MCC on validation ≥ 0.85

---

**Ready to start! Begin with Phase 0, Step 1. Good luck! 🚀**
