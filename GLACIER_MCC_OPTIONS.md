# Glacier MCC 0.8+ Blueprint

## Priority Upgrades

1. **Encoder & Architecture**: Switch to a stronger encoder (e.g., `timm-efficientnet-b3`, `mit_b2`) and enable Unet++/DeepLabV3+ heads for sharper debris/lake boundaries. Keep in_channels=7 so NDVI/NDWI remain available.
2. **Curriculum & Loss**: Start MCC supervision by epoch 10 with a linear ramp, pair focal+dice with a boundary-aware term (Lovasz or clDice), and replace static 80× class weights with logit-adjusted scaling derived from label histograms each epoch.
3. **Sampling Strategy**: Track minority pixel counts per tile, target 35-40% minority coverage per epoch, and use a `WeightedRandomSampler` with adaptive weights rather than hard duplication.
4. **Optimization**: Use AdamW+Lookahead or SAM, warmup for 5 epochs before cosine restarts, freeze encoder for the first 5 epochs, and train with AMP so the batch size can rise to 12-16 without OOM.
5. **Hard-Negative Focus**: Every 5 epochs, mine crops with high glacier↔debris or debris↔background confusion and run short booster epochs at higher MCC weight. Maintain a replay buffer for persistent failure cases.

## Validation & Monitoring

- Maintain three stratified validation folds; report mean ± std MCC and per-class MCC to confirm stability.
- Track minority precision/recall, calibration curves, and confusion heatmaps; adjust lake weighting only when recall dips below 0.45.
- Log cosine LR, loss-component trends, and minority sample share to spot imbalance regressions early.

## Fast Iteration Checklist

- [ ] Profile data loader throughput; cache Band5/Band10 to RAM if IO bound.
- [ ] Run ablations on NDVI/NDWI vs raw bands to confirm incremental value.
- [ ] Export top failure tiles after each run for manual QA and targeted augmentation.
- [ ] Automate seed sweeps (3-5 seeds) before promoting a checkpoint.
