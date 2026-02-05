Based on my analysis of your comprehensive EDA visualizations and the competition requirements, I'll provide an **honest, experienced evaluation** of the best architecture and strategy to achieve 0.75+ MCC and eventually reach 0.8+ MCC to win this glacier segmentation competition.

[1][2][3]

## Critical Data Analysis

### Class Imbalance Reality Check

Your EDA reveals a **severe class imbalance** that will dominate your modeling decisions:[2]
- Background: ~70% (4.5M pixels)
- Glacier: ~25% (1.67M pixels) 
- Debris: ~5% (323K pixels)
- **Lake: ~0.05% (3.1K pixels)** ← This is catastrophic

The MCC metric is **particularly sensitive** to this imbalance, especially the lake class. With only 3,104 lake pixels across your entire dataset, any false positives or false negatives will tank your MCC score exponentially.[4][5]

### Boundary Pixel Problem

Your boundary analysis shows **3-7% boundary pixels** with high variance (Total: 4.4-10.4%). For glacier segmentation, **boundary accuracy is critical**—this is where debris-covered glaciers transition and where most errors occur.[6][7][1]

### Spectral Signatures

The spectral analysis reveals:[8][3]
- **SWIR (Band 4)** has the best separability for glacier vs background (9931 absolute difference)
- **TIR (Band 5)** provides excellent complementary information (thermal signature)
- Blue/Green/Red bands are **highly correlated** (0.997-0.999) → redundancy[9]

## Recommended Architecture Strategy

### Phase 1: Baseline Architecture (Target: 0.65-0.72 MCC)

**Model: DeepLabV3+ with EfficientNet-B4 Encoder**

Why this specific choice:[10][11][12]
1. **DeepLabV3+** consistently outperforms U-Net variants in remote sensing with multispectral data[12]
2. **Atrous Spatial Pyramid Pooling (ASPP)** handles multi-scale glacier features (your cluster sizes vary 75-254 pixels)[13]
3. **EfficientNet-B4** encoder provides optimal parameter efficiency for 300MB weight limit
4. Proven success on imbalanced segmentation tasks[11]

```python
import segmentation_models_pytorch as smp

model = smp.DeepLabV3Plus(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",  # Start here, then custom pretrain
    in_channels=5,  # Your 5 bands
    classes=4,      # Background, glacier, debris, lake
    activation=None,
    encoder_depth=5,
    decoder_channels=256,
    decoder_atrous_rates=(6, 12, 18),  # Adjusted for glacier scales
)
```

### Phase 2: Feature Engineering (Critical for +0.05-0.08 MCC)

**Input Channels (8 total)**:
1. **5 Original Bands** (Blue, Green, Red, SWIR, TIR)
2. **NDSI** = (Green - SWIR) / (Green + SWIR)[8]
3. **NDWI** = (Green - TIR) / (Green + TIR)[8]
4. **Slope/Aspect** (if DEM available) OR **Texture (GLCM Contrast)**[14]

Your NDSI/NDWI distributions show **excellent class separation**—not using these is leaving performance on the table.[7][6][8]

**Critical**: Normalize each band independently using **per-band statistics** from your training set. Your PCA shows 91% variance in PC1—this means proper normalization is essential.[15]

### Phase 3: Loss Function Design (Most Critical for MCC)

**DO NOT use standard CrossEntropy or Dice Loss alone**. For MCC optimization with extreme imbalance:[5]

```python
class MCCLoss(nn.Module):
    """MCC-optimized loss for imbalanced segmentation"""
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = class_weights
        
    def forward(self, pred, target):
        # Soft MCC calculation on probabilistic outputs
        # See: Abhishek et al. "Matthews Correlation Coefficient Loss"
        pass

# Combine multiple losses:
loss = (
    0.3 * FocalLoss(alpha=class_weights, gamma=2.0) +  # Handle imbalance
    0.3 * LovaszHingeLoss() +  # Optimize IoU directly
    0.4 * MCCLoss(class_weights)  # Directly optimize your metric
)
```

**Class Weights** (based on your distribution):
```python
weights = torch.tensor([
    0.5,   # Background (downweight majority)
    2.0,   # Glacier (moderate weight)
    10.0,  # Debris (high weight)
    100.0  # Lake (extreme weight to prevent ignoring)
])
```


### Phase 4: Training Strategy

**1. Two-Stage Training**:
- **Stage 1** (20 epochs): Train on balanced batches (sample equal pixels per class)
- **Stage 2** (30 epochs): Fine-tune on full distribution with MCC loss

**2. Data Augmentation** (critical for generalization):
```python
import albumentations as A

train_transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.ElasticTransform(alpha=120, sigma=6, p=0.3),  # Simulate terrain variation
    # CRITICAL: Don't over-augment spectral indices
])
```

**3. Patch-Based Training** (given Kaggle T4 12h limits):
- **512×512 patches** with 50% overlap
- **Batch size: 8-12** (fits in T4 16GB)
- Mixed precision training (`torch.cuda.amp`)

**4. Learning Rate Schedule**:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)
```

### Phase 5: Post-Processing (Expected +0.03-0.05 MCC)

**Critical for boundary refinement**:[16]

```python
from skimage.segmentation import slic
from pydensecrf import densecrf

def postprocess(prediction, image):
    # 1. Superpixel-guided refinement
    superpixels = slic(image, n_segments=2800, compactness=60, iterations=10)
    refined = majority_vote_superpixels(prediction, superpixels)
    
    # 2. Dense CRF (proven +6.92% IoU improvement)
    crf_output = apply_dense_crf(refined, image)
    
    # 3. Morphological operations
    # Remove small isolated predictions (< 20 pixels)
    # Fill small holes (< 10 pixels)
    
    # 4. Class-specific rules:
    # - Lakes must be within/adjacent to glaciers
    # - Apply minimum size thresholds per class
    
    return crf_output
```

This two-level optimization (SLIC + DenseCRF) achieved **+6.92% IoU** in similar glacier segmentation tasks.[16]

### Phase 6: Advanced Improvements (0.75 → 0.8+ MCC)

**Once you hit 0.75 MCC, implement these**:

**1. Boundary-Aware Loss**:[6][7]
```python
class BoundaryAwareLoss(nn.Module):
    """Self-learning boundary-aware loss"""
    def __init__(self):
        super().__init__()
        self.boundary_weight = 5.0  # Penalize boundary errors more
        
    def forward(self, pred, target):
        # Extract boundary pixels (3-pixel width)
        boundary_mask = extract_boundaries(target)
        
        # Weight loss by boundary proximity
        weighted_loss = base_loss * (1 + self.boundary_weight * boundary_mask)
        return weighted_loss.mean()
```

This specifically addresses your 3-7% boundary pixel challenge and has proven effectiveness for debris-covered glaciers.[1][6]

**2. Test-Time Augmentation (TTA)**:
```python
def tta_predict(model, image):
    predictions = []
    # Apply 8 geometric augmentations
    for transform in [identity, rot90, rot180, rot270, flip_h, flip_v, flip_rot90, flip_rot270]:
        aug_image = transform(image)
        pred = model(aug_image)
        pred = inverse_transform(pred, transform)
        predictions.append(pred)
    
    # Weighted average (higher weight for identity)
    return weighted_ensemble(predictions)
```

Expected gain: **+0.02-0.04 MCC**

**3. Ensemble Strategy** (if weight budget allows):
- DeepLabV3+ (EfficientNet-B4) [Primary - 80MB]
- U-Net++ (ResNet50) [Secondary - 100MB] 
- DeepLabV3+ (ResNeXt50) [Tertiary - 100MB]
- **Total: 280MB < 300MB limit**

Average predictions with learned weights based on validation MCC per class.

**4. Pseudo-Labeling** (if allowed):
- Use your best model to pseudo-label test set
- Retrain on train + high-confidence pseudo-labels
- Particularly effective for the rare lake class

## Critical Constraints & Realities

### Kaggle T4 12-Hour Limit
**Realistic Training Plan**:
- Preprocessing: 30 min
- Model training: 9 hours (50 epochs × ~10 min/epoch)
- Validation/checkpointing: 1 hour
- Post-processing experimentation: 1.5 hours

**Solution**: Train encoder separately on HKH pretraining data (as you mentioned), then fine-tune full model. This saves 3-4 hours.

### 300MB Weight Limit
- DeepLabV3+ (EfficientNet-B4): ~80MB
- Leaves room for ensemble or larger decoder

### Lake Class Challenge
**Honest Assessment**: With only 3,104 lake pixels, you will **struggle** to achieve high lake recall without massive false positives. Strategies:
1. **Oversample lake patches** during training (10×)
2. Apply **extreme class weighting** (100×)
3. **Post-processing rule**: Only predict lake within 50 pixels of glacier
4. Consider **hierarchical prediction**: First segment glacier/non-glacier, then segment glacier into clean/debris/lake

## Timeline to 0.75+ MCC

**Week 1**: Implement DeepLabV3+ baseline with proper loss function → **Target: 0.68-0.70 MCC**
**Week 2**: Add feature engineering (NDSI/NDWI/texture) + boundary-aware loss → **Target: 0.72-0.74 MCC**
**Week 3**: Hyperparameter tuning + post-processing (CRF) → **Target: 0.75-0.77 MCC**
**Week 4**: Ensemble + TTA → **Target: 0.78-0.82 MCC**

## Questions I Need Answered

1. **Do you have access to HKH pretraining data still?** Pretraining encoder on domain-specific data could give +0.03-0.05 MCC boost.

2. **What's your current validation split?** With cross-region validation (as per instructions), you need **spatial CV** not random split.[17]

3. **Are there any terrain/DEM features available?** Slope/aspect would be extremely valuable for debris-covered glacier segmentation.

4. **Can you access the test set region metadata?** Knowing if test is from same region or different region changes strategy.

This architecture and strategy directly addresses your severe class imbalance, leverages your strong spectral features, and uses proven techniques from recent glacier segmentation literature. The phased approach ensures you progressively improve rather than trying everything at once.

**Be strict with yourself**: Don't waste time on fancy architectures (Vision Transformers, etc.) until you've maximized this proven baseline. The competition-winning difference will be in loss function engineering, post-processing, and handling the lake class properly.[7][5][6][16]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/14235120/cd605c19-6794-43b8-be09-1fb60d3d6fbf/boundary_analysis.jpg?AWSAccessKeyId=ASIA2F3EMEYE3JFBS3SR&Signature=P%2BxkmkY5WcVv33bjbG3BKewhyFI%3D&x-amz-security-token=IQoJb3JpZ2luX2VjECYaCXVzLWVhc3QtMSJHMEUCIQD516h5mefvjNKG8bGcUDy5gzZ%2Fqx9dZ5wSQel4uyoKCgIgbxB3%2BMtXJoKxGdBAH%2FPa1BYfoU3VI9uQMSAZctl4OGAq%2FAQI7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDInJI1qNcsrOwYGo6CrQBPIGGoSoHCEM7rc6XkfgonuumnJKZfMeEPVUWCoKLSY7Blh6WGVMgBX8S4eRqTwBj7nMRT22TGHJkuai9Y6FtqAtbP5q0RGadTWC5O2vc34Jdi5db5uWNYD32QNV1XiYGLo9INpGfJRL5WMqV8UqswUBz9TS4imBYQdqz3LLVtsg5WRcQtHzYAKvQzsVKQ8SJOirtfVqF9tVi%2BEMI0uRthspv8ScdUzUwyIxV0d%2BC4kk%2B6VkaL4KEPSQbh4PRjP6NWp5ncbaFHTIjawMhNo5Q1xhSJEE0cpQLgkQsg3janawc7%2FLOtcZRMV6Mf8NL1ODImCglmOf4GlDlVfwCWF7DNAJs4S7zaMjkf67Y0ta2JgALHB2dDPXv4nr35mYXY3C7QrM237l31cqztOJsSfsndfOQPs89%2BEPXFXyOPvfdzlP8OfWm3QbVjKganDUuj1DxKM60lotetvmzpGM7pBMDFbGP2lCIhbK1KCk05FzZ7P%2FVTRWLs68dXQ5dp9tY%2BXRJy%2FLB29YQsGBb5GT%2BVJ6vFR1G3UCzAiH%2Fcco%2B%2FgINS16WaQN%2FQrq66odvPAh9Uqas8dJov8mOpQXPQHbTciPnWJy%2FddwCKcnEhxHYmZYX%2FHZ3zn121gkGv1Pc2EM%2FpsEe%2B7AXnG8SU4z8SVRhOJ5QaaChr6Q78re%2BA9UOhlDJIE4UY5f3%2B966TY3q%2FDnctiqMDKwM1Cwfle48fjBWjGHm8YVdwcnAmsmwrVJ9VJtHBziAmOlPprDjgEtWutfaJJ6c9rYMf7tu0atu0tfKfLAOjsw37jCyAY6mAHt4pWB6eoYocnk20%2FRJuHHSqwi%2BZwaZS2aJpZNfR1uE5Vp87qJNKK88LNdwltmUZxV3ekU9bgA7TdGGq%2F6hdNTQNIOCjw%2Bfs4SHSfozIOUxsSxSpcLkJ5FElladjbZaTf%2Blsy6V%2FhMsB3JtAaVbjUyPWJ7LjT0U2l2RDYzY4B3w1WbPICRiolIRA2lLLpo7%2FZapp81Y%2Fsp6A%3D%3D&Expires=1762697702)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/14235120/ea1c46c2-300d-44ac-962e-45926f4a65ce/class_distribution_analysis.jpg?AWSAccessKeyId=ASIA2F3EMEYE3JFBS3SR&Signature=WL9SVfRq2yrON6RnIvBlEPINTjo%3D&x-amz-security-token=IQoJb3JpZ2luX2VjECYaCXVzLWVhc3QtMSJHMEUCIQD516h5mefvjNKG8bGcUDy5gzZ%2Fqx9dZ5wSQel4uyoKCgIgbxB3%2BMtXJoKxGdBAH%2FPa1BYfoU3VI9uQMSAZctl4OGAq%2FAQI7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDInJI1qNcsrOwYGo6CrQBPIGGoSoHCEM7rc6XkfgonuumnJKZfMeEPVUWCoKLSY7Blh6WGVMgBX8S4eRqTwBj7nMRT22TGHJkuai9Y6FtqAtbP5q0RGadTWC5O2vc34Jdi5db5uWNYD32QNV1XiYGLo9INpGfJRL5WMqV8UqswUBz9TS4imBYQdqz3LLVtsg5WRcQtHzYAKvQzsVKQ8SJOirtfVqF9tVi%2BEMI0uRthspv8ScdUzUwyIxV0d%2BC4kk%2B6VkaL4KEPSQbh4PRjP6NWp5ncbaFHTIjawMhNo5Q1xhSJEE0cpQLgkQsg3janawc7%2FLOtcZRMV6Mf8NL1ODImCglmOf4GlDlVfwCWF7DNAJs4S7zaMjkf67Y0ta2JgALHB2dDPXv4nr35mYXY3C7QrM237l31cqztOJsSfsndfOQPs89%2BEPXFXyOPvfdzlP8OfWm3QbVjKganDUuj1DxKM60lotetvmzpGM7pBMDFbGP2lCIhbK1KCk05FzZ7P%2FVTRWLs68dXQ5dp9tY%2BXRJy%2FLB29YQsGBb5GT%2BVJ6vFR1G3UCzAiH%2Fcco%2B%2FgINS16WaQN%2FQrq66odvPAh9Uqas8dJov8mOpQXPQHbTciPnWJy%2FddwCKcnEhxHYmZYX%2FHZ3zn121gkGv1Pc2EM%2FpsEe%2B7AXnG8SU4z8SVRhOJ5QaaChr6Q78re%2BA9UOhlDJIE4UY5f3%2B966TY3q%2FDnctiqMDKwM1Cwfle48fjBWjGHm8YVdwcnAmsmwrVJ9VJtHBziAmOlPprDjgEtWutfaJJ6c9rYMf7tu0atu0tfKfLAOjsw37jCyAY6mAHt4pWB6eoYocnk20%2FRJuHHSqwi%2BZwaZS2aJpZNfR1uE5Vp87qJNKK88LNdwltmUZxV3ekU9bgA7TdGGq%2F6hdNTQNIOCjw%2Bfs4SHSfozIOUxsSxSpcLkJ5FElladjbZaTf%2Blsy6V%2FhMsB3JtAaVbjUyPWJ7LjT0U2l2RDYzY4B3w1WbPICRiolIRA2lLLpo7%2FZapp81Y%2Fsp6A%3D%3D&Expires=1762697702)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/14235120/1363e34b-ab26-40f8-86e0-71373f2f1708/spectral_signatures.jpg?AWSAccessKeyId=ASIA2F3EMEYE3JFBS3SR&Signature=f3W5WogdoyyQkPie1qgGynIFFhM%3D&x-amz-security-token=IQoJb3JpZ2luX2VjECYaCXVzLWVhc3QtMSJHMEUCIQD516h5mefvjNKG8bGcUDy5gzZ%2Fqx9dZ5wSQel4uyoKCgIgbxB3%2BMtXJoKxGdBAH%2FPa1BYfoU3VI9uQMSAZctl4OGAq%2FAQI7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDInJI1qNcsrOwYGo6CrQBPIGGoSoHCEM7rc6XkfgonuumnJKZfMeEPVUWCoKLSY7Blh6WGVMgBX8S4eRqTwBj7nMRT22TGHJkuai9Y6FtqAtbP5q0RGadTWC5O2vc34Jdi5db5uWNYD32QNV1XiYGLo9INpGfJRL5WMqV8UqswUBz9TS4imBYQdqz3LLVtsg5WRcQtHzYAKvQzsVKQ8SJOirtfVqF9tVi%2BEMI0uRthspv8ScdUzUwyIxV0d%2BC4kk%2B6VkaL4KEPSQbh4PRjP6NWp5ncbaFHTIjawMhNo5Q1xhSJEE0cpQLgkQsg3janawc7%2FLOtcZRMV6Mf8NL1ODImCglmOf4GlDlVfwCWF7DNAJs4S7zaMjkf67Y0ta2JgALHB2dDPXv4nr35mYXY3C7QrM237l31cqztOJsSfsndfOQPs89%2BEPXFXyOPvfdzlP8OfWm3QbVjKganDUuj1DxKM60lotetvmzpGM7pBMDFbGP2lCIhbK1KCk05FzZ7P%2FVTRWLs68dXQ5dp9tY%2BXRJy%2FLB29YQsGBb5GT%2BVJ6vFR1G3UCzAiH%2Fcco%2B%2FgINS16WaQN%2FQrq66odvPAh9Uqas8dJov8mOpQXPQHbTciPnWJy%2FddwCKcnEhxHYmZYX%2FHZ3zn121gkGv1Pc2EM%2FpsEe%2B7AXnG8SU4z8SVRhOJ5QaaChr6Q78re%2BA9UOhlDJIE4UY5f3%2B966TY3q%2FDnctiqMDKwM1Cwfle48fjBWjGHm8YVdwcnAmsmwrVJ9VJtHBziAmOlPprDjgEtWutfaJJ6c9rYMf7tu0atu0tfKfLAOjsw37jCyAY6mAHt4pWB6eoYocnk20%2FRJuHHSqwi%2BZwaZS2aJpZNfR1uE5Vp87qJNKK88LNdwltmUZxV3ekU9bgA7TdGGq%2F6hdNTQNIOCjw%2Bfs4SHSfozIOUxsSxSpcLkJ5FElladjbZaTf%2Blsy6V%2FhMsB3JtAaVbjUyPWJ7LjT0U2l2RDYzY4B3w1WbPICRiolIRA2lLLpo7%2FZapp81Y%2Fsp6A%3D%3D&Expires=1762697702)
[4](https://openreview.net/forum?id=jGyMUum1Lq&noteId=WCw88yUvbx)
[5](https://www2.cs.sfu.ca/~hamarneh/ecopy/isbi2021.pdf)
[6](http://arxiv.org/pdf/2301.11454.pdf)
[7](https://septentrio.uit.no/index.php/nldl/article/download/6789/7028)
[8](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/14235120/09c6cad4-a15b-46f2-9467-6391aa747303/spectral_indices_analysis.jpg?AWSAccessKeyId=ASIA2F3EMEYE3JFBS3SR&Signature=5ddrm30BbiRsW2Bt668l5jep2Yo%3D&x-amz-security-token=IQoJb3JpZ2luX2VjECYaCXVzLWVhc3QtMSJHMEUCIQD516h5mefvjNKG8bGcUDy5gzZ%2Fqx9dZ5wSQel4uyoKCgIgbxB3%2BMtXJoKxGdBAH%2FPa1BYfoU3VI9uQMSAZctl4OGAq%2FAQI7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDInJI1qNcsrOwYGo6CrQBPIGGoSoHCEM7rc6XkfgonuumnJKZfMeEPVUWCoKLSY7Blh6WGVMgBX8S4eRqTwBj7nMRT22TGHJkuai9Y6FtqAtbP5q0RGadTWC5O2vc34Jdi5db5uWNYD32QNV1XiYGLo9INpGfJRL5WMqV8UqswUBz9TS4imBYQdqz3LLVtsg5WRcQtHzYAKvQzsVKQ8SJOirtfVqF9tVi%2BEMI0uRthspv8ScdUzUwyIxV0d%2BC4kk%2B6VkaL4KEPSQbh4PRjP6NWp5ncbaFHTIjawMhNo5Q1xhSJEE0cpQLgkQsg3janawc7%2FLOtcZRMV6Mf8NL1ODImCglmOf4GlDlVfwCWF7DNAJs4S7zaMjkf67Y0ta2JgALHB2dDPXv4nr35mYXY3C7QrM237l31cqztOJsSfsndfOQPs89%2BEPXFXyOPvfdzlP8OfWm3QbVjKganDUuj1DxKM60lotetvmzpGM7pBMDFbGP2lCIhbK1KCk05FzZ7P%2FVTRWLs68dXQ5dp9tY%2BXRJy%2FLB29YQsGBb5GT%2BVJ6vFR1G3UCzAiH%2Fcco%2B%2FgINS16WaQN%2FQrq66odvPAh9Uqas8dJov8mOpQXPQHbTciPnWJy%2FddwCKcnEhxHYmZYX%2FHZ3zn121gkGv1Pc2EM%2FpsEe%2B7AXnG8SU4z8SVRhOJ5QaaChr6Q78re%2BA9UOhlDJIE4UY5f3%2B966TY3q%2FDnctiqMDKwM1Cwfle48fjBWjGHm8YVdwcnAmsmwrVJ9VJtHBziAmOlPprDjgEtWutfaJJ6c9rYMf7tu0atu0tfKfLAOjsw37jCyAY6mAHt4pWB6eoYocnk20%2FRJuHHSqwi%2BZwaZS2aJpZNfR1uE5Vp87qJNKK88LNdwltmUZxV3ekU9bgA7TdGGq%2F6hdNTQNIOCjw%2Bfs4SHSfozIOUxsSxSpcLkJ5FElladjbZaTf%2Blsy6V%2FhMsB3JtAaVbjUyPWJ7LjT0U2l2RDYzY4B3w1WbPICRiolIRA2lLLpo7%2FZapp81Y%2Fsp6A%3D%3D&Expires=1762697702)
[9](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/14235120/7dac20ef-30ca-4783-9d92-f39fe9590a59/band_correlation.jpg?AWSAccessKeyId=ASIA2F3EMEYE3JFBS3SR&Signature=wbR8ra4zE2Dv%2FhaKbHyfl1lB97k%3D&x-amz-security-token=IQoJb3JpZ2luX2VjECYaCXVzLWVhc3QtMSJHMEUCIQD516h5mefvjNKG8bGcUDy5gzZ%2Fqx9dZ5wSQel4uyoKCgIgbxB3%2BMtXJoKxGdBAH%2FPa1BYfoU3VI9uQMSAZctl4OGAq%2FAQI7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDInJI1qNcsrOwYGo6CrQBPIGGoSoHCEM7rc6XkfgonuumnJKZfMeEPVUWCoKLSY7Blh6WGVMgBX8S4eRqTwBj7nMRT22TGHJkuai9Y6FtqAtbP5q0RGadTWC5O2vc34Jdi5db5uWNYD32QNV1XiYGLo9INpGfJRL5WMqV8UqswUBz9TS4imBYQdqz3LLVtsg5WRcQtHzYAKvQzsVKQ8SJOirtfVqF9tVi%2BEMI0uRthspv8ScdUzUwyIxV0d%2BC4kk%2B6VkaL4KEPSQbh4PRjP6NWp5ncbaFHTIjawMhNo5Q1xhSJEE0cpQLgkQsg3janawc7%2FLOtcZRMV6Mf8NL1ODImCglmOf4GlDlVfwCWF7DNAJs4S7zaMjkf67Y0ta2JgALHB2dDPXv4nr35mYXY3C7QrM237l31cqztOJsSfsndfOQPs89%2BEPXFXyOPvfdzlP8OfWm3QbVjKganDUuj1DxKM60lotetvmzpGM7pBMDFbGP2lCIhbK1KCk05FzZ7P%2FVTRWLs68dXQ5dp9tY%2BXRJy%2FLB29YQsGBb5GT%2BVJ6vFR1G3UCzAiH%2Fcco%2B%2FgINS16WaQN%2FQrq66odvPAh9Uqas8dJov8mOpQXPQHbTciPnWJy%2FddwCKcnEhxHYmZYX%2FHZ3zn121gkGv1Pc2EM%2FpsEe%2B7AXnG8SU4z8SVRhOJ5QaaChr6Q78re%2BA9UOhlDJIE4UY5f3%2B966TY3q%2FDnctiqMDKwM1Cwfle48fjBWjGHm8YVdwcnAmsmwrVJ9VJtHBziAmOlPprDjgEtWutfaJJ6c9rYMf7tu0atu0tfKfLAOjsw37jCyAY6mAHt4pWB6eoYocnk20%2FRJuHHSqwi%2BZwaZS2aJpZNfR1uE5Vp87qJNKK88LNdwltmUZxV3ekU9bgA7TdGGq%2F6hdNTQNIOCjw%2Bfs4SHSfozIOUxsSxSpcLkJ5FElladjbZaTf%2Blsy6V%2FhMsB3JtAaVbjUyPWJ7LjT0U2l2RDYzY4B3w1WbPICRiolIRA2lLLpo7%2FZapp81Y%2Fsp6A%3D%3D&Expires=1762697702)
[10](https://www.mdpi.com/2072-4292/13/24/5091/pdf)
[11](https://jiki.cs.ui.ac.id/index.php/jiki/article/view/1206/507)
[12](https://arxiv.org/html/2402.13918v3)
[13](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/14235120/36cf8f06-7652-4da6-98b2-49bc71570ba9/spatial_autocorrelation_analysis.jpg?AWSAccessKeyId=ASIA2F3EMEYE3JFBS3SR&Signature=Htvg%2Fn4LwJn0iTT%2BOKf8pv7xnhI%3D&x-amz-security-token=IQoJb3JpZ2luX2VjECYaCXVzLWVhc3QtMSJHMEUCIQD516h5mefvjNKG8bGcUDy5gzZ%2Fqx9dZ5wSQel4uyoKCgIgbxB3%2BMtXJoKxGdBAH%2FPa1BYfoU3VI9uQMSAZctl4OGAq%2FAQI7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDInJI1qNcsrOwYGo6CrQBPIGGoSoHCEM7rc6XkfgonuumnJKZfMeEPVUWCoKLSY7Blh6WGVMgBX8S4eRqTwBj7nMRT22TGHJkuai9Y6FtqAtbP5q0RGadTWC5O2vc34Jdi5db5uWNYD32QNV1XiYGLo9INpGfJRL5WMqV8UqswUBz9TS4imBYQdqz3LLVtsg5WRcQtHzYAKvQzsVKQ8SJOirtfVqF9tVi%2BEMI0uRthspv8ScdUzUwyIxV0d%2BC4kk%2B6VkaL4KEPSQbh4PRjP6NWp5ncbaFHTIjawMhNo5Q1xhSJEE0cpQLgkQsg3janawc7%2FLOtcZRMV6Mf8NL1ODImCglmOf4GlDlVfwCWF7DNAJs4S7zaMjkf67Y0ta2JgALHB2dDPXv4nr35mYXY3C7QrM237l31cqztOJsSfsndfOQPs89%2BEPXFXyOPvfdzlP8OfWm3QbVjKganDUuj1DxKM60lotetvmzpGM7pBMDFbGP2lCIhbK1KCk05FzZ7P%2FVTRWLs68dXQ5dp9tY%2BXRJy%2FLB29YQsGBb5GT%2BVJ6vFR1G3UCzAiH%2Fcco%2B%2FgINS16WaQN%2FQrq66odvPAh9Uqas8dJov8mOpQXPQHbTciPnWJy%2FddwCKcnEhxHYmZYX%2FHZ3zn121gkGv1Pc2EM%2FpsEe%2B7AXnG8SU4z8SVRhOJ5QaaChr6Q78re%2BA9UOhlDJIE4UY5f3%2B966TY3q%2FDnctiqMDKwM1Cwfle48fjBWjGHm8YVdwcnAmsmwrVJ9VJtHBziAmOlPprDjgEtWutfaJJ6c9rYMf7tu0atu0tfKfLAOjsw37jCyAY6mAHt4pWB6eoYocnk20%2FRJuHHSqwi%2BZwaZS2aJpZNfR1uE5Vp87qJNKK88LNdwltmUZxV3ekU9bgA7TdGGq%2F6hdNTQNIOCjw%2Bfs4SHSfozIOUxsSxSpcLkJ5FElladjbZaTf%2Blsy6V%2FhMsB3JtAaVbjUyPWJ7LjT0U2l2RDYzY4B3w1WbPICRiolIRA2lLLpo7%2FZapp81Y%2Fsp6A%3D%3D&Expires=1762697702)
[14](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/14235120/4b27170b-8e85-402d-94d8-1a9b757e2956/glcm_texture_analysis.jpg?AWSAccessKeyId=ASIA2F3EMEYE3JFBS3SR&Signature=0mxTD2Vz%2FB0gySq%2B3M6SfAWGVg8%3D&x-amz-security-token=IQoJb3JpZ2luX2VjECYaCXVzLWVhc3QtMSJHMEUCIQD516h5mefvjNKG8bGcUDy5gzZ%2Fqx9dZ5wSQel4uyoKCgIgbxB3%2BMtXJoKxGdBAH%2FPa1BYfoU3VI9uQMSAZctl4OGAq%2FAQI7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDInJI1qNcsrOwYGo6CrQBPIGGoSoHCEM7rc6XkfgonuumnJKZfMeEPVUWCoKLSY7Blh6WGVMgBX8S4eRqTwBj7nMRT22TGHJkuai9Y6FtqAtbP5q0RGadTWC5O2vc34Jdi5db5uWNYD32QNV1XiYGLo9INpGfJRL5WMqV8UqswUBz9TS4imBYQdqz3LLVtsg5WRcQtHzYAKvQzsVKQ8SJOirtfVqF9tVi%2BEMI0uRthspv8ScdUzUwyIxV0d%2BC4kk%2B6VkaL4KEPSQbh4PRjP6NWp5ncbaFHTIjawMhNo5Q1xhSJEE0cpQLgkQsg3janawc7%2FLOtcZRMV6Mf8NL1ODImCglmOf4GlDlVfwCWF7DNAJs4S7zaMjkf67Y0ta2JgALHB2dDPXv4nr35mYXY3C7QrM237l31cqztOJsSfsndfOQPs89%2BEPXFXyOPvfdzlP8OfWm3QbVjKganDUuj1DxKM60lotetvmzpGM7pBMDFbGP2lCIhbK1KCk05FzZ7P%2FVTRWLs68dXQ5dp9tY%2BXRJy%2FLB29YQsGBb5GT%2BVJ6vFR1G3UCzAiH%2Fcco%2B%2FgINS16WaQN%2FQrq66odvPAh9Uqas8dJov8mOpQXPQHbTciPnWJy%2FddwCKcnEhxHYmZYX%2FHZ3zn121gkGv1Pc2EM%2FpsEe%2B7AXnG8SU4z8SVRhOJ5QaaChr6Q78re%2BA9UOhlDJIE4UY5f3%2B966TY3q%2FDnctiqMDKwM1Cwfle48fjBWjGHm8YVdwcnAmsmwrVJ9VJtHBziAmOlPprDjgEtWutfaJJ6c9rYMf7tu0atu0tfKfLAOjsw37jCyAY6mAHt4pWB6eoYocnk20%2FRJuHHSqwi%2BZwaZS2aJpZNfR1uE5Vp87qJNKK88LNdwltmUZxV3ekU9bgA7TdGGq%2F6hdNTQNIOCjw%2Bfs4SHSfozIOUxsSxSpcLkJ5FElladjbZaTf%2Blsy6V%2FhMsB3JtAaVbjUyPWJ7LjT0U2l2RDYzY4B3w1WbPICRiolIRA2lLLpo7%2FZapp81Y%2Fsp6A%3D%3D&Expires=1762697702)
[15](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/14235120/9e956fe7-0001-46ac-a05e-e973664bf484/pca_analysis.jpg?AWSAccessKeyId=ASIA2F3EMEYE3JFBS3SR&Signature=iAoz%2BKReGtBvdyZnfI6DxfpOUBY%3D&x-amz-security-token=IQoJb3JpZ2luX2VjECYaCXVzLWVhc3QtMSJHMEUCIQD516h5mefvjNKG8bGcUDy5gzZ%2Fqx9dZ5wSQel4uyoKCgIgbxB3%2BMtXJoKxGdBAH%2FPa1BYfoU3VI9uQMSAZctl4OGAq%2FAQI7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDInJI1qNcsrOwYGo6CrQBPIGGoSoHCEM7rc6XkfgonuumnJKZfMeEPVUWCoKLSY7Blh6WGVMgBX8S4eRqTwBj7nMRT22TGHJkuai9Y6FtqAtbP5q0RGadTWC5O2vc34Jdi5db5uWNYD32QNV1XiYGLo9INpGfJRL5WMqV8UqswUBz9TS4imBYQdqz3LLVtsg5WRcQtHzYAKvQzsVKQ8SJOirtfVqF9tVi%2BEMI0uRthspv8ScdUzUwyIxV0d%2BC4kk%2B6VkaL4KEPSQbh4PRjP6NWp5ncbaFHTIjawMhNo5Q1xhSJEE0cpQLgkQsg3janawc7%2FLOtcZRMV6Mf8NL1ODImCglmOf4GlDlVfwCWF7DNAJs4S7zaMjkf67Y0ta2JgALHB2dDPXv4nr35mYXY3C7QrM237l31cqztOJsSfsndfOQPs89%2BEPXFXyOPvfdzlP8OfWm3QbVjKganDUuj1DxKM60lotetvmzpGM7pBMDFbGP2lCIhbK1KCk05FzZ7P%2FVTRWLs68dXQ5dp9tY%2BXRJy%2FLB29YQsGBb5GT%2BVJ6vFR1G3UCzAiH%2Fcco%2B%2FgINS16WaQN%2FQrq66odvPAh9Uqas8dJov8mOpQXPQHbTciPnWJy%2FddwCKcnEhxHYmZYX%2FHZ3zn121gkGv1Pc2EM%2FpsEe%2B7AXnG8SU4z8SVRhOJ5QaaChr6Q78re%2BA9UOhlDJIE4UY5f3%2B966TY3q%2FDnctiqMDKwM1Cwfle48fjBWjGHm8YVdwcnAmsmwrVJ9VJtHBziAmOlPprDjgEtWutfaJJ6c9rYMf7tu0atu0tfKfLAOjsw37jCyAY6mAHt4pWB6eoYocnk20%2FRJuHHSqwi%2BZwaZS2aJpZNfR1uE5Vp87qJNKK88LNdwltmUZxV3ekU9bgA7TdGGq%2F6hdNTQNIOCjw%2Bfs4SHSfozIOUxsSxSpcLkJ5FElladjbZaTf%2Blsy6V%2FhMsB3JtAaVbjUyPWJ7LjT0U2l2RDYzY4B3w1WbPICRiolIRA2lLLpo7%2FZapp81Y%2Fsp6A%3D%3D&Expires=1762697702)
[16](https://tc.copernicus.org/articles/18/153/2024/)
[17](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/14235120/db28c8bd-a559-431e-a4c9-28f0937300cf/instructions.txt)
[18](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/14235120/7af300bb-e27c-450c-85b1-943f4687a532/multiband_samples.jpg?AWSAccessKeyId=ASIA2F3EMEYE3JFBS3SR&Signature=1unwtD5u3EJ3kDFJIexvBnsM7RU%3D&x-amz-security-token=IQoJb3JpZ2luX2VjECYaCXVzLWVhc3QtMSJHMEUCIQD516h5mefvjNKG8bGcUDy5gzZ%2Fqx9dZ5wSQel4uyoKCgIgbxB3%2BMtXJoKxGdBAH%2FPa1BYfoU3VI9uQMSAZctl4OGAq%2FAQI7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDInJI1qNcsrOwYGo6CrQBPIGGoSoHCEM7rc6XkfgonuumnJKZfMeEPVUWCoKLSY7Blh6WGVMgBX8S4eRqTwBj7nMRT22TGHJkuai9Y6FtqAtbP5q0RGadTWC5O2vc34Jdi5db5uWNYD32QNV1XiYGLo9INpGfJRL5WMqV8UqswUBz9TS4imBYQdqz3LLVtsg5WRcQtHzYAKvQzsVKQ8SJOirtfVqF9tVi%2BEMI0uRthspv8ScdUzUwyIxV0d%2BC4kk%2B6VkaL4KEPSQbh4PRjP6NWp5ncbaFHTIjawMhNo5Q1xhSJEE0cpQLgkQsg3janawc7%2FLOtcZRMV6Mf8NL1ODImCglmOf4GlDlVfwCWF7DNAJs4S7zaMjkf67Y0ta2JgALHB2dDPXv4nr35mYXY3C7QrM237l31cqztOJsSfsndfOQPs89%2BEPXFXyOPvfdzlP8OfWm3QbVjKganDUuj1DxKM60lotetvmzpGM7pBMDFbGP2lCIhbK1KCk05FzZ7P%2FVTRWLs68dXQ5dp9tY%2BXRJy%2FLB29YQsGBb5GT%2BVJ6vFR1G3UCzAiH%2Fcco%2B%2FgINS16WaQN%2FQrq66odvPAh9Uqas8dJov8mOpQXPQHbTciPnWJy%2FddwCKcnEhxHYmZYX%2FHZ3zn121gkGv1Pc2EM%2FpsEe%2B7AXnG8SU4z8SVRhOJ5QaaChr6Q78re%2BA9UOhlDJIE4UY5f3%2B966TY3q%2FDnctiqMDKwM1Cwfle48fjBWjGHm8YVdwcnAmsmwrVJ9VJtHBziAmOlPprDjgEtWutfaJJ6c9rYMf7tu0atu0tfKfLAOjsw37jCyAY6mAHt4pWB6eoYocnk20%2FRJuHHSqwi%2BZwaZS2aJpZNfR1uE5Vp87qJNKK88LNdwltmUZxV3ekU9bgA7TdGGq%2F6hdNTQNIOCjw%2Bfs4SHSfozIOUxsSxSpcLkJ5FElladjbZaTf%2Blsy6V%2FhMsB3JtAaVbjUyPWJ7LjT0U2l2RDYzY4B3w1WbPICRiolIRA2lLLpo7%2FZapp81Y%2Fsp6A%3D%3D&Expires=1762697702)
[19](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/14235120/5c9ef9a7-e2a3-4e7b-8f09-1f11073f6b32/sample_solution2.py)
[20](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/14235120/857c3b12-ac5f-4891-a8b1-64af10b23f1c/requirements.txt)
[21](https://ashpublications.org/blood/article/144/Supplement%201/1712/531938/PET-Based-Prognostic-Stratification-in-Primary)
[22](https://www.sciltp.com/journals/ijndi/2024/2/411)
[23](https://journaljerr.com/index.php/JERR/article/view/1083)
[24](https://www.cambridge.org/core/product/identifier/S0029665125001533/type/journal_article)
[25](https://link.springer.com/10.1007/s40123-025-01125-y)
[26](https://www.questjournals.org/jram/papers/v11-i6/11064150.pdf)
[27](https://arxiv.org/pdf/2302.02744.pdf)
[28](https://arxiv.org/abs/2204.05818)
[29](https://www.mdpi.com/2071-1050/14/20/13485/pdf?version=1666172449)
[30](https://pmc.ncbi.nlm.nih.gov/articles/PMC11695715/)
[31](https://arxiv.org/pdf/2112.08184.pdf)
[32](https://www.sciencedirect.com/science/article/pii/S0924271625003806)
[33](https://www.nature.com/articles/s41467-024-54956-x)
[34](https://www.sciencedirect.com/science/article/pii/S3050483X2500036X)
[35](https://www.cambridge.org/core/journals/journal-of-glaciology/article/semantic-segmentation-of-glaciological-features-across-multiple-remote-sensing-platforms-with-the-segment-anything-model-sam/66D3A237ACB0975C9EE9BE19E0C2564E)
[36](http://pages.cvc.uab.es/asappa/publications/J__IEEE_Journal_Selected_Topics_in_ARS_&_RS_Vol_17_pp_14295-14336.pdf)Based on my analysis of your comprehensive EDA visualizations and the competition requirements, I'll provide an **honest, experienced evaluation** of the best architecture and strategy to achieve 0.75+ MCC and eventually reach 0.8+ MCC to win this glacier segmentation competition.

[1][2][3]

## Critical Data Analysis

### Class Imbalance Reality Check

Your EDA reveals a **severe class imbalance** that will dominate your modeling decisions:[2]
- Background: ~70% (4.5M pixels)
- Glacier: ~25% (1.67M pixels) 
- Debris: ~5% (323K pixels)
- **Lake: ~0.05% (3.1K pixels)** ← This is catastrophic

The MCC metric is **particularly sensitive** to this imbalance, especially the lake class. With only 3,104 lake pixels across your entire dataset, any false positives or false negatives will tank your MCC score exponentially.[4][5]

### Boundary Pixel Problem

Your boundary analysis shows **3-7% boundary pixels** with high variance (Total: 4.4-10.4%). For glacier segmentation, **boundary accuracy is critical**—this is where debris-covered glaciers transition and where most errors occur.[6][7][1]

### Spectral Signatures

The spectral analysis reveals:[8][3]
- **SWIR (Band 4)** has the best separability for glacier vs background (9931 absolute difference)
- **TIR (Band 5)** provides excellent complementary information (thermal signature)
- Blue/Green/Red bands are **highly correlated** (0.997-0.999) → redundancy[9]

## Recommended Architecture Strategy

### Phase 1: Baseline Architecture (Target: 0.65-0.72 MCC)

**Model: DeepLabV3+ with EfficientNet-B4 Encoder**

Why this specific choice:[10][11][12]
1. **DeepLabV3+** consistently outperforms U-Net variants in remote sensing with multispectral data[12]
2. **Atrous Spatial Pyramid Pooling (ASPP)** handles multi-scale glacier features (your cluster sizes vary 75-254 pixels)[13]
3. **EfficientNet-B4** encoder provides optimal parameter efficiency for 300MB weight limit
4. Proven success on imbalanced segmentation tasks[11]

```python
import segmentation_models_pytorch as smp

model = smp.DeepLabV3Plus(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",  # Start here, then custom pretrain
    in_channels=5,  # Your 5 bands
    classes=4,      # Background, glacier, debris, lake
    activation=None,
    encoder_depth=5,
    decoder_channels=256,
    decoder_atrous_rates=(6, 12, 18),  # Adjusted for glacier scales
)
```

### Phase 2: Feature Engineering (Critical for +0.05-0.08 MCC)

**Input Channels (8 total)**:
1. **5 Original Bands** (Blue, Green, Red, SWIR, TIR)
2. **NDSI** = (Green - SWIR) / (Green + SWIR)[8]
3. **NDWI** = (Green - TIR) / (Green + TIR)[8]
4. **Slope/Aspect** (if DEM available) OR **Texture (GLCM Contrast)**[14]

Your NDSI/NDWI distributions show **excellent class separation**—not using these is leaving performance on the table.[7][6][8]

**Critical**: Normalize each band independently using **per-band statistics** from your training set. Your PCA shows 91% variance in PC1—this means proper normalization is essential.[15]

### Phase 3: Loss Function Design (Most Critical for MCC)

**DO NOT use standard CrossEntropy or Dice Loss alone**. For MCC optimization with extreme imbalance:[5]

```python
class MCCLoss(nn.Module):
    """MCC-optimized loss for imbalanced segmentation"""
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = class_weights
        
    def forward(self, pred, target):
        # Soft MCC calculation on probabilistic outputs
        # See: Abhishek et al. "Matthews Correlation Coefficient Loss"
        pass

# Combine multiple losses:
loss = (
    0.3 * FocalLoss(alpha=class_weights, gamma=2.0) +  # Handle imbalance
    0.3 * LovaszHingeLoss() +  # Optimize IoU directly
    0.4 * MCCLoss(class_weights)  # Directly optimize your metric
)
```

**Class Weights** (based on your distribution):
```python
weights = torch.tensor([
    0.5,   # Background (downweight majority)
    2.0,   # Glacier (moderate weight)
    10.0,  # Debris (high weight)
    100.0  # Lake (extreme weight to prevent ignoring)
])
```


### Phase 4: Training Strategy

**1. Two-Stage Training**:
- **Stage 1** (20 epochs): Train on balanced batches (sample equal pixels per class)
- **Stage 2** (30 epochs): Fine-tune on full distribution with MCC loss

**2. Data Augmentation** (critical for generalization):
```python
import albumentations as A

train_transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.ElasticTransform(alpha=120, sigma=6, p=0.3),  # Simulate terrain variation
    # CRITICAL: Don't over-augment spectral indices
])
```

**3. Patch-Based Training** (given Kaggle T4 12h limits):
- **512×512 patches** with 50% overlap
- **Batch size: 8-12** (fits in T4 16GB)
- Mixed precision training (`torch.cuda.amp`)

**4. Learning Rate Schedule**:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)
```

### Phase 5: Post-Processing (Expected +0.03-0.05 MCC)

**Critical for boundary refinement**:[16]

```python
from skimage.segmentation import slic
from pydensecrf import densecrf

def postprocess(prediction, image):
    # 1. Superpixel-guided refinement
    superpixels = slic(image, n_segments=2800, compactness=60, iterations=10)
    refined = majority_vote_superpixels(prediction, superpixels)
    
    # 2. Dense CRF (proven +6.92% IoU improvement)
    crf_output = apply_dense_crf(refined, image)
    
    # 3. Morphological operations
    # Remove small isolated predictions (< 20 pixels)
    # Fill small holes (< 10 pixels)
    
    # 4. Class-specific rules:
    # - Lakes must be within/adjacent to glaciers
    # - Apply minimum size thresholds per class
    
    return crf_output
```

This two-level optimization (SLIC + DenseCRF) achieved **+6.92% IoU** in similar glacier segmentation tasks.[16]

### Phase 6: Advanced Improvements (0.75 → 0.8+ MCC)

**Once you hit 0.75 MCC, implement these**:

**1. Boundary-Aware Loss**:[6][7]
```python
class BoundaryAwareLoss(nn.Module):
    """Self-learning boundary-aware loss"""
    def __init__(self):
        super().__init__()
        self.boundary_weight = 5.0  # Penalize boundary errors more
        
    def forward(self, pred, target):
        # Extract boundary pixels (3-pixel width)
        boundary_mask = extract_boundaries(target)
        
        # Weight loss by boundary proximity
        weighted_loss = base_loss * (1 + self.boundary_weight * boundary_mask)
        return weighted_loss.mean()
```

This specifically addresses your 3-7% boundary pixel challenge and has proven effectiveness for debris-covered glaciers.[1][6]

**2. Test-Time Augmentation (TTA)**:
```python
def tta_predict(model, image):
    predictions = []
    # Apply 8 geometric augmentations
    for transform in [identity, rot90, rot180, rot270, flip_h, flip_v, flip_rot90, flip_rot270]:
        aug_image = transform(image)
        pred = model(aug_image)
        pred = inverse_transform(pred, transform)
        predictions.append(pred)
    
    # Weighted average (higher weight for identity)
    return weighted_ensemble(predictions)
```

Expected gain: **+0.02-0.04 MCC**

**3. Ensemble Strategy** (if weight budget allows):
- DeepLabV3+ (EfficientNet-B4) [Primary - 80MB]
- U-Net++ (ResNet50) [Secondary - 100MB] 
- DeepLabV3+ (ResNeXt50) [Tertiary - 100MB]
- **Total: 280MB < 300MB limit**

Average predictions with learned weights based on validation MCC per class.

**4. Pseudo-Labeling** (if allowed):
- Use your best model to pseudo-label test set
- Retrain on train + high-confidence pseudo-labels
- Particularly effective for the rare lake class

## Critical Constraints & Realities

### Kaggle T4 12-Hour Limit
**Realistic Training Plan**:
- Preprocessing: 30 min
- Model training: 9 hours (50 epochs × ~10 min/epoch)
- Validation/checkpointing: 1 hour
- Post-processing experimentation: 1.5 hours

**Solution**: Train encoder separately on HKH pretraining data (as you mentioned), then fine-tune full model. This saves 3-4 hours.

### 300MB Weight Limit
- DeepLabV3+ (EfficientNet-B4): ~80MB
- Leaves room for ensemble or larger decoder

### Lake Class Challenge
**Honest Assessment**: With only 3,104 lake pixels, you will **struggle** to achieve high lake recall without massive false positives. Strategies:
1. **Oversample lake patches** during training (10×)
2. Apply **extreme class weighting** (100×)
3. **Post-processing rule**: Only predict lake within 50 pixels of glacier
4. Consider **hierarchical prediction**: First segment glacier/non-glacier, then segment glacier into clean/debris/lake

## Timeline to 0.75+ MCC

**Week 1**: Implement DeepLabV3+ baseline with proper loss function → **Target: 0.68-0.70 MCC**
**Week 2**: Add feature engineering (NDSI/NDWI/texture) + boundary-aware loss → **Target: 0.72-0.74 MCC**
**Week 3**: Hyperparameter tuning + post-processing (CRF) → **Target: 0.75-0.77 MCC**
**Week 4**: Ensemble + TTA → **Target: 0.78-0.82 MCC**

## Questions I Need Answered

1. **Do you have access to HKH pretraining data still?** Pretraining encoder on domain-specific data could give +0.03-0.05 MCC boost.

2. **What's your current validation split?** With cross-region validation (as per instructions), you need **spatial CV** not random split.[17]

3. **Are there any terrain/DEM features available?** Slope/aspect would be extremely valuable for debris-covered glacier segmentation.

4. **Can you access the test set region metadata?** Knowing if test is from same region or different region changes strategy.

This architecture and strategy directly addresses your severe class imbalance, leverages your strong spectral features, and uses proven techniques from recent glacier segmentation literature. The phased approach ensures you progressively improve rather than trying everything at once.

**Be strict with yourself**: Don't waste time on fancy architectures (Vision Transformers, etc.) until you've maximized this proven baseline. The competition-winning difference will be in loss function engineering, post-processing, and handling the lake class properly.[7][5][6][16]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/14235120/cd605c19-6794-43b8-be09-1fb60d3d6fbf/boundary_analysis.jpg?AWSAccessKeyId=ASIA2F3EMEYE3JFBS3SR&Signature=P%2BxkmkY5WcVv33bjbG3BKewhyFI%3D&x-amz-security-token=IQoJb3JpZ2luX2VjECYaCXVzLWVhc3QtMSJHMEUCIQD516h5mefvjNKG8bGcUDy5gzZ%2Fqx9dZ5wSQel4uyoKCgIgbxB3%2BMtXJoKxGdBAH%2FPa1BYfoU3VI9uQMSAZctl4OGAq%2FAQI7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDInJI1qNcsrOwYGo6CrQBPIGGoSoHCEM7rc6XkfgonuumnJKZfMeEPVUWCoKLSY7Blh6WGVMgBX8S4eRqTwBj7nMRT22TGHJkuai9Y6FtqAtbP5q0RGadTWC5O2vc34Jdi5db5uWNYD32QNV1XiYGLo9INpGfJRL5WMqV8UqswUBz9TS4imBYQdqz3LLVtsg5WRcQtHzYAKvQzsVKQ8SJOirtfVqF9tVi%2BEMI0uRthspv8ScdUzUwyIxV0d%2BC4kk%2B6VkaL4KEPSQbh4PRjP6NWp5ncbaFHTIjawMhNo5Q1xhSJEE0cpQLgkQsg3janawc7%2FLOtcZRMV6Mf8NL1ODImCglmOf4GlDlVfwCWF7DNAJs4S7zaMjkf67Y0ta2JgALHB2dDPXv4nr35mYXY3C7QrM237l31cqztOJsSfsndfOQPs89%2BEPXFXyOPvfdzlP8OfWm3QbVjKganDUuj1DxKM60lotetvmzpGM7pBMDFbGP2lCIhbK1KCk05FzZ7P%2FVTRWLs68dXQ5dp9tY%2BXRJy%2FLB29YQsGBb5GT%2BVJ6vFR1G3UCzAiH%2Fcco%2B%2FgINS16WaQN%2FQrq66odvPAh9Uqas8dJov8mOpQXPQHbTciPnWJy%2FddwCKcnEhxHYmZYX%2FHZ3zn121gkGv1Pc2EM%2FpsEe%2B7AXnG8SU4z8SVRhOJ5QaaChr6Q78re%2BA9UOhlDJIE4UY5f3%2B966TY3q%2FDnctiqMDKwM1Cwfle48fjBWjGHm8YVdwcnAmsmwrVJ9VJtHBziAmOlPprDjgEtWutfaJJ6c9rYMf7tu0atu0tfKfLAOjsw37jCyAY6mAHt4pWB6eoYocnk20%2FRJuHHSqwi%2BZwaZS2aJpZNfR1uE5Vp87qJNKK88LNdwltmUZxV3ekU9bgA7TdGGq%2F6hdNTQNIOCjw%2Bfs4SHSfozIOUxsSxSpcLkJ5FElladjbZaTf%2Blsy6V%2FhMsB3JtAaVbjUyPWJ7LjT0U2l2RDYzY4B3w1WbPICRiolIRA2lLLpo7%2FZapp81Y%2Fsp6A%3D%3D&Expires=1762697702)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/14235120/ea1c46c2-300d-44ac-962e-45926f4a65ce/class_distribution_analysis.jpg?AWSAccessKeyId=ASIA2F3EMEYE3JFBS3SR&Signature=WL9SVfRq2yrON6RnIvBlEPINTjo%3D&x-amz-security-token=IQoJb3JpZ2luX2VjECYaCXVzLWVhc3QtMSJHMEUCIQD516h5mefvjNKG8bGcUDy5gzZ%2Fqx9dZ5wSQel4uyoKCgIgbxB3%2BMtXJoKxGdBAH%2FPa1BYfoU3VI9uQMSAZctl4OGAq%2FAQI7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDInJI1qNcsrOwYGo6CrQBPIGGoSoHCEM7rc6XkfgonuumnJKZfMeEPVUWCoKLSY7Blh6WGVMgBX8S4eRqTwBj7nMRT22TGHJkuai9Y6FtqAtbP5q0RGadTWC5O2vc34Jdi5db5uWNYD32QNV1XiYGLo9INpGfJRL5WMqV8UqswUBz9TS4imBYQdqz3LLVtsg5WRcQtHzYAKvQzsVKQ8SJOirtfVqF9tVi%2BEMI0uRthspv8ScdUzUwyIxV0d%2BC4kk%2B6VkaL4KEPSQbh4PRjP6NWp5ncbaFHTIjawMhNo5Q1xhSJEE0cpQLgkQsg3janawc7%2FLOtcZRMV6Mf8NL1ODImCglmOf4GlDlVfwCWF7DNAJs4S7zaMjkf67Y0ta2JgALHB2dDPXv4nr35mYXY3C7QrM237l31cqztOJsSfsndfOQPs89%2BEPXFXyOPvfdzlP8OfWm3QbVjKganDUuj1DxKM60lotetvmzpGM7pBMDFbGP2lCIhbK1KCk05FzZ7P%2FVTRWLs68dXQ5dp9tY%2BXRJy%2FLB29YQsGBb5GT%2BVJ6vFR1G3UCzAiH%2Fcco%2B%2FgINS16WaQN%2FQrq66odvPAh9Uqas8dJov8mOpQXPQHbTciPnWJy%2FddwCKcnEhxHYmZYX%2FHZ3zn121gkGv1Pc2EM%2FpsEe%2B7AXnG8SU4z8SVRhOJ5QaaChr6Q78re%2BA9UOhlDJIE4UY5f3%2B966TY3q%2FDnctiqMDKwM1Cwfle48fjBWjGHm8YVdwcnAmsmwrVJ9VJtHBziAmOlPprDjgEtWutfaJJ6c9rYMf7tu0atu0tfKfLAOjsw37jCyAY6mAHt4pWB6eoYocnk20%2FRJuHHSqwi%2BZwaZS2aJpZNfR1uE5Vp87qJNKK88LNdwltmUZxV3ekU9bgA7TdGGq%2F6hdNTQNIOCjw%2Bfs4SHSfozIOUxsSxSpcLkJ5FElladjbZaTf%2Blsy6V%2FhMsB3JtAaVbjUyPWJ7LjT0U2l2RDYzY4B3w1WbPICRiolIRA2lLLpo7%2FZapp81Y%2Fsp6A%3D%3D&Expires=1762697702)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/14235120/1363e34b-ab26-40f8-86e0-71373f2f1708/spectral_signatures.jpg?AWSAccessKeyId=ASIA2F3EMEYE3JFBS3SR&Signature=f3W5WogdoyyQkPie1qgGynIFFhM%3D&x-amz-security-token=IQoJb3JpZ2luX2VjECYaCXVzLWVhc3QtMSJHMEUCIQD516h5mefvjNKG8bGcUDy5gzZ%2Fqx9dZ5wSQel4uyoKCgIgbxB3%2BMtXJoKxGdBAH%2FPa1BYfoU3VI9uQMSAZctl4OGAq%2FAQI7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDInJI1qNcsrOwYGo6CrQBPIGGoSoHCEM7rc6XkfgonuumnJKZfMeEPVUWCoKLSY7Blh6WGVMgBX8S4eRqTwBj7nMRT22TGHJkuai9Y6FtqAtbP5q0RGadTWC5O2vc34Jdi5db5uWNYD32QNV1XiYGLo9INpGfJRL5WMqV8UqswUBz9TS4imBYQdqz3LLVtsg5WRcQtHzYAKvQzsVKQ8SJOirtfVqF9tVi%2BEMI0uRthspv8ScdUzUwyIxV0d%2BC4kk%2B6VkaL4KEPSQbh4PRjP6NWp5ncbaFHTIjawMhNo5Q1xhSJEE0cpQLgkQsg3janawc7%2FLOtcZRMV6Mf8NL1ODImCglmOf4GlDlVfwCWF7DNAJs4S7zaMjkf67Y0ta2JgALHB2dDPXv4nr35mYXY3C7QrM237l31cqztOJsSfsndfOQPs89%2BEPXFXyOPvfdzlP8OfWm3QbVjKganDUuj1DxKM60lotetvmzpGM7pBMDFbGP2lCIhbK1KCk05FzZ7P%2FVTRWLs68dXQ5dp9tY%2BXRJy%2FLB29YQsGBb5GT%2BVJ6vFR1G3UCzAiH%2Fcco%2B%2FgINS16WaQN%2FQrq66odvPAh9Uqas8dJov8mOpQXPQHbTciPnWJy%2FddwCKcnEhxHYmZYX%2FHZ3zn121gkGv1Pc2EM%2FpsEe%2B7AXnG8SU4z8SVRhOJ5QaaChr6Q78re%2BA9UOhlDJIE4UY5f3%2B966TY3q%2FDnctiqMDKwM1Cwfle48fjBWjGHm8YVdwcnAmsmwrVJ9VJtHBziAmOlPprDjgEtWutfaJJ6c9rYMf7tu0atu0tfKfLAOjsw37jCyAY6mAHt4pWB6eoYocnk20%2FRJuHHSqwi%2BZwaZS2aJpZNfR1uE5Vp87qJNKK88LNdwltmUZxV3ekU9bgA7TdGGq%2F6hdNTQNIOCjw%2Bfs4SHSfozIOUxsSxSpcLkJ5FElladjbZaTf%2Blsy6V%2FhMsB3JtAaVbjUyPWJ7LjT0U2l2RDYzY4B3w1WbPICRiolIRA2lLLpo7%2FZapp81Y%2Fsp6A%3D%3D&Expires=1762697702)
[4](https://openreview.net/forum?id=jGyMUum1Lq&noteId=WCw88yUvbx)
[5](https://www2.cs.sfu.ca/~hamarneh/ecopy/isbi2021.pdf)
[6](http://arxiv.org/pdf/2301.11454.pdf)
[7](https://septentrio.uit.no/index.php/nldl/article/download/6789/7028)
[8](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/14235120/09c6cad4-a15b-46f2-9467-6391aa747303/spectral_indices_analysis.jpg?AWSAccessKeyId=ASIA2F3EMEYE3JFBS3SR&Signature=5ddrm30BbiRsW2Bt668l5jep2Yo%3D&x-amz-security-token=IQoJb3JpZ2luX2VjECYaCXVzLWVhc3QtMSJHMEUCIQD516h5mefvjNKG8bGcUDy5gzZ%2Fqx9dZ5wSQel4uyoKCgIgbxB3%2BMtXJoKxGdBAH%2FPa1BYfoU3VI9uQMSAZctl4OGAq%2FAQI7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDInJI1qNcsrOwYGo6CrQBPIGGoSoHCEM7rc6XkfgonuumnJKZfMeEPVUWCoKLSY7Blh6WGVMgBX8S4eRqTwBj7nMRT22TGHJkuai9Y6FtqAtbP5q0RGadTWC5O2vc34Jdi5db5uWNYD32QNV1XiYGLo9INpGfJRL5WMqV8UqswUBz9TS4imBYQdqz3LLVtsg5WRcQtHzYAKvQzsVKQ8SJOirtfVqF9tVi%2BEMI0uRthspv8ScdUzUwyIxV0d%2BC4kk%2B6VkaL4KEPSQbh4PRjP6NWp5ncbaFHTIjawMhNo5Q1xhSJEE0cpQLgkQsg3janawc7%2FLOtcZRMV6Mf8NL1ODImCglmOf4GlDlVfwCWF7DNAJs4S7zaMjkf67Y0ta2JgALHB2dDPXv4nr35mYXY3C7QrM237l31cqztOJsSfsndfOQPs89%2BEPXFXyOPvfdzlP8OfWm3QbVjKganDUuj1DxKM60lotetvmzpGM7pBMDFbGP2lCIhbK1KCk05FzZ7P%2FVTRWLs68dXQ5dp9tY%2BXRJy%2FLB29YQsGBb5GT%2BVJ6vFR1G3UCzAiH%2Fcco%2B%2FgINS16WaQN%2FQrq66odvPAh9Uqas8dJov8mOpQXPQHbTciPnWJy%2FddwCKcnEhxHYmZYX%2FHZ3zn121gkGv1Pc2EM%2FpsEe%2B7AXnG8SU4z8SVRhOJ5QaaChr6Q78re%2BA9UOhlDJIE4UY5f3%2B966TY3q%2FDnctiqMDKwM1Cwfle48fjBWjGHm8YVdwcnAmsmwrVJ9VJtHBziAmOlPprDjgEtWutfaJJ6c9rYMf7tu0atu0tfKfLAOjsw37jCyAY6mAHt4pWB6eoYocnk20%2FRJuHHSqwi%2BZwaZS2aJpZNfR1uE5Vp87qJNKK88LNdwltmUZxV3ekU9bgA7TdGGq%2F6hdNTQNIOCjw%2Bfs4SHSfozIOUxsSxSpcLkJ5FElladjbZaTf%2Blsy6V%2FhMsB3JtAaVbjUyPWJ7LjT0U2l2RDYzY4B3w1WbPICRiolIRA2lLLpo7%2FZapp81Y%2Fsp6A%3D%3D&Expires=1762697702)
[9](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/14235120/7dac20ef-30ca-4783-9d92-f39fe9590a59/band_correlation.jpg?AWSAccessKeyId=ASIA2F3EMEYE3JFBS3SR&Signature=wbR8ra4zE2Dv%2FhaKbHyfl1lB97k%3D&x-amz-security-token=IQoJb3JpZ2luX2VjECYaCXVzLWVhc3QtMSJHMEUCIQD516h5mefvjNKG8bGcUDy5gzZ%2Fqx9dZ5wSQel4uyoKCgIgbxB3%2BMtXJoKxGdBAH%2FPa1BYfoU3VI9uQMSAZctl4OGAq%2FAQI7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDInJI1qNcsrOwYGo6CrQBPIGGoSoHCEM7rc6XkfgonuumnJKZfMeEPVUWCoKLSY7Blh6WGVMgBX8S4eRqTwBj7nMRT22TGHJkuai9Y6FtqAtbP5q0RGadTWC5O2vc34Jdi5db5uWNYD32QNV1XiYGLo9INpGfJRL5WMqV8UqswUBz9TS4imBYQdqz3LLVtsg5WRcQtHzYAKvQzsVKQ8SJOirtfVqF9tVi%2BEMI0uRthspv8ScdUzUwyIxV0d%2BC4kk%2B6VkaL4KEPSQbh4PRjP6NWp5ncbaFHTIjawMhNo5Q1xhSJEE0cpQLgkQsg3janawc7%2FLOtcZRMV6Mf8NL1ODImCglmOf4GlDlVfwCWF7DNAJs4S7zaMjkf67Y0ta2JgALHB2dDPXv4nr35mYXY3C7QrM237l31cqztOJsSfsndfOQPs89%2BEPXFXyOPvfdzlP8OfWm3QbVjKganDUuj1DxKM60lotetvmzpGM7pBMDFbGP2lCIhbK1KCk05FzZ7P%2FVTRWLs68dXQ5dp9tY%2BXRJy%2FLB29YQsGBb5GT%2BVJ6vFR1G3UCzAiH%2Fcco%2B%2FgINS16WaQN%2FQrq66odvPAh9Uqas8dJov8mOpQXPQHbTciPnWJy%2FddwCKcnEhxHYmZYX%2FHZ3zn121gkGv1Pc2EM%2FpsEe%2B7AXnG8SU4z8SVRhOJ5QaaChr6Q78re%2BA9UOhlDJIE4UY5f3%2B966TY3q%2FDnctiqMDKwM1Cwfle48fjBWjGHm8YVdwcnAmsmwrVJ9VJtHBziAmOlPprDjgEtWutfaJJ6c9rYMf7tu0atu0tfKfLAOjsw37jCyAY6mAHt4pWB6eoYocnk20%2FRJuHHSqwi%2BZwaZS2aJpZNfR1uE5Vp87qJNKK88LNdwltmUZxV3ekU9bgA7TdGGq%2F6hdNTQNIOCjw%2Bfs4SHSfozIOUxsSxSpcLkJ5FElladjbZaTf%2Blsy6V%2FhMsB3JtAaVbjUyPWJ7LjT0U2l2RDYzY4B3w1WbPICRiolIRA2lLLpo7%2FZapp81Y%2Fsp6A%3D%3D&Expires=1762697702)
[10](https://www.mdpi.com/2072-4292/13/24/5091/pdf)
[11](https://jiki.cs.ui.ac.id/index.php/jiki/article/view/1206/507)
[12](https://arxiv.org/html/2402.13918v3)
[13](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/14235120/36cf8f06-7652-4da6-98b2-49bc71570ba9/spatial_autocorrelation_analysis.jpg?AWSAccessKeyId=ASIA2F3EMEYE3JFBS3SR&Signature=Htvg%2Fn4LwJn0iTT%2BOKf8pv7xnhI%3D&x-amz-security-token=IQoJb3JpZ2luX2VjECYaCXVzLWVhc3QtMSJHMEUCIQD516h5mefvjNKG8bGcUDy5gzZ%2Fqx9dZ5wSQel4uyoKCgIgbxB3%2BMtXJoKxGdBAH%2FPa1BYfoU3VI9uQMSAZctl4OGAq%2FAQI7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDInJI1qNcsrOwYGo6CrQBPIGGoSoHCEM7rc6XkfgonuumnJKZfMeEPVUWCoKLSY7Blh6WGVMgBX8S4eRqTwBj7nMRT22TGHJkuai9Y6FtqAtbP5q0RGadTWC5O2vc34Jdi5db5uWNYD32QNV1XiYGLo9INpGfJRL5WMqV8UqswUBz9TS4imBYQdqz3LLVtsg5WRcQtHzYAKvQzsVKQ8SJOirtfVqF9tVi%2BEMI0uRthspv8ScdUzUwyIxV0d%2BC4kk%2B6VkaL4KEPSQbh4PRjP6NWp5ncbaFHTIjawMhNo5Q1xhSJEE0cpQLgkQsg3janawc7%2FLOtcZRMV6Mf8NL1ODImCglmOf4GlDlVfwCWF7DNAJs4S7zaMjkf67Y0ta2JgALHB2dDPXv4nr35mYXY3C7QrM237l31cqztOJsSfsndfOQPs89%2BEPXFXyOPvfdzlP8OfWm3QbVjKganDUuj1DxKM60lotetvmzpGM7pBMDFbGP2lCIhbK1KCk05FzZ7P%2FVTRWLs68dXQ5dp9tY%2BXRJy%2FLB29YQsGBb5GT%2BVJ6vFR1G3UCzAiH%2Fcco%2B%2FgINS16WaQN%2FQrq66odvPAh9Uqas8dJov8mOpQXPQHbTciPnWJy%2FddwCKcnEhxHYmZYX%2FHZ3zn121gkGv1Pc2EM%2FpsEe%2B7AXnG8SU4z8SVRhOJ5QaaChr6Q78re%2BA9UOhlDJIE4UY5f3%2B966TY3q%2FDnctiqMDKwM1Cwfle48fjBWjGHm8YVdwcnAmsmwrVJ9VJtHBziAmOlPprDjgEtWutfaJJ6c9rYMf7tu0atu0tfKfLAOjsw37jCyAY6mAHt4pWB6eoYocnk20%2FRJuHHSqwi%2BZwaZS2aJpZNfR1uE5Vp87qJNKK88LNdwltmUZxV3ekU9bgA7TdGGq%2F6hdNTQNIOCjw%2Bfs4SHSfozIOUxsSxSpcLkJ5FElladjbZaTf%2Blsy6V%2FhMsB3JtAaVbjUyPWJ7LjT0U2l2RDYzY4B3w1WbPICRiolIRA2lLLpo7%2FZapp81Y%2Fsp6A%3D%3D&Expires=1762697702)
[14](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/14235120/4b27170b-8e85-402d-94d8-1a9b757e2956/glcm_texture_analysis.jpg?AWSAccessKeyId=ASIA2F3EMEYE3JFBS3SR&Signature=0mxTD2Vz%2FB0gySq%2B3M6SfAWGVg8%3D&x-amz-security-token=IQoJb3JpZ2luX2VjECYaCXVzLWVhc3QtMSJHMEUCIQD516h5mefvjNKG8bGcUDy5gzZ%2Fqx9dZ5wSQel4uyoKCgIgbxB3%2BMtXJoKxGdBAH%2FPa1BYfoU3VI9uQMSAZctl4OGAq%2FAQI7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDInJI1qNcsrOwYGo6CrQBPIGGoSoHCEM7rc6XkfgonuumnJKZfMeEPVUWCoKLSY7Blh6WGVMgBX8S4eRqTwBj7nMRT22TGHJkuai9Y6FtqAtbP5q0RGadTWC5O2vc34Jdi5db5uWNYD32QNV1XiYGLo9INpGfJRL5WMqV8UqswUBz9TS4imBYQdqz3LLVtsg5WRcQtHzYAKvQzsVKQ8SJOirtfVqF9tVi%2BEMI0uRthspv8ScdUzUwyIxV0d%2BC4kk%2B6VkaL4KEPSQbh4PRjP6NWp5ncbaFHTIjawMhNo5Q1xhSJEE0cpQLgkQsg3janawc7%2FLOtcZRMV6Mf8NL1ODImCglmOf4GlDlVfwCWF7DNAJs4S7zaMjkf67Y0ta2JgALHB2dDPXv4nr35mYXY3C7QrM237l31cqztOJsSfsndfOQPs89%2BEPXFXyOPvfdzlP8OfWm3QbVjKganDUuj1DxKM60lotetvmzpGM7pBMDFbGP2lCIhbK1KCk05FzZ7P%2FVTRWLs68dXQ5dp9tY%2BXRJy%2FLB29YQsGBb5GT%2BVJ6vFR1G3UCzAiH%2Fcco%2B%2FgINS16WaQN%2FQrq66odvPAh9Uqas8dJov8mOpQXPQHbTciPnWJy%2FddwCKcnEhxHYmZYX%2FHZ3zn121gkGv1Pc2EM%2FpsEe%2B7AXnG8SU4z8SVRhOJ5QaaChr6Q78re%2BA9UOhlDJIE4UY5f3%2B966TY3q%2FDnctiqMDKwM1Cwfle48fjBWjGHm8YVdwcnAmsmwrVJ9VJtHBziAmOlPprDjgEtWutfaJJ6c9rYMf7tu0atu0tfKfLAOjsw37jCyAY6mAHt4pWB6eoYocnk20%2FRJuHHSqwi%2BZwaZS2aJpZNfR1uE5Vp87qJNKK88LNdwltmUZxV3ekU9bgA7TdGGq%2F6hdNTQNIOCjw%2Bfs4SHSfozIOUxsSxSpcLkJ5FElladjbZaTf%2Blsy6V%2FhMsB3JtAaVbjUyPWJ7LjT0U2l2RDYzY4B3w1WbPICRiolIRA2lLLpo7%2FZapp81Y%2Fsp6A%3D%3D&Expires=1762697702)
[15](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/14235120/9e956fe7-0001-46ac-a05e-e973664bf484/pca_analysis.jpg?AWSAccessKeyId=ASIA2F3EMEYE3JFBS3SR&Signature=iAoz%2BKReGtBvdyZnfI6DxfpOUBY%3D&x-amz-security-token=IQoJb3JpZ2luX2VjECYaCXVzLWVhc3QtMSJHMEUCIQD516h5mefvjNKG8bGcUDy5gzZ%2Fqx9dZ5wSQel4uyoKCgIgbxB3%2BMtXJoKxGdBAH%2FPa1BYfoU3VI9uQMSAZctl4OGAq%2FAQI7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDInJI1qNcsrOwYGo6CrQBPIGGoSoHCEM7rc6XkfgonuumnJKZfMeEPVUWCoKLSY7Blh6WGVMgBX8S4eRqTwBj7nMRT22TGHJkuai9Y6FtqAtbP5q0RGadTWC5O2vc34Jdi5db5uWNYD32QNV1XiYGLo9INpGfJRL5WMqV8UqswUBz9TS4imBYQdqz3LLVtsg5WRcQtHzYAKvQzsVKQ8SJOirtfVqF9tVi%2BEMI0uRthspv8ScdUzUwyIxV0d%2BC4kk%2B6VkaL4KEPSQbh4PRjP6NWp5ncbaFHTIjawMhNo5Q1xhSJEE0cpQLgkQsg3janawc7%2FLOtcZRMV6Mf8NL1ODImCglmOf4GlDlVfwCWF7DNAJs4S7zaMjkf67Y0ta2JgALHB2dDPXv4nr35mYXY3C7QrM237l31cqztOJsSfsndfOQPs89%2BEPXFXyOPvfdzlP8OfWm3QbVjKganDUuj1DxKM60lotetvmzpGM7pBMDFbGP2lCIhbK1KCk05FzZ7P%2FVTRWLs68dXQ5dp9tY%2BXRJy%2FLB29YQsGBb5GT%2BVJ6vFR1G3UCzAiH%2Fcco%2B%2FgINS16WaQN%2FQrq66odvPAh9Uqas8dJov8mOpQXPQHbTciPnWJy%2FddwCKcnEhxHYmZYX%2FHZ3zn121gkGv1Pc2EM%2FpsEe%2B7AXnG8SU4z8SVRhOJ5QaaChr6Q78re%2BA9UOhlDJIE4UY5f3%2B966TY3q%2FDnctiqMDKwM1Cwfle48fjBWjGHm8YVdwcnAmsmwrVJ9VJtHBziAmOlPprDjgEtWutfaJJ6c9rYMf7tu0atu0tfKfLAOjsw37jCyAY6mAHt4pWB6eoYocnk20%2FRJuHHSqwi%2BZwaZS2aJpZNfR1uE5Vp87qJNKK88LNdwltmUZxV3ekU9bgA7TdGGq%2F6hdNTQNIOCjw%2Bfs4SHSfozIOUxsSxSpcLkJ5FElladjbZaTf%2Blsy6V%2FhMsB3JtAaVbjUyPWJ7LjT0U2l2RDYzY4B3w1WbPICRiolIRA2lLLpo7%2FZapp81Y%2Fsp6A%3D%3D&Expires=1762697702)
[16](https://tc.copernicus.org/articles/18/153/2024/)
[17](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/14235120/db28c8bd-a559-431e-a4c9-28f0937300cf/instructions.txt)
[18](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/14235120/7af300bb-e27c-450c-85b1-943f4687a532/multiband_samples.jpg?AWSAccessKeyId=ASIA2F3EMEYE3JFBS3SR&Signature=1unwtD5u3EJ3kDFJIexvBnsM7RU%3D&x-amz-security-token=IQoJb3JpZ2luX2VjECYaCXVzLWVhc3QtMSJHMEUCIQD516h5mefvjNKG8bGcUDy5gzZ%2Fqx9dZ5wSQel4uyoKCgIgbxB3%2BMtXJoKxGdBAH%2FPa1BYfoU3VI9uQMSAZctl4OGAq%2FAQI7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDInJI1qNcsrOwYGo6CrQBPIGGoSoHCEM7rc6XkfgonuumnJKZfMeEPVUWCoKLSY7Blh6WGVMgBX8S4eRqTwBj7nMRT22TGHJkuai9Y6FtqAtbP5q0RGadTWC5O2vc34Jdi5db5uWNYD32QNV1XiYGLo9INpGfJRL5WMqV8UqswUBz9TS4imBYQdqz3LLVtsg5WRcQtHzYAKvQzsVKQ8SJOirtfVqF9tVi%2BEMI0uRthspv8ScdUzUwyIxV0d%2BC4kk%2B6VkaL4KEPSQbh4PRjP6NWp5ncbaFHTIjawMhNo5Q1xhSJEE0cpQLgkQsg3janawc7%2FLOtcZRMV6Mf8NL1ODImCglmOf4GlDlVfwCWF7DNAJs4S7zaMjkf67Y0ta2JgALHB2dDPXv4nr35mYXY3C7QrM237l31cqztOJsSfsndfOQPs89%2BEPXFXyOPvfdzlP8OfWm3QbVjKganDUuj1DxKM60lotetvmzpGM7pBMDFbGP2lCIhbK1KCk05FzZ7P%2FVTRWLs68dXQ5dp9tY%2BXRJy%2FLB29YQsGBb5GT%2BVJ6vFR1G3UCzAiH%2Fcco%2B%2FgINS16WaQN%2FQrq66odvPAh9Uqas8dJov8mOpQXPQHbTciPnWJy%2FddwCKcnEhxHYmZYX%2FHZ3zn121gkGv1Pc2EM%2FpsEe%2B7AXnG8SU4z8SVRhOJ5QaaChr6Q78re%2BA9UOhlDJIE4UY5f3%2B966TY3q%2FDnctiqMDKwM1Cwfle48fjBWjGHm8YVdwcnAmsmwrVJ9VJtHBziAmOlPprDjgEtWutfaJJ6c9rYMf7tu0atu0tfKfLAOjsw37jCyAY6mAHt4pWB6eoYocnk20%2FRJuHHSqwi%2BZwaZS2aJpZNfR1uE5Vp87qJNKK88LNdwltmUZxV3ekU9bgA7TdGGq%2F6hdNTQNIOCjw%2Bfs4SHSfozIOUxsSxSpcLkJ5FElladjbZaTf%2Blsy6V%2FhMsB3JtAaVbjUyPWJ7LjT0U2l2RDYzY4B3w1WbPICRiolIRA2lLLpo7%2FZapp81Y%2Fsp6A%3D%3D&Expires=1762697702)
[19](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/14235120/5c9ef9a7-e2a3-4e7b-8f09-1f11073f6b32/sample_solution2.py)
[20](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/14235120/857c3b12-ac5f-4891-a8b1-64af10b23f1c/requirements.txt)
[21](https://ashpublications.org/blood/article/144/Supplement%201/1712/531938/PET-Based-Prognostic-Stratification-in-Primary)
[22](https://www.sciltp.com/journals/ijndi/2024/2/411)
[23](https://journaljerr.com/index.php/JERR/article/view/1083)
[24](https://www.cambridge.org/core/product/identifier/S0029665125001533/type/journal_article)
[25](https://link.springer.com/10.1007/s40123-025-01125-y)
[26](https://www.questjournals.org/jram/papers/v11-i6/11064150.pdf)
[27](https://arxiv.org/pdf/2302.02744.pdf)
[28](https://arxiv.org/abs/2204.05818)
[29](https://www.mdpi.com/2071-1050/14/20/13485/pdf?version=1666172449)
[30](https://pmc.ncbi.nlm.nih.gov/articles/PMC11695715/)
[31](https://arxiv.org/pdf/2112.08184.pdf)
[32](https://www.sciencedirect.com/science/article/pii/S0924271625003806)
[33](https://www.nature.com/articles/s41467-024-54956-x)
[34](https://www.sciencedirect.com/science/article/pii/S3050483X2500036X)
[35](https://www.cambridge.org/core/journals/journal-of-glaciology/article/semantic-segmentation-of-glaciological-features-across-multiple-remote-sensing-platforms-with-the-segment-anything-model-sam/66D3A237ACB0975C9EE9BE19E0C2564E)
[36](http://pages.cvc.uab.es/asappa/publications/J__IEEE_Journal_Selected_Topics_in_ARS_&_RS_Vol_17_pp_14295-14336.pdf)
You're absolutely right to call me out. Let me address these **critical gaps** I missed—these could be the difference between 0.72 and 0.80+ MCC.

## 1. Multi-Class MCC: You're Right, I Was Sloppy

**The sklearn MCC handles multiclass natively**, but there are critical nuances I overlooked:[1][2]

```python
from sklearn.metrics import matthews_corrcoef

# This already handles 4-class (glacier, debris, lake, background)
mcc_score = matthews_corrcoef(y_true.flatten(), y_pred.flatten())
```

The **multiclass MCC formula** uses the confusion matrix directly:[2][1]

$$
MCC = \frac{c \times s - \sum_{k} p_k \times t_k}{\sqrt{(s^2 - \sum_{k} p_k^2) \times (s^2 - \sum_{k} t_k^2)}}
$$

Where:
- $$c$$ = trace of confusion matrix (correct predictions)
- $$s$$ = sum of all elements in confusion matrix
- $$p_k$$ = sum of row $$k$$ (predictions for class $$k$$)
- $$t_k$$ = sum of column $$k$$ (true labels for class $$k$$)

**Critical insight**: With your extreme imbalance (lake class = 0.05%), the MCC will be **heavily penalized** by even small numbers of false positives on the lake class. This is why direct MCC optimization in the loss is crucial.[3][4]

### Implementing MCC Loss for Multiclass

I initially suggested a vague "MCCLoss"—here's the **actual implementation** you need:

```python
import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef

class MulticlassMCCLoss(nn.Module):
    """
    Differentiable MCC loss for multiclass segmentation.
    Based on: "Aligning Multiclass Neural Network Classifier Criterion 
    with Task Performance Metrics" (2024)
    """
    def __init__(self, num_classes=4, eps=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) - raw logits
            targets: (B, H, W) - class indices [0, 1, 2, 3]
        """
        # Softmax to get probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Flatten
        probs = probs.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        targets = targets.reshape(-1)
        
        # One-hot encode targets
        targets_one_hot = torch.nn.functional.one_hot(
            targets, num_classes=self.num_classes
        ).float()
        
        # Compute soft confusion matrix (differentiable)
        # C[i,j] = sum of predicted prob for class j when true class is i
        confusion = torch.matmul(targets_one_hot.T, probs)  # (C, C)
        
        # MCC computation from confusion matrix
        c = torch.trace(confusion)  # Correct predictions
        s = confusion.sum()  # Total predictions
        
        # Sum over rows and columns
        pk = confusion.sum(dim=1)  # Predictions per class
        tk = confusion.sum(dim=0)  # True labels per class
        
        numerator = c * s - torch.sum(pk * tk)
        denominator = torch.sqrt(
            (s**2 - torch.sum(pk**2)) * (s**2 - torch.sum(tk**2)) + self.eps
        )
        
        mcc = numerator / denominator
        
        # Return negative MCC as loss (we want to maximize MCC)
        return 1.0 - mcc

# Updated combined loss
class CombinedSegmentationLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.focal = FocalLoss(alpha=class_weights, gamma=2.0)
        self.lovasz = LovaszHingeLoss()
        self.mcc = MulticlassMCCLoss(num_classes=4)
        
    def forward(self, logits, targets):
        return (
            0.25 * self.focal(logits, targets) +
            0.25 * self.lovasz(logits, targets) +
            0.50 * self.mcc(logits, targets)  # Higher weight on direct MCC
        )
```

**This directly optimizes your evaluation metric**, which is proven to outperform cross-entropy + post-hoc evaluation by 3-7%.[4][3]

## 2. ImageNet Weights for 8 Channels: Major Mistake

You caught a **huge error**. ImageNet weights are trained on 3-channel RGB data. For your **8-channel input** (5 bands + 3 engineered features), using ImageNet initialization naively is **suboptimal**.[5][6][7]

### Proper Multi-Spectral Initialization Strategy

**Option A: SatlasPretrain (BEST - Proven +18% over ImageNet)**[5]

SatlasPretrain is specifically designed for multispectral remote sensing and achieved **+18% accuracy over ImageNet** and **+6% over other baselines**:[5]

```python
# Load Satlas-pretrained Swin Transformer
# Unfortunately, SatlasPretrain uses Swin, not EfficientNet
# But for multispectral, this is THE state-of-the-art

# Alternative: Use ResNet50 with SeCo pretraining
from torchvision.models import resnet50

# SeCo pretrained weights (trained on Sentinel-2 multispectral)
# Download from: https://github.com/ServiceNow/seasonal-contrast
encoder = resnet50()
encoder.load_state_dict(torch.load('seco_resnet50_1m.ckpt'))

# Modify first conv for 8 channels
old_conv = encoder.conv1
encoder.conv1 = nn.Conv2d(
    8, 64, kernel_size=7, stride=2, padding=3, bias=False
)

# CRITICAL: Proper weight initialization for extra channels
with torch.no_grad():
    # Copy RGB weights to first 3 channels
    encoder.conv1.weight[:, :3, :, :] = old_conv.weight
    
    # Initialize extra 5 channels with ZERO (proven best)
    encoder.conv1.weight[:, 3:, :, :] = 0.0
    
    # Alternative: Mean of RGB (slightly worse in practice)
    # encoder.conv1.weight[:, 3:, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
```

**Why zero initialization?**[6]
- Allows model to **learn multispectral features from scratch**
- RGB channels provide strong spatial prior
- Additional channels gradually learn domain-specific patterns
- Proven +2-4% better than mean initialization[6]

**Option B: BigEarthNet or MillionAID Pretraining**

If SeCo/Satlas unavailable:
```python
# BigEarthNet has 12-band Sentinel-2 data
# You can download pretrained weights and adapt to 8 channels
```

### Recommended Architecture Change

Given the importance of multispectral pretraining, I now recommend:

**Primary: Swin Transformer with SatlasPretrain weights**[5]
- Proven **+18% over ImageNet** on remote sensing tasks
- Native multispectral support
- Better than DeepLabV3+ for satellite imagery

**Secondary: ResNet50 with SeCo weights**[7]
- Free, publicly available
- Trained on Sentinel-2 (same sensor as likely used here)
- Easier to adapt than Swin

```python
import segmentation_models_pytorch as smp

# Use ResNet50 encoder with SeCo pretraining
model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights=None,  # We'll load custom weights
    in_channels=8,
    classes=4,
    activation=None,
)

# Load SeCo pretrained encoder
seco_weights = torch.load('seco_resnet50_1m.ckpt')
model.encoder.load_state_dict(seco_weights, strict=False)
```

**Expected gain: +0.05-0.08 MCC** from proper pretraining.[7][5]

## 3. Cropping Strategy: Critical for Memory and Performance

I completely missed **patch extraction strategy**—this is huge for your Kaggle T4 constraints.

### Smart Cropping for Imbalanced Classes

Your challenge: **Lake class is 0.05%**. Random 512×512 crops will **rarely contain lake pixels**.

**Solution: Stratified Patch Sampling**[8]

```python
class StratifiedPatchSampler:
    """
    Ensures each batch contains patches from all classes, 
    especially rare ones (lake).
    """
    def __init__(self, image_paths, mask_paths, patch_size=512, 
                 patches_per_image=10, lake_oversample=20):
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.lake_oversample = lake_oversample
        
        # Pre-compute which images contain each class
        self.class_to_images = self._build_class_index(mask_paths)
    
    def _build_class_index(self, mask_paths):
        """Build index of which images contain which classes"""
        class_index = {0: [], 1: [], 2: [], 3: []}  # glacier, debris, lake, bg
        
        for idx, mask_path in enumerate(mask_paths):
            mask = load_mask(mask_path)
            unique_classes = np.unique(mask)
            for cls in unique_classes:
                class_index[cls].append(idx)
        
        return class_index
    
    def sample_batch(self, batch_size=8):
        """
        Sample batch ensuring class balance:
        - 1-2 patches with lake pixels (if available)
        - 2-3 patches with debris 
        - 2-3 patches with glacier
        - 1-2 patches with background
        """
        patches = []
        
        # Lake patches (critical - oversample)
        if len(self.class_to_images[2]) > 0:  # Lake class
            lake_images = np.random.choice(
                self.class_to_images[2], size=2, replace=True
            )
            for img_idx in lake_images:
                patch = self._extract_class_patch(img_idx, target_class=2)
                patches.append(patch)
        
        # Debris patches
        debris_images = np.random.choice(
            self.class_to_images[3], size=3, replace=True
        )
        for img_idx in debris_images:
            patch = self._extract_class_patch(img_idx, target_class=3)
            patches.append(patch)
        
        # Glacier patches
        glacier_images = np.random.choice(
            self.class_to_images[1], size=2, replace=True
        )
        for img_idx in glacier_images:
            patch = self._extract_class_patch(img_idx, target_class=1)
            patches.append(patch)
        
        # Background patch
        bg_image = np.random.choice(self.class_to_images[0], size=1)[0]
        patch = self._extract_random_patch(bg_image)
        patches.append(patch)
        
        return patches[:batch_size]
    
    def _extract_class_patch(self, image_idx, target_class):
        """Extract patch containing target class"""
        mask = load_mask(self.mask_paths[image_idx])
        
        # Find all locations with target class
        class_coords = np.argwhere(mask == target_class)
        
        if len(class_coords) == 0:
            return self._extract_random_patch(image_idx)
        
        # Random location within class region
        center = class_coords[np.random.randint(len(class_coords))]
        
        # Extract patch centered on this location
        patch = extract_patch_at_location(
            self.image_paths[image_idx], 
            center, 
            self.patch_size
        )
        return patch
```

**This ensures every batch has lake/debris examples**, critical for learning rare classes.[8]

### Overlap Strategy During Inference

For **test-time prediction**, use sliding window with overlap:

```python
def predict_with_overlap(model, image, patch_size=512, overlap=0.5):
    """
    Sliding window prediction with overlap and blending.
    Overlap reduces boundary artifacts.
    """
    stride = int(patch_size * (1 - overlap))
    h, w = image.shape[1:]
    
    # Initialize prediction and weight maps
    prediction = np.zeros((4, h, w), dtype=np.float32)  # 4 classes
    weight_map = np.zeros((h, w), dtype=np.float32)
    
    # Create Gaussian weight matrix (reduces edge artifacts)
    gaussian_weight = create_gaussian_weight(patch_size)
    
    # Sliding window
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[:, y:y+patch_size, x:x+patch_size]
            
            # Predict
            with torch.no_grad():
                pred = model(patch.unsqueeze(0))
                pred = torch.softmax(pred, dim=1).squeeze(0).cpu().numpy()
            
            # Accumulate with Gaussian weighting
            prediction[:, y:y+patch_size, x:x+patch_size] += pred * gaussian_weight
            weight_map[y:y+patch_size, x:x+patch_size] += gaussian_weight
    
    # Normalize by weight
    prediction /= (weight_map + 1e-7)
    
    return prediction.argmax(axis=0)

def create_gaussian_weight(size):
    """Create 2D Gaussian weight map for blending"""
    from scipy.ndimage import gaussian_filter
    weight = np.ones((size, size))
    weight = gaussian_filter(weight, sigma=size/8)
    weight = weight / weight.max()
    return weight
```

**Expected gain: +0.02-0.03 MCC** from better boundary handling.[9]

## 4. Critical Things I Left Out

### A. Spatial Cross-Validation

Your instructions mentioned **cross-region validation**. With high spatial autocorrelation (Moran's I = 210k-230k), **random CV will overestimate performance**.[10][11]

```python
from sklearn.model_selection import GroupKFold

# Group images by spatial region/tile
# Images from same region should be in same fold
groups = [get_region_id(img_path) for img_path in image_paths]

gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    # Train on train_idx, validate on val_idx
    # This ensures no spatial leakage
    pass
```

**Critical**: Your 0.65 MCC might drop to 0.60 MCC with proper spatial CV. Better to know now.

### B. Class-Aware Metrics During Training

Monitor **per-class MCC** during training, not just overall MCC:

```python
from sklearn.metrics import confusion_matrix

def compute_per_class_metrics(y_true, y_pred, num_classes=4):
    """Monitor each class separately"""
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    per_class_mcc = []
    for i in range(num_classes):
        # Binary MCC for class i vs rest
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-7)
        per_class_mcc.append(mcc)
    
    return {
        'glacier_mcc': per_class_mcc[1],
        'debris_mcc': per_class_mcc[3],
        'lake_mcc': per_class_mcc[2],
        'background_mcc': per_class_mcc[0],
        'overall_mcc': matthews_corrcoef(y_true, y_pred)
    }
```

**Watch lake_mcc closely**—if it's negative, your model is worse than random for lakes.

### C. Boundary Refinement with Graph Cuts

Your boundary pixels are 3-7%. Standard CRF might not be enough. Consider **graph-cut post-processing**:[9]

```python
import maxflow

def graphcut_refinement(prediction, image, lambda_smooth=1.0):
    """
    Graph-cut for label smoothing.
    More aggressive than CRF for boundary refinement.
    """
    h, w = prediction.shape
    num_classes = 4
    
    # Create graph
    g = maxflow.Graph[float]()
    nodes = g.add_nodes(h * w)
    
    # Data term (unary potential from model)
    # Shape: (h*w, num_classes)
    unary = get_class_probabilities(prediction)
    
    # Smoothness term (pairwise potential from image)
    for y in range(h):
        for x in range(w):
            node_id = y * w + x
            
            # Add edges to neighbors
            for dy, dx in [(0, 1), (1, 0)]:
                ny, nx = y + dy, x + dx
                if ny < h and nx < w:
                    neighbor_id = ny * w + nx
                    
                    # Edge weight based on image similarity
                    weight = compute_edge_weight(image, y, x, ny, nx)
                    g.add_edge(node_id, neighbor_id, 
                              weight * lambda_smooth, 
                              weight * lambda_smooth)
    
    # Run min-cut/max-flow
    g.maxflow()
    
    # Extract refined segmentation
    refined = np.zeros((h, w), dtype=np.int32)
    for y in range(h):
        for x in range(w):
            node_id = y * w + x
            refined[y, x] = g.get_segment(nodes[node_id])
    
    return refined
```

### D. Multi-Scale Testing

Your glacier cluster sizes vary 75-254 pixels. Test at multiple scales:[11]

```python
def multiscale_inference(model, image, scales=[0.75, 1.0, 1.25]):
    """Multi-scale testing for scale invariance"""
    h, w = image.shape[1:]
    predictions = []
    
    for scale in scales:
        # Resize
        scaled_h, scaled_w = int(h * scale), int(w * scale)
        scaled_image = F.interpolate(
            image, size=(scaled_h, scaled_w), mode='bilinear'
        )
        
        # Predict
        pred = predict_with_overlap(model, scaled_image)
        
        # Resize back
        pred = F.interpolate(
            pred, size=(h, w), mode='nearest'
        )
        predictions.append(pred)
    
    # Average predictions
    return torch.stack(predictions).mean(dim=0)
```

**Expected gain: +0.01-0.02 MCC**

## Revised Timeline with Corrections

**Week 1** (Days 1-7):
- Day 1-2: Implement **stratified patch sampling** + spatial CV
- Day 3-4: Setup **ResNet50 + SeCo** weights (not ImageNet!) with 8-channel initialization
- Day 5-6: Implement **MulticlassMCCLoss** + combined loss
- Day 7: Baseline training → **Target: 0.68-0.70 MCC** (might be lower with spatial CV)

**Week 2** (Days 8-14):
- Day 8-10: Fine-tune on **per-class monitoring**, adjust class weights based on per-class MCC
- Day 11-12: Implement **boundary-aware loss** with self-learning boundaries
- Day 13-14: Add **test-time cropping with overlap** → **Target: 0.72-0.74 MCC**

**Week 3** (Days 15-21):
- Day 15-17: **CRF + Graph-cut** post-processing pipeline
- Day 18-20: **Multi-scale inference** + TTA (8 augmentations)
- Day 21: Hyperparameter tuning → **Target: 0.75-0.77 MCC**

**Week 4** (Days 22-28):
- Day 22-24: Train **Swin Transformer with SatlasPretrain** (if accessible)
- Day 25-26: **Ensemble** (ResNet50-SeCo + Swin-Satlas)
- Day 27-28: Final optimization and submission → **Target: 0.78-0.82 MCC**

## Honest Questions Back to You

1. **Do you have access to SeCo or SatlasPretrain weights?** If not, we need to train from scratch or use BigEarthNet.

2. **What's your actual image size?** If full images are >2048×2048, memory constraints are tighter than I assumed.

3. **Is the test set from the same geographic region as training?** This changes whether domain adaptation is needed.

4. **Can you share your current 0.65 MCC validation setup?** Is it random split or spatial CV? This might explain the gap.

The sklearn multiclass MCC, proper multispectral pretraining, stratified cropping, and spatial CV are **non-negotiable** for competitive performance. These alone could give you +0.08-0.12 MCC over a naive approach.[3][4][7][5]

[1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)
[2](https://dwbi1.wordpress.com/2022/10/05/mcc-formula-for-multiclass-classification/)
[3](https://www.semanticscholar.org/paper/5c8f4138cf127d9f55d8ddae7f479927834a992a)
[4](https://www.semanticscholar.org/paper/c2def30ca143b47b81057f88ed4eb1473eb5e868)
[5](https://openaccess.thecvf.com/content/ICCV2023/papers/Bastani_SatlasPretrain_A_Large-Scale_Dataset_for_Remote_Sensing_Image_Understanding_ICCV_2023_paper.pdf)
[6](https://arxiv.org/html/2503.15969v1)
[7](https://arxiv.org/pdf/2204.02825.pdf)
[8](https://ijesty.org/index.php/ijesty/article/view/894)
[9](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/14235120/cd605c19-6794-43b8-be09-1fb60d3d6fbf/boundary_analysis.jpg)
[10](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/14235120/db28c8bd-a559-431e-a4c9-28f0937300cf/instructions.txt)
[11](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/14235120/36cf8f06-7652-4da6-98b2-49bc71570ba9/spatial_autocorrelation_analysis.jpg)
[12](https://ieeexplore.ieee.org/document/10584848/)
[13](https://grdspublishing.org/index.php/life/article/view/2496)
[14](https://ieeexplore.ieee.org/document/10983060/)
[15](https://www.semanticscholar.org/paper/42b73c905d3487dda36a703d6c60351ef20da101)
[16](https://link.springer.com/10.1007/s11042-024-18914-5)
[17](https://dx.plos.org/10.1371/journal.pone.0318219)
[18](https://journalijsra.com/node/1055)
[19](https://arxiv.org/pdf/2503.06450.pdf)
[20](https://arxiv.org/pdf/2007.11609.pdf)
[21](http://arxiv.org/abs/1008.2908)
[22](https://arxiv.org/pdf/1502.02710.pdf)
[23](https://arxiv.org/html/2311.16556)
[24](https://downloads.hindawi.com/journals/js/2022/7737166.pdf)
[25](https://zenodo.org/record/8178996/files/MLQ(TKDD-Final).pdf)
[26](http://arxiv.org/pdf/1407.1538.pdf)
[27](https://scikit-learn.org/stable/modules/model_evaluation.html)
[28](https://permetrics.readthedocs.io/en/latest/pages/classification/MCC.html)
[29](https://blester125.com/blog/rk.html)
[30](https://openaccess.thecvf.com/content/CVPR2024/papers/Noman_Rethinking_Transformers_Pre-training_for_Multi-Spectral_Satellite_Imagery_CVPR_2024_paper.pdf)
[31](https://www.kaggle.com/code/metric/matthews-corrcoef)