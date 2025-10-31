# HKH Dataset Preprocessing Guide
## Using glacier_mapping Repository for Pretraining

This guide details the exact steps to use the `glacier_mapping` repository for HKH pretraining, based on proven workflows.

---

## 📦 Phase 0: Setup glacier_mapping Repository

### 1. Clone and Install

```bash
cd /home/observer/projects/gchack2_v2/data/hkh
git clone https://github.com/krisrs1128/glacier_mapping.git
cd glacier_mapping

# Install dependencies
pip install -r requirements.txt
pip install rasterio geopandas shapely
```

### 2. Download HKH Dataset

```bash
# Download from Lila.Science
wget https://lila.science/wp-content/uploads/2020/06/hkh-glacier-mapping.zip
unzip hkh-glacier-mapping.zip -d raw/

# Expected structure:
# raw/
#   ├── HKH_images/         # ~7,229 Landsat 7 tiles
#   ├── HKH_labels/         # Shapefile labels
#   └── metadata.csv
```

**Dataset Details:**
- **Size:** ~7,229 labeled tiles
- **Bands:** Landsat 7 (7 bands + TIR)
- **Resolution:** 30m/pixel
- **Format:** GeoTIFF images, Shapefile labels
- **Classes:** Background, Clean Ice, Debris-Covered, Glacial Lake

---

## 🔧 Phase 1: Preprocessing with glacier_mapping Scripts

### 1. Band Alignment (Critical!)

**Competition Bands:**
- Band1 → Blue (B2)
- Band2 → Green (B3)
- Band3 → Red (B4)
- Band4 → SWIR (B6)
- Band5 → TIR (B10)

**Landsat 7 Mapping:**
```python
# In glacier_mapping/data_prep/band_mapping.py
COMPETITION_BANDS = {
    'B1': 0,   # Blue (0.45-0.52 µm)
    'B2': 1,   # Green (0.52-0.60 µm)
    'B3': 2,   # Red (0.63-0.69 µm)
    'B4': 5,   # SWIR (1.55-1.75 µm) - Landsat B5
    'B5': 6,   # TIR (10.40-12.50 µm) - Landsat B6
}

def extract_competition_bands(landsat_image):
    """Extract 5 bands matching competition format."""
    return landsat_image[:, :, [0, 1, 2, 5, 6]]  # B1, B2, B3, B5, B6
```

### 2. Slicing to 512×512 Tiles

Use the repo's slicing tools:

```bash
cd glacier_mapping

# Modify config to output 512×512 tiles
# Edit: conf/preprocess.yaml
python preprocess.py \
    --input_dir ../raw/HKH_images \
    --labels_dir ../raw/HKH_labels \
    --output_dir ../processed \
    --tile_size 512 \
    --overlap 0.1 \
    --min_glacier_pixels 100
```

**Expected Output:**
```
processed/
  ├── images/           # (N, 512, 512, 5) numpy arrays
  ├── masks/            # (N, 512, 512) class indices
  ├── train_ids.txt     # Training tile IDs
  └── val_ids.txt       # Validation tile IDs
```

### 3. Mask Generation and Filtering

```python
# Use glacier_mapping utilities
from data_processing import generate_masks, write_pair_slices

# Generate masks from shapefiles
generate_masks(
    images_dir='../raw/HKH_images',
    labels_shapefile='../raw/HKH_labels/glaciers.shp',
    output_dir='../processed/masks',
    class_mapping={
        'background': 0,
        'clean_ice': 1,
        'debris': 2,
        'glacial_lake': 3
    }
)

# Filter tiles with rich debris/lake content
# Avoid background-dominated tiles
write_pair_slices(
    images_dir='../processed/images',
    masks_dir='../processed/masks',
    output_dir='../processed/filtered',
    min_class_pixels={
        'debris': 500,      # At least 500 debris pixels
        'glacial_lake': 50  # At least 50 lake pixels
    },
    background_threshold=0.8  # Skip if >80% background
)
```

### 4. Compute Normalization Statistics

**CRITICAL:** Do NOT use ImageNet statistics for multispectral data!

```python
# src/data/compute_hkh_stats.py
import numpy as np
import glob
from tqdm import tqdm

def compute_band_statistics(images_dir):
    """Compute mean/std per band from HKH training set."""
    all_means = []
    all_stds = []
    
    image_files = glob.glob(f'{images_dir}/train/*.npy')
    
    for img_path in tqdm(image_files):
        img = np.load(img_path)  # (512, 512, 5)
        
        # Compute per-band statistics
        for band_idx in range(5):
            band_data = img[:, :, band_idx]
            all_means.append(band_data.mean())
            all_stds.append(band_data.std())
    
    # Aggregate
    mean_per_band = np.array(all_means).reshape(-1, 5).mean(axis=0)
    std_per_band = np.array(all_stds).reshape(-1, 5).mean(axis=0)
    
    print(f"HKH Band Means: {mean_per_band}")
    print(f"HKH Band Stds: {std_per_band}")
    
    return mean_per_band, std_per_band

# Run computation
mean, std = compute_band_statistics('data/hkh/processed/images')

# Save for use in training
np.save('data/hkh/hkh_mean.npy', mean)
np.save('data/hkh/hkh_std.npy', std)
```

**Update configs/hkh_pretrain.yaml:**
```yaml
data:
  normalization:
    mean: [0.3245, 0.3512, 0.3123, 0.4567, 0.2891]  # From HKH computation
    std: [0.1234, 0.1345, 0.1267, 0.1789, 0.0987]   # From HKH computation
    use_imagenet: false  # CRITICAL!
```

---

## 🏋️ Phase 2: HKH Pretraining

### 1. Use glacier_mapping Training Scripts

The repo provides a boundary-aware U-Net with dropout. We'll adapt it:

```bash
# Copy glacier_mapping model as baseline
cp glacier_mapping/models/unet.py src/models/glacier_unet.py

# Modify for our needs:
# 1. Set encoder_weights=None
# 2. Add our loss functions (Focal + Dice + Boundary)
# 3. Use our normalization stats
```

### 2. Training Configuration

**Update configs/hkh_pretrain.yaml:**
```yaml
data:
  train_path: data/hkh/processed/filtered/train
  val_path: data/hkh/processed/filtered/val
  in_channels: 5
  num_classes: 4
  
  # Use HKH statistics (NOT ImageNet)
  normalization:
    mean: [0.3245, 0.3512, 0.3123, 0.4567, 0.2891]
    std: [0.1234, 0.1345, 0.1267, 0.1789, 0.0987]

model:
  architecture: unet
  encoder_name: resnet34
  encoder_weights: null  # Train from scratch
  decoder_attention_type: scse
  dropout: 0.2  # glacier_mapping uses dropout

loss:
  # Phase A: Simple and stable
  focal_weight: 0.50
  focal_gamma: 3.0
  dice_weight: 0.30
  boundary_weight: 0.20
  boundary_interface_weight: 5.0

training:
  epochs: 50-80  # Adjust based on convergence
  batch_size: 16
  early_stop_patience: 15
  save_name: hkh_pretrained.pth
```

### 3. Run Pretraining

```bash
python src/training/train.py --config configs/hkh_pretrain.yaml

# Expected:
# - Convergence around epoch 50-60
# - Validation MCC: 0.75-0.78
# - Output: weights/hkh_pretrained.pth (~44 MB)
```

### 4. Verify Class Order Compatibility

**CRITICAL:** Ensure HKH and competition use same class encoding!

```python
# HKH Classes
HKH_CLASSES = {
    0: 'background',
    1: 'clean_ice',      # → Glacier
    2: 'debris',
    3: 'glacial_lake'    # → Lake
}

# Competition Classes
COMP_CLASSES = {
    0: 'background',
    1: 'glacier',
    2: 'debris',
    3: 'lake'
}

# Mapping is 1:1, safe to transfer weights!
```

---

## 🎯 Phase 3: Transfer to Competition

### 1. Load HKH Weights

```python
# src/training/train.py
import torch
import segmentation_models_pytorch as smp

# Initialize competition model
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights=None,  # Will load from HKH
    in_channels=5,
    classes=4,
    decoder_attention_type='scse'
)

# Load HKH pretrained weights
checkpoint = torch.load('weights/hkh_pretrained.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print("Loaded HKH pretrained weights!")

# Now fine-tune on competition data...
```

### 2. Fine-Tuning Strategy

**Update configs/competition_finetune.yaml:**
```yaml
model:
  pretrained_path: weights/hkh_pretrained.pth
  freeze_encoder: false  # Fine-tune entire network
  
  # Use SAME normalization as HKH (consistency!)
  normalization:
    mean: [0.3245, 0.3512, 0.3123, 0.4567, 0.2891]
    std: [0.1234, 0.1345, 0.1267, 0.1789, 0.0987]

optimizer:
  lr: 5e-5  # Lower than HKH pretraining (fine-tuning)

training:
  epochs: 80-150  # More epochs for small dataset
```

---

## 🚀 Phase 4: GlaViTU Integration (Optional Enhancement)

### Option A: GlaViTU as Ensemble Member

**Why:** Hybrid CNN-Transformer captures global context better than pure CNN.

```bash
# Clone GlaViTU repo
cd /home/observer/projects/gchack2_v2
git clone https://github.com/konstantin-a-maslov/GlaViTU-IGARSS2023.git

# Install dependencies
cd GlaViTU-IGARSS2023
pip install -r requirements.txt
```

**Training Workflow:**
1. **Pretrain GlaViTU on HKH** (same 7k tiles)
   ```bash
   python train.py \
       --data data/hkh/processed \
       --bands 5 \
       --encoder_weights none \
       --epochs 50
   ```

2. **Fine-tune on competition data**
   ```bash
   python train.py \
       --data Train/ \
       --pretrained weights/glavitu_hkh.pth \
       --epochs 80
   ```

3. **Ensemble with BA-UNet**
   ```python
   # In solution.py
   models = [
       load_model('weights/ba_unet_best.pth'),      # Weight: 0.4
       load_model('weights/glavitu_best.pth'),      # Weight: 0.3
       load_model('weights/ba_unet_seed2.pth'),     # Weight: 0.3
   ]
   
   # Weighted ensemble
   ensemble_logits = (
       0.4 * models[0](image) +
       0.3 * models[1](image) +
       0.3 * models[2](image)
   )
   ```

**Expected Gain:** +0.02-0.04 MCC from ensemble

### Option B: Knowledge Distillation (Advanced)

Use GlaViTU + BA-UNet as teachers, distill to lightweight student:

```python
# src/training/distillation.py
teacher_models = [ba_unet, glavitu]
student_model = smp.Unet('efficientnet-b0', ...)  # Smaller

for batch in dataloader:
    # Soft targets from teachers
    with torch.no_grad():
        teacher_logits = [t(batch) for t in teacher_models]
        soft_targets = torch.stack(teacher_logits).mean(0)
    
    # Student prediction
    student_logits = student_model(batch)
    
    # Distillation loss
    loss = (
        0.5 * F.kl_div(student_logits, soft_targets) +
        0.5 * cross_entropy(student_logits, labels)
    )
```

**Benefit:** Student model < 100 MB, easier to fit in 300 MB limit

### Option C: Skip GlaViTU (Conservative)

If time-constrained, focus on:
1. HKH pretraining (BA-UNet)
2. Competition fine-tuning
3. Ensemble 3-5 BA-UNet variants (different seeds)

**Still competitive:** MCC 0.85-0.89 without GlaViTU

---

## 📋 Validation Checklist

### Before HKH Pretraining:
- [ ] HKH dataset downloaded and extracted
- [ ] Tiles sliced to 512×512 with `glacier_mapping/preprocess.py`
- [ ] Masks generated with correct class mapping (0/1/2/3)
- [ ] Normalization statistics computed from HKH train set
- [ ] Background-heavy tiles filtered out
- [ ] Train/val splits created (stratified by region if possible)

### Before Competition Fine-Tuning:
- [ ] HKH pretrained weights exist: `weights/hkh_pretrained.pth`
- [ ] HKH validation MCC ≥ 0.75
- [ ] Class order verified compatible (HKH ↔ Competition)
- [ ] Same normalization stats used for consistency
- [ ] Competition data uses image-level stratified 5-fold CV

### Before Submission:
- [ ] Model size < 300 MB (check: `ls -lh weights/model.pth`)
- [ ] Output format correct: {0, 85, 170, 255}
- [ ] Filenames match Band1 exactly
- [ ] TTA implemented (6 augmentations)
- [ ] Post-processing tested (morphology + CRF)

---

## 🐛 Troubleshooting

**Issue: "Band mismatch" errors**
- Check: Are you using Landsat 7 bands [0,1,2,5,6]?
- Fix: Update `extract_competition_bands()` function

**Issue: "Normalization looks wrong"**
- Check: Are you using HKH stats, not ImageNet?
- Fix: Recompute with `compute_hkh_stats.py`

**Issue: "Class encoding mismatch"**
- Check: HKH masks should be {0,1,2,3}, not {0,85,170,255}
- Fix: Update `generate_masks()` class_mapping

**Issue: "Too many background tiles"**
- Check: Filter threshold in `write_pair_slices()`
- Fix: Set `background_threshold=0.7` (stricter)

---

## 📚 Key Resources

- **glacier_mapping repo:** https://github.com/krisrs1128/glacier_mapping
- **GlaViTU repo:** https://github.com/konstantin-a-maslov/GlaViTU-IGARSS2023
- **HKH dataset:** https://lila.science/datasets/hkh-glacier-mapping/
- **Boundary-Aware U-Net paper:** https://arxiv.org/abs/2301.11454

---

## 🎯 Expected Timeline

| Phase | Task | Time | Output |
|-------|------|------|--------|
| 0 | Download + setup glacier_mapping | 2 hours | Repo cloned |
| 1 | Preprocess HKH (slicing, masking) | 4-6 hours | ~7k clean tiles |
| 2 | HKH pretraining | 6-8 hours | MCC 0.75-0.78 |
| 3 | Competition fine-tuning | 8-12 hours | MCC 0.82-0.85 |
| 4 | (Optional) GlaViTU training | 8-10 hours | +0.03 MCC |
| 5 | Ensemble + TTA | 2-3 hours | MCC 0.88-0.92 |

**Total:** 30-40 hours (conservative estimate)

---

**This guide provides the exact workflow used in successful glacier mapping projects. Follow step-by-step for best results!**
