# 🎯 Dual-Platform Training Strategy

## Your Resources

### Google Colab
- **GPU:** 15GB T4 (single)
- **RAM:** 12.7 GB system
- **Disk:** 112.6 GB
- **Time:** 5 hours limit
- **Cost:** FREE

### Kaggle
- **GPU:** 2× 15GB T4 (dual)
- **RAM:** 30 GB system
- **Disk:** 50 GB
- **Time:** Ongoing
- **Cost:** FREE

---

## 📋 Complete Workflow

### Phase 1: Google Colab (HKH Pretraining)
**Duration:** 2.5 hours  
**Notebook:** `notebooks/colab_hkh_pretrain.ipynb`

#### Steps:
1. **Upload notebook to Colab** (0:00-0:05)
   - Go to https://colab.research.google.com
   - File → Upload notebook
   - Select `colab_hkh_pretrain.ipynb`

2. **Download HKH dataset** (0:05-0:25)
   - 29.4 GB from Azure blob storage
   - Extract to 14,190 numpy patches
   - Each patch: 512×512×15 channels

3. **Train model** (0:25-2:25)
   - Select 5 bands from 15: [B1_Blue, B2_Green, B3_Red, B5_SWIR1, B6_high_TIR]
   - Train Boundary-Aware U-Net (ResNet34)
   - Batch size: 32 (single T4)
   - 50 epochs × ~3 min/epoch = 150 min
   - **Expected MCC: 0.65-0.75**

4. **Export weights** (2:25-2:30)
   - Download `hkh_pretrained_weights.tar.gz` (~44 MB)
   - Or save to Google Drive

### Phase 2: Kaggle (Competition Fine-Tuning)
**Duration:** 2 hours  
**Notebook:** `notebooks/kaggle_competition_train.ipynb` (to create)

#### Steps:
1. **Setup** (0:00-0:05)
   - Create Kaggle notebook with dual T4 GPUs
   - Upload HKH weights
   - Clone gchack2_v2 repo

2. **Fine-tune** (0:05-2:00)
   - Load HKH pretrained weights
   - Train on 25 competition images
   - 5-fold CV, pixel-balanced sampling
   - Batch size: 32 (dual T4)
   - 100 epochs × ~1 min/epoch × 5 folds = 120 min
   - **Expected MCC: 0.85-0.92**

3. **Ensemble** (optional, if time permits)
   - Train 3 seeds
   - Implement TTA
   - **Expected MCC: 0.88-0.94**

---

## 🔑 Key Configuration Changes

### HKH Dataset (Colab)

**Your friend's band mapping (CORRECT!):**
```
HKH Channel  →  Competition Band  →  Wavelength
-------------------------------------------------
LE7 B1       →  Band1 (B2)        →  Blue
LE7 B2       →  Band2 (B3)        →  Green  
LE7 B3       →  Band3 (B4)        →  Red
LE7 B5       →  Band4 (B6)        →  SWIR1
LE7 B6_VCID_1 → Band5 (B10)       →  TIR (low-gain)
```

**Band selection in code:**
```python
# HKH has 15 channels at indices 0-14:
# [B1, B2, B3, B4_NIR, B5_SWIR1, B6_low_TIR, B6_high_TIR, B7_SWIR2, 
#  B8_pan, BQA, NDVI, NDSI, NDWI, elev, slope]

# Select 5 matching competition:
selected_channels = [0, 1, 2, 4, 6]  
# Maps to: B1_Blue, B2_Green, B3_Red, B5_SWIR1, B6_high_TIR
```

### Model Configuration

**Colab (Single GPU):**
```yaml
training:
  batch_size: 32  # Single T4 has 15GB
  use_amp: true
  
device:
  use_parallel: false
  gpu_ids: [0]
```

**Kaggle (Dual GPU):**
```yaml
training:
  batch_size: 48  # 24 per GPU × 2 = 48 total
  use_amp: true
  
device:
  use_parallel: true
  gpu_ids: [0, 1]
```

---

## 📊 Expected Performance

| Stage | Platform | Dataset | MCC | Time |
|-------|----------|---------|-----|------|
| HKH Pretrain | Colab | 14,190 patches | 0.65-0.75 | 2.5h |
| Competition | Kaggle | 25 images | 0.82-0.87 | 1.5h |
| Ensemble | Kaggle | 25 images | 0.85-0.92 | 2.5h |

**Final submission:** MCC 0.85-0.92 = **Top 3 position** 🏆

---

## ⚠️ Troubleshooting

### Colab Issues

**"Runtime disconnected"**
- Save checkpoint every 10 epochs
- Use Google Drive for persistent storage

**"Out of memory"**
- Reduce batch_size from 32 to 24
- Disable gradient accumulation

**"Download too slow"**
- Try GCP mirror instead of Azure
- Or use direct upload from local if you have fast internet

### Kaggle Issues

**"Module not found"**
- Run: `!pip install -q segmentation-models-pytorch albumentations`

**"Weights don't load"**
- Check checkpoint keys match model architecture
- Use `strict=False` for partial loading

**"Low MCC on competition"**
- Ensure pixel-balanced sampling is enabled
- Check that 5 bands are correctly aligned
- Verify HKH weights loaded successfully

---

## 🚀 Action Plan

1. **NOW:** Open Colab, upload `colab_hkh_pretrain.ipynb`
2. **Start training** (will take 2.5 hours)
3. **Meanwhile:** Prepare Kaggle notebook for Phase 2
4. **After Colab finishes:** Upload weights to Kaggle
5. **Run fine-tuning** on Kaggle (2 hours)
6. **Submit predictions**

**Total time:** ~5 hours (fits in Colab's limit!)

---

## 📁 Files You Need

**Colab:**
- `notebooks/colab_hkh_pretrain.ipynb` ✅ (created)
- Your GitHub repo (will clone)

**Kaggle:**
- `hkh_pretrained_weights.tar.gz` (from Colab)
- Your GitHub repo (will clone)
- Competition data (already on Kaggle)

---

**Ready to start?** Upload the Colab notebook and let's go! 🚀
