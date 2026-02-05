#!/usr/bin/env python3
"""
Test HKH Pretrained Model on Competition Data (Baseline Evaluation)

Purpose: Evaluate how well the HKH pretrained model (3-class) performs
         on competition data BEFORE fine-tuning.

Expected: Poor performance because:
1. Missing Lake class (4th class)
2. Different data distribution (HKH vs Competition)
3. Different spectral characteristics

This establishes the baseline for measuring fine-tuning improvement.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from skimage import io
from tqdm import tqdm
import segmentation_models_pytorch as smp
from sklearn.metrics import matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt

# =====================================================================
# Configuration
# =====================================================================

class Config:
    # Paths (adjust for your local setup)
    COMP_DATA_ROOT = Path('/home/observer/projects/gchack2_v2/Train')
    HKH_WEIGHTS_PATH = Path('/home/observer/projects/gchack2_v2/weights/hkh_pretrained_best.pth')  # Adjust path!
    
    # Model
    ENCODER = 'resnet34'
    IN_CHANNELS = 5
    NUM_CLASSES = 3  # HKH model has 3 classes (no lake!)
    
    # Hardware
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # HKH Normalization (from data analysis)
    HKH_MEAN = [-0.1758, -0.1622, -0.1597, 0.0151, 0.1778]
    HKH_STD = [0.8870, 0.8610, 0.9044, 0.7954, 0.7620]

# =====================================================================
# Helper Functions
# =====================================================================

def load_competition_image(img_id, data_root):
    """Load competition image (5 bands from separate folders)"""
    band_dirs = [
        data_root / 'Band1',  # B2 Blue
        data_root / 'Band2',  # B3 Green
        data_root / 'Band3',  # B4 Red
        data_root / 'Band4',  # B6 SWIR
        data_root / 'Band5',  # B10 Thermal
    ]
    
    bands = []
    for band_dir in band_dirs:
        band_path = band_dir / f"{img_id}.tif"
        band = io.imread(str(band_path))
        bands.append(band)
    
    image = np.stack(bands, axis=-1).astype(np.float32)
    
    # Normalize (DN → reflectance-like)
    image = image / 10000.0
    image = np.clip(image, 0, 1.5)
    
    return image

def load_competition_mask(img_id, data_root):
    """Load competition mask (3 channels: glacier, debris, lake)"""
    mask_path = data_root / 'labels' / f"{img_id}.png"
    mask = io.imread(str(mask_path))  # (H, W, 3)
    
    # Convert to 4-class labels (priority: Lake > Debris > Glacier > Background)
    mask_classes = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
    mask_classes[mask[:, :, 0] > 0.5] = 1  # Glacier
    mask_classes[mask[:, :, 1] > 0.5] = 2  # Debris
    mask_classes[mask[:, :, 2] > 0.5] = 3  # Lake
    
    return mask_classes

def normalize_image(image, mean, std):
    """Apply HKH normalization"""
    mean = np.array(mean).reshape(1, 1, -1)
    std = np.array(std).reshape(1, 1, -1)
    return (image - mean) / std

def predict_image(model, image, device):
    """Run inference on a single image"""
    # Normalize
    image_norm = normalize_image(image, Config.HKH_MEAN, Config.HKH_STD)
    
    # To tensor: (H, W, C) → (1, C, H, W)
    image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0).float()
    image_tensor = image_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(image_tensor)  # (1, 3, H, W) - only 3 classes!
        preds = torch.argmax(logits, dim=1).squeeze(0)  # (H, W)
    
    return preds.cpu().numpy()

def map_3class_to_4class(pred_3class, mask_4class):
    """
    Map HKH 3-class predictions to competition 4-class for evaluation
    
    HKH classes: 0=Background, 1=Glacier, 2=Debris
    Competition: 0=Background, 1=Glacier, 2=Debris, 3=Lake
    
    Strategy: Map HKH predictions directly, treat all Lake pixels as misclassified
    """
    # Direct mapping (Lake pixels will be predicted as 0, 1, or 2 - all wrong!)
    return pred_3class  # No change needed, just different interpretation

def calculate_metrics(preds, targets, num_classes=4):
    """Calculate per-class and overall MCC"""
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    
    # Overall MCC
    overall_mcc = matthews_corrcoef(targets_flat, preds_flat)
    
    # Per-class MCC
    per_class_mcc = []
    for c in range(num_classes):
        pred_c = (preds_flat == c).astype(int)
        target_c = (targets_flat == c).astype(int)
        try:
            mcc = matthews_corrcoef(target_c, pred_c)
        except:
            mcc = 0.0
        per_class_mcc.append(mcc)
    
    # Confusion matrix
    cm = confusion_matrix(targets_flat, preds_flat, labels=list(range(num_classes)))
    
    return overall_mcc, per_class_mcc, cm

# =====================================================================
# Main Evaluation
# =====================================================================

def main():
    print("="*80)
    print("HKH PRETRAINED MODEL - BASELINE EVALUATION ON COMPETITION DATA")
    print("="*80)
    print(f"\nDevice: {Config.DEVICE}")
    print(f"HKH Weights: {Config.HKH_WEIGHTS_PATH}")
    print(f"Competition Data: {Config.COMP_DATA_ROOT}\n")
    
    # ---------------------------------------------------------------
    # 1. Load HKH Pretrained Model
    # ---------------------------------------------------------------
    print("Loading HKH pretrained model (3 classes)...")
    
    model = smp.Unet(
        encoder_name=Config.ENCODER,
        encoder_weights=None,
        in_channels=Config.IN_CHANNELS,
        classes=Config.NUM_CLASSES,  # 3 classes (no lake!)
        activation=None
    )
    
    # Load weights
    checkpoint = torch.load(Config.HKH_WEIGHTS_PATH, map_location='cpu')
    
    # Handle both full checkpoint and state_dict only
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded from checkpoint (Epoch {checkpoint.get('epoch', 'N/A')}, MCC: {checkpoint.get('best_mcc', 'N/A'):.4f})")
    else:
        model.load_state_dict(checkpoint)
        print("✓ Loaded state dict")
    
    model = model.to(Config.DEVICE)
    model.eval()
    
    # ---------------------------------------------------------------
    # 2. Get Competition Image IDs
    # ---------------------------------------------------------------
    train_img_dir = Config.COMP_DATA_ROOT / 'Band1'
    all_image_ids = sorted([f.stem for f in train_img_dir.glob('*.tif')])
    
    print(f"\nFound {len(all_image_ids)} competition images")
    print(f"Sample IDs: {all_image_ids[:3]}\n")
    
    # ---------------------------------------------------------------
    # 3. Run Inference on ALL Competition Images
    # ---------------------------------------------------------------
    print("Running inference on competition dataset...")
    print("(This will be SLOW - HKH model doesn't know Lake class!)\n")
    
    all_preds = []
    all_targets = []
    
    for img_id in tqdm(all_image_ids, desc='Evaluating'):
        # Load image and mask
        image = load_competition_image(img_id, Config.COMP_DATA_ROOT)
        mask = load_competition_mask(img_id, Config.COMP_DATA_ROOT)
        
        # Predict
        pred = predict_image(model, image, Config.DEVICE)
        
        # Store
        all_preds.append(pred)
        all_targets.append(mask)
    
    # Concatenate all predictions
    all_preds = np.concatenate([p.flatten() for p in all_preds])
    all_targets = np.concatenate([m.flatten() for m in all_targets])
    
    # ---------------------------------------------------------------
    # 4. Calculate Metrics
    # ---------------------------------------------------------------
    print("\n" + "="*80)
    print("BASELINE RESULTS (HKH 3-Class Model on Competition 4-Class Data)")
    print("="*80)
    
    overall_mcc, per_class_mcc, cm = calculate_metrics(all_preds, all_targets, num_classes=4)
    
    print(f"\n📊 Overall MCC: {overall_mcc:.4f}")
    print(f"\n📈 Per-Class MCC:")
    print(f"  Background (0): {per_class_mcc[0]:.4f}")
    print(f"  Glacier (1):    {per_class_mcc[1]:.4f}")
    print(f"  Debris (2):     {per_class_mcc[2]:.4f}")
    print(f"  Lake (3):       {per_class_mcc[3]:.4f} ⚠️ (model can't predict this!)")
    
    print(f"\n🔢 Confusion Matrix:")
    print("   Rows = True, Cols = Predicted")
    print("   [BG, Glacier, Debris, Lake]\n")
    print(cm)
    
    # ---------------------------------------------------------------
    # 5. Analysis
    # ---------------------------------------------------------------
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    # Lake pixel statistics
    lake_pixels = (all_targets == 3).sum()
    total_pixels = len(all_targets)
    lake_pct = lake_pixels / total_pixels * 100
    
    print(f"\n🌊 Lake Statistics:")
    print(f"  Lake pixels: {lake_pixels:,} ({lake_pct:.3f}% of dataset)")
    print(f"  Lake MCC: {per_class_mcc[3]:.4f} (expected to be very poor!)")
    
    # What did model predict for lake pixels?
    lake_mask = all_targets == 3
    lake_predictions = all_preds[lake_mask]
    pred_as_bg = (lake_predictions == 0).sum()
    pred_as_glacier = (lake_predictions == 1).sum()
    pred_as_debris = (lake_predictions == 2).sum()
    
    print(f"\n  Lake pixels predicted as:")
    print(f"    Background: {pred_as_bg:,} ({pred_as_bg/lake_pixels*100:.1f}%)")
    print(f"    Glacier:    {pred_as_glacier:,} ({pred_as_glacier/lake_pixels*100:.1f}%)")
    print(f"    Debris:     {pred_as_debris:,} ({pred_as_debris/lake_pixels*100:.1f}%)")
    
    print(f"\n🎯 Expected Improvement After Fine-Tuning:")
    print(f"  Current MCC:  {overall_mcc:.4f} (3-class model on 4-class data)")
    print(f"  Target MCC:   ≥0.88 (after fine-tuning with lake class)")
    print(f"  Required Gain: +{0.88 - overall_mcc:.4f} MCC")
    
    if overall_mcc < 0.70:
        print(f"\n  Status: ⚠️ LOW baseline (expected - no lake class!)")
        print(f"          Fine-tuning will add lake class + adapt to competition")
    elif overall_mcc < 0.80:
        print(f"\n  Status: ✓ Decent baseline (HKH features transferring)")
        print(f"          Fine-tuning should push to 0.85-0.92")
    else:
        print(f"\n  Status: 🎉 Strong baseline (rare - HKH already similar!)")
        print(f"          Fine-tuning should easily reach 0.88+")
    
    print("\n" + "="*80)
    print("NEXT STEP: Fine-tune with competition data to add lake class!")
    print("="*80)

if __name__ == '__main__':
    main()
