#!/usr/bin/env python3
"""
Validation script for baseline_mcc_training.ipynb setup.
Tests all critical components before running full training.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

print("=" * 80)
print("BASELINE MCC TRAINING - PRE-FLIGHT VALIDATION")
print("=" * 80)

# Test 1: PyTorch and CUDA
print("\n[1/8] Testing PyTorch and CUDA...")
try:
    import torch
    import torch.nn as nn
    print(f"  ✓ PyTorch version: {torch.__version__}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✓ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("  ⚠ WARNING: No CUDA available, training will be VERY slow!")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)

# Test 2: Required libraries
print("\n[2/8] Testing required libraries...")
required_libs = {
    'segmentation_models_pytorch': 'smp',
    'albumentations': 'A',
    'sklearn': 'sklearn',
    'tqdm': 'tqdm'
}
for lib_name, import_name in required_libs.items():
    try:
        __import__(import_name if lib_name != import_name else lib_name)
        print(f"  ✓ {lib_name}")
    except ImportError:
        print(f"  ✗ MISSING: {lib_name}")
        print(f"    Install: pip install {lib_name}")
        sys.exit(1)

# Test 3: Data directory structure
print("\n[3/8] Testing data directory structure...")
data_root = Path('/home/observer/projects/gchack2_v2/Train')
if not data_root.exists():
    print(f"  ✗ ERROR: Data root not found: {data_root}")
    sys.exit(1)

required_dirs = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'labels']
for dir_name in required_dirs:
    dir_path = data_root / dir_name
    if not dir_path.exists():
        print(f"  ✗ MISSING: {dir_path}")
        sys.exit(1)
    file_count = len(list(dir_path.glob('*.tif')))
    print(f"  ✓ {dir_name}/: {file_count} files")

# Test 4: Image count consistency
print("\n[4/8] Testing image count consistency...")
image_counts = {}
for dir_name in required_dirs:
    count = len(list((data_root / dir_name).glob('*.tif')))
    image_counts[dir_name] = count

if len(set(image_counts.values())) != 1:
    print(f"  ✗ ERROR: Inconsistent image counts: {image_counts}")
    sys.exit(1)
else:
    print(f"  ✓ All directories have {image_counts['Band1']} images")

# Test 5: Label values
print("\n[5/8] Testing label values...")
label_file = list((data_root / 'labels').glob('*.tif'))[0]
mask = np.array(Image.open(label_file))
unique_vals = sorted(np.unique(mask))
expected_vals = [0, 85, 170, 255]
if list(unique_vals) == expected_vals:
    print(f"  ✓ Label values correct: {unique_vals}")
else:
    print(f"  ⚠ WARNING: Unexpected label values: {unique_vals}")
    print(f"    Expected: {expected_vals}")

# Test 6: 7-channel stacking test
print("\n[6/8] Testing 7-channel stacking...")
try:
    tile_id = list((data_root / 'Band1').glob('*.tif'))[0].stem
    band1 = np.array(Image.open(data_root / 'Band1' / f'{tile_id}.tif'))
    band2 = np.array(Image.open(data_root / 'Band2' / f'{tile_id}.tif'))
    band3 = np.array(Image.open(data_root / 'Band3' / f'{tile_id}.tif'))
    band4 = np.array(Image.open(data_root / 'Band4' / f'{tile_id}.tif'))
    band5 = np.array(Image.open(data_root / 'Band5' / f'{tile_id}.tif'))
    
    # Stack with indices (7 channels)
    eps = 1e-8
    green = band2.astype(np.float32)
    swir = band4.astype(np.float32)
    tir = band5.astype(np.float32)
    
    green_swir = np.clip(green / (swir + eps), 0.0, 10.0)
    swir_tir = np.clip(swir / (tir + eps), 0.0, 1e10)
    swir_tir_log = np.clip(np.log1p(swir_tir), 0.0, 25.0)
    
    stacked = np.stack([
        band1, band2, band3, band4, band5,
        green_swir, swir_tir_log
    ], axis=-1)
    
    print(f"  ✓ 7-channel stack successful: shape {stacked.shape}")
    print(f"    - RGB range: [{band1.min()}, {band1.max()}]")
    print(f"    - SWIR range: [{band4.min()}, {band4.max()}]")
    print(f"    - TIR range: [{band5.min()}, {band5.max()}]")
    print(f"    - Green/SWIR ratio range: [{green_swir.min():.2f}, {green_swir.max():.2f}]")
    print(f"    - log(SWIR/TIR) range: [{swir_tir_log.min():.2f}, {swir_tir_log.max():.2f}]")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)

# Test 7: Model initialization test
print("\n[7/8] Testing model initialization...")
try:
    import segmentation_models_pytorch as smp
    
    # Test 3-channel model with noisy-student
    model_3ch = smp.Unet(
        encoder_name='timm-efficientnet-b3',
        encoder_weights='noisy-student',
        in_channels=3,
        classes=4
    )
    print(f"  ✓ 3-channel model loaded with noisy-student weights")
    
    # Test expanding to 7 channels
    first_conv = model_3ch.encoder.conv_stem
    old_weight = first_conv.weight.data.clone()
    
    new_conv = nn.Conv2d(7, first_conv.out_channels,
                        kernel_size=first_conv.kernel_size,
                        stride=first_conv.stride,
                        padding=first_conv.padding,
                        bias=first_conv.bias is not None)
    
    with torch.no_grad():
        # Zero-init strategy
        new_conv.weight[:, :3, :, :] = old_weight
        new_conv.weight[:, 3:, :, :] = 0.0
        if first_conv.bias is not None:
            new_conv.bias.copy_(first_conv.bias)
    
    model_3ch.encoder.conv_stem = new_conv
    
    # Test forward pass
    dummy_input = torch.randn(1, 7, 256, 256)
    output = model_3ch(dummy_input)
    
    print(f"  ✓ 7-channel model expansion successful")
    print(f"  ✓ Forward pass successful: output shape {output.shape}")
    
    total_params = sum(p.numel() for p in model_3ch.parameters())
    print(f"  ✓ Total parameters: {total_params:,}")
    
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Loss function test
print("\n[8/8] Testing loss functions...")
try:
    import segmentation_models_pytorch as smp
    
    # Test inputs
    batch_size = 2
    logits = torch.randn(batch_size, 4, 64, 64)
    targets = torch.randint(0, 4, (batch_size, 64, 64))
    
    # Test Focal Loss
    focal = smp.losses.FocalLoss(mode='multiclass', gamma=2.0)
    focal_loss = focal(logits, targets)
    print(f"  ✓ Focal Loss: {focal_loss.item():.4f}")
    
    # Test Dice Loss
    dice = smp.losses.DiceLoss(mode='multiclass')
    dice_loss = dice(logits, targets)
    print(f"  ✓ Dice Loss: {dice_loss.item():.4f}")
    
    # Test Lovász Loss (boundary)
    lovasz = smp.losses.LovaszLoss(mode='multiclass')
    lovasz_loss = lovasz(logits, targets)
    print(f"  ✓ Lovász Loss: {lovasz_loss.item():.4f}")
    
    # Test MCC calculation
    from sklearn.metrics import matthews_corrcoef
    preds = logits.argmax(dim=1).cpu().numpy().flatten()
    targets_np = targets.cpu().numpy().flatten()
    mcc = matthews_corrcoef(targets_np, preds)
    print(f"  ✓ MCC calculation: {mcc:.4f}")
    
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final summary
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)
print("✅ All tests passed! Notebook is ready to run.")
print("\nNext steps:")
print("  1. Open notebooks/baseline_mcc_training.ipynb")
print("  2. Run all cells sequentially")
print("  3. Monitor training progress (target: 0.8+ MCC)")
print("\nExpected timeline:")
print("  - ~1 hour per epoch on dual T4 GPUs")
print("  - ~100-150 epochs for convergence")
print("  - Early stopping may finish at epoch 80-100")
print("\nGood luck! 🎯")
print("=" * 80)
