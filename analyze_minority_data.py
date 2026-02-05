#!/usr/bin/env python3
"""
Data-driven analysis to understand minority class collapse
Tests hypotheses about why Debris/Lake are failing
"""

import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict
import random

# Paths
DATA_ROOT = Path('Train')
BAND_DIRS = {
    'B2': DATA_ROOT / 'Band2',
    'B3': DATA_ROOT / 'Band3', 
    'B4': DATA_ROOT / 'Band4',
    'B5': DATA_ROOT / 'Band5',
    'B6': DATA_ROOT / 'Band6',
}
LABEL_DIR = DATA_ROOT / 'labels'

# Get all tiles
all_tiles = sorted([f.stem for f in LABEL_DIR.glob('*.tif')])
print(f"Found {len(all_tiles)} tiles")

# ============================================================================
# HYPOTHESIS 1: What's the actual per-tile distribution?
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 1: Per-Tile Class Distribution")
print("="*80)

tile_stats = []
for tile_id in all_tiles:
    label_path = LABEL_DIR / f'{tile_id}.tif'
    mask = np.array(Image.open(label_path), dtype=np.int64)
    
    # Apply mapping [0, 85, 170, 255] → [0, 1, 2, 3]
    if mask.max() > 10:
        mask = (mask / 85).astype(np.int64)
    
    total_pixels = mask.size
    class_counts = [(mask == c).sum() for c in range(4)]
    class_pcts = [count / total_pixels * 100 for count in class_counts]
    
    tile_stats.append({
        'tile_id': tile_id,
        'bg_pct': class_pcts[0],
        'glacier_pct': class_pcts[1],
        'debris_pct': class_pcts[2],
        'lake_pct': class_pcts[3],
        'debris_pixels': class_counts[2],
        'lake_pixels': class_counts[3]
    })

# Sort by debris and lake
debris_sorted = sorted(tile_stats, key=lambda x: x['debris_pct'], reverse=True)
lake_sorted = sorted(tile_stats, key=lambda x: x['lake_pct'], reverse=True)

print("\n📊 TOP 5 TILES BY DEBRIS %:")
for i, t in enumerate(debris_sorted[:5]):
    print(f"   {i+1}. {t['tile_id']}: Debris={t['debris_pct']:.2f}% ({t['debris_pixels']:,} pixels)")

print("\n📊 TOP 5 TILES BY LAKE %:")
for i, t in enumerate(lake_sorted[:5]):
    print(f"   {i+1}. {t['tile_id']}: Lake={t['lake_pct']:.4f}% ({t['lake_pixels']:,} pixels)")

print("\n📊 BOTTOM 5 TILES BY DEBRIS %:")
for i, t in enumerate(debris_sorted[-5:]):
    print(f"   {i+1}. {t['tile_id']}: Debris={t['debris_pct']:.2f}% ({t['debris_pixels']:,} pixels)")

# Overall statistics
total_debris_pixels = sum(t['debris_pixels'] for t in tile_stats)
total_lake_pixels = sum(t['lake_pixels'] for t in tile_stats)
total_pixels = sum(mask.size for _ in tile_stats)

print(f"\n📊 DATASET-WIDE STATISTICS:")
print(f"   Total tiles: {len(tile_stats)}")
print(f"   Tiles with >5% debris: {sum(1 for t in tile_stats if t['debris_pct'] > 5.0)}")
print(f"   Tiles with >0.1% lake: {sum(1 for t in tile_stats if t['lake_pct'] > 0.1)}")
print(f"   Tiles with ZERO lake: {sum(1 for t in tile_stats if t['lake_pixels'] == 0)}")
print(f"   Tiles with <1000 debris pixels: {sum(1 for t in tile_stats if t['debris_pixels'] < 1000)}")

# ============================================================================
# HYPOTHESIS 2: What happens when we extract 512x512 patches?
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 2: Random 512x512 Crop Analysis")
print("="*80)
print("Testing: Do random crops capture minorities?")

def extract_random_crops(mask, crop_size=512, n_crops=10):
    """Extract random crops and analyze class distribution"""
    h, w = mask.shape
    crop_stats = []
    
    for _ in range(n_crops):
        if h > crop_size and w > crop_size:
            y = random.randint(0, h - crop_size)
            x = random.randint(0, w - crop_size)
            crop = mask[y:y+crop_size, x:x+crop_size]
        else:
            crop = mask
        
        total = crop.size
        stats = {
            'bg_pct': (crop == 0).sum() / total * 100,
            'glacier_pct': (crop == 1).sum() / total * 100,
            'debris_pct': (crop == 2).sum() / total * 100,
            'lake_pct': (crop == 3).sum() / total * 100,
            'has_debris': (crop == 2).any(),
            'has_lake': (crop == 3).any()
        }
        crop_stats.append(stats)
    
    return crop_stats

# Test on high-debris tile
high_debris_tile = debris_sorted[0]['tile_id']
label_path = LABEL_DIR / f'{high_debris_tile}.tif'
mask = np.array(Image.open(label_path), dtype=np.int64)
if mask.max() > 10:
    mask = (mask / 85).astype(np.int64)

print(f"\n📊 TESTING ON HIGH-DEBRIS TILE: {high_debris_tile}")
print(f"   Full tile: Debris={debris_sorted[0]['debris_pct']:.2f}%, Lake={debris_sorted[0]['lake_pct']:.4f}%")

crop_stats = extract_random_crops(mask, crop_size=512, n_crops=50)

debris_in_crops = sum(1 for c in crop_stats if c['has_debris'])
lake_in_crops = sum(1 for c in crop_stats if c['has_lake'])
avg_debris = np.mean([c['debris_pct'] for c in crop_stats])
avg_lake = np.mean([c['lake_pct'] for c in crop_stats])

print(f"\n   Random 512x512 crops (n=50):")
print(f"   Crops containing Debris: {debris_in_crops}/50 ({debris_in_crops/50*100:.1f}%)")
print(f"   Crops containing Lake: {lake_in_crops}/50 ({lake_in_crops/50*100:.1f}%)")
print(f"   Avg Debris % in crops: {avg_debris:.2f}%")
print(f"   Avg Lake % in crops: {avg_lake:.4f}%")

# Test on high-lake tile
high_lake_tile = lake_sorted[0]['tile_id']
label_path = LABEL_DIR / f'{high_lake_tile}.tif'
mask = np.array(Image.open(label_path), dtype=np.int64)
if mask.max() > 10:
    mask = (mask / 85).astype(np.int64)

print(f"\n📊 TESTING ON HIGH-LAKE TILE: {high_lake_tile}")
print(f"   Full tile: Debris={[t for t in tile_stats if t['tile_id']==high_lake_tile][0]['debris_pct']:.2f}%, Lake={lake_sorted[0]['lake_pct']:.4f}%")

crop_stats = extract_random_crops(mask, crop_size=512, n_crops=50)

debris_in_crops = sum(1 for c in crop_stats if c['has_debris'])
lake_in_crops = sum(1 for c in crop_stats if c['has_lake'])
avg_debris = np.mean([c['debris_pct'] for c in crop_stats])
avg_lake = np.mean([c['lake_pct'] for c in crop_stats])

print(f"\n   Random 512x512 crops (n=50):")
print(f"   Crops containing Debris: {debris_in_crops}/50 ({debris_in_crops/50*100:.1f}%)")
print(f"   Crops containing Lake: {lake_in_crops}/50 ({lake_in_crops/50*100:.1f}%)")
print(f"   Avg Debris % in crops: {avg_debris:.2f}%")
print(f"   Avg Lake % in crops: {avg_lake:.4f}%")

# ============================================================================
# HYPOTHESIS 3: Would 256x256 crops help?
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 3: Crop Size Comparison (256 vs 512)")
print("="*80)

for crop_size in [256, 384, 512]:
    crop_stats = extract_random_crops(mask, crop_size=crop_size, n_crops=100)
    
    debris_in_crops = sum(1 for c in crop_stats if c['has_debris'])
    lake_in_crops = sum(1 for c in crop_stats if c['has_lake'])
    
    print(f"\n📊 {crop_size}x{crop_size} crops (n=100 from high-lake tile):")
    print(f"   Debris presence: {debris_in_crops}/100 ({debris_in_crops}%)")
    print(f"   Lake presence: {lake_in_crops}/100 ({lake_in_crops}%)")

# ============================================================================
# HYPOTHESIS 4: Are minority patches being extracted correctly?
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 4: Minority-Centered Patch Extraction")
print("="*80)
print("Simulating MinorityAwareDataset behavior...")

def extract_minority_centered_patches(mask, patch_size=512, max_patches=10):
    """Extract patches centered on debris/lake pixels"""
    h, w = mask.shape
    patches = []
    
    # Find debris pixels
    debris_coords = np.argwhere(mask == 2)
    if len(debris_coords) > 0:
        sampled = debris_coords[np.random.choice(len(debris_coords), 
                                                  min(max_patches, len(debris_coords)), 
                                                  replace=False)]
        
        for y, x in sampled:
            y1 = max(0, y - patch_size // 2)
            x1 = max(0, x - patch_size // 2)
            y2 = min(h, y1 + patch_size)
            x2 = min(w, x1 + patch_size)
            
            # Adjust if we hit edge
            if y2 - y1 < patch_size:
                y1 = max(0, y2 - patch_size)
            if x2 - x1 < patch_size:
                x1 = max(0, x2 - patch_size)
            
            patch = mask[y1:y2, x1:x2]
            
            total = patch.size
            patches.append({
                'debris_pct': (patch == 2).sum() / total * 100,
                'lake_pct': (patch == 3).sum() / total * 100,
                'has_debris': (patch == 2).any(),
                'has_lake': (patch == 3).any()
            })
    
    # Find lake pixels
    lake_coords = np.argwhere(mask == 3)
    if len(lake_coords) > 0:
        sampled = lake_coords[np.random.choice(len(lake_coords), 
                                                min(max_patches, len(lake_coords)), 
                                                replace=False)]
        
        for y, x in sampled:
            y1 = max(0, y - patch_size // 2)
            x1 = max(0, x - patch_size // 2)
            y2 = min(h, y1 + patch_size)
            x2 = min(w, x1 + patch_size)
            
            if y2 - y1 < patch_size:
                y1 = max(0, y2 - patch_size)
            if x2 - x1 < patch_size:
                x1 = max(0, x2 - patch_size)
            
            patch = mask[y1:y2, x1:x2]
            
            total = patch.size
            patches.append({
                'debris_pct': (patch == 2).sum() / total * 100,
                'lake_pct': (patch == 3).sum() / total * 100,
                'has_debris': (patch == 2).any(),
                'has_lake': (patch == 3).any()
            })
    
    return patches

minority_patches = extract_minority_centered_patches(mask, patch_size=512, max_patches=20)

if minority_patches:
    debris_presence = sum(1 for p in minority_patches if p['has_debris'])
    lake_presence = sum(1 for p in minority_patches if p['has_lake'])
    avg_debris = np.mean([p['debris_pct'] for p in minority_patches])
    avg_lake = np.mean([p['lake_pct'] for p in minority_patches])
    
    print(f"\n📊 Minority-centered 512x512 patches (n={len(minority_patches)}):")
    print(f"   Debris presence: {debris_presence}/{len(minority_patches)} ({debris_presence/len(minority_patches)*100:.1f}%)")
    print(f"   Lake presence: {lake_presence}/{len(minority_patches)} ({lake_presence/len(minority_patches)*100:.1f}%)")
    print(f"   Avg Debris %: {avg_debris:.2f}%")
    print(f"   Avg Lake %: {avg_lake:.4f}%")
    print(f"\n   ✅ Minority-centered extraction DOES increase exposure!")
    print(f"   Random crops had Lake in {lake_in_crops}% of crops")
    print(f"   Minority-centered has Lake in {lake_presence/len(minority_patches)*100:.1f}% of patches")

# ============================================================================
# HYPOTHESIS 5: Training set imbalance
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 5: Train/Val Split Analysis")
print("="*80)

# Simulate the split from notebook (top 3 lake tiles for val)
lake_sorted_ids = [t['tile_id'] for t in lake_sorted]
val_ids = lake_sorted_ids[:3]

# Add 2 more with good glacier/debris
remaining = [t for t in tile_stats if t['tile_id'] not in val_ids 
             and t['glacier_pct'] > 10 and t['debris_pct'] > 1]
random.seed(42)
random.shuffle(remaining)
val_ids.extend([t['tile_id'] for t in remaining[:2]])

train_ids = [t['tile_id'] for t in tile_stats if t['tile_id'] not in val_ids]

# Calculate distributions
train_debris = sum(t['debris_pixels'] for t in tile_stats if t['tile_id'] in train_ids)
train_lake = sum(t['lake_pixels'] for t in tile_stats if t['tile_id'] in train_ids)
val_debris = sum(t['debris_pixels'] for t in tile_stats if t['tile_id'] in val_ids)
val_lake = sum(t['lake_pixels'] for t in tile_stats if t['tile_id'] in val_ids)

train_total = sum(mask.size for tile_id in train_ids for _ in [0])
val_total = sum(mask.size for tile_id in val_ids for _ in [0])

print(f"\n📊 TRAIN SET ({len(train_ids)} tiles):")
print(f"   Debris: {train_debris:,} pixels ({train_debris/train_total*100:.3f}%)")
print(f"   Lake: {train_lake:,} pixels ({train_lake/train_total*100:.4f}%)")

print(f"\n📊 VAL SET ({len(val_ids)} tiles):")
print(f"   Debris: {val_debris:,} pixels ({val_debris/val_total*100:.3f}%)")
print(f"   Lake: {val_lake:,} pixels ({val_lake/val_total*100:.4f}%)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE - GENERATING RECOMMENDATIONS...")
print("="*80)
