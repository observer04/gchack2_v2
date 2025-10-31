#!/bin/bash
# Quick setup script for Kaggle
# Run this as first cell in Kaggle notebook

set -e

echo "=================================================="
echo "Glacier Segmentation - Kaggle Setup"
echo "=================================================="

# 1. Clone repository
echo -e "\n[1/5] Cloning repository..."
if [ ! -d "gchack2_v2" ]; then
    git clone https://github.com/observer04/gchack2_v2.git
    cd gchack2_v2
else
    cd gchack2_v2
    git pull
fi

# 2. Install dependencies
echo -e "\n[2/5] Installing dependencies..."
pip install -q segmentation-models-pytorch albumentations timm scikit-learn rasterio geopandas PyYAML tqdm

# 3. Verify GPU setup
echo -e "\n[3/5] Verifying GPU setup..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# 4. Create directories
echo -e "\n[4/5] Creating directories..."
mkdir -p data/competition
mkdir -p data/hkh/raw
mkdir -p data/hkh/processed/images
mkdir -p data/hkh/processed/masks
mkdir -p weights
mkdir -p logs

# 5. Download competition data
echo -e "\n[5/5] Downloading competition data..."
cd /kaggle/working
if [ ! -f "train2.zip" ]; then
    wget -q https://www.glacier-hack.in/train2.zip
    unzip -q train2.zip
fi

# Move to proper location
if [ -d "Train" ]; then
    mv Train/* gchack2_v2/data/competition/
fi

cd gchack2_v2

echo -e "\n=================================================="
echo "✓ Setup complete!"
echo "=================================================="
echo ""
echo "Data structure:"
ls -l data/competition/

echo -e "\nNext steps:"
echo "1. Download HKH dataset (optional)"
echo "2. Run training: python src/training/train.py --config configs/hkh_pretrain_kaggle.yaml"
echo ""
