# Glacier Semantic Segmentation

A deep learning solution for semantic segmentation of glaciers in satellite imagery, addressing the critical task of monitoring glacial regions for climate change research and disaster risk management.

## Overview

This project implements a U-Net based architecture for multi-class glacier segmentation from multispectral satellite imagery. The model classifies pixels into four categories:
- **Background** (0): Non-glacier regions
- **Glacier** (1): Clean glacier ice
- **Debris** (2): Debris-covered glacier regions
- **Lake** (3): Glacial lake regions

The solution is evaluated using the Matthews Correlation Coefficient (MCC), which provides a balanced measure suitable for imbalanced class distributions.

## Key Features

- **Multispectral Input**: Processes 5-band satellite imagery (B2, B3, B4, B6 SWIR, B10 TIR)
- **Advanced Architecture**: U-Net with channel-spatial attention mechanisms
- **Transfer Learning**: Leverages pretraining on external glacier datasets
- **Robust Training**: Implements balanced sampling, mixed precision training, and adaptive learning rate scheduling
- **Ensemble Predictions**: Supports test-time augmentation (TTA) and model ensembling

## Requirements

### Core Dependencies
```bash
torch>=2.0.0
torchvision>=0.15.0
segmentation-models-pytorch>=0.3.3
albumentations>=1.3.0
rasterio>=1.3.0
opencv-python>=4.8.0
scikit-learn>=1.3.0
numpy>=1.24.0
PyYAML>=6.0
```

See `requirements.txt` for the complete list of dependencies.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/observer04/gchack2_v2.git
cd gchack2_v2
```

2. Create and activate a virtual environment:
```bash
python -m venv gc
source gc/bin/activate  # On Windows: gc\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
gchack2_v2/
├── configs/                 # Training configuration files (YAML)
├── src/                     # Source code
│   ├── models/              # Model architectures
│   ├── losses/              # Loss functions
│   └── training/            # Training utilities and metrics
├── notebooks/               # Jupyter notebooks for training and EDA
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Usage

### Training

The project uses YAML configuration files to specify training parameters. Example configurations are provided in the `configs/` directory.

**Basic Training:**
```bash
python src/training/train.py --config configs/hkh_pretrain.yaml
```

**Fine-tuning with Pretrained Weights:**
```bash
python src/training/train.py --config configs/competition_finetune.yaml \
    --pretrained weights/hkh_pretrained.pth
```

### Inference

For generating predictions on test data:
```bash
python solution.py --data <path_to_test_data> --out <output_directory>
```

**Input Format:**
- Test data should be organized in folders: `Band1/`, `Band2/`, `Band3/`, `Band4/`, `Band5/`
- Each folder contains `.tif` files with consistent naming across bands

**Output Format:**
- Predicted masks saved as `.tif` files
- Pixel values: 0 (background), 85 (glacier), 170 (debris), 255 (lake)
- Filenames match the input Band1 filenames

## Model Architecture

The solution employs a **Boundary-Aware U-Net** architecture with:
- **Encoder**: ResNet34 backbone (trained from scratch for multispectral data)
- **Decoder**: U-Net style with skip connections
- **Attention**: Channel-Spatial Squeeze & Excitation (cSE) blocks
- **Loss Functions**: Combination of Focal Loss, Dice Loss, MCC Loss, and Boundary Loss

## Dataset

The model is designed to work with multispectral satellite imagery containing 5 bands:
- **B2** (Blue): 450-520 nm
- **B3** (Green): 520-600 nm
- **B4** (Red): 630-690 nm
- **B6** (SWIR): 1560-1660 nm
- **B10** (TIR): 10600-11190 nm

Training data should include corresponding ground truth masks with pixel values representing the four classes.

## Performance

The model achieves competitive performance on glacier segmentation tasks:
- Balanced accuracy across all four classes
- Robust handling of class imbalance (especially rare lake class)
- Strong generalization to unseen geographical regions

## Configuration

Key training parameters can be adjusted in the YAML configuration files:

```yaml
data:
  in_channels: 5
  num_classes: 4
  batch_size: 8
  image_size: 512

model:
  architecture: unet
  encoder_name: resnet34
  encoder_weights: null  # Train from scratch for multispectral

optimizer:
  name: AdamW
  lr: 1.0e-4
  weight_decay: 1.0e-4
```

## Notebooks

Jupyter notebooks are provided for:
- **Training workflows**: Step-by-step training procedures
- **Exploratory Data Analysis**: Dataset statistics and visualizations

## License

This project is developed for the IEEE GRSS GlacierHack Challenge 2025.

## Acknowledgments

This project addresses glacier semantic segmentation for climate change monitoring and disaster risk management in high-altitude regions. The solution supports:
- Monitoring glacial retreat
- Estimating freshwater reserves
- Predicting glacial lake outburst floods (GLOFs)

## Contact

For questions or issues, please open an issue on the GitHub repository.
