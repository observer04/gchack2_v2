# Glacier Segmentation - Multi-Class Solution

## Model Architecture
- **Type**: UNet++ with Attention
- **Encoder**: efficientnet-b2
- **Decoder**: scse attention
- **Classes**: 4 (debris, ice, snow, background)
- **Input**: 5-band multi-spectral imagery

## Performance Metrics
- **MCC**: 0.5747
- **mIoU**: 0.3938
- **Epochs**: 100

## Files
- `unetpp_best.pth` - Trained model weights
- `config.json` - Model configuration
- `solution.py` - Inference script
- `README.md` - This file

## Usage

```python
from solution import Solution

# Initialize
sol = Solution('.')

# Predict single image with TTA
band_paths = ['Band1/img.tif', 'Band2/img.tif', ..., 'Band5/img.tif']
mask = sol.predict(band_paths, use_tta=True)
# Returns: (H,W) array with class labels 0-3

# Batch prediction
image_list = [
    ('img1', [band1_path, ..., band5_path]),
    ('img2', [band1_path, ..., band5_path]),
]
results = sol.predict_batch(image_list)
```

## Notes
- Multi-class segmentation (no threshold needed - uses argmax)
- Test-Time Augmentation with horizontal flip recommended
- Percentile normalization (2-98) applied to each band
- GPU accelerated when available
