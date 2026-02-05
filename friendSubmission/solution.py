import torch
import numpy as np
import segmentation_models_pytorch as smp
import tifffile
import json
from pathlib import Path

class Solution:
    """4-Class Glacier Segmentation Solution"""
    
    def __init__(self, model_dir='./'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        with open(Path(model_dir) / 'config.json') as f:
            cfg = json.load(f)
        
        # Build model
        self.model = smp.UnetPlusPlus(
            encoder_name=cfg['encoder'],
            encoder_weights=None,
            in_channels=cfg['in_channels'],
            classes=cfg['num_classes'],
            decoder_attention_type=cfg['attention']
        )
        
        # Load weights
        ckpt = torch.load(Path(model_dir) / f"{cfg['model_name']}_best.pth", 
                         map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device).eval()
        
        print(f"✓ Model loaded: {cfg['model_name']}")
        print(f"  MCC: {cfg['best_mcc']:.4f} | mIoU: {cfg['best_miou']:.4f}")
    
    def load_image(self, band_paths):
        """Load and normalize 5-band multi-spectral image"""
        bands = [tifffile.imread(p) for p in band_paths]
        img = np.stack(bands, axis=0).astype(np.float32)
        
        # Percentile normalization for each band
        for i in range(5):
            pmin, pmax = np.percentile(img[i], [2, 98])
            if pmax > pmin:
                img[i] = np.clip((img[i] - pmin) / (pmax - pmin), 0, 1)
        
        return img
    
    def predict(self, band_paths, use_tta=True):
        """
        Predict 4-class segmentation mask
        
        Args:
            band_paths: List of 5 paths to band TIFF files
            use_tta: Use test-time augmentation (horizontal flip)
        
        Returns:
            mask: (H, W) numpy array with class labels 0-3
        """
        img = self.load_image(band_paths)
        x = torch.from_numpy(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if use_tta:
                # Original prediction
                logits1 = self.model(x)
                # Horizontally flipped prediction
                logits2 = torch.flip(self.model(torch.flip(x, dims=[3])), dims=[3])
                # Average logits and apply softmax
                probs = torch.softmax((logits1 + logits2) / 2, dim=1)
            else:
                probs = torch.softmax(self.model(x), dim=1)
        
        # Get class with highest probability (argmax)
        mask = torch.argmax(probs, dim=1)[0].cpu().numpy()
        return mask.astype(np.uint8)
    
    def predict_batch(self, image_list, use_tta=True):
        """
        Batch prediction for multiple images
        
        Args:
            image_list: List of (image_id, [band_paths])
            use_tta: Use test-time augmentation
        
        Returns:
            results: Dict {image_id: mask}
        """
        results = {}
        for image_id, band_paths in image_list:
            mask = self.predict(band_paths, use_tta=use_tta)
            results[image_id] = mask
        return results

# Example usage
if __name__ == "__main__":
    sol = Solution('.')
    
    # Example paths
    band_paths = [
        'path/to/Band1/image.tif',
        'path/to/Band2/image.tif', 
        'path/to/Band3/image.tif',
        'path/to/Band4/image.tif',
        'path/to/Band5/image.tif'
    ]
    
    mask = sol.predict(band_paths, use_tta=True)
    print(f"Prediction shape: {mask.shape}")
    print(f"Classes found: {np.unique(mask)}")
