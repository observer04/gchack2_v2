"""
Boundary-Aware U-Net with Channel-Spatial Squeeze & Excitation (cSE) attention.

Architecture:
- Encoder: ResNet34 (or other timm encoders)
- Decoder: U-Net style with skip connections
- Attention: cSE blocks in decoder
- Output: Multi-class segmentation logits

Key Features:
- encoder_weights=None for multispectral data (NO ImageNet)
- Flexible input channels (5 for competition, 7 for HKH)
- Progressive feature fusion with attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Optional, List


class ChannelSpatialSELayer(nn.Module):
    """
    Channel-Spatial Squeeze & Excitation block.
    
    Combines channel attention (cSE) and spatial attention (sSE)
    to recalibrate feature maps.
    
    Args:
        num_channels: Number of input channels
        reduction_ratio: Reduction factor for channel attention
    """
    
    def __init__(self, num_channels: int, reduction_ratio: int = 2):
        super().__init__()
        
        # Channel Squeeze & Excitation
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_channels, num_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // reduction_ratio, num_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial Squeeze & Excitation
        self.sSE = nn.Sequential(
            nn.Conv2d(num_channels, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        
        Returns:
            attended: (B, C, H, W)
        """
        # Channel attention
        cse_out = self.cSE(x) * x
        
        # Spatial attention
        sse_out = self.sSE(x) * x
        
        # Combine
        return cse_out + sse_out


class DecoderBlock(nn.Module):
    """
    Decoder block with skip connection and cSE attention.
    
    Args:
        in_channels: Input channels from previous decoder level
        skip_channels: Channels from encoder skip connection
        out_channels: Output channels
        use_attention: Whether to use cSE attention
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_attention: bool = True
    ):
        super().__init__()
        
        # Upsample previous decoder features
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Convolution after concatenation
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention
        self.use_attention = use_attention
        if use_attention:
            self.attention = ChannelSpatialSELayer(out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) - from previous decoder
            skip: (B, skip_channels, 2H, 2W) - from encoder
        
        Returns:
            out: (B, out_channels, 2H, 2W)
        """
        # Upsample
        x = self.upsample(x)
        
        # Concatenate with skip
        x = torch.cat([x, skip], dim=1)
        
        # Convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Attention
        if self.use_attention:
            x = self.attention(x)
        
        return x


class BoundaryAwareUNet(nn.Module):
    """
    Boundary-Aware U-Net for glacier segmentation.
    
    Uses segmentation_models_pytorch encoder with custom decoder
    that includes cSE attention for boundary refinement.
    
    Args:
        encoder_name: Name of encoder (e.g., 'resnet34', 'efficientnet-b0')
        encoder_weights: Pretrained weights ('imagenet' or None)
        in_channels: Input channels (5 for competition, 7 for HKH)
        num_classes: Output classes (4: background, glacier, debris, lake)
        decoder_channels: List of decoder channel counts
        use_attention: Whether to use cSE attention in decoder
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet34',
        encoder_weights: Optional[str] = None,
        in_channels: int = 5,
        num_classes: int = 4,
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        use_attention: bool = True
    ):
        super().__init__()
        
        # Create encoder using segmentation_models_pytorch
        # This gives us a pretrained encoder with proper skip connections
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights
        )
        
        # Get encoder output channels for each stage
        encoder_channels = self.encoder.out_channels  # e.g., [3, 64, 64, 128, 256, 512]
        
        # Build decoder
        self.decoder_blocks = nn.ModuleList()
        
        # Start from deepest level
        in_ch = encoder_channels[-1]  # e.g., 512
        
        for i, out_ch in enumerate(decoder_channels):
            # Skip connection from encoder at corresponding level
            skip_ch = encoder_channels[-(i + 2)] if i < len(encoder_channels) - 1 else 0
            
            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    use_attention=use_attention
                )
            )
            
            in_ch = out_ch
        
        # Final segmentation head
        self.segmentation_head = nn.Conv2d(
            decoder_channels[-1],
            num_classes,
            kernel_size=1
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize decoder weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) - input image
        
        Returns:
            logits: (B, num_classes, H, W)
        """
        # Encoder forward pass
        features = self.encoder(x)  # List of features at each stage
        
        # Start from deepest features
        x = features[-1]
        
        # Decoder forward pass
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Get skip connection (if available)
            skip_idx = -(i + 2)
            skip = features[skip_idx] if abs(skip_idx) <= len(features) else None
            
            if skip is not None:
                x = decoder_block(x, skip)
            else:
                # Final upsampling without skip
                x = decoder_block.upsample(x)
                x = decoder_block.conv1(x)
                x = decoder_block.conv2(x)
                if decoder_block.use_attention:
                    x = decoder_block.attention(x)
        
        # Final segmentation
        logits = self.segmentation_head(x)
        
        return logits


class BoundaryAwareUNetWithAux(nn.Module):
    """
    Boundary-Aware U-Net with auxiliary boundary prediction head.
    
    Adds explicit boundary prediction to help model learn boundaries.
    Can be used during training for additional supervision.
    
    Args:
        Same as BoundaryAwareUNet
        aux_weight: Weight for auxiliary boundary loss
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet34',
        encoder_weights: Optional[str] = None,
        in_channels: int = 5,
        num_classes: int = 4,
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        use_attention: bool = True,
        aux_weight: float = 0.4
    ):
        super().__init__()
        
        # Main segmentation network
        self.main_net = BoundaryAwareUNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
            decoder_channels=decoder_channels,
            use_attention=use_attention
        )
        
        # Auxiliary boundary prediction head
        self.boundary_head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
        self.aux_weight = aux_weight
    
    def forward(self, x: torch.Tensor, return_aux: bool = False):
        """
        Args:
            x: (B, C, H, W)
            return_aux: Whether to return auxiliary outputs
        
        Returns:
            If return_aux:
                (logits, boundary_pred)
            Else:
                logits
        """
        # Get intermediate features
        features = self.main_net.encoder(x)
        
        # Decoder
        x_dec = features[-1]
        for i, decoder_block in enumerate(self.main_net.decoder_blocks):
            skip_idx = -(i + 2)
            skip = features[skip_idx] if abs(skip_idx) <= len(features) else None
            if skip is not None:
                x_dec = decoder_block(x_dec, skip)
        
        # Main segmentation output
        logits = self.main_net.segmentation_head(x_dec)
        
        if return_aux:
            # Auxiliary boundary output
            boundary_pred = self.boundary_head(x_dec)
            return logits, boundary_pred
        else:
            return logits


def create_model(config: dict) -> nn.Module:
    """
    Factory function to create model from config.
    
    Args:
        config: Model configuration dict with keys:
            - architecture: 'boundary_aware_unet' or 'boundary_aware_unet_aux'
            - encoder: encoder name (e.g., 'resnet34')
            - encoder_weights: None or 'imagenet'
            - in_channels: input channels
            - num_classes: output classes
            - attention: 'cse' or None
    
    Returns:
        model: nn.Module
    """
    architecture = config.get('architecture', 'boundary_aware_unet')
    encoder = config.get('encoder', 'resnet34')
    encoder_weights = config.get('encoder_weights', None)
    in_channels = config.get('in_channels', 5)
    num_classes = config.get('num_classes', 4)
    use_attention = config.get('attention', 'cse') == 'cse'
    
    if architecture == 'boundary_aware_unet_aux':
        model = BoundaryAwareUNetWithAux(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
            use_attention=use_attention
        )
    else:
        model = BoundaryAwareUNet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
            use_attention=use_attention
        )
    
    return model


if __name__ == '__main__':
    # Test model
    print("Testing Boundary-Aware U-Net...")
    
    # Competition model (5 channels)
    model_comp = BoundaryAwareUNet(
        encoder_name='resnet34',
        encoder_weights=None,  # CRITICAL: No ImageNet for multispectral
        in_channels=5,
        num_classes=4
    )
    
    # Test forward pass
    x_comp = torch.randn(2, 5, 512, 512)
    logits_comp = model_comp(x_comp)
    print(f"Competition model output: {logits_comp.shape}")
    
    # HKH model (7 channels)
    model_hkh = BoundaryAwareUNet(
        encoder_name='resnet34',
        encoder_weights=None,
        in_channels=7,
        num_classes=4
    )
    
    x_hkh = torch.randn(2, 7, 512, 512)
    logits_hkh = model_hkh(x_hkh)
    print(f"HKH model output: {logits_hkh.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model_comp.parameters())
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.1f}M)")
    
    # Test auxiliary model
    model_aux = BoundaryAwareUNetWithAux(
        encoder_name='resnet34',
        encoder_weights=None,
        in_channels=5,
        num_classes=4
    )
    
    logits, boundary = model_aux(x_comp, return_aux=True)
    print(f"Auxiliary model outputs: {logits.shape}, {boundary.shape}")
    
    print("\n✓ Model architecture working!")
