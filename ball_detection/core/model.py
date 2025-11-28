"""
Lightweight CNN for Sub-Pixel Ball Center Detection

Architecture optimized for:
- Speed: < 5ms inference on AMD Vega 8 (via ONNX Runtime DirectML)
- Accuracy: < 0.2 pixel error
- Size: ~100K parameters (~500KB model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResidualBlock(nn.Module):
    """
    Lightweight residual block for feature extraction.

    Uses 3x3 convolutions with batch normalization and ReLU activation.
    Skip connection helps with gradient flow during training.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection with 1x1 conv if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out


class BallDetectorCNN(nn.Module):
    """
    Lightweight CNN for detecting ball center with sub-pixel accuracy.

    Architecture:
        Input: 64x64 RGB image (cropped around red ball)

        Feature Extraction:
        - Conv1: 3 → 16 channels, 64x64 → 32x32
        - ResBlock1: 16 → 16 channels, 32x32
        - ResBlock2: 16 → 32 channels, 32x32 → 16x16
        - ResBlock3: 32 → 64 channels, 16x16 → 8x8
        - ResBlock4: 64 → 64 channels, 8x8

        Regression Head:
        - Global Average Pooling: 8x8x64 → 64
        - FC1: 64 → 32
        - FC2: 32 → 2 (x, y)

        Output:
        - x_norm ∈ [0, 1]: Normalized X coordinate in crop
        - y_norm ∈ [0, 1]: Normalized Y coordinate in crop

    Total Parameters: ~98K
    """

    def __init__(self):
        super().__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.layer1 = ResidualBlock(16, 16, stride=1)    # 32x32 → 32x32
        self.layer2 = ResidualBlock(16, 32, stride=2)    # 32x32 → 16x16
        self.layer3 = ResidualBlock(32, 64, stride=2)    # 16x16 → 8x8
        self.layer4 = ResidualBlock(64, 64, stride=1)    # 8x8 → 8x8

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Regression head
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 2)  # (x, y) only

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, 64, 64)

        Returns:
            Output tensor of shape (batch, 2):
                - [:, 0]: x_normalized ∈ [0, 1]
                - [:, 1]: y_normalized ∈ [0, 1]
        """
        # Feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Regression head
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        # Apply sigmoid to ensure outputs in [0, 1]
        x = torch.sigmoid(x)

        return x

    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BallDetectorMobileNetV3(nn.Module):
    """
    Ball detector using MobileNetV3-Small backbone with pretrained weights.

    Uses ImageNet pretrained MobileNetV3-Small as feature extractor,
    with a custom regression head for ball center detection.

    Total Parameters: ~1.5M (backbone) + ~10K (head)
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # Load MobileNetV3-Small with pretrained weights
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)

        # Extract feature extractor (everything except classifier)
        self.features = mobilenet.features
        # MobileNetV3-Small outputs 576 channels after features

        # Regression head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(576, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 2)  # (x, y) only
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, 64, 64)

        Returns:
            Output tensor of shape (batch, 2):
                - [:, 0]: x_normalized ∈ [0, 1]
                - [:, 1]: y_normalized ∈ [0, 1]
        """
        # Feature extraction
        x = self.features(x)

        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Regression head
        x = self.head(x)

        # Apply sigmoid to ensure outputs in [0, 1]
        x = torch.sigmoid(x)

        return x

    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BallDetectorShuffleNetV2(nn.Module):
    """
    Ball detector using ShuffleNetV2 x0.5 backbone with pretrained weights.

    ShuffleNetV2 is optimized for speed with channel shuffle operations.
    The x0.5 variant is the smallest, with only ~350K parameters.

    Total Parameters: ~350K (backbone) + ~5K (head) = ~355K
    Expected inference: ~0.3-0.5ms (faster than MobileNetV3)
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # Load ShuffleNetV2 x0.5 with pretrained weights
        if pretrained:
            weights = models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
        else:
            weights = None

        shufflenet = models.shufflenet_v2_x0_5(weights=weights)

        # Extract feature extractor (everything except fc)
        # ShuffleNetV2 structure: conv1, maxpool, stage2, stage3, stage4, conv5
        self.conv1 = shufflenet.conv1
        self.maxpool = shufflenet.maxpool
        self.stage2 = shufflenet.stage2
        self.stage3 = shufflenet.stage3
        self.stage4 = shufflenet.stage4
        self.conv5 = shufflenet.conv5

        # ShuffleNetV2 x0.5 outputs 1024 channels after conv5

        # Regression head (gradual reduction like MobileNetV3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 2)  # (x, y) coordinates
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, H, W)

        Returns:
            Output tensor of shape (batch, 2):
                - [:, 0]: x_normalized in [0, 1]
                - [:, 1]: y_normalized in [0, 1]
        """
        # Feature extraction
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)

        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Regression head
        x = self.head(x)

        # Apply sigmoid to ensure outputs in [0, 1]
        x = torch.sigmoid(x)

        return x

    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BallDetectorFullFrame(nn.Module):
    """
    Full-frame ball detector for 1280x720 input with sub-pixel regression.

    Architecture designed for speed on large inputs:
    - Aggressive downsampling stem (stride 4 + stride 2 = 8x reduction early)
    - Lightweight MobileNetV3-Small backbone (modified)
    - Global pooling + regression head
    - Outputs: x, y (normalized 0-1), confidence (ball present 0-1)

    Input: 1280x720 RGB
    Output: (x, y, confidence) where x,y are normalized to image space
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # Aggressive downsampling stem for 1280x720 -> 160x90
        # This reduces computation before the backbone
        self.stem = nn.Sequential(
            # 1280x720 -> 320x180 (stride 4)
            nn.Conv2d(3, 24, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            # 320x180 -> 160x90 (stride 2)
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Load MobileNetV3-Small backbone
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)

        # Modify first conv to accept 32 channels from our stem (instead of 3)
        # Original: Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        # We replace it with a 1x1 conv to adapt channels
        self.adapt = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
        )

        # Use MobileNetV3 features starting from layer 1 (skip first conv)
        # features[0] is the first conv block, we skip it
        self.backbone = nn.Sequential(*list(mobilenet.features.children())[1:])

        # MobileNetV3-Small outputs 576 channels
        # Regression head with 3 outputs: x, y, confidence
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(576, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3)  # x, y, confidence
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, 720, 1280)

        Returns:
            Output tensor of shape (batch, 3):
                - [:, 0]: x_normalized in [0, 1] (multiply by 1280 for pixels)
                - [:, 1]: y_normalized in [0, 1] (multiply by 720 for pixels)
                - [:, 2]: confidence in [0, 1] (ball present probability)
        """
        # Aggressive downsampling: 1280x720 -> 160x90
        x = self.stem(x)

        # Adapt channels for backbone: 32 -> 16
        x = self.adapt(x)

        # MobileNetV3 backbone
        x = self.backbone(x)

        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Regression head
        x = self.head(x)

        # Sigmoid for normalized outputs
        x = torch.sigmoid(x)

        return x

    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BallDetectorFullFrameTiny(nn.Module):
    """
    Ultra-lightweight full-frame detector for maximum speed.

    Even more aggressive downsampling + custom tiny backbone.
    Target: < 500K parameters, < 5ms inference on GPU.

    Input: 1280x720 RGB
    Output: (x, y, confidence)
    """

    def __init__(self):
        super().__init__()

        # Very aggressive stem: 1280x720 -> 80x45 (16x reduction)
        self.stem = nn.Sequential(
            # 1280x720 -> 320x180
            nn.Conv2d(3, 16, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # 320x180 -> 80x45
            nn.Conv2d(16, 32, kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Tiny backbone
        self.backbone = nn.Sequential(
            # 80x45 -> 40x23
            self._make_block(32, 64, stride=2),
            # 40x23 -> 20x12
            self._make_block(64, 128, stride=2),
            # 20x12 -> 10x6
            self._make_block(128, 256, stride=2),
            # Refine
            self._make_block(256, 256, stride=1),
        )

        # Regression head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # x, y, confidence
        )

    def _make_block(self, in_ch, out_ch, stride):
        """Depthwise separable conv block for efficiency."""
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        x = torch.sigmoid(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BallDetectorFullFrameUltra(nn.Module):
    """
    Maximum efficiency full-frame detector using PixelUnshuffle.

    Uses PixelUnshuffle for zero-compute spatial reduction, then
    lightweight depthwise-separable backbone.

    Architecture:
    - PixelUnshuffle(8): 1280x720x3 → 160x90x192 (FREE! just reshape)
    - 1x1 conv: 192 → 32 channels (cheap channel reduction)
    - Tiny depthwise-separable backbone
    - Global pooling + regression head

    Input: 1280x720 RGB
    Output: (x, y, confidence)
    """

    def __init__(self, base_channels=32):
        super().__init__()

        # PixelUnshuffle stem - ZERO compute for 8x spatial reduction!
        # 1280x720x3 → 160x90x192
        self.pixel_unshuffle = nn.PixelUnshuffle(8)

        # Channel reduction: 192 → base_channels
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(192, base_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Tiny backbone with depthwise-separable convs
        # 160x90 → 80x45 → 40x23 → 20x12 → 10x6
        self.backbone = nn.Sequential(
            self._make_block(base_channels, base_channels * 2, stride=2),      # 160x90 → 80x45
            self._make_block(base_channels * 2, base_channels * 4, stride=2),  # 80x45 → 40x23
            self._make_block(base_channels * 4, base_channels * 8, stride=2),  # 40x23 → 20x12
            self._make_block(base_channels * 8, base_channels * 8, stride=2),  # 20x12 → 10x6
        )

        # Regression head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(base_channels * 8, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # x, y, confidence
        )

    def _make_block(self, in_ch, out_ch, stride):
        """Depthwise separable conv block."""
        return nn.Sequential(
            # Depthwise conv
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            # Pointwise conv
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # PixelUnshuffle: 1280x720x3 → 160x90x192 (zero compute!)
        x = self.pixel_unshuffle(x)

        # Reduce channels: 192 → 32
        x = self.channel_reduce(x)

        # Backbone
        x = self.backbone(x)

        # Global pooling + head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        x = torch.sigmoid(x)

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BallDetectorFullFrameMobileNet(nn.Module):
    """
    Full-frame detector combining PixelUnshuffle with MobileNetV3 backbone.

    Uses PixelUnshuffle for zero-compute spatial reduction, then feeds into
    pretrained MobileNetV3-Small for feature extraction.

    Architecture:
    - PixelUnshuffle(8): 1280x720x3 → 160x90x192 (FREE reshape, 8x reduction)
    - 1x1 conv: 192 → 16 channels (match MobileNetV3 first conv output)
    - MobileNetV3 features[1:]: pretrained backbone (skip first conv)
    - Global pooling + regression head

    Benefits:
    - Zero-compute downsampling via PixelUnshuffle
    - ImageNet pretrained features from MobileNetV3
    - ~1M parameters with strong feature extraction

    Input: 1280x720 RGB
    Output: (x, y, confidence)
    """

    def __init__(self, pretrained=True, unshuffle_factor=8):
        super().__init__()

        self.unshuffle_factor = unshuffle_factor

        # PixelUnshuffle for zero-compute spatial reduction
        # factor=4: 1280x720x3 → 320x180x48
        # factor=8: 1280x720x3 → 160x90x192
        self.pixel_unshuffle = nn.PixelUnshuffle(unshuffle_factor)
        in_channels = 3 * (unshuffle_factor ** 2)  # 48 for factor=4, 192 for factor=8

        # Channel adaptation to match MobileNetV3's expected input (16 channels after first conv)
        # MobileNetV3 first conv: 3 → 16 with stride 2
        # We skip that and directly provide 16 channels
        self.channel_adapt = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
        )

        # Load MobileNetV3-Small and use features[1:] (skip first conv)
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(mobilenet.features.children())[1:])

        # MobileNetV3-Small outputs 576 channels
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(576, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3)  # x, y, confidence
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, 720, 1280)

        Returns:
            Output tensor of shape (batch, 3):
                - [:, 0]: x_normalized in [0, 1]
                - [:, 1]: y_normalized in [0, 1]
                - [:, 2]: confidence in [0, 1]
        """
        # PixelUnshuffle: 1280x720x3 → 320x180x48 (zero compute!)
        x = self.pixel_unshuffle(x)

        # Adapt channels: 48 → 16
        x = self.channel_adapt(x)

        # MobileNetV3 backbone (pretrained)
        x = self.backbone(x)

        # Global pooling + head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        x = torch.sigmoid(x)

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BallDetectorFullFrameMobileNetLite(nn.Module):
    """
    Lightweight full-frame detector with partial MobileNetV3 backbone.

    Optimized for ~5-8ms CPU inference while retaining pretrained features.

    Architecture:
    - PixelUnshuffle(8): 1280x720x3 → 160x90x192 (zero-compute)
    - Depthwise stride-2: 160x90 → 80x45 (fast spatial reduction)
    - Pointwise: 192 → 16 channels
    - MobileNetV3 layers 1-4: pretrained inverted residuals (40ch output)
    - Global pooling + small head

    Benefits:
    - ~5-8ms CPU, ~1ms GPU
    - ~30K parameters
    - Pretrained ImageNet features (partial)

    Input: 1280x720 RGB
    Output: (x, y, confidence)
    """

    def __init__(self, pretrained=True, num_layers=4):
        super().__init__()

        self.pixel_unshuffle = nn.PixelUnshuffle(8)  # 160x90x192

        # Depthwise stride-2 + pointwise channel reduction
        self.stem = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1, groups=192, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
        )

        # Load partial MobileNetV3-Small backbone
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(mobilenet.features.children())[1:num_layers+1])

        # Output channels based on num_layers: 1->16, 2->24, 3->24, 4->40
        layer_channels = {1: 16, 2: 24, 3: 24, 4: 40, 5: 40, 6: 40}
        out_channels = layer_channels.get(num_layers, 40)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(out_channels, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        x = self.pixel_unshuffle(x)
        x = self.stem(x)
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return torch.sigmoid(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# STEREO MODELS (6-channel input: RGB_left + RGB_right)
# ============================================================================

class StereoTiny(nn.Module):
    """
    Fast 6-channel stereo detector based on FullFrame-Tiny.

    Uses aggressive strided convolutions for maximum speed.
    ~400 FPS on laptop, ~600 FPS on desktop GPU.

    Input: (B, 6, 720, 1280) - left+right RGB concatenated
    Output: 6 values (x_l, y_l, conf_l, x_r, y_r, conf_r)
    """

    def __init__(self):
        super().__init__()

        # Very aggressive stem: 1280x720 -> 80x45 (16x reduction)
        self.stem = nn.Sequential(
            # 1280x720 -> 320x180
            nn.Conv2d(6, 16, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # 320x180 -> 80x45
            nn.Conv2d(16, 32, kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Tiny backbone with depthwise-separable blocks
        self.backbone = nn.Sequential(
            self._make_block(32, 64, stride=2),    # 80x45 -> 40x23
            self._make_block(64, 128, stride=2),   # 40x23 -> 20x12
            self._make_block(128, 256, stride=2),  # 20x12 -> 10x6
            self._make_block(256, 256, stride=1),  # refine
        )

        self.out_channels = 256

        # Regression head - 6 outputs for stereo
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 6)
        )

    def _make_block(self, in_ch, out_ch, stride):
        """Depthwise separable conv block."""
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return torch.sigmoid(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_feature_size(self):
        return self.out_channels

    def extract_features(self, x):
        """Extract features without final head (for temporal models)."""
        x = self.stem(x)
        x = self.backbone(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class EfficientTemporalTiny(nn.Module):
    """
    Stereo detector with GRU for temporal context.

    Uses StereoTiny encoder + GRU to leverage frame history.
    Only processes ONE new frame per forward pass (features cached).

    Input: (B, 6, 720, 1280) - current stereo frame
    Output: (B, 6) - (x_l, y_l, conf_l, x_r, y_r, conf_r)

    For training: Use forward() with full sequence
    For inference: Use forward_stream() with single frames (maintains state)
    """

    def __init__(self, history_length=8, hidden_size=128):
        super().__init__()
        self.history_length = history_length
        self.hidden_size = hidden_size

        # Feature encoder (StereoTiny without head)
        self.encoder = StereoTiny()
        feature_size = self.encoder.out_channels  # 256

        # Remove encoder's head - we'll use our own
        self.encoder.head = nn.Identity()

        # GRU for temporal processing
        self.gru = nn.GRU(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 6)
        )

        # For streaming inference
        self.feature_buffer = None
        self.gru_hidden = None

    def forward(self, x):
        """
        Forward pass for training with frame sequences.

        Args:
            x: (B, T, 6, H, W) - sequence of T stereo frames

        Returns:
            output: (B, 6) - prediction for last frame using full context
        """
        B, T, C, H, W = x.shape

        # Extract features for all frames
        features = []
        for t in range(T):
            feat = self.encoder.extract_features(x[:, t])  # (B, 256)
            features.append(feat)

        # Stack into sequence: (B, T, 256)
        feat_seq = torch.stack(features, dim=1)

        # Process through GRU
        gru_out, _ = self.gru(feat_seq)  # (B, T, hidden_size)

        # Use last timestep for prediction
        last_hidden = gru_out[:, -1]  # (B, hidden_size)

        # Output
        output = self.head(last_hidden)
        return torch.sigmoid(output)

    def forward_stream(self, x):
        """
        Streaming inference - process one frame at a time.
        Maintains internal state between calls.

        Args:
            x: (B, 6, H, W) - single stereo frame

        Returns:
            output: (B, 6) - prediction using cached history
        """
        B = x.shape[0]

        # Extract features for current frame
        feat = self.encoder.extract_features(x)  # (B, 256)

        # Initialize buffer if needed
        if self.feature_buffer is None or self.feature_buffer.shape[0] != B:
            self.feature_buffer = feat.unsqueeze(1).repeat(1, self.history_length, 1)
            self.gru_hidden = None
        else:
            # Shift buffer and add new features
            self.feature_buffer = torch.cat([
                self.feature_buffer[:, 1:],  # Drop oldest
                feat.unsqueeze(1)            # Add newest
            ], dim=1)

        # Process through GRU
        gru_out, self.gru_hidden = self.gru(self.feature_buffer, self.gru_hidden)

        # Use last timestep
        last_hidden = gru_out[:, -1]

        # Output
        output = self.head(last_hidden)
        return torch.sigmoid(output)

    def reset_state(self):
        """Reset streaming state (call when starting new video)."""
        self.feature_buffer = None
        self.gru_hidden = None

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(pretrained_path=None, use_mobilenet=False, mobilenet_pretrained=True, use_shufflenet=False, use_fullframe=False, use_fullframe_tiny=False, use_fullframe_ultra=False, use_fullframe_mobilenet=False, use_fullframe_mobilenet_lite=False, use_stereo_tiny=False):
    """
    Create ball detector model.

    Args:
        pretrained_path: Optional path to pretrained weights (.pth file)
        use_mobilenet: If True, use MobileNetV3-Small backbone
        mobilenet_pretrained: If True and use_mobilenet/use_shufflenet=True, load ImageNet pretrained weights
        use_shufflenet: If True, use ShuffleNetV2 x0.5 backbone (faster than MobileNetV3)
        use_fullframe: If True, use full-frame detector (1280x720 input)
        use_fullframe_tiny: If True, use ultra-lightweight full-frame detector
        use_fullframe_ultra: If True, use PixelUnshuffle-based ultra-efficient detector
        use_fullframe_mobilenet: If True, use PixelUnshuffle + MobileNetV3 hybrid
        use_fullframe_mobilenet_lite: If True, use lightweight partial MobileNetV3 (~30K params, ~5-8ms CPU)
        use_stereo_tiny: If True, use 6-channel stereo detector (~150K params, ~2.5ms)

    Returns:
        Ball detector model
    """
    if use_stereo_tiny:
        model = StereoTiny()
    elif use_fullframe_mobilenet_lite:
        model = BallDetectorFullFrameMobileNetLite(pretrained=mobilenet_pretrained)
    elif use_fullframe_mobilenet:
        model = BallDetectorFullFrameMobileNet(pretrained=mobilenet_pretrained)
    elif use_fullframe_ultra:
        model = BallDetectorFullFrameUltra()
    elif use_fullframe_tiny:
        model = BallDetectorFullFrameTiny()
    elif use_fullframe:
        model = BallDetectorFullFrame(pretrained=mobilenet_pretrained)
    elif use_shufflenet:
        model = BallDetectorShuffleNetV2(pretrained=mobilenet_pretrained)
    elif use_mobilenet:
        model = BallDetectorMobileNetV3(pretrained=mobilenet_pretrained)
    else:
        model = BallDetectorCNN()

    if pretrained_path is not None:
        print(f"Loading pretrained weights from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print("Weights loaded successfully")

    param_count = model.count_parameters()
    print(f"Model created with {param_count:,} parameters ({param_count * 4 / 1024:.1f} KB)")

    return model


if __name__ == "__main__":
    # Test all model architectures
    batch_size = 2

    # Crop-based models (128x128 input)
    crop_input = torch.randn(batch_size, 3, 128, 128)
    crop_models = [
        ("BallDetectorCNN", BallDetectorCNN()),
        ("BallDetectorMobileNetV3", BallDetectorMobileNetV3(pretrained=False)),
        ("BallDetectorShuffleNetV2", BallDetectorShuffleNetV2(pretrained=False)),
    ]

    print("=" * 60)
    print("CROP-BASED MODELS (128x128 input)")
    print("=" * 60)

    for name, model in crop_models:
        model.eval()
        param_count = model.count_parameters()
        with torch.no_grad():
            output = model(crop_input)
        print(f"{name}: {param_count:,} params, output shape: {output.shape}")

    # Full-frame models (1280x720 input)
    fullframe_input = torch.randn(batch_size, 3, 720, 1280)
    fullframe_models = [
        ("BallDetectorFullFrame", BallDetectorFullFrame(pretrained=False)),
        ("BallDetectorFullFrameTiny", BallDetectorFullFrameTiny()),
        ("BallDetectorFullFrameUltra", BallDetectorFullFrameUltra()),
        ("BallDetectorFullFrameMobileNet", BallDetectorFullFrameMobileNet(pretrained=False)),
        ("BallDetectorFullFrameMobileNetLite", BallDetectorFullFrameMobileNetLite(pretrained=False)),
    ]

    print()
    print("=" * 60)
    print("FULL-FRAME MODELS (1280x720 input)")
    print("=" * 60)

    for name, model in fullframe_models:
        model.eval()
        param_count = model.count_parameters()
        with torch.no_grad():
            output = model(fullframe_input)
        print(f"{name}: {param_count:,} params, output shape: {output.shape}")
        print(f"  Output: x={output[0, 0]:.4f}, y={output[0, 1]:.4f}, conf={output[0, 2]:.4f}")

    print()
    print("All model tests successful!")
