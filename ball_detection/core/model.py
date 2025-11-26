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


def create_model(pretrained_path=None, use_mobilenet=False, mobilenet_pretrained=True, use_shufflenet=False):
    """
    Create ball detector model.

    Args:
        pretrained_path: Optional path to pretrained weights (.pth file)
        use_mobilenet: If True, use MobileNetV3-Small backbone
        mobilenet_pretrained: If True and use_mobilenet/use_shufflenet=True, load ImageNet pretrained weights
        use_shufflenet: If True, use ShuffleNetV2 x0.5 backbone (faster than MobileNetV3)

    Returns:
        BallDetectorCNN, BallDetectorMobileNetV3, or BallDetectorShuffleNetV2 model
    """
    if use_shufflenet:
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
    dummy_input = torch.randn(batch_size, 3, 128, 128)

    models_to_test = [
        ("BallDetectorCNN", BallDetectorCNN()),
        ("BallDetectorMobileNetV3", BallDetectorMobileNetV3(pretrained=False)),
        ("BallDetectorShuffleNetV2", BallDetectorShuffleNetV2(pretrained=False)),
    ]

    for name, model in models_to_test:
        print("=" * 60)
        print(f"Testing {name}...")
        print("=" * 60)

        model.eval()
        param_count = model.count_parameters()
        print(f"Parameters: {param_count:,} ({param_count * 4 / 1024:.1f} KB)")

        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)

        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Example output: x={output[0, 0]:.4f}, y={output[0, 1]:.4f}")
        print()

    print("All model tests successful!")
