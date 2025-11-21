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
        - FC2: 32 → 3 (x, y, confidence)

        Output:
        - x_norm ∈ [0, 1]: Normalized X coordinate in crop
        - y_norm ∈ [0, 1]: Normalized Y coordinate in crop
        - confidence ∈ [0, 1]: Detection confidence

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


def create_model(pretrained_path=None, use_mobilenet=False, mobilenet_pretrained=True):
    """
    Create ball detector model.

    Args:
        pretrained_path: Optional path to pretrained weights (.pth file)
        use_mobilenet: If True, use MobileNetV3-Small backbone
        mobilenet_pretrained: If True and use_mobilenet=True, load ImageNet pretrained weights

    Returns:
        BallDetectorCNN or BallDetectorMobileNetV3 model
    """
    if use_mobilenet:
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
    # Test model creation and forward pass
    print("Testing BallDetectorCNN...")

    model = create_model()
    model.eval()

    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 64, 64)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nExample outputs:")
    for i in range(batch_size):
        x, y, conf = output[i].numpy()
        print(f"  Sample {i+1}: x={x:.4f}, y={y:.4f}, confidence={conf:.4f}")

    print("\nModel test successful!")
