"""
PyTorch Dataset for Ball Detection Training

Handles loading labeled ball images and applies augmentation for robust training.
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BallDetectionDataset(Dataset):
    """
    Dataset for ball center detection training.

    Expected data format:
        data_dir/
            images/
                img_0001.jpg
                img_0002.jpg
                ...
            labels.json  # {"img_0001.jpg": {"x": 32.5, "y": 28.3}, ...}

    or:
        data_dir/
            img_0001.jpg
            img_0002.jpg
            ...
            labels.json

    Labels JSON format:
        {
            "img_name.jpg": {
                "x": <x_pixel>,      # Ball center X coordinate
                "y": <y_pixel>,      # Ball center Y coordinate
                "valid": true/false  # Optional: whether detection is valid
            },
            ...
        }
    """

    def __init__(self, data_dir, labels_file='labels.json',
                 image_size=64, crop_size=128, augment=True, split='train'):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing images and labels
            labels_file: Name of labels JSON file
            image_size: Target image size (assumes square images) - final size after resize
            crop_size: Size of crop to extract around ball before resizing
            augment: Whether to apply data augmentation
            split: 'train' or 'val' (affects augmentation)
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.crop_size = crop_size
        self.augment = augment and (split == 'train')

        # Check for images in subdirectory or root
        if (self.data_dir / 'images').exists():
            self.image_dir = self.data_dir / 'images'
        else:
            self.image_dir = self.data_dir

        # Load labels
        labels_path = self.data_dir / labels_file
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        with open(labels_path, 'r') as f:
            self.labels = json.load(f)

        # Filter valid samples
        self.samples = []
        for img_name, label in self.labels.items():
            # Skip if explicitly marked as invalid
            if not label.get('valid', True):
                continue

            img_path = self.image_dir / img_name
            if img_path.exists():
                self.samples.append((str(img_path), label))
            else:
                print(f"Warning: Image not found: {img_path}")

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {data_dir}")

        print(f"Loaded {len(self.samples)} samples from {data_dir}")

        # Setup augmentation pipeline
        self.transform = self._get_transforms()

    def _get_transforms(self):
        """Create augmentation pipeline."""
        if self.augment:
            # Training augmentations
            return A.Compose([
                # Resize crop to final input size
                A.Resize(self.image_size, self.image_size),

                # Geometric
                A.Rotate(limit=15, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=0,
                    p=0.5
                ),

                # Color/brightness
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.7
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=20,
                    p=0.5
                ),

                # Blur and noise
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ], p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),

                # Normalization
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        else:
            # Validation/test: no augmentation, just resize
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get training sample.

        Returns:
            image: Tensor of shape (3, H, W)
            target: Tensor of shape (3,) containing (x_norm, y_norm, confidence=1.0)
        """
        img_path, label = self.samples[idx]

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        # Get ball center coordinates
        ball_x = label['x']
        ball_y = label['y']

        # Add random offset during training for robustness
        if self.augment:
            # Random offset up to 20% of crop size
            max_offset = int(self.crop_size * 0.2)
            offset_x = np.random.randint(-max_offset, max_offset + 1)
            offset_y = np.random.randint(-max_offset, max_offset + 1)
        else:
            offset_x = 0
            offset_y = 0

        # Calculate crop bounds (centered on ball + offset)
        crop_center_x = int(ball_x + offset_x)
        crop_center_y = int(ball_y + offset_y)

        half_crop = self.crop_size // 2
        x1 = crop_center_x - half_crop
        y1 = crop_center_y - half_crop
        x2 = x1 + self.crop_size
        y2 = y1 + self.crop_size

        # Handle boundary conditions with padding
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - width)
        pad_bottom = max(0, y2 - height)

        # Adjust crop bounds to valid image region
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        # Extract crop
        crop = image[y1:y2, x1:x2]

        # Apply padding if needed
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            crop = cv2.copyMakeBorder(
                crop,
                pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REFLECT_101
            )

        # Calculate ball position relative to crop
        ball_x_in_crop = ball_x - (crop_center_x - half_crop)
        ball_y_in_crop = ball_y - (crop_center_y - half_crop)

        # Apply transforms (includes resize to image_size and augmentations with keypoint tracking)
        transformed = self.transform(image=crop, keypoints=[(ball_x_in_crop, ball_y_in_crop)])
        image_tensor = transformed['image']

        # Get transformed keypoint (already scaled to image_size by A.Resize)
        if len(transformed['keypoints']) > 0:
            x_transformed, y_transformed = transformed['keypoints'][0]
        else:
            # Keypoint went out of bounds after augmentation
            x_transformed, y_transformed = self.image_size / 2, self.image_size / 2

        # Normalize coordinates to [0, 1] relative to image_size
        x_norm = x_transformed / self.image_size
        y_norm = y_transformed / self.image_size

        # Clip to valid range (in case of numerical issues)
        x_norm = np.clip(x_norm, 0.0, 1.0)
        y_norm = np.clip(y_norm, 0.0, 1.0)

        # Create target tensor: (x_norm, y_norm, confidence)
        target = torch.tensor([x_norm, y_norm, 1.0], dtype=torch.float32)

        return image_tensor, target


def create_dataloaders(data_dir, batch_size=32, image_size=64, crop_size=128,
                       train_split=0.8, num_workers=4):
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Directory containing images and labels
        batch_size: Batch size for training
        image_size: Final input image size
        crop_size: Size of crop to extract around ball
        train_split: Fraction of data for training (rest for validation)
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader
    """
    # Load full dataset
    full_dataset = BallDetectionDataset(
        data_dir=data_dir,
        image_size=image_size,
        crop_size=crop_size,
        augment=False,  # We'll split first
        split='train'
    )

    # Split into train/val
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, dataset_size))

    # Create separate datasets with proper augmentation
    train_dataset = BallDetectionDataset(
        data_dir=data_dir,
        image_size=image_size,
        crop_size=crop_size,
        augment=True,
        split='train'
    )
    train_dataset.samples = [full_dataset.samples[i] for i in train_indices]

    val_dataset = BallDetectionDataset(
        data_dir=data_dir,
        image_size=image_size,
        crop_size=crop_size,
        augment=False,
        split='val'
    )
    val_dataset.samples = [full_dataset.samples[i] for i in val_indices]

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset.py <data_dir>")
        print("Example: python dataset.py ./ball_data")
        sys.exit(1)

    data_dir = sys.argv[1]

    print(f"Testing dataset loading from: {data_dir}\n")

    # Create dataset
    dataset = BallDetectionDataset(
        data_dir=data_dir,
        image_size=64,
        crop_size=128,
        augment=True,
        split='train'
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test loading a few samples
    print("\nTesting sample loading...")
    for i in range(min(3, len(dataset))):
        image, target = dataset[i]
        x_norm, y_norm, conf = target.numpy()

        print(f"\nSample {i+1}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Target: x={x_norm:.4f}, y={y_norm:.4f}, conf={conf:.4f}")

    # Test dataloader
    print("\n\nTesting dataloader...")
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=8,
        num_workers=0  # Single-threaded for testing
    )

    # Get one batch
    batch = next(iter(train_loader))
    images, targets = batch

    print(f"Batch images shape: {images.shape}")
    print(f"Batch targets shape: {targets.shape}")
    print(f"\nFirst target in batch: {targets[0].numpy()}")

    print("\nDataset test successful!")
