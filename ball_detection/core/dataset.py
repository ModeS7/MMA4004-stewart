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
from functools import lru_cache


class ImageTearing(A.ImageOnlyTransform):
    """
    Simulate realistic camera tearing artifact - horizontal and/or vertical frame splits.

    When applied, the image becomes invalid for detection (confidence should be 0).
    Supports horizontal tears (row splits), vertical tears (column splits), or both.
    """
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        h, w = img.shape[:2]
        result = img.copy()

        # Random direction: 50% horizontal, 30% vertical, 20% both
        direction = np.random.choice(['horizontal', 'vertical', 'both'], p=[0.5, 0.3, 0.2])

        if direction in ['horizontal', 'both']:
            result = self._apply_horizontal_tears(result)
        if direction in ['vertical', 'both']:
            result = self._apply_vertical_tears(result)

        return result

    def _apply_horizontal_tears(self, img):
        """Apply 1-2 horizontal tears (row-wise splits with horizontal shifts)."""
        h, w = img.shape[:2]
        result = img.copy()

        # 1-2 tear lines (75% chance of 1 tear)
        num_tears = np.random.choice([1, 1, 1, 2])

        if num_tears == 1:
            tear_y = np.random.randint(int(h * 0.1), int(h * 0.9))
            tear_positions = [tear_y]
        else:
            tear1 = np.random.randint(int(h * 0.15), int(h * 0.45))
            tear2 = np.random.randint(int(h * 0.55), int(h * 0.85))
            tear_positions = [tear1, tear2]

        # Shift amount (scaled to image size)
        min_shift = max(20, w // 6)
        max_shift = max(40, w // 3)

        prev_y = 0
        for i, tear_y in enumerate(tear_positions + [h]):
            if i > 0:
                shift = np.random.randint(min_shift, max_shift + 1)
                if np.random.random() > 0.5:
                    shift = -shift
                # Horizontal shift (roll along axis=1)
                result[prev_y:tear_y] = np.roll(result[prev_y:tear_y], shift, axis=1)
            prev_y = tear_y

        return result

    def _apply_vertical_tears(self, img):
        """Apply 1-2 vertical tears (column-wise splits with vertical shifts)."""
        h, w = img.shape[:2]
        result = img.copy()

        # 1-2 tear lines (75% chance of 1 tear)
        num_tears = np.random.choice([1, 1, 1, 2])

        if num_tears == 1:
            tear_x = np.random.randint(int(w * 0.1), int(w * 0.9))
            tear_positions = [tear_x]
        else:
            tear1 = np.random.randint(int(w * 0.15), int(w * 0.45))
            tear2 = np.random.randint(int(w * 0.55), int(w * 0.85))
            tear_positions = [tear1, tear2]

        # Shift amount (scaled to image size)
        min_shift = max(20, h // 6)
        max_shift = max(40, h // 3)

        prev_x = 0
        for i, tear_x in enumerate(tear_positions + [w]):
            if i > 0:
                shift = np.random.randint(min_shift, max_shift + 1)
                if np.random.random() > 0.5:
                    shift = -shift
                # Vertical shift (roll along axis=0)
                result[:, prev_x:tear_x] = np.roll(result[:, prev_x:tear_x], shift, axis=0)
            prev_x = tear_x

        return result

    def get_transform_init_args_names(self):
        return ()


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
                 crop_size=128, use_spatial_aug=True, use_appearance_aug=True,
                 use_color_invariance_aug=False, use_tearing_aug=False, tearing_probability=0.05,
                 split='train', disable_normalize=False):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing images and labels
            labels_file: Name of labels JSON file
            crop_size: Crop size (scale variation handled by ShiftScaleRotate augmentation)
            use_spatial_aug: Enable spatial augmentations (offset, rotate, scale, shift)
            use_appearance_aug: Enable appearance augmentations (brightness, hue, blur, noise)
            use_color_invariance_aug: Enable color invariance augmentations (hue shift, channel shuffle, grayscale)
            use_tearing_aug: Enable tearing augmentation (simulates camera tearing, sets confidence=0)
            tearing_probability: Probability of applying tearing (default 5%)
            split: 'train' or 'val' (affects augmentation)
            disable_normalize: If True, skip normalization (for GPU augmentations)
        """
        self.data_dir = Path(data_dir)
        self.crop_size = crop_size
        self.use_spatial_aug = use_spatial_aug and (split == 'train')
        self.use_appearance_aug = use_appearance_aug and (split == 'train')
        self.use_color_invariance_aug = use_color_invariance_aug and (split == 'train')
        self.use_tearing_aug = use_tearing_aug and (split == 'train')
        self.tearing_probability = tearing_probability
        self.disable_normalize = disable_normalize

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

        # Setup augmentation pipelines (separate for spatial and appearance)
        self.spatial_transform = self._get_spatial_transforms()
        self.color_invariance_transform = self._get_color_invariance_transforms()
        self.appearance_transform = self._get_appearance_transforms()

    def _get_spatial_transforms(self):
        """Create spatial augmentation pipeline (applied to FULL image before cropping).

        For validation: Will use seeded randomness (deterministic per sample)
        For training: Uses normal randomness (different every epoch)
        """
        # Spatial augmentations: shift, scale, rotate
        # Applied to full image, then we crop centered on augmented ball position
        # Using BORDER_CONSTANT (black padding) is acceptable
        return A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.1,  # ±10% = ~±64px on 640px image
                scale_limit=0.1,     # ±10% scale
                rotate_limit=180,    # Full 360° rotation (±180 covers all angles)
                p=1.0,
                border_mode=cv2.BORDER_CONSTANT  # Black padding (default)
            ),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def _get_normalize_transform(self):
        """Get normalize-only transform (for difficult samples that skip appearance aug)."""
        transforms_list = []
        if self.disable_normalize:
            transforms_list.append(A.ToFloat(max_value=255.0))
        else:
            transforms_list.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        transforms_list.append(ToTensorV2())
        return A.Compose(transforms_list)

    def _get_color_invariance_transforms(self):
        """Create color invariance augmentation pipeline (train model to work with any color ball).

        Applies aggressive color changes:
        - Full hue rotation (red -> any color)
        - Channel shuffle (swap R, G, B channels)
        - Random grayscale (force shape-based detection)
        """
        if not self.use_color_invariance_aug:
            return A.Compose([])  # No-op transform

        return A.Compose([
            # Full hue rotation - covers entire color spectrum (70% chance)
            # hue_shift_limit=180 means ±180° which covers all colors
            A.HueSaturationValue(
                hue_shift_limit=180,  # Full spectrum rotation
                sat_shift_limit=30,   # Mild saturation change
                val_shift_limit=20,   # Mild value change
                p=0.7
            ),
            # Channel shuffle - swaps RGB channels (30% chance)
            # Red ball becomes green or blue instantly
            A.ChannelShuffle(p=0.3),
            # Random grayscale - forces model to learn shape, not color (20% chance)
            A.ToGray(p=0.2),
        ])

    def _get_appearance_transforms(self):
        """Create appearance augmentation pipeline (applied to crop after spatial aug)."""
        transforms_list = []

        # Appearance augmentations (brightness, hue, blur, noise)
        if self.use_appearance_aug:
            transforms_list.extend([
                # Lighting variations
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.7
                ),
                # Color temperature / white balance simulation
                A.OneOf([
                    A.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.1,
                        p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=15,
                        sat_shift_limit=25,
                        val_shift_limit=25,
                        p=1.0
                    ),
                    A.RGBShift(
                        r_shift_limit=20,
                        g_shift_limit=20,
                        b_shift_limit=20,
                        p=1.0
                    ),
                ], p=0.8),
                # HEAVY augmentations - only ONE of these applies per sample
                A.OneOf([
                    A.RandomShadow(
                        shadow_roi=(0, 0.5, 1, 1),
                        num_shadows_limit=(1, 2),
                        shadow_dimension=5,
                        p=1.0
                    ),
                    A.RandomSunFlare(
                        flare_roi=(0, 0, 1, 0.5),
                        angle_range=(0, 1),
                        num_flare_circles_range=(3, 6),
                        src_radius=100,
                        p=1.0
                    ),
                    A.CLAHE(
                        clip_limit=2.0,
                        tile_grid_size=(8, 8),
                        p=1.0
                    ),
                    A.ISONoise(
                        color_shift=(0.01, 0.05),
                        intensity=(0.1, 0.2),
                        p=1.0
                    ),
                    A.Downscale(
                        scale_range=(0.75, 0.95),
                        interpolation_pair={
                            'downscale': cv2.INTER_LINEAR,
                            'upscale': cv2.INTER_LINEAR
                        },
                        p=1.0
                    ),
                ], p=0.4),  # 40% chance of ONE heavy augmentation
                # Blur variations (mild)
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ], p=0.3),
                # Mild noise (always available)
                A.GaussNoise(std_range=(10.0/255, 20.0/255), p=0.3),
                # Mild quality degradation
                A.ImageCompression(quality_range=(70, 100), p=0.2),
            ])

        # Normalize and convert to tensor (skip normalize if using GPU augmentations)
        if self.disable_normalize:
            # When using GPU augmentations, ensure conversion to float [0, 1]
            # Add a dummy normalize that just converts to float without changing values
            transforms_list.append(A.ToFloat(max_value=255.0))
        else:
            # Standard ImageNet normalization
            transforms_list.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        transforms_list.append(ToTensorV2())

        # No keypoints needed for appearance transforms (ball position already set)
        return A.Compose(transforms_list)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get training sample with NEW two-stage augmentation pipeline:
        1. Apply spatial augmentations to FULL image (shift/scale/rotate)
        2. Crop centered on augmented ball position (no padding)
        3. Apply color invariance augmentations (if enabled)
        4. Apply appearance augmentations to crop (skipped for difficult samples)

        Returns:
            image: Tensor of shape (3, H, W)
            target: Tensor of shape (2,) containing (x_norm, y_norm)
        """
        img_path, label = self.samples[idx]

        # Check if sample is marked as difficult (skip appearance augmentations)
        is_difficult = label.get('difficult', False)

        # Load image
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        # Original ball coordinates
        ball_x = label['x']
        ball_y = label['y']

        # STAGE 1: Apply spatial augmentations to FULL image
        # For validation: seed randomness for deterministic augmentation
        # For training: use natural randomness (different every epoch)
        if not self.use_spatial_aug:
            # Validation: seed based on sample index for deterministic augmentation
            seed_val = int(idx)  # Convert to Python int (idx might be numpy int64)
            np.random.seed(seed_val)
            # Also seed albumentations random state
            import random
            random.seed(seed_val)

        spatial_transformed = self.spatial_transform(image=image, keypoints=[(ball_x, ball_y)])

        # Reset random state for training (validation seed is only for this sample)
        if not self.use_spatial_aug:
            np.random.seed(None)
            import random
            random.seed(None)
        augmented_image = spatial_transformed['image']
        aug_height, aug_width = augmented_image.shape[:2]

        # Get augmented ball coordinates
        if len(spatial_transformed['keypoints']) > 0:
            aug_ball_x, aug_ball_y = spatial_transformed['keypoints'][0]
        else:
            # Keypoint went out of bounds - fall back to image center
            aug_ball_x, aug_ball_y = aug_width / 2, aug_height / 2

        # STAGE 2: Crop with random offset from augmented ball position
        # Add random offset to create ball position variation in the crop
        half_crop = self.crop_size // 2

        # Both training and validation use random offset for ball position variation
        max_offset = int(self.crop_size * 0.2)  # ±20% = ±25.6 pixels

        if not self.use_spatial_aug:
            # Validation: use seeded randomness (deterministic)
            np.random.seed(int(idx) + 1000)  # Different seed than spatial aug

        offset_x = np.random.randint(-max_offset, max_offset + 1)
        offset_y = np.random.randint(-max_offset, max_offset + 1)

        if not self.use_spatial_aug:
            # Reset seed
            np.random.seed(None)

        x1 = int(aug_ball_x) - half_crop + offset_x
        y1 = int(aug_ball_y) - half_crop + offset_y
        x2 = x1 + self.crop_size
        y2 = y1 + self.crop_size

        # Calculate ball position relative to crop (before any clamping)
        ball_x_in_crop = aug_ball_x - x1
        ball_y_in_crop = aug_ball_y - y1

        # Extract crop, handling out-of-bounds cases
        # Calculate what part of the desired crop is within the image
        crop_in_img_y1 = max(0, y1)
        crop_in_img_x1 = max(0, x1)
        crop_in_img_y2 = min(aug_height, y2)
        crop_in_img_x2 = min(aug_width, x2)

        # Handle case where crop is completely outside the image
        if crop_in_img_y2 <= crop_in_img_y1 or crop_in_img_x2 <= crop_in_img_x1:
            # Entire crop is outside image - create black crop
            crop = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
        else:
            # Extract the visible part
            crop = augmented_image[crop_in_img_y1:crop_in_img_y2, crop_in_img_x1:crop_in_img_x2]

            # Calculate padding needed
            pad_top = max(0, -y1)
            pad_bottom = max(0, y2 - aug_height)
            pad_left = max(0, -x1)
            pad_right = max(0, x2 - aug_width)

            # Apply padding
            if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                crop = cv2.copyMakeBorder(
                    crop, pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_CONSTANT, value=0  # Black padding
                )

        # Final safety check - ensure exactly crop_size
        assert crop.shape[0] == self.crop_size and crop.shape[1] == self.crop_size, \
            f"Crop size error: got {crop.shape}, expected ({self.crop_size}, {self.crop_size}). " \
            f"y1={y1}, y2={y2}, x1={x1}, x2={x2}, aug_size=({aug_height}, {aug_width})"

        # Default confidence is 1.0 (valid detection)
        confidence = 1.0

        # STAGE 3: Apply tearing BEFORE other augmentations (on raw uint8 image)
        # This avoids expensive denormalize/renormalize operations
        if self.use_tearing_aug and np.random.random() < self.tearing_probability:
            tearing = ImageTearing(p=1.0)
            crop = tearing(image=crop)['image']
            confidence = 0.0  # Torn image = invalid

        # STAGE 4: Apply color invariance augmentations (before appearance)
        # Color invariance is OK for difficult samples (doesn't reduce visibility)
        if self.use_color_invariance_aug:
            color_transformed = self.color_invariance_transform(image=crop)
            crop = color_transformed['image']

        # STAGE 5: Apply appearance augmentations to crop
        # Skip for difficult samples (they're already hard to detect)
        if is_difficult and self.use_appearance_aug:
            # Difficult sample: only normalize and convert to tensor (skip augmentations)
            normalize_only = self._get_normalize_transform()
            appearance_transformed = normalize_only(image=crop)
        else:
            # Normal sample: apply full appearance augmentations
            # For validation: seed for deterministic augmentation
            if not self.use_spatial_aug:
                seed_val = int(idx) + 2000  # Different seed than spatial/crop
                np.random.seed(seed_val)
                import random
                random.seed(seed_val)

            appearance_transformed = self.appearance_transform(image=crop)

        image_tensor = appearance_transformed['image']

        # Reset seed
        if not self.use_spatial_aug:
            np.random.seed(None)
            import random
            random.seed(None)

        # Normalize coordinates to [0, 1] relative to crop_size
        x_norm = np.clip(ball_x_in_crop / self.crop_size, 0.0, 1.0)
        y_norm = np.clip(ball_y_in_crop / self.crop_size, 0.0, 1.0)

        # Create target tensor: (x_norm, y_norm, confidence)
        target = torch.tensor([x_norm, y_norm, confidence], dtype=torch.float32)

        return image_tensor, target


def create_dataloaders(data_dir, batch_size=32, crop_size=128,
                       train_split=0.8, num_workers=4, use_spatial_augmentation=True,
                       use_appearance_augmentation=True, use_color_invariance_augmentation=False,
                       use_tearing_augmentation=False, tearing_probability=0.05,
                       disable_normalize=False):
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Directory containing images and labels
        batch_size: Batch size for training
        crop_size: Crop size (scale variation handled by ShiftScaleRotate augmentation)
        train_split: Fraction of data for training (rest for validation)
        num_workers: Number of workers for data loading
        use_spatial_augmentation: Enable spatial augmentations (offset, rotate, scale, shift)
        use_appearance_augmentation: Enable appearance augmentations (brightness, hue, blur, noise)
        use_color_invariance_augmentation: Enable color invariance augmentations (hue shift, channel shuffle, grayscale)
        use_tearing_augmentation: Enable tearing augmentation (simulates camera tearing, sets confidence=0)
        tearing_probability: Probability of applying tearing (default 5%)
        disable_normalize: If True, skip normalization (for GPU augmentations)

    Returns:
        train_loader, val_loader
    """
    # Load full dataset
    full_dataset = BallDetectionDataset(
        data_dir=data_dir,
        crop_size=crop_size,
        use_spatial_aug=False,  # We'll split first
        use_appearance_aug=False,
        split='train',
        disable_normalize=disable_normalize
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
        crop_size=crop_size,
        use_spatial_aug=use_spatial_augmentation,
        use_appearance_aug=use_appearance_augmentation,
        use_color_invariance_aug=use_color_invariance_augmentation,
        use_tearing_aug=use_tearing_augmentation,
        tearing_probability=tearing_probability,
        split='train',
        disable_normalize=disable_normalize
    )
    train_dataset.samples = [full_dataset.samples[i] for i in train_indices]

    val_dataset = BallDetectionDataset(
        data_dir=data_dir,
        crop_size=crop_size,
        use_spatial_aug=False,
        use_appearance_aug=False,
        use_color_invariance_aug=False,
        use_tearing_aug=False,
        split='val',
        disable_normalize=disable_normalize
    )
    val_dataset.samples = [full_dataset.samples[i] for i in val_indices]

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create dataloaders with WSL2 optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=8 if num_workers > 0 else None,  # Increased prefetch for better GPU utilization
        drop_last=True,  # Avoid small last batch for consistent performance
        multiprocessing_context='fork' if num_workers > 0 else None  # Faster on Linux
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers // 2 if num_workers > 0 else 0,  # Fewer workers for validation
        pin_memory=True,
        persistent_workers=True if num_workers > 1 else False,
        prefetch_factor=4 if num_workers > 1 else None,
        drop_last=False,  # Keep all validation samples
        multiprocessing_context='fork' if num_workers > 1 else None  # Faster on Linux
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
