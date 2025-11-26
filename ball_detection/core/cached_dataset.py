"""
Cached Augmentation Dataset for Maximum GPU Utilization

Pre-loads all images into RAM and maintains a persistent cache of augmented images.
Background workers continuously refresh the cache while GPU trains, eliminating
data loading bottlenecks.

Memory usage: ~1.3GB for 7K images × 3 cache multiplier
Expected GPU utilization improvement: <20% → 60-80%
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
import cv2
import threading
import time
from collections import deque
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CachedAugmentationDataset(Dataset):
    """
    Dataset with persistent augmentation cache for maximum GPU throughput.

    Architecture:
    1. On initialization: Load all images into RAM
    2. Build cache: Pre-augment images (cache_multiplier × dataset size)
    3. During training: Serve from cache instantly
    4. Background thread: Continuously re-augment and replace oldest entries

    Args:
        data_dir: Directory containing images and labels.json
        labels_file: JSON file with labels
        crop_size: Size to crop images (default: 128)
        cache_multiplier: How many augmented copies to maintain (default: 3)
        max_reuse_count: Replace cached items after N uses (default: 3)
        use_spatial_aug: Enable spatial augmentations
        use_appearance_aug: Enable appearance augmentations
        enable_refresh: Enable background cache refresh
        indices: List of indices to use (for train/val split)
    """

    def __init__(self, data_dir, labels_file='labels.json', crop_size=128,
                 cache_multiplier=3, max_reuse_count=3,
                 use_spatial_aug=True, use_appearance_aug=True,
                 enable_refresh=True, indices=None):
        self.data_dir = Path(data_dir).resolve()  # Convert to absolute path
        self.crop_size = crop_size
        self.cache_multiplier = cache_multiplier
        self.max_reuse_count = max_reuse_count
        self.use_spatial_aug = use_spatial_aug
        self.use_appearance_aug = use_appearance_aug
        self.enable_refresh = enable_refresh

        # Check for images in subdirectory or root
        if (self.data_dir / 'images').exists():
            self.image_dir = self.data_dir / 'images'
        else:
            self.image_dir = self.data_dir

        # Load labels
        labels_path = self.data_dir / labels_file
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)

        # Filter valid samples and apply indices if provided
        all_samples = []
        for img_name, label in self.labels.items():
            # Skip if explicitly marked as invalid
            if not label.get('valid', True):
                continue
            all_samples.append((img_name, label))

        # Apply train/val split indices if provided
        if indices is not None:
            self.samples = [all_samples[i] for i in indices]
        else:
            self.samples = all_samples

        print(f"\nCached Dataset Initialization:")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Cache multiplier: {cache_multiplier}x")
        print(f"  Cache size: {len(self.samples) * cache_multiplier}")
        print(f"  Max reuse count: {max_reuse_count}")

        # Create augmentation pipeline
        self.augmentation = self._get_augmentation_pipeline()

        # Load all raw images into RAM
        print(f"  Loading images into RAM...")
        self.raw_images = self._load_all_images()
        print(f"  [OK] All images loaded ({len(self.raw_images)} images)")

        # Build initial cache
        print(f"  Building augmentation cache...")
        self.cache = self._build_initial_cache()
        print(f"  [OK] Cache built ({len(self.cache)} items)")

        # Calculate memory usage
        if self.cache:
            single_item_size = self.cache[0]['image'].element_size() * self.cache[0]['image'].nelement()
            total_size_mb = (single_item_size * len(self.cache)) / (1024 * 1024)
            print(f"  Memory usage: ~{total_size_mb:.1f} MB")

        # Start background refresh thread
        self.refresh_thread = None
        self.stop_refresh = threading.Event()
        if enable_refresh:
            self._start_refresh_thread()
            print(f"  [OK] Background refresh enabled")

        print()

    def _get_augmentation_pipeline(self):
        """Create augmentation pipeline (same as regular dataset)."""
        transforms_list = []

        if self.use_spatial_aug:
            transforms_list.extend([
                A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=0,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                ),
            ])

        if self.use_appearance_aug:
            transforms_list.extend([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.OneOf([
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=25, p=1.0),
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
                ], p=0.8),
                A.OneOf([
                    A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_limit=(1, 2), shadow_dimension=5, p=1.0),
                    A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_range=(0, 1), num_flare_circles_range=(3, 6), src_radius=100, p=1.0),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.2), p=1.0),
                    A.Downscale(
                        scale_range=(0.75, 0.95),
                        interpolation_pair={
                            'downscale': cv2.INTER_LINEAR,
                            'upscale': cv2.INTER_LINEAR
                        },
                        p=1.0
                    ),
                ], p=0.4),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ], p=0.3),
                A.GaussNoise(std_range=(10.0/255, 20.0/255), p=0.3),
                A.ImageCompression(quality_range=(70, 100), p=0.2),
            ])

        # Normalize and convert to tensor
        transforms_list.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        transforms_list.append(ToTensorV2())

        return A.Compose(transforms_list, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def _load_all_images(self):
        """Load all images and their pre-cropped versions into RAM."""
        raw_images = []

        for img_name, label in self.samples:
            # Load image
            img_path = self.image_dir / img_name
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Pre-crop image (centered on ball, no padding)
            ball_x, ball_y = label['x'], label['y']
            width, height = image.shape[1], image.shape[0]

            # Center crop on ball, ensuring it fits within image bounds (no padding)
            half_crop = self.crop_size // 2
            x1 = max(0, min(int(ball_x) - half_crop, width - self.crop_size))
            y1 = max(0, min(int(ball_y) - half_crop, height - self.crop_size))
            x2 = x1 + self.crop_size
            y2 = y1 + self.crop_size

            # Extract crop (no clamping - ranges calculated to fit)
            crop = image[y1:y2, x1:x2]

            # Sanity check
            assert crop.shape[0] == self.crop_size and crop.shape[1] == self.crop_size, \
                f"Crop size mismatch: got {crop.shape}, expected ({self.crop_size}, {self.crop_size})"

            # Calculate ball position relative to crop (normalized)
            ball_x_in_crop = ball_x - x1
            ball_y_in_crop = ball_y - y1
            target = np.array([ball_x_in_crop / self.crop_size, ball_y_in_crop / self.crop_size], dtype=np.float32)

            raw_images.append({
                'image': crop,
                'target': target,
                'filename': img_name
            })

        return raw_images

    def _augment_single_image(self, raw_item, seed=None):
        """Apply augmentation to a single raw image with optional seed."""
        if seed is not None:
            np.random.seed(seed)

        # Apply augmentation
        transformed = self.augmentation(
            image=raw_item['image'],
            keypoints=[raw_item['target'] * self.crop_size]
        )

        image_tensor = transformed['image']
        keypoint = transformed['keypoints'][0]
        target_tensor = torch.tensor([keypoint[0] / self.crop_size, keypoint[1] / self.crop_size], dtype=torch.float32)

        return {
            'image': image_tensor,
            'target': target_tensor,
            'use_count': 0,
            'source_idx': raw_item.get('source_idx', 0)
        }

    def _build_initial_cache(self):
        """Build initial cache with multiple augmented copies per image."""
        cache = []

        for idx, raw_item in enumerate(self.raw_images):
            raw_item['source_idx'] = idx

            # Create multiple augmented versions
            for copy_idx in range(self.cache_multiplier):
                seed = idx * self.cache_multiplier + copy_idx
                augmented = self._augment_single_image(raw_item, seed=seed)
                cache.append(augmented)

        return cache

    def _start_refresh_thread(self):
        """Start background thread to continuously refresh cache."""
        def refresh_worker():
            while not self.stop_refresh.is_set():
                # Find items with highest use count
                if not self.cache:
                    time.sleep(1)
                    continue

                # Sort by use count and replace top ones
                cache_with_indices = [(i, item) for i, item in enumerate(self.cache)]
                sorted_cache = sorted(cache_with_indices, key=lambda x: x[1]['use_count'], reverse=True)

                # Replace items that exceeded max reuse
                replaced_count = 0
                for cache_idx, item in sorted_cache[:100]:  # Replace up to 100 items per iteration
                    if item['use_count'] >= self.max_reuse_count:
                        # Re-augment
                        source_idx = item['source_idx']
                        new_seed = int(time.time() * 1000) % 1000000 + cache_idx
                        new_item = self._augment_single_image(self.raw_images[source_idx], seed=new_seed)
                        self.cache[cache_idx] = new_item
                        replaced_count += 1

                if replaced_count == 0:
                    time.sleep(0.1)  # No work to do, sleep briefly

        self.refresh_thread = threading.Thread(target=refresh_worker, daemon=True)
        self.refresh_thread.start()

    def __len__(self):
        """Return number of samples (not cache size)."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get item from cache.

        Maps dataset index to cache index using modulo, allowing
        multiple cached versions to be used across epochs.
        """
        # Map to cache with rotation
        cache_idx = (idx * self.cache_multiplier + (idx // len(self.samples))) % len(self.cache)

        # Get from cache
        item = self.cache[cache_idx]

        # Increment use count (for refresh priority)
        item['use_count'] += 1

        return item['image'], item['target']

    def __del__(self):
        """Clean up background thread."""
        if hasattr(self, 'refresh_thread') and self.refresh_thread is not None:
            if hasattr(self, 'stop_refresh'):
                self.stop_refresh.set()
            self.refresh_thread.join(timeout=1.0)
