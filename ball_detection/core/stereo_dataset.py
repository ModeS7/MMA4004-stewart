"""
Stereo Dataset for Full-Frame Ball Detection Training

Handles loading stereo image pairs (left + right) and creates 6-channel input tensors.
Output: 6 values (x_left, y_left, conf_left, x_right, y_right, conf_right)
"""

import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class StereoFullFrameDataset(Dataset):
    """
    Dataset for stereo full-frame ball detection.

    Input: 6-channel tensor (left RGB + right RGB)
    Output: 6 values (x_l, y_l, conf_l, x_r, y_r, conf_r) normalized to [0,1]

    Expected data format:
        data_dir/
            images/
                frame_000000_left.jpg
                frame_000000_right.jpg
                ...
            stereo_labels.json

    stereo_labels.json format:
        {
            "frame_000000": {
                "x_left": 640.5, "y_left": 360.2, "valid_left": true,
                "x_right": 580.3, "y_right": 358.1, "valid_right": true
            },
            ...
        }
    """

    def __init__(self, data_dir, labels_file='stereo_labels.json',
                 image_height=720, image_width=1280,
                 use_augmentation=True, split='train'):
        """
        Initialize stereo dataset.

        Includes ALL frames - model must learn to detect when ball is NOT present.
        - Both cameras detect ball → conf_l=1, conf_r=1
        - Only left detects → conf_l=1, conf_r=0
        - Only right detects → conf_l=0, conf_r=1
        - Neither detects → conf_l=0, conf_r=0

        Args:
            data_dir: Directory containing images and stereo_labels.json
            labels_file: Name of stereo labels JSON file
            image_height: Expected image height (720 for ZED)
            image_width: Expected image width (1280 for ZED)
            use_augmentation: Enable augmentations for training
            split: 'train' or 'val'
        """
        self.data_dir = Path(data_dir)
        self.image_height = image_height
        self.image_width = image_width
        self.use_augmentation = use_augmentation and (split == 'train')

        # Check for images directory
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

        # Build sample list - include ALL frames
        self.samples = []
        stats = {'both': 0, 'left_only': 0, 'right_only': 0, 'neither': 0}

        for frame_id, label in self.labels.items():
            # Check if images exist
            left_path = self.image_dir / f"{frame_id}_left.jpg"
            right_path = self.image_dir / f"{frame_id}_right.jpg"

            if not left_path.exists() or not right_path.exists():
                # Try .png extension
                left_path = self.image_dir / f"{frame_id}_left.png"
                right_path = self.image_dir / f"{frame_id}_right.png"
                if not left_path.exists() or not right_path.exists():
                    continue

            self.samples.append({
                'frame_id': frame_id,
                'left_path': str(left_path),
                'right_path': str(right_path),
                'label': label
            })

            # Track stats
            valid_left = label.get('valid_left', False)
            valid_right = label.get('valid_right', False)
            if valid_left and valid_right:
                stats['both'] += 1
            elif valid_left:
                stats['left_only'] += 1
            elif valid_right:
                stats['right_only'] += 1
            else:
                stats['neither'] += 1

        if len(self.samples) == 0:
            raise ValueError(f"No valid stereo samples found in {data_dir}")

        print(f"Loaded {len(self.samples)} stereo samples from {data_dir}")
        print(f"  Both: {stats['both']}, Left only: {stats['left_only']}, Right only: {stats['right_only']}, Neither: {stats['neither']}")

        # Setup augmentation
        self.transform = self._get_transforms()

    def _get_transforms(self):
        """Create augmentation pipeline for stereo images."""
        transforms_list = []

        if self.use_augmentation:
            # Stereo-safe augmentations (same transform applied to both images)
            transforms_list.extend([
                # Brightness/contrast (applied identically to both cameras)
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                # Color jitter
                A.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05,
                    p=0.5
                ),
                # Mild blur
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ], p=0.2),
                # Noise
                A.GaussNoise(std_range=(5.0/255, 15.0/255), p=0.2),
            ])

        # Always normalize
        transforms_list.append(
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        transforms_list.append(ToTensorV2())

        return A.Compose(transforms_list, additional_targets={'image_right': 'image'})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get stereo training sample.

        Returns:
            image: Tensor of shape (6, H, W) - concatenated left and right
            target: Tensor of shape (6,) - (x_l, y_l, conf_l, x_r, y_r, conf_r)
        """
        sample = self.samples[idx]
        label = sample['label']

        # Load images
        left_img = cv2.imread(sample['left_path'], cv2.IMREAD_COLOR)
        right_img = cv2.imread(sample['right_path'], cv2.IMREAD_COLOR)

        if left_img is None:
            raise ValueError(f"Failed to load left image: {sample['left_path']}")
        if right_img is None:
            raise ValueError(f"Failed to load right image: {sample['right_path']}")

        # Convert BGR to RGB
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

        # Apply same augmentation to both images
        transformed = self.transform(image=left_img, image_right=right_img)
        left_tensor = transformed['image']  # (3, H, W)
        right_tensor = transformed['image_right']  # (3, H, W)

        # Concatenate to 6-channel tensor: (6, H, W)
        stereo_tensor = torch.cat([left_tensor, right_tensor], dim=0)

        # Extract coordinates and normalize to [0, 1]
        x_left = label.get('x_left', -1) / self.image_width
        y_left = label.get('y_left', -1) / self.image_height
        conf_left = 1.0 if label.get('valid_left', False) else 0.0

        x_right = label.get('x_right', -1) / self.image_width
        y_right = label.get('y_right', -1) / self.image_height
        conf_right = 1.0 if label.get('valid_right', False) else 0.0

        # Clamp coordinates to [0, 1]
        x_left = np.clip(x_left, 0.0, 1.0)
        y_left = np.clip(y_left, 0.0, 1.0)
        x_right = np.clip(x_right, 0.0, 1.0)
        y_right = np.clip(y_right, 0.0, 1.0)

        # Target: (x_l, y_l, conf_l, x_r, y_r, conf_r)
        target = torch.tensor([x_left, y_left, conf_left, x_right, y_right, conf_right],
                              dtype=torch.float32)

        return stereo_tensor, target


class MultiStereoDataset(Dataset):
    """
    Combines multiple stereo datasets from different directories.
    Useful for merging old_labels and new_labels.
    """

    def __init__(self, data_dirs, **kwargs):
        """
        Initialize multi-source stereo dataset.

        Args:
            data_dirs: List of directories containing stereo data
            **kwargs: Arguments passed to StereoFullFrameDataset
        """
        self.datasets = []
        self.cumulative_lengths = [0]

        for data_dir in data_dirs:
            try:
                dataset = StereoFullFrameDataset(data_dir, **kwargs)
                self.datasets.append(dataset)
                self.cumulative_lengths.append(
                    self.cumulative_lengths[-1] + len(dataset)
                )
            except Exception as e:
                print(f"Warning: Failed to load {data_dir}: {e}")

        if not self.datasets:
            raise ValueError("No valid datasets found")

        print(f"Combined {len(self.datasets)} datasets, total {len(self)} samples")

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        for i, (start, end) in enumerate(zip(self.cumulative_lengths[:-1],
                                             self.cumulative_lengths[1:])):
            if start <= idx < end:
                return self.datasets[i][idx - start]
        raise IndexError(f"Index {idx} out of range")


def create_stereo_dataloaders(data_dirs, batch_size=8, train_split=0.8,
                               num_workers=4, **kwargs):
    """
    Create train and validation dataloaders for stereo training.

    Args:
        data_dirs: List of directories or single directory
        batch_size: Batch size
        train_split: Fraction for training
        num_workers: Number of data loading workers
        **kwargs: Additional arguments for dataset

    Returns:
        train_loader, val_loader
    """
    if isinstance(data_dirs, (str, Path)):
        data_dirs = [data_dirs]

    # Create full dataset
    full_dataset = MultiStereoDataset(
        data_dirs,
        use_augmentation=False,
        split='train',
        **kwargs
    )

    # Split indices
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)

    # Shuffle indices for random split
    indices = np.random.permutation(dataset_size)
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:].tolist()

    # Create train dataset with augmentation
    train_dataset = MultiStereoDataset(
        data_dirs,
        use_augmentation=True,
        split='train',
        **kwargs
    )

    # Create val dataset without augmentation
    val_dataset = MultiStereoDataset(
        data_dirs,
        use_augmentation=False,
        split='val',
        **kwargs
    )

    # Create subset samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers // 2 if num_workers > 0 else 0,
        pin_memory=True,
        drop_last=False
    )

    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")

    return train_loader, val_loader


class TemporalStereoDataset(Dataset):
    """
    Temporal dataset for training with frame sequences.

    Features:
    - Non-overlapping sequences within each epoch
    - Random offset per epoch (call reshuffle_epoch())
    - Same augmentation applied to all frames in sequence
    - Optional frame skipping for robustness
    - Stereo-aware tearing (applied to concatenated stereo feed)
    """

    def __init__(self, data_dir, labels_file='stereo_labels.json',
                 image_height=720, image_width=1280,
                 sequence_length=30, use_augmentation=True,
                 frame_skip_prob=0.0, reverse_prob=0.5,
                 tearing_prob=0.01):
        """
        Args:
            data_dir: Directory with images/ and stereo_labels.json
            sequence_length: Frames per sequence (30 = 30 stereo pairs)
            frame_skip_prob: Probability of skipping each frame (0.1 = 10%)
            reverse_prob: Probability of reversing sequence (0.5 = 50%)
            tearing_prob: Probability of tearing per frame (0.01 = 1%)
        """
        self.data_dir = Path(data_dir)
        self.image_height = image_height
        self.image_width = image_width
        self.sequence_length = sequence_length
        self.use_augmentation = use_augmentation
        self.frame_skip_prob = frame_skip_prob
        self.reverse_prob = reverse_prob
        self.tearing_prob = tearing_prob

        if (self.data_dir / 'images').exists():
            self.image_dir = self.data_dir / 'images'
        else:
            self.image_dir = self.data_dir

        labels_path = self.data_dir / labels_file
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)

        # Build sorted frame list
        self.frames = []
        for frame_id, label in self.labels.items():
            left_path = self.image_dir / f"{frame_id}_left.jpg"
            right_path = self.image_dir / f"{frame_id}_right.jpg"

            if not left_path.exists() or not right_path.exists():
                left_path = self.image_dir / f"{frame_id}_left.png"
                right_path = self.image_dir / f"{frame_id}_right.png"
                if not left_path.exists() or not right_path.exists():
                    continue

            self.frames.append({
                'frame_id': frame_id,
                'left_path': str(left_path),
                'right_path': str(right_path),
                'label': label
            })

        # Sort by frame_id for temporal order
        self.frames.sort(key=lambda x: x['frame_id'])

        # Find continuous segments (same video prefix)
        self.segments = self._find_continuous_segments()

        print(f"Loaded {len(self.frames)} frames from {data_dir}")
        print(f"  Found {len(self.segments)} continuous segments")

        # Initialize sequence starts for first epoch
        self.sequence_starts = []
        self.reshuffle_epoch()

        self.transform = self._get_transforms()

    def _parse_frame_id(self, frame_id):
        """Extract prefix and number from frame_id."""
        parts = frame_id.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0], int(parts[1])
        return frame_id, 0

    def _find_continuous_segments(self):
        """Find continuous frame segments (no gaps, same video)."""
        if not self.frames:
            return []

        segments = []
        seg_start = 0

        for i in range(1, len(self.frames)):
            prev_prefix, prev_num = self._parse_frame_id(self.frames[i-1]['frame_id'])
            curr_prefix, curr_num = self._parse_frame_id(self.frames[i]['frame_id'])

            # Check if continuous (same prefix, consecutive numbers)
            if curr_prefix != prev_prefix or curr_num != prev_num + 1:
                # End of segment
                if i - seg_start >= self.sequence_length:
                    segments.append((seg_start, i))
                seg_start = i

        # Last segment
        if len(self.frames) - seg_start >= self.sequence_length:
            segments.append((seg_start, len(self.frames)))

        return segments

    def reshuffle_epoch(self):
        """Generate new sequence starts with random offset. Call at start of each epoch."""
        self.sequence_starts = []

        for seg_start, seg_end in self.segments:
            seg_length = seg_end - seg_start

            # Random offset (0 to sequence_length-1)
            max_offset = min(self.sequence_length, seg_length - self.sequence_length)
            if max_offset <= 0:
                offset = 0
            else:
                offset = np.random.randint(0, max_offset)

            # Generate non-overlapping starts within this segment
            pos = seg_start + offset
            while pos + self.sequence_length <= seg_end:
                self.sequence_starts.append(pos)
                pos += self.sequence_length

        # Shuffle all sequences
        np.random.shuffle(self.sequence_starts)

    def _get_transforms(self):
        """Full augmentation pipeline matching previous training setup."""
        transforms_list = []

        if self.use_augmentation:
            # Spatial augmentations (applied identically to both cameras)
            transforms_list.append(
                A.ShiftScaleRotate(
                    shift_limit=0.1,      # ±10% shift
                    scale_limit=0.1,      # ±10% scale
                    rotate_limit=10,      # ±10° rotation
                    p=1.0,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0  # Black padding
                )
            )

            # Color invariance augmentations (train model to work with any color ball)
            transforms_list.extend([
                A.HueSaturationValue(
                    hue_shift_limit=180,  # Full spectrum rotation
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.7
                ),
                A.ChannelShuffle(p=0.3),
                A.ToGray(p=0.2),
            ])

            # Appearance augmentations
            transforms_list.extend([
                # Lighting variations
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.7
                ),
                # Color temperature / white balance
                A.OneOf([
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=25, p=1.0),
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
                ], p=0.8),
                # Heavy augmentations (one of)
                A.OneOf([
                    A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_limit=(1, 2), shadow_dimension=5, p=1.0),
                    A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_range=(0, 1), num_flare_circles_range=(3, 6), src_radius=100, p=1.0),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.2), p=1.0),
                    A.Downscale(scale_range=(0.75, 0.95), interpolation_pair={'downscale': cv2.INTER_LINEAR, 'upscale': cv2.INTER_LINEAR}, p=1.0),
                ], p=0.4),
                # Blur variations
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ], p=0.3),
                # Noise
                A.GaussNoise(std_range=(10.0/255, 20.0/255), p=0.3),
                # Quality degradation
                A.ImageCompression(quality_range=(70, 100), p=0.2),
            ])

        # Use ReplayCompose to allow replaying exact same augmentation across sequence
        # NOTE: Normalize and ToTensor are separate so we can apply tearing after augmentation
        self.augment_transform = A.ReplayCompose(
            transforms_list,
            additional_targets={'image_right': 'image'},
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, label_fields=['keypoint_labels'])
        )

        # Final normalization (applied after tearing)
        self.normalize_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'image_right': 'image'})

        return self.augment_transform

    def _apply_stereo_tearing(self, left_img, right_img):
        """
        Apply tearing to concatenated stereo feed (simulates real camera tearing).

        In real stereo systems, left and right frames are concatenated into one
        video stream, so tearing affects both identically at the same positions.

        Returns:
            left_img, right_img: Torn images (or originals if no tearing)
            is_torn: True if tearing was applied
        """
        if not self.use_augmentation or np.random.random() >= self.tearing_prob:
            return left_img, right_img, False

        h, w = left_img.shape[:2]

        # Concatenate horizontally (simulating real stereo feed)
        combined = np.concatenate([left_img, right_img], axis=1)  # (H, W*2, 3)
        combined_w = combined.shape[1]

        # Random direction: 50% horizontal, 50% vertical
        direction = np.random.choice(['horizontal', 'vertical'])

        if direction == 'horizontal':
            # Horizontal tear (row-wise split with horizontal shift)
            tear_y = np.random.randint(int(h * 0.1), int(h * 0.9))
            shift = np.random.randint(100, 300)
            if np.random.random() > 0.5:
                shift = -shift
            combined[tear_y:] = np.roll(combined[tear_y:], shift, axis=1)
        else:
            # Vertical tear (column-wise split with vertical shift)
            tear_x = np.random.randint(int(combined_w * 0.1), int(combined_w * 0.9))
            shift = np.random.randint(50, 150)
            if np.random.random() > 0.5:
                shift = -shift
            combined[:, tear_x:] = np.roll(combined[:, tear_x:], shift, axis=0)

        # Split back into left and right
        left_torn = combined[:, :w]
        right_torn = combined[:, w:]

        return left_torn, right_torn, True

    def __len__(self):
        return len(self.sequence_starts)

    def __getitem__(self, idx):
        """
        Returns:
            frames: (seq_len, 6, H, W) tensor
            targets: (seq_len, 6) tensor
        """
        start_idx = self.sequence_starts[idx]

        frames = []
        targets = []
        replay_data = None  # Will store augmentation params from first frame

        for t in range(self.sequence_length):
            # Frame skipping for robustness
            if self.frame_skip_prob > 0 and np.random.random() < self.frame_skip_prob:
                continue  # Skip this frame

            frame_data = self.frames[start_idx + t]

            # Load images
            left_img = cv2.imread(frame_data['left_path'], cv2.IMREAD_COLOR)
            right_img = cv2.imread(frame_data['right_path'], cv2.IMREAD_COLOR)

            if left_img is None or right_img is None:
                continue  # Skip if load fails

            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

            # Get original coordinates
            label = frame_data['label']
            x_left_orig = label.get('x_left', -1)
            y_left_orig = label.get('y_left', -1)
            x_right_orig = label.get('x_right', -1)
            y_right_orig = label.get('y_right', -1)
            valid_left = label.get('valid_left', False)
            valid_right = label.get('valid_right', False)

            # Build keypoints list (only valid ones)
            keypoints = []
            keypoint_labels = []
            if valid_left and x_left_orig >= 0 and y_left_orig >= 0:
                keypoints.append((x_left_orig, y_left_orig))
                keypoint_labels.append('left')
            if valid_right and x_right_orig >= 0 and y_right_orig >= 0:
                keypoints.append((x_right_orig, y_right_orig))
                keypoint_labels.append('right')

            # Apply augmentation - first frame generates params, rest replay them
            if replay_data is None:
                # First frame: run transform and capture replay data
                transformed = self.augment_transform(
                    image=left_img,
                    image_right=right_img,
                    keypoints=keypoints,
                    keypoint_labels=keypoint_labels
                )
                replay_data = transformed['replay']
            else:
                # Subsequent frames: replay exact same augmentation
                transformed = A.ReplayCompose.replay(
                    replay_data,
                    image=left_img,
                    image_right=right_img,
                    keypoints=keypoints,
                    keypoint_labels=keypoint_labels
                )

            left_aug = transformed['image']
            right_aug = transformed['image_right']

            # Apply stereo tearing AFTER augmentation (on uint8 images)
            left_aug, right_aug, is_torn = self._apply_stereo_tearing(left_aug, right_aug)

            # Final normalization and convert to tensor
            normalized = self.normalize_transform(image=left_aug, image_right=right_aug)
            left_tensor = normalized['image']
            right_tensor = normalized['image_right']

            stereo_tensor = torch.cat([left_tensor, right_tensor], dim=0)
            frames.append(stereo_tensor)

            # Extract transformed keypoints
            transformed_kps = transformed['keypoints']
            transformed_labels = transformed['keypoint_labels']

            # Default values (invalid)
            x_left, y_left, conf_left = 0.0, 0.0, 0.0
            x_right, y_right, conf_right = 0.0, 0.0, 0.0

            for kp, lbl in zip(transformed_kps, transformed_labels):
                kp_x, kp_y = kp[0], kp[1]
                # Normalize to [0, 1] and check bounds
                kp_x_norm = kp_x / self.image_width
                kp_y_norm = kp_y / self.image_height

                # Only valid if within image bounds
                if 0 <= kp_x_norm <= 1 and 0 <= kp_y_norm <= 1:
                    if lbl == 'left':
                        x_left = kp_x_norm
                        y_left = kp_y_norm
                        conf_left = 1.0
                    elif lbl == 'right':
                        x_right = kp_x_norm
                        y_right = kp_y_norm
                        conf_right = 1.0

            # If frame is torn, set both confidences to 0 (invalid detection)
            if is_torn:
                conf_left = 0.0
                conf_right = 0.0

            target = torch.tensor([x_left, y_left, conf_left, x_right, y_right, conf_right],
                                  dtype=torch.float32)
            targets.append(target)

        # Stack into tensors
        frames = torch.stack(frames, dim=0)  # (T, 6, H, W)
        targets = torch.stack(targets, dim=0)  # (T, 6)

        # 50% chance to reverse the sequence (only during training with augmentation)
        if self.use_augmentation and self.reverse_prob > 0 and np.random.random() < self.reverse_prob:
            frames = torch.flip(frames, dims=[0])
            targets = torch.flip(targets, dims=[0])

        return frames, targets


class MultiTemporalDataset(Dataset):
    """Combines multiple temporal datasets."""

    def __init__(self, data_dirs, **kwargs):
        self.datasets = []
        self.cumulative_lengths = [0]

        for data_dir in data_dirs:
            try:
                ds = TemporalStereoDataset(data_dir, **kwargs)
                if len(ds) > 0:
                    self.datasets.append(ds)
                    self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(ds))
            except Exception as e:
                print(f"Warning: Failed to load {data_dir}: {e}")

        print(f"Combined {len(self.datasets)} datasets, total {len(self)} sequences")

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        for i, (start, end) in enumerate(zip(self.cumulative_lengths[:-1],
                                             self.cumulative_lengths[1:])):
            if start <= idx < end:
                return self.datasets[i][idx - start]
        raise IndexError(f"Index {idx} out of range")

    def reshuffle_epoch(self):
        """Reshuffle all datasets for new epoch."""
        for ds in self.datasets:
            ds.reshuffle_epoch()
        # Rebuild cumulative lengths (may change slightly due to random offsets)
        self.cumulative_lengths = [0]
        for ds in self.datasets:
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(ds))


if __name__ == "__main__":
    # Test stereo dataset
    import sys

    data_dir = [
        "./ball_detection/data/full_dataset"
    ]

    print("Testing stereo dataset...")

    train_loader, val_loader = create_stereo_dataloaders(
        data_dir,
        batch_size=4,
        num_workers=0
    )

    batch = next(iter(train_loader))
    images, targets = batch

    print(f"\nBatch images shape: {images.shape}")
    print(f"Batch targets shape: {targets.shape}")
    print(f"\nFirst target: {targets[0].numpy()}")

    print("\nStereo dataset test successful!")
