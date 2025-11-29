"""
View Augmentation Examples

Loads random images from the dataset and shows how augmentations affect them.
Useful for debugging and tuning augmentation parameters.

Usage:
    python -m ball_detection.tools.view_a
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ball_detection.core.dataset import BallDetectionDataset


class FullFrameDataset:
    """Simple full-frame dataset for visualization (no cropping)."""

    def __init__(self, data_dir, width=1280, height=720,
                 use_spatial_aug=True, use_appearance_aug=True,
                 use_color_invariance_aug=False, use_tearing_aug=False,
                 tearing_probability=0.05, split='train'):
        self.data_dir = Path(data_dir)
        self.width = width
        self.height = height
        self.use_spatial_aug = use_spatial_aug and (split == 'train')
        self.use_appearance_aug = use_appearance_aug and (split == 'train')
        self.use_color_invariance_aug = use_color_invariance_aug and (split == 'train')
        self.use_tearing_aug = use_tearing_aug and (split == 'train')
        self.tearing_probability = tearing_probability

        # Find images
        if (self.data_dir / 'images').exists():
            self.image_dir = self.data_dir / 'images'
        else:
            self.image_dir = self.data_dir

        # Load labels
        labels_path = self.data_dir / 'labels.json'
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)

        self.samples = []
        for img_name, label in self.labels.items():
            if not label.get('valid', True):
                continue
            img_path = self.image_dir / img_name
            if img_path.exists():
                self.samples.append((str(img_path), label))

        # Build augmentation pipeline
        self.transform = self._build_transform()

    def _build_transform(self):
        transforms = []

        if self.use_spatial_aug:
            transforms.append(A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.7,
                border_mode=cv2.BORDER_CONSTANT
            ))

        if self.use_color_invariance_aug:
            transforms.extend([
                A.HueSaturationValue(hue_shift_limit=180, sat_shift_limit=30, val_shift_limit=20, p=0.7),
                A.ChannelShuffle(p=0.3),
                A.ToGray(p=0.2),
            ])

        if self.use_appearance_aug:
            transforms.extend([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ], p=0.3),
                A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
            ])

        transforms.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        return A.Compose(transforms, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def _apply_tearing(self, img):
        """Apply realistic tearing effect - horizontal or vertical frame splits."""
        h, w = img.shape[:2]
        result = img.copy()

        # 50% horizontal, 50% vertical (never both)
        if np.random.random() < 0.5:
            result = self._apply_horizontal_tears(result)
        else:
            result = self._apply_vertical_tears(result)

        return result

    def _apply_horizontal_tears(self, img):
        """Apply 1-2 horizontal tears (row-wise splits with horizontal shifts)."""
        h, w = img.shape[:2]
        result = img.copy()

        num_tears = np.random.choice([1, 1, 1, 2])

        if num_tears == 1:
            tear_y = np.random.randint(int(h * 0.1), int(h * 0.9))
            tear_positions = [tear_y]
        else:
            tear1 = np.random.randint(int(h * 0.15), int(h * 0.45))
            tear2 = np.random.randint(int(h * 0.55), int(h * 0.85))
            tear_positions = [tear1, tear2]

        prev_y = 0
        for i, tear_y in enumerate(tear_positions + [h]):
            if i > 0:
                shift = np.random.randint(100, 300)
                if np.random.random() > 0.5:
                    shift = -shift
                result[prev_y:tear_y] = np.roll(result[prev_y:tear_y], shift, axis=1)
            prev_y = tear_y

        return result

    def _apply_vertical_tears(self, img):
        """Apply 1-2 vertical tears (column-wise splits with vertical shifts)."""
        h, w = img.shape[:2]
        result = img.copy()

        num_tears = np.random.choice([1, 1, 1, 2])

        if num_tears == 1:
            tear_x = np.random.randint(int(w * 0.1), int(w * 0.9))
            tear_positions = [tear_x]
        else:
            tear1 = np.random.randint(int(w * 0.15), int(w * 0.45))
            tear2 = np.random.randint(int(w * 0.55), int(w * 0.85))
            tear_positions = [tear1, tear2]

        prev_x = 0
        for i, tear_x in enumerate(tear_positions + [w]):
            if i > 0:
                shift = np.random.randint(50, 150)
                if np.random.random() > 0.5:
                    shift = -shift
                result[:, prev_x:tear_x] = np.roll(result[:, prev_x:tear_x], shift, axis=0)
            prev_x = tear_x

        return result

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load and resize image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height))

        # Get ball coordinates and scale to new size
        orig_h, orig_w = cv2.imread(img_path).shape[:2]
        ball_x = label['x'] * self.width / orig_w
        ball_y = label['y'] * self.height / orig_h

        # Apply augmentations
        transformed = self.transform(image=image, keypoints=[(ball_x, ball_y)])
        image_tensor = transformed['image']

        if len(transformed['keypoints']) > 0:
            aug_x, aug_y = transformed['keypoints'][0]
        else:
            aug_x, aug_y = self.width / 2, self.height / 2

        # Normalize coordinates
        x_norm = np.clip(aug_x / self.width, 0.0, 1.0)
        y_norm = np.clip(aug_y / self.height, 0.0, 1.0)
        confidence = 1.0

        # Apply tearing
        if self.use_tearing_aug and np.random.random() < self.tearing_probability:
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = image_tensor.numpy().transpose(1, 2, 0)
            img_np = (img_np * std + mean) * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

            # Apply tearing
            img_np = self._apply_tearing(img_np)

            # Re-normalize
            img_np = img_np.astype(np.float32) / 255.0
            img_np = (img_np - mean) / std
            image_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
            confidence = 0.0

        target = torch.tensor([x_norm, y_norm, confidence], dtype=torch.float32)
        return image_tensor, target


# ============================================================
# SETTINGS - Edit these
# ============================================================
DATA_DIR = "./ball_detection/data/full_dataset/training_data_full"
FULLFRAME_MODE = True  # True for 1280x720 full frames, False for 128x128 crops
CROP_SIZE = 128
FULLFRAME_WIDTH = 1280
FULLFRAME_HEIGHT = 720
MODE = "train"  # "train" or "val"
USE_SPATIAL_AUGMENTATION = True
USE_APPEARANCE_AUGMENTATION = True
USE_COLOR_INVARIANCE_AUGMENTATION = True
USE_TEARING_AUGMENTATION = True
TEARING_PROBABILITY = 0.5  # High for visualization (normally 0.01-0.05)
NUM_SAMPLES = 3
NUM_AUGMENTATIONS = 4
FIGSIZE = (20, 12)
SAVE_OUTPUT = True
OUTPUT_PATH = "./ball_detection/augmentation_examples.png"
# ============================================================


def denormalize(img):
    """Denormalize image from ImageNet stats back to [0, 255]."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = img.transpose(1, 2, 0)  # CHW -> HWC
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    return img


def draw_keypoint(img, x, y, color=(0, 255, 0), fullframe=False):
    """Draw keypoint on image."""
    img = img.copy()
    x_pixel = int(x * img.shape[1])
    y_pixel = int(y * img.shape[0])

    if fullframe:
        radius = 8
        crosshair = 20
        thickness = 2
    else:
        radius = 2
        crosshair = 5
        thickness = 1

    cv2.circle(img, (x_pixel, y_pixel), radius, color, -1)
    cv2.line(img, (x_pixel - crosshair, y_pixel), (x_pixel + crosshair, y_pixel), color, thickness)
    cv2.line(img, (x_pixel, y_pixel - crosshair), (x_pixel, y_pixel + crosshair), color, thickness)
    return img


def main():
    """Visualize augmentations."""
    print("=" * 60)
    print("AUGMENTATION VISUALIZATION")
    print("=" * 60)
    print(f"Data: {DATA_DIR}")
    print(f"Mode: {MODE.upper()}")
    if FULLFRAME_MODE:
        print(f"Full frame: {FULLFRAME_WIDTH}x{FULLFRAME_HEIGHT}")
    else:
        print(f"Crop size: {CROP_SIZE}x{CROP_SIZE}")

    # Validation mode: force no spatial augmentation
    if MODE == "val":
        spatial_aug = False
        appearance_aug = USE_APPEARANCE_AUGMENTATION
        color_invariance_aug = False
        tearing_aug = False
        print(f"Spatial augmentation: Disabled (validation mode)")
        print(f"Appearance augmentation: {'Enabled' if appearance_aug else 'Disabled'}")
        print(f"Color invariance augmentation: Disabled (validation mode)")
        print(f"Tearing augmentation: Disabled (validation mode)")
    else:
        spatial_aug = USE_SPATIAL_AUGMENTATION
        appearance_aug = USE_APPEARANCE_AUGMENTATION
        color_invariance_aug = USE_COLOR_INVARIANCE_AUGMENTATION
        tearing_aug = USE_TEARING_AUGMENTATION
        print(f"Spatial augmentation: {'Enabled' if spatial_aug else 'Disabled'}")
        print(f"Appearance augmentation: {'Enabled' if appearance_aug else 'Disabled'}")
        print(f"Color invariance augmentation: {'Enabled' if color_invariance_aug else 'Disabled'}")
        print(f"Tearing augmentation: {'Enabled' if tearing_aug else 'Disabled'}" + (f" ({TEARING_PROBABILITY*100:.0f}%)" if tearing_aug else ""))

    print(f"Samples: {NUM_SAMPLES}")
    print(f"Augmentations per sample: {NUM_AUGMENTATIONS}")
    print("=" * 60)
    print()

    # Create datasets based on mode
    if FULLFRAME_MODE:
        dataset_no_aug = FullFrameDataset(
            data_dir=DATA_DIR,
            width=FULLFRAME_WIDTH,
            height=FULLFRAME_HEIGHT,
            use_spatial_aug=False,
            use_appearance_aug=False,
            use_color_invariance_aug=False,
            use_tearing_aug=False,
            split=MODE
        )

        dataset_with_aug = FullFrameDataset(
            data_dir=DATA_DIR,
            width=FULLFRAME_WIDTH,
            height=FULLFRAME_HEIGHT,
            use_spatial_aug=spatial_aug,
            use_appearance_aug=appearance_aug,
            use_color_invariance_aug=color_invariance_aug,
            use_tearing_aug=tearing_aug,
            tearing_probability=TEARING_PROBABILITY,
            split=MODE
        )
    else:
        dataset_no_aug = BallDetectionDataset(
            data_dir=DATA_DIR,
            crop_size=CROP_SIZE,
            use_spatial_aug=False,
            use_appearance_aug=False,
            split=MODE
        )

        dataset_with_aug = BallDetectionDataset(
            data_dir=DATA_DIR,
            crop_size=CROP_SIZE,
            use_spatial_aug=spatial_aug,
            use_appearance_aug=appearance_aug,
            use_color_invariance_aug=color_invariance_aug,
            use_tearing_aug=tearing_aug,
            tearing_probability=TEARING_PROBABILITY,
            split=MODE
        )

    print(f"Dataset loaded: {len(dataset_no_aug)} samples")
    print()

    # Select random samples
    np.random.seed(42)
    sample_indices = np.random.choice(len(dataset_no_aug), NUM_SAMPLES, replace=False)

    # Create figure
    fig, axes = plt.subplots(NUM_SAMPLES, NUM_AUGMENTATIONS + 1, figsize=FIGSIZE)
    title = f'{MODE.upper()} Augmentation Examples (Green crosshair = ball center)'
    fig.suptitle(title, fontsize=14, y=0.995)

    for row, idx in enumerate(sample_indices):
        # Get original (no augmentation)
        img_orig, target_orig = dataset_no_aug[idx]
        img_orig_display = denormalize(img_orig.numpy())
        img_orig_display = draw_keypoint(
            img_orig_display,
            target_orig[0].item(),
            target_orig[1].item(),
            fullframe=FULLFRAME_MODE
        )

        # Display original
        if NUM_SAMPLES == 1:
            ax = axes[0]
        else:
            ax = axes[row, 0]

        ax.imshow(img_orig_display)
        ax.set_title('Original', fontsize=10)
        ax.axis('off')

        # Get augmented versions
        for col in range(NUM_AUGMENTATIONS):
            # Force dataset to reload same sample with different augmentation
            dataset_with_aug.samples[idx] = dataset_no_aug.samples[idx]
            img_aug, target_aug = dataset_with_aug[idx]

            img_aug_display = denormalize(img_aug.numpy())

            # Check confidence (3rd value) - if 0, image is torn/invalid
            confidence = target_aug[2].item() if len(target_aug) > 2 else 1.0
            is_torn = confidence < 0.5

            if not is_torn:
                img_aug_display = draw_keypoint(
                    img_aug_display,
                    target_aug[0].item(),
                    target_aug[1].item(),
                    fullframe=FULLFRAME_MODE
                )

            # Display augmented
            if NUM_SAMPLES == 1:
                ax = axes[col + 1]
            else:
                ax = axes[row, col + 1]

            ax.imshow(img_aug_display)
            title = f'Aug {col + 1}'
            if is_torn:
                title += ' [TORN]'
            ax.set_title(title, fontsize=10, color='red' if is_torn else 'black')
            ax.axis('off')

        print(f"Processed sample {row + 1}/{NUM_SAMPLES}")

    plt.tight_layout()

    # Save or show
    if SAVE_OUTPUT:
        plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
        print()
        print(f"Saved visualization to: {OUTPUT_PATH}")
    else:
        print()
        print("Displaying visualization...")
        plt.show()

    print("=" * 60)


if __name__ == "__main__":
    main()
