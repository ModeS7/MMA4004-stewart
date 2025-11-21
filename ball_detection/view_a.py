"""
View Augmentation Examples

Loads random images from the dataset and shows how augmentations affect them.
Useful for debugging and tuning augmentation parameters.

Usage: python ball_detection/view_a.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

from dataset import BallDetectionDataset

# ============================================================
# SETTINGS - Edit these
# ============================================================
DATA_DIR = "./ball_detection/data/final"
CROP_SIZE = 128
USE_SPATIAL_AUGMENTATION = True  # Offset, rotate, scale, shift
USE_APPEARANCE_AUGMENTATION = True  # Brightness, hue, blur, noise
NUM_SAMPLES = 5  # Number of different images to show
NUM_AUGMENTATIONS = 4  # Number of augmented versions per image
FIGSIZE = (16, 10)  # Figure size for display
SAVE_OUTPUT = True  # Save visualization to file
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


def draw_keypoint(img, x, y, color=(0, 255, 0), radius=2, thickness=-1):
    """Draw keypoint on image."""
    img = img.copy()
    x_pixel = int(x * img.shape[1])
    y_pixel = int(y * img.shape[0])
    cv2.circle(img, (x_pixel, y_pixel), radius, color, thickness)
    # Draw crosshair
    cv2.line(img, (x_pixel - 5, y_pixel), (x_pixel + 5, y_pixel), color, 1)
    cv2.line(img, (x_pixel, y_pixel - 5), (x_pixel, y_pixel + 5), color, 1)
    return img


def main():
    """Visualize augmentations."""
    print("=" * 60)
    print("AUGMENTATION VISUALIZATION")
    print("=" * 60)
    print(f"Data: {DATA_DIR}")
    print(f"Crop size: {CROP_SIZE}x{CROP_SIZE}")
    print(f"Spatial augmentation: {'Enabled' if USE_SPATIAL_AUGMENTATION else 'Disabled'}")
    print(f"Appearance augmentation: {'Enabled' if USE_APPEARANCE_AUGMENTATION else 'Disabled'}")
    print(f"Samples: {NUM_SAMPLES}")
    print(f"Augmentations per sample: {NUM_AUGMENTATIONS}")
    print("=" * 60)
    print()

    # Create datasets
    dataset_no_aug = BallDetectionDataset(
        data_dir=DATA_DIR,
        crop_size=CROP_SIZE,
        use_spatial_aug=False,
        use_appearance_aug=False,
        split='train'
    )

    dataset_with_aug = BallDetectionDataset(
        data_dir=DATA_DIR,
        crop_size=CROP_SIZE,
        use_spatial_aug=USE_SPATIAL_AUGMENTATION,
        use_appearance_aug=USE_APPEARANCE_AUGMENTATION,
        split='train'
    )

    print(f"Dataset loaded: {len(dataset_no_aug)} samples")
    print()

    # Select random samples
    np.random.seed(42)
    sample_indices = np.random.choice(len(dataset_no_aug), NUM_SAMPLES, replace=False)

    # Create figure
    fig, axes = plt.subplots(NUM_SAMPLES, NUM_AUGMENTATIONS + 1, figsize=FIGSIZE)
    fig.suptitle('Augmentation Examples (Green crosshair = ball center)', fontsize=14, y=0.995)

    for row, idx in enumerate(sample_indices):
        # Get original (no augmentation)
        img_orig, target_orig = dataset_no_aug[idx]
        img_orig_display = denormalize(img_orig.numpy())
        img_orig_display = draw_keypoint(
            img_orig_display,
            target_orig[0].item(),
            target_orig[1].item()
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
            img_aug_display = draw_keypoint(
                img_aug_display,
                target_aug[0].item(),
                target_aug[1].item()
            )

            # Display augmented
            if NUM_SAMPLES == 1:
                ax = axes[col + 1]
            else:
                ax = axes[row, col + 1]

            ax.imshow(img_aug_display)
            ax.set_title(f'Aug {col + 1}', fontsize=10)
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
