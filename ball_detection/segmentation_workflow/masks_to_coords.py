#!/usr/bin/env python3
"""
Convert segmentation masks to center coordinates for regression training

Run: python ball_detection/masks_to_coords.py
"""

import cv2
import numpy as np
import json
import shutil
from pathlib import Path
from tqdm import tqdm

# ============================================================
# SETTINGS - Edit these
# ============================================================
MASKS_DIR = "./ball_detection/data/full_dataset/auto_labeled/masks"
SOURCE_IMAGES_DIR = "./ball_detection/data/full_dataset/images"
OUTPUT_DIR = "./ball_detection/data/full_dataset/training_data_full"
MIN_AREA = 50  # Minimum mask area to be considered valid
# ============================================================


def get_mask_center(mask):
    """Calculate center of mass from binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Use largest contour
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < MIN_AREA:
        return None

    # Calculate center
    M = cv2.moments(largest)
    if M['m00'] == 0:
        return None

    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']

    return (cx, cy)


def main():
    print("=" * 60)
    print("CONVERTING MASKS TO COORDINATES")
    print("=" * 60)
    print(f"Masks: {MASKS_DIR}")
    print(f"Images: {SOURCE_IMAGES_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Min area: {MIN_AREA} pixels")
    print("=" * 60)
    print()

    masks_dir = Path(MASKS_DIR)
    source_images_dir = Path(SOURCE_IMAGES_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_images_dir = output_dir / "images"
    output_path = output_dir / "labels.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(exist_ok=True)

    # Get all masks
    mask_files = list(masks_dir.glob("*.jpg")) + list(masks_dir.glob("*.png"))
    print(f"Found {len(mask_files)} masks")
    print()

    # Process masks
    labels = {}
    valid_count = 0
    invalid_count = 0

    for mask_path in tqdm(mask_files, desc="Converting"):
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Failed to load {mask_path.name}")
            continue

        # Get center
        center = get_mask_center(mask)

        if center is None:
            labels[mask_path.name] = {'x': -1, 'y': -1, 'valid': False}
            invalid_count += 1
        else:
            cx, cy = center
            labels[mask_path.name] = {'x': float(cx), 'y': float(cy), 'valid': True}
            valid_count += 1

    # Copy images
    print("\nCopying images...")
    for mask_file in tqdm(mask_files, desc="Copying"):
        img_file = source_images_dir / mask_file.name
        if img_file.exists():
            shutil.copy(img_file, output_images_dir / mask_file.name)

    # Save labels
    with open(output_path, 'w') as f:
        json.dump(labels, f, indent=2)

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE!")
    print("=" * 60)
    print(f"Total masks: {len(mask_files)}")
    print(f"Valid detections: {valid_count} ({valid_count/len(mask_files)*100:.1f}%)")
    print(f"No ball/invalid: {invalid_count} ({invalid_count/len(mask_files)*100:.1f}%)")
    print(f"\nLabels saved to: {output_path}")
    print(f"Images copied to: {output_images_dir}")
    print("\nNext step:")
    print("  Train regression model:")
    print("  python -m ball_detection.train --data-dir ./ball_detection/training_data_full")
    print("=" * 60)


if __name__ == "__main__":
    main()
