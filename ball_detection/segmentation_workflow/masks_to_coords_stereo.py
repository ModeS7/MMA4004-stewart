#!/usr/bin/env python3
"""
Convert stereo segmentation masks to center coordinates for regression training.

Creates a stereo labels.json with format:
{
    "frame_000000": {
        "x_left": 640.5, "y_left": 360.2, "valid_left": true,
        "x_right": 580.3, "y_right": 358.1, "valid_right": true
    },
    ...
}

Run: python -m ball_detection.segmentation_workflow.masks_to_coords_stereo
"""

import cv2
import numpy as np
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import re
import argparse

# ============================================================
# SETTINGS - Edit these or use command line args
# ============================================================
DEFAULT_DATA_DIR = "./ball_detection/data/full_dataset"
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


def extract_frame_id(filename):
    """
    Extract frame identifier from filename.

    Handles formats:
    - frame_000000_left.jpg -> frame_000000
    - recording_20251126_205350_000000_left.jpg -> recording_20251126_205350_000000
    """
    name = Path(filename).stem  # Remove extension

    # Remove _left or _right suffix
    if name.endswith('_left'):
        return name[:-5]
    elif name.endswith('_right'):
        return name[:-6]
    else:
        return name


def process_stereo_masks(data_dir, output_dir=None):
    """
    Process stereo mask pairs and create labels.json.

    Args:
        data_dir: Directory containing auto_labeled/masks/ and images/
        output_dir: Output directory (default: data_dir)
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir) if output_dir else data_dir

    masks_dir = data_dir / "auto_labeled" / "masks"
    images_dir = data_dir / "images"
    output_path = output_dir / "stereo_labels.json"

    print("=" * 60)
    print("CONVERTING STEREO MASKS TO COORDINATES")
    print("=" * 60)
    print(f"Data dir: {data_dir}")
    print(f"Masks: {masks_dir}")
    print(f"Images: {images_dir}")
    print(f"Output: {output_path}")
    print(f"Min area: {MIN_AREA} pixels")
    print("=" * 60)
    print()

    # Get all masks
    mask_files = list(masks_dir.glob("*.jpg")) + list(masks_dir.glob("*.png"))
    print(f"Found {len(mask_files)} mask files")

    # Group by frame ID
    frames = {}
    for mask_path in mask_files:
        frame_id = extract_frame_id(mask_path.name)
        if frame_id not in frames:
            frames[frame_id] = {'left': None, 'right': None}

        if '_left' in mask_path.name:
            frames[frame_id]['left'] = mask_path
        elif '_right' in mask_path.name:
            frames[frame_id]['right'] = mask_path

    print(f"Found {len(frames)} stereo frame pairs")
    print()

    # Process each frame pair
    labels = {}
    stats = {
        'both_valid': 0,
        'left_only': 0,
        'right_only': 0,
        'neither_valid': 0,
        'missing_pair': 0
    }

    for frame_id, paths in tqdm(frames.items(), desc="Processing"):
        label = {
            'x_left': -1.0, 'y_left': -1.0, 'valid_left': False,
            'x_right': -1.0, 'y_right': -1.0, 'valid_right': False
        }

        # Check if we have both left and right
        if paths['left'] is None or paths['right'] is None:
            stats['missing_pair'] += 1
            labels[frame_id] = label
            continue

        # Process left mask
        left_mask = cv2.imread(str(paths['left']), cv2.IMREAD_GRAYSCALE)
        if left_mask is not None:
            center = get_mask_center(left_mask)
            if center is not None:
                label['x_left'] = float(center[0])
                label['y_left'] = float(center[1])
                label['valid_left'] = True

        # Process right mask
        right_mask = cv2.imread(str(paths['right']), cv2.IMREAD_GRAYSCALE)
        if right_mask is not None:
            center = get_mask_center(right_mask)
            if center is not None:
                label['x_right'] = float(center[0])
                label['y_right'] = float(center[1])
                label['valid_right'] = True

        # Update stats
        if label['valid_left'] and label['valid_right']:
            stats['both_valid'] += 1
        elif label['valid_left']:
            stats['left_only'] += 1
        elif label['valid_right']:
            stats['right_only'] += 1
        else:
            stats['neither_valid'] += 1

        labels[frame_id] = label

    # Save labels
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(labels, f, indent=2)

    # Print summary
    total = len(frames)
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE!")
    print("=" * 60)
    print(f"Total stereo pairs: {total}")
    print(f"Both cameras valid: {stats['both_valid']} ({stats['both_valid']/total*100:.1f}%)")
    print(f"Left only valid: {stats['left_only']} ({stats['left_only']/total*100:.1f}%)")
    print(f"Right only valid: {stats['right_only']} ({stats['right_only']/total*100:.1f}%)")
    print(f"Neither valid: {stats['neither_valid']} ({stats['neither_valid']/total*100:.1f}%)")
    print(f"Missing pair: {stats['missing_pair']}")
    print()
    print(f"Labels saved to: {output_path}")
    print()
    print("For training, you need stereo pairs where BOTH cameras are valid.")
    print(f"Usable samples: {stats['both_valid']} stereo pairs")
    print("=" * 60)

    return labels, stats


def main():
    parser = argparse.ArgumentParser(description='Convert stereo masks to coordinates')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR,
                       help='Data directory containing auto_labeled/masks/ and images/')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: same as data-dir)')
    args = parser.parse_args()

    process_stereo_masks(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
