"""
Preprocess fullframe dataset to numpy memmap for fast training.

Converts 1280x720 images to 320x180 and stores as memory-mapped arrays.
Eliminates JPEG decoding and resize overhead during training.

Supports two modes:
- Normal: Single images, 3-channel, (x, y, valid) labels
- Stereo: Paired left/right images, 6-channel, (x_left, y_left, x_right, y_right, confidence) labels

Usage:
    python -m ball_detection.tools.preprocess_to_memmap

Output:
    Normal mode:
        data/fullframe_memmap/
            images.npy   - (N, 180, 320, 3) uint8
            labels.npy   - (N, 3) float32 [x_norm, y_norm, valid]
            metadata.json

    Stereo mode:
        data/stereo_memmap/
            images.npy   - (N, 180, 320, 6) uint8 [left_RGB + right_RGB]
            labels.npy   - (N, 5) float32 [x_left, y_left, x_right, y_right, confidence]
            metadata.json
"""

import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import re

# ============================================================
# SETTINGS
# ============================================================

INPUT_DIR = "./ball_detection/data/full_dataset/training_data_full"
OUTPUT_DIR = "./ball_detection/data/fullframe_memmap"

# Stereo mode settings
STEREO_MODE = True
STEREO_OUTPUT_DIR = "./ball_detection/data/stereo_memmap"

TARGET_WIDTH = 320
TARGET_HEIGHT = 180


def process_normal_mode(input_path, output_path, labels, image_dir):
    """Process single images (non-stereo mode)."""
    output_path.mkdir(parents=True, exist_ok=True)

    # Filter samples (images that exist)
    samples = []
    for img_name, label in labels.items():
        img_path = image_dir / img_name
        if img_path.exists():
            samples.append((img_name, str(img_path), label))

    n_samples = len(samples)
    print(f"Found {n_samples} samples")
    print(f"Target: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"Output: {output_path}")

    # Create memmap files
    images_path = output_path / "images.npy"
    labels_path = output_path / "labels.npy"

    size_gb = (n_samples * TARGET_HEIGHT * TARGET_WIDTH * 3) / (1024**3)
    print(f"Creating memmap: ({n_samples}, {TARGET_HEIGHT}, {TARGET_WIDTH}, 3) = {size_gb:.2f} GB")

    images_memmap = np.lib.format.open_memmap(
        str(images_path), mode='w+', dtype=np.uint8,
        shape=(n_samples, TARGET_HEIGHT, TARGET_WIDTH, 3)
    )

    labels_memmap = np.lib.format.open_memmap(
        str(labels_path), mode='w+', dtype=np.float32,
        shape=(n_samples, 3)
    )

    # Process
    filenames = []
    valid_count = 0

    for i, (img_name, img_path, label) in enumerate(tqdm(samples)):
        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Could not load {img_path}")
            continue

        orig_h, orig_w = img.shape[:2]

        # Resize
        img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images_memmap[i] = img

        # Labels
        is_valid = label.get('valid', True) and label['x'] >= 0 and label['y'] >= 0

        if is_valid:
            x_norm = (label['x'] / orig_w)
            y_norm = (label['y'] / orig_h)
            valid_count += 1
        else:
            x_norm, y_norm = 0.5, 0.5

        labels_memmap[i] = [x_norm, y_norm, 1.0 if is_valid else 0.0]
        filenames.append(img_name)

    images_memmap.flush()
    labels_memmap.flush()

    # Metadata
    metadata = {
        'n_samples': n_samples,
        'valid_count': valid_count,
        'invalid_count': n_samples - valid_count,
        'image_shape': [TARGET_HEIGHT, TARGET_WIDTH, 3],
        'target_resolution': [TARGET_WIDTH, TARGET_HEIGHT],
        'filenames': filenames,
    }

    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 50)
    print(f"Done! {n_samples} samples ({valid_count} valid)")
    print(f"Images: {images_path}")
    print(f"Labels: {labels_path}")
    print("=" * 50)


def process_stereo_mode(input_path, output_path, labels, image_dir):
    """Process stereo image pairs (left + right)."""
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse frame IDs and pair left/right images
    # Expected naming: frame_000001_left.png, frame_000001_right.png
    left_images = {}
    right_images = {}

    for img_name, label in labels.items():
        # Extract frame ID and side
        match = re.match(r'(.+)_(left|right)\.(png|jpg|jpeg)$', img_name, re.IGNORECASE)
        if match:
            frame_id = match.group(1)
            side = match.group(2).lower()
            if side == 'left':
                left_images[frame_id] = (img_name, label)
            else:
                right_images[frame_id] = (img_name, label)

    # Find matching pairs
    pairs = []
    for frame_id in left_images:
        if frame_id in right_images:
            left_name, left_label = left_images[frame_id]
            right_name, right_label = right_images[frame_id]

            left_path = image_dir / left_name
            right_path = image_dir / right_name

            if left_path.exists() and right_path.exists():
                pairs.append((frame_id, left_name, str(left_path), left_label,
                             right_name, str(right_path), right_label))

    n_samples = len(pairs)
    print(f"Found {n_samples} stereo pairs")
    print(f"Target: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"Output: {output_path}")

    if n_samples == 0:
        print("ERROR: No stereo pairs found!")
        print("Expected naming: frame_XXXX_left.png and frame_XXXX_right.png")
        return

    # Create memmap files - 6 channels for stereo
    images_path = output_path / "images.npy"
    labels_path = output_path / "labels.npy"

    size_gb = (n_samples * TARGET_HEIGHT * TARGET_WIDTH * 6) / (1024**3)
    print(f"Creating memmap: ({n_samples}, {TARGET_HEIGHT}, {TARGET_WIDTH}, 6) = {size_gb:.2f} GB")

    images_memmap = np.lib.format.open_memmap(
        str(images_path), mode='w+', dtype=np.uint8,
        shape=(n_samples, TARGET_HEIGHT, TARGET_WIDTH, 6)
    )

    # 5 labels: x_left, y_left, x_right, y_right, confidence
    labels_memmap = np.lib.format.open_memmap(
        str(labels_path), mode='w+', dtype=np.float32,
        shape=(n_samples, 5)
    )

    # Process
    filenames = []
    valid_left_count = 0
    valid_right_count = 0
    valid_both_count = 0
    valid_either_count = 0

    for i, (frame_id, left_name, left_path, left_label,
            right_name, right_path, right_label) in enumerate(tqdm(pairs)):
        # Load left image
        left_img = cv2.imread(left_path, cv2.IMREAD_COLOR)
        if left_img is None:
            print(f"Warning: Could not load {left_path}")
            continue

        # Load right image
        right_img = cv2.imread(right_path, cv2.IMREAD_COLOR)
        if right_img is None:
            print(f"Warning: Could not load {right_path}")
            continue

        orig_h, orig_w = left_img.shape[:2]

        # Resize both
        left_img = cv2.resize(left_img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)

        right_img = cv2.resize(right_img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

        # Stack left and right (6 channels: left_R, left_G, left_B, right_R, right_G, right_B)
        stereo_img = np.concatenate([left_img, right_img], axis=2)
        images_memmap[i] = stereo_img

        # Labels for left
        left_valid = left_label.get('valid', True) and left_label.get('x', -1) >= 0 and left_label.get('y', -1) >= 0
        if left_valid:
            x_left = left_label['x'] / orig_w
            y_left = left_label['y'] / orig_h
            valid_left_count += 1
        else:
            x_left, y_left = 0.5, 0.5

        # Labels for right
        right_valid = right_label.get('valid', True) and right_label.get('x', -1) >= 0 and right_label.get('y', -1) >= 0
        if right_valid:
            x_right = right_label['x'] / orig_w
            y_right = right_label['y'] / orig_h
            valid_right_count += 1
        else:
            x_right, y_right = 0.5, 0.5

        # Confidence: 1.0 if either left or right is valid
        confidence = 1.0 if (left_valid or right_valid) else 0.0

        if left_valid and right_valid:
            valid_both_count += 1
        if left_valid or right_valid:
            valid_either_count += 1

        labels_memmap[i] = [x_left, y_left, x_right, y_right, confidence]
        filenames.append(f"{left_name}|{right_name}")

    images_memmap.flush()
    labels_memmap.flush()

    # Metadata
    metadata = {
        'n_samples': n_samples,
        'stereo': True,
        'valid_left_count': valid_left_count,
        'valid_right_count': valid_right_count,
        'valid_both_count': valid_both_count,
        'valid_either_count': valid_either_count,
        'invalid_count': n_samples - valid_either_count,
        'image_shape': [TARGET_HEIGHT, TARGET_WIDTH, 6],
        'target_resolution': [TARGET_WIDTH, TARGET_HEIGHT],
        'label_format': ['x_left', 'y_left', 'x_right', 'y_right', 'confidence'],
        'filenames': filenames,
    }

    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 50)
    print(f"Done! {n_samples} stereo pairs")
    print(f"  Valid left: {valid_left_count}")
    print(f"  Valid right: {valid_right_count}")
    print(f"  Valid both: {valid_both_count}")
    print(f"  Valid either (confidence=1): {valid_either_count}")
    print(f"Images: {images_path}")
    print(f"Labels: {labels_path}")
    print("=" * 50)


def main():
    input_path = Path(INPUT_DIR)

    # Load labels
    labels_file = input_path / "labels.json"
    with open(labels_file, 'r') as f:
        labels = json.load(f)

    # Find image directory
    if (input_path / 'images').exists():
        image_dir = input_path / 'images'
    else:
        image_dir = input_path

    if STEREO_MODE:
        print("=" * 50)
        print("STEREO MODE")
        print("=" * 50)
        output_path = Path(STEREO_OUTPUT_DIR)
        process_stereo_mode(input_path, output_path, labels, image_dir)
    else:
        print("=" * 50)
        print("NORMAL MODE (single images)")
        print("=" * 50)
        output_path = Path(OUTPUT_DIR)
        process_normal_mode(input_path, output_path, labels, image_dir)


if __name__ == "__main__":
    main()
