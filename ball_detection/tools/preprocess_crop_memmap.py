"""
Preprocess dataset to numpy memmap for crop-based training.

Crops 256x256 patches centered on the ball from full images.
Only includes POSITIVE samples (ball visible) - no invalid samples.

The 256x256 size allows for random offset augmentation during training
when using 128x128 crops.

Usage:
    python -m ball_detection.tools.preprocess_crop_memmap

Output:
    data/crop_memmap/
        images.npy   - (N, 256, 256, 3) uint8
        labels.npy   - (N, 2) float32 [x_norm, y_norm] ball position in crop
        metadata.json
"""

import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# ============================================================
# SETTINGS
# ============================================================

INPUT_DIR = "./ball_detection/data/full_dataset/training_data_full"
OUTPUT_DIR = "./ball_detection/data/crop_memmap"

CROP_SIZE = 200  # Larger than 128 training crop to allow offset augmentation


def main():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load labels
    labels_file = input_path / "labels.json"
    with open(labels_file, 'r') as f:
        labels = json.load(f)

    # Find image directory
    if (input_path / 'images').exists():
        image_dir = input_path / 'images'
    else:
        image_dir = input_path

    # Filter to ONLY positive samples (ball visible)
    samples = []
    skipped_invalid = 0
    skipped_missing = 0

    for img_name, label in labels.items():
        img_path = image_dir / img_name

        # Skip if image doesn't exist
        if not img_path.exists():
            skipped_missing += 1
            continue

        # Skip invalid samples (no ball)
        is_valid = label.get('valid', True) and label.get('x', -1) >= 0 and label.get('y', -1) >= 0
        if not is_valid:
            skipped_invalid += 1
            continue

        samples.append((img_name, str(img_path), label))

    n_samples = len(samples)
    print(f"Found {n_samples} positive samples")
    print(f"Skipped: {skipped_invalid} invalid, {skipped_missing} missing")
    print(f"Crop size: {CROP_SIZE}x{CROP_SIZE}")
    print(f"Output: {output_path}")

    if n_samples == 0:
        print("ERROR: No valid samples found!")
        return

    # Create memmap files
    images_path = output_path / "images.npy"
    labels_path = output_path / "labels.npy"

    size_mb = (n_samples * CROP_SIZE * CROP_SIZE * 3) / (1024**2)
    print(f"Creating memmap: ({n_samples}, {CROP_SIZE}, {CROP_SIZE}, 3) = {size_mb:.1f} MB")

    images_memmap = np.lib.format.open_memmap(
        str(images_path), mode='w+', dtype=np.uint8,
        shape=(n_samples, CROP_SIZE, CROP_SIZE, 3)
    )

    # Only x, y for crop mode (no confidence needed - all positive)
    labels_memmap = np.lib.format.open_memmap(
        str(labels_path), mode='w+', dtype=np.float32,
        shape=(n_samples, 2)
    )

    # Process
    filenames = []
    half_crop = CROP_SIZE // 2

    for i, (img_name, img_path, label) in enumerate(tqdm(samples, desc="Processing")):
        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Could not load {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Ball position in original image
        ball_x = int(label['x'])
        ball_y = int(label['y'])

        # Calculate crop bounds (center on ball as much as possible)
        x1 = ball_x - half_crop
        y1 = ball_y - half_crop
        x2 = x1 + CROP_SIZE
        y2 = y1 + CROP_SIZE

        # Clamp to image bounds
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > w:
            x1 -= (x2 - w)
            x2 = w
        if y2 > h:
            y1 -= (y2 - h)
            y2 = h

        # Final clamp
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Extract crop
        crop = img[y1:y2, x1:x2]

        # Handle edge cases where crop is smaller than expected
        if crop.shape[0] != CROP_SIZE or crop.shape[1] != CROP_SIZE:
            # Pad with black if needed
            padded = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded

        images_memmap[i] = crop

        # Ball position relative to crop, normalized to [0, 1]
        ball_x_in_crop = (ball_x - x1) / CROP_SIZE
        ball_y_in_crop = (ball_y - y1) / CROP_SIZE

        # Clamp to valid range
        ball_x_in_crop = np.clip(ball_x_in_crop, 0.0, 1.0)
        ball_y_in_crop = np.clip(ball_y_in_crop, 0.0, 1.0)

        labels_memmap[i] = [ball_x_in_crop, ball_y_in_crop]
        filenames.append(img_name)

    images_memmap.flush()
    labels_memmap.flush()

    # Metadata
    metadata = {
        'n_samples': n_samples,
        'crop_size': CROP_SIZE,
        'image_shape': [CROP_SIZE, CROP_SIZE, 3],
        'label_format': ['x_norm', 'y_norm'],
        'all_positive': True,
        'filenames': filenames,
    }

    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 50)
    print(f"Done! {n_samples} positive samples")
    print(f"Images: {images_path}")
    print(f"Labels: {labels_path}")
    print(f"All samples have ball visible (no negatives)")
    print("=" * 50)


if __name__ == "__main__":
    main()
