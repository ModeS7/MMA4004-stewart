"""
Preprocess fullframe dataset to numpy memmap for fast training.

Converts 1280x720 images to 320x180 and stores as memory-mapped arrays.
Eliminates JPEG decoding and resize overhead during training.

Usage:
    python -m ball_detection.tools.preprocess_to_memmap

Output:
    data/fullframe_memmap/
        images.npy   - (N, 180, 320, 3) uint8
        labels.npy   - (N, 3) float32 [x_norm, y_norm, valid]
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
OUTPUT_DIR = "./ball_detection/data/fullframe_memmap"

TARGET_WIDTH = 320
TARGET_HEIGHT = 180


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


if __name__ == "__main__":
    main()
