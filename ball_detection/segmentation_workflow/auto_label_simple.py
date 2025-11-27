#!/usr/bin/env python3
"""
Auto-label all video frames using trained segmentation model

Run: python ball_detection/auto_label_simple.py
"""

import torch
import torch.nn as nn
from torch.amp import autocast
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import segmentation_models_pytorch as smp

# ============================================================
# SETTINGS - Edit these
# ============================================================
IMAGES_DIR = "./ball_detection/data/new_labels/images"  # Directory with extracted images
OUTPUT_DIR = "./ball_detection/data/new_labels/auto_labeled"
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
DEVICE = "cuda"
BATCH_SIZE = 32  # Process multiple images at once for speed

# Model settings - single model or averaged checkpoints
USE_CHECKPOINT_AVERAGING = True  # Set to False to use single MODEL_PATH
MODEL_PATH = "./ball_detection/models/segmentation/best_segmentation.pth"  # Single model path
CHECKPOINT_PATHS = [  # Paths for checkpoint averaging
    "./ball_detection/models/segmentation/v2/best_segmentation.pth",
    "./ball_detection/models/segmentation/best_interval_600-700.pth",
    "./ball_detection/models/segmentation/best_interval_800-900.pth",
]

# Post-processing
KEEP_LARGEST_BLOB = True  # Filter out small detections, keep only largest
MIN_BLOB_AREA = 100  # Minimum blob area in pixels (ignore smaller)
# ============================================================


def average_checkpoints(paths, device):
    """Average weights from multiple checkpoint files."""
    print(f"Averaging {len(paths)} checkpoints...")
    state_dicts = []

    for path in paths:
        if not Path(path).exists():
            print(f"  Warning: {path} not found, skipping")
            continue
        sd = torch.load(path, map_location=device)
        # Handle torch.compile saved models
        if any(k.startswith('_orig_mod.') for k in sd.keys()):
            sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        state_dicts.append(sd)
        print(f"  Loaded: {path}")

    if len(state_dicts) == 0:
        raise ValueError("No valid checkpoints found!")

    if len(state_dicts) == 1:
        print("  Only 1 checkpoint found, using as-is")
        return state_dicts[0]

    # Average all weights
    avg_state = {}
    for key in state_dicts[0].keys():
        tensors = [sd[key].float() for sd in state_dicts]
        avg_state[key] = torch.stack(tensors).mean(0)

    print(f"  Averaged {len(state_dicts)} checkpoints")
    return avg_state


def keep_largest_blob(mask, min_area=100):
    """Keep only the largest connected component in a binary mask."""
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:  # Only background
        return mask, None

    # Find largest component (skip label 0 = background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = np.argmax(areas)
    largest_area = areas[largest_idx]

    if largest_area < min_area:
        return np.zeros_like(mask), None  # Too small = no ball

    largest_label = largest_idx + 1

    # Get centroid of largest blob
    centroid = centroids[largest_label]

    # Create mask with only largest component
    result = np.zeros_like(mask)
    result[labels == largest_label] = 255

    return result, centroid


def predict_masks_batch(model, images, device):
    """Predict masks for batch of images."""
    batch_tensors = []

    for image in images:
        # Resize
        image_resized = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # Normalize
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_norm = image_rgb.astype(np.float32) / 255.0
        image_tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(torch.from_numpy(image_norm).permute(2, 0, 1))

        batch_tensors.append(image_tensor)

    # Stack into batch
    batch = torch.stack(batch_tensors).to(device)

    # Predict
    with torch.no_grad():
        with autocast('cuda', dtype=torch.bfloat16):
            outputs = model(batch)
        preds = torch.argmax(outputs, dim=1)
        masks = preds.cpu().numpy().astype(np.uint8) * 255

    return masks


def main():
    print("=" * 60)
    print("AUTO-LABELING WITH SEGMENTATION MODEL")
    print("=" * 60)
    print(f"Images: {IMAGES_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Resolution: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"Checkpoint averaging: {'Enabled' if USE_CHECKPOINT_AVERAGING else 'Disabled'}")
    print(f"Keep largest blob: {'Enabled' if KEEP_LARGEST_BLOB else 'Disabled'}" +
          (f" (min area: {MIN_BLOB_AREA}px)" if KEEP_LARGEST_BLOB else ""))
    print("=" * 60)
    print()

    # Create output directories
    output_dir = Path(OUTPUT_DIR)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    model = smp.Unet(encoder_name='efficientnet-b0', encoder_weights=None, classes=2, activation=None)

    if USE_CHECKPOINT_AVERAGING:
        state_dict = average_checkpoints(CHECKPOINT_PATHS, DEVICE)
    else:
        print(f"Loading single model: {MODEL_PATH}")
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        # Handle torch.compile saved models (remove _orig_mod prefix if present)
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}")
    print()

    # Get all images
    images_dir = Path(IMAGES_DIR)
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"Found {len(image_files)} images")
    print()

    if not image_files:
        print(f"ERROR: No images found in {IMAGES_DIR}")
        return

    # Process images in batches
    print(f"Generating masks (batch size: {BATCH_SIZE})...")

    detected_count = 0
    empty_count = 0

    # Process in batches
    for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Auto-labeling"):
        batch_paths = image_files[i:i+BATCH_SIZE]
        batch_images = []
        valid_paths = []

        # Load batch
        for img_path in batch_paths:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Failed to load {img_path.name}")
                continue
            batch_images.append(image)
            valid_paths.append(img_path)

        if not batch_images:
            continue

        # Generate masks for batch
        masks = predict_masks_batch(model, batch_images, DEVICE)

        # Save masks
        for mask, img_path in zip(masks, valid_paths):
            # Apply largest blob filter if enabled
            if KEEP_LARGEST_BLOB:
                mask, centroid = keep_largest_blob(mask, min_area=MIN_BLOB_AREA)

            # Check if mask has any detections
            if np.sum(mask > 127) > 0:
                detected_count += 1
            else:
                empty_count += 1

            # Save mask
            mask_path = masks_dir / img_path.name
            cv2.imwrite(str(mask_path), mask)

    print("\n" + "=" * 60)
    print("AUTO-LABELING COMPLETE!")
    print("=" * 60)
    print(f"Processed {len(image_files)} images")
    print(f"Ball detected: {detected_count} ({detected_count/len(image_files)*100:.1f}%)")
    print(f"No ball: {empty_count} ({empty_count/len(image_files)*100:.1f}%)")
    print(f"Masks saved to: {masks_dir}")

    if detected_count == 0:
        print("\nWARNING: No balls detected in any image!")
        print("This could mean:")
        print("  1. Model not trained properly")
        print("  2. Images are different from training data")
        print("  3. Ball not visible in these images")
    else:
        print("\nNext step:")
        print("  View masks: python ball_detection/view_masks.py")
        print("  Convert to coordinates: python ball_detection/masks_to_coords.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
