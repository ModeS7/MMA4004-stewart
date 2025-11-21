#!/usr/bin/env python3
"""
Quick diagnostic - test model on training images
"""

import torch
import torch.nn as nn
from torch.amp import autocast
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
import json
import segmentation_models_pytorch as smp

# ============================================================
# SETTINGS - Edit these
# ============================================================
MODEL_PATH = "./ball_detection/models/segmentation/best_segmentation.pth"
TRAIN_DIR = "./ball_detection/training_data/train"
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
DEVICE = "cuda"
# ============================================================

def test_on_training_image():
    """Test model on a single training image."""

    # Load COCO annotations to get ground truth
    ann_path = Path(TRAIN_DIR) / "_annotations.coco.json"
    with open(ann_path, 'r') as f:
        coco = json.load(f)

    # Find an image WITH ball annotation
    images_dict = {img['id']: img for img in coco['images']}

    # Group annotations by image
    img_annotations = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    # Find first image that has annotations
    test_img_info = None
    test_img_id = None
    for img_id, img_info in images_dict.items():
        if img_id in img_annotations and len(img_annotations[img_id]) > 0:
            test_img_info = img_info
            test_img_id = img_id
            break

    if test_img_info is None:
        print("ERROR: No images with annotations found!")
        return

    img_path = Path(TRAIN_DIR) / test_img_info['file_name']
    print(f"\nTesting on: {test_img_info['file_name']}")
    print(f"Ground truth: {len(img_annotations[test_img_id])} annotation(s)")

    # Load model
    print("\nLoading model...")
    model = smp.Unet(encoder_name='efficientnet-b4', encoder_weights=None, classes=2, activation=None)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded")

    # Load image
    image = cv2.imread(str(img_path))
    original = image.copy()
    h, w = image.shape[:2]

    # Create ground truth mask
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    for ann in img_annotations[test_img_id]:
        if 'segmentation' in ann and ann['segmentation']:
            for seg in ann['segmentation']:
                if len(seg) >= 6:
                    try:
                        pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(gt_mask, [pts], 1)
                    except:
                        continue

    gt_mask_resized = cv2.resize(gt_mask, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    gt_pixels = np.sum(gt_mask_resized > 0)
    print(f"Ground truth mask: {gt_pixels} / {IMAGE_WIDTH*IMAGE_HEIGHT} pixels ({gt_pixels/(IMAGE_WIDTH*IMAGE_HEIGHT)*100:.2f}%)")

    # Preprocess
    image_resized = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_norm = image_rgb.astype(np.float32) / 255.0
    image_tensor = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(torch.from_numpy(image_norm).permute(2, 0, 1))

    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        with autocast('cuda', dtype=torch.bfloat16):
            outputs = model(image_tensor)

        print(f"Output shape: {outputs.shape}")
        print(f"Output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")

        # Check raw logits (convert from bfloat16 to float32 first)
        logits_bg = outputs[0, 0].float().cpu().numpy()
        logits_ball = outputs[0, 1].float().cpu().numpy()

        print(f"\nBackground logits - min: {logits_bg.min():.4f}, max: {logits_bg.max():.4f}, mean: {logits_bg.mean():.4f}")
        print(f"Ball logits - min: {logits_ball.min():.4f}, max: {logits_ball.max():.4f}, mean: {logits_ball.mean():.4f}")

        preds = torch.argmax(outputs, dim=1)
        mask = preds[0].cpu().numpy().astype(np.uint8) * 255

        num_ball_pixels = np.sum(mask > 127)
        total_pixels = mask.shape[0] * mask.shape[1]

        print(f"\nMask statistics:")
        print(f"  Ball pixels: {num_ball_pixels} / {total_pixels} ({num_ball_pixels/total_pixels*100:.2f}%)")
        print(f"  Unique values: {np.unique(mask)}")

    # Create visualizations
    image_display = cv2.resize(original, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # Ground truth overlay (blue)
    gt_overlay = image_display.copy()
    gt_colored = np.zeros_like(gt_overlay)
    gt_colored[gt_mask_resized > 0] = [255, 0, 0]  # Blue for ground truth
    gt_result = cv2.addWeighted(gt_overlay, 0.7, gt_colored, 0.3, 0)

    # Prediction overlay (green)
    pred_overlay = image_display.copy()
    pred_colored = np.zeros_like(pred_overlay)
    pred_colored[mask > 127] = [0, 255, 0]  # Green for prediction
    pred_result = cv2.addWeighted(pred_overlay, 0.7, pred_colored, 0.3, 0)

    # Side by side
    comparison = np.hstack([gt_result, pred_result])

    # Save to file
    output_path = "ball_detection/test_result.jpg"
    cv2.imwrite(output_path, comparison)
    print(f"\nResult saved to: {output_path}")
    print("Left: Ground truth (blue), Right: Prediction (green)")

if __name__ == "__main__":
    test_on_training_image()
