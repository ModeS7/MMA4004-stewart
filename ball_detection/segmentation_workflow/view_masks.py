#!/usr/bin/env python3
"""
View masks overlaid on images to verify quality

Run: python ball_detection/view_masks.py
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ============================================================
# SETTINGS - Edit these
# ============================================================
IMAGES_DIR = "./ball_detection/data/old_labels/images"
MASKS_DIR = "./ball_detection/data/old_labels/auto_labeled/masks"
OUTPUT_VIDEO = "./ball_detection/segmentation_workflow/mask_review_old.mp4"
FPS = 20  # Frames per second for output video
# ============================================================


def overlay_mask(image, mask):
    """Create image with mask overlay."""
    # Create colored overlay (green for ball)
    overlay = image.copy()
    mask_colored = np.zeros_like(image)
    mask_colored[mask > 127] = [0, 255, 0]  # Green

    # Blend
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)

    # Draw contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    # Calculate and draw center
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)
            cv2.circle(overlay, (cx, cy), 20, (0, 0, 255), 2)

    return overlay


def main():
    print("=" * 60)
    print("MASK VISUALIZATION - STEREO VIDEO OUTPUT")
    print("=" * 60)
    print(f"Images: {IMAGES_DIR}")
    print(f"Masks: {MASKS_DIR}")
    print(f"Output: {OUTPUT_VIDEO}")
    print(f"FPS: {FPS}")
    print("=" * 60)
    print()

    images_dir = Path(IMAGES_DIR)
    masks_dir = Path(MASKS_DIR)

    # Group images by frame number (supports prefix_000001_left.jpg naming)
    left_files = {}
    right_files = {}

    all_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    for img_path in all_images:
        name = img_path.name
        if "_left" in name:
            # Extract frame identifier (e.g., "prefix_000001" from "prefix_000001_left.jpg")
            frame_num = name.split("_left")[0]
            left_files[frame_num] = img_path
        elif "_right" in name:
            frame_num = name.split("_right")[0]
            right_files[frame_num] = img_path

    # Find frames that have both left and right
    common_frames = sorted(set(left_files.keys()) & set(right_files.keys()))

    if not common_frames:
        print("ERROR: No matching left/right pairs found!")
        return

    print(f"Found {len(common_frames)} stereo frame pairs")
    print()

    # Get video dimensions from first image
    first_img = cv2.imread(str(left_files[common_frames[0]]))
    h, w = first_img.shape[:2]

    # Create video writer (side-by-side = 2x width)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w * 2, h))

    print(f"Creating stereo video: {w*2}x{h} @ {FPS} fps")
    print()

    # Process each frame pair
    for idx, frame_num in enumerate(tqdm(common_frames, desc="Creating video"), 1):
        left_path = left_files[frame_num]
        right_path = right_files[frame_num]

        left_mask_path = masks_dir / left_path.name
        right_mask_path = masks_dir / right_path.name

        # Load left
        left_img = cv2.imread(str(left_path))
        left_mask = cv2.imread(str(left_mask_path), cv2.IMREAD_GRAYSCALE)

        # Load right
        right_img = cv2.imread(str(right_path))
        right_mask = cv2.imread(str(right_mask_path), cv2.IMREAD_GRAYSCALE)

        if left_img is None or right_img is None or left_mask is None or right_mask is None:
            continue

        # Create overlays
        left_overlay = overlay_mask(left_img, left_mask)
        right_overlay = overlay_mask(right_img, right_mask)

        # Add labels
        cv2.putText(left_overlay, f"LEFT - Frame {idx}/{len(common_frames)}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(right_overlay, f"RIGHT - Frame {idx}/{len(common_frames)}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Combine side-by-side
        combined = np.hstack([left_overlay, right_overlay])

        # Write frame
        video_writer.write(combined)

    video_writer.release()

    print("\n" + "=" * 60)
    print("STEREO VIDEO CREATED!")
    print("=" * 60)
    print(f"Output: {OUTPUT_VIDEO}")
    print(f"Frames: {len(common_frames)} stereo pairs")
    print(f"Duration: {len(common_frames)/FPS:.1f} seconds")
    print("\nReview the video to check mask quality.")
    print("Left camera on left, right camera on right")
    print("Green overlay = detected ball")
    print("Red dot = calculated center")
    print("\nIf masks look good, continue with:")
    print("  python ball_detection/masks_to_coords.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
