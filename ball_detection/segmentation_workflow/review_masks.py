#!/usr/bin/env python3
"""
Training Data Review Tool

Supports two data formats:
1. Masks: Segmentation masks in a separate folder
2. Labels JSON: Ball coordinates in labels.json file

Usage:
    # Generate video from labels.json (fullframe training data)
    python -m ball_detection.segmentation_workflow.review_masks --data-dir ./ball_detection/data/full_dataset/training_data_full

    # Generate video from masks (stereo workflow)
    python -m ball_detection.segmentation_workflow.review_masks --images-dir ./images --masks-dir ./masks

    # Filter by camera
    python -m ball_detection.segmentation_workflow.review_masks --data-dir ./data --camera left

    # Delete specific frames (by frame number from video)
    python -m ball_detection.segmentation_workflow.review_masks --data-dir ./data --delete 42,156,789

    # Interactive mode (requires X server)
    python -m ball_detection.segmentation_workflow.review_masks --data-dir ./data --interactive
"""

import json
import re
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

# ============================================================
# SETTINGS - Edit these for mask mode
# ============================================================
IMAGES_DIR = "./ball_detection/data/full_dataset/training_data_full/images"
MASKS_DIR = "./ball_detection/data/old_labels/auto_labeled/masks"
OUTPUT_VIDEO = "./ball_detection/segmentation_workflow/mask_review.mp4"
MASK_RADIUS = 20
FPS = 15
# ============================================================


def natural_sort_key(s):
    """Sort strings with embedded numbers naturally."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(s))]


def overlay_mask(image, mask):
    """Create image with mask overlay."""
    overlay = image.copy()

    if mask is None or mask.max() == 0:
        cv2.putText(overlay, "NO DETECTION", (image.shape[1]//2 - 150, image.shape[0]//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
        return overlay

    # Green overlay for ball region
    mask_colored = np.zeros_like(image)
    mask_colored[mask > 127] = [0, 255, 0]
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)

    # Draw contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    # Draw center
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)
            cv2.circle(overlay, (cx, cy), 20, (0, 0, 255), 2)

    return overlay


def overlay_label(image, x, y, valid, marker_size=8):
    """Create image with ball center marker from labels.json."""
    overlay = image.copy()

    if not valid:
        cv2.putText(overlay, "NO BALL", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        return overlay

    x, y = int(x), int(y)

    # Draw crosshair marker
    cv2.circle(overlay, (x, y), marker_size, (0, 255, 0), 2)
    cv2.circle(overlay, (x, y), 3, (0, 255, 0), -1)
    cv2.line(overlay, (x - marker_size - 5, y), (x - marker_size, y), (0, 255, 0), 2)
    cv2.line(overlay, (x + marker_size, y), (x + marker_size + 5, y), (0, 255, 0), 2)
    cv2.line(overlay, (x, y - marker_size - 5), (x, y - marker_size), (0, 255, 0), 2)
    cv2.line(overlay, (x, y + marker_size), (x, y + marker_size + 5), (0, 255, 0), 2)

    return overlay


def load_stereo_frames(images_dir, masks_dir):
    """Load stereo frame pairs with masks."""
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    left_files = {}
    right_files = {}

    all_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

    for img_path in all_images:
        name = img_path.name
        if "_left" in name:
            frame_id = name.split("_left")[0]
            left_files[frame_id] = img_path
        elif "_right" in name:
            frame_id = name.split("_right")[0]
            right_files[frame_id] = img_path

    common_frames = sorted(set(left_files.keys()) & set(right_files.keys()))

    frames = []
    for frame_id in common_frames:
        frames.append({
            'id': frame_id,
            'left_img': left_files[frame_id],
            'right_img': right_files[frame_id],
            'left_mask': masks_dir / left_files[frame_id].name,
            'right_mask': masks_dir / right_files[frame_id].name,
        })

    return frames


def load_labeled_frames(data_dir, camera='both'):
    """Load frames with labels.json coordinates."""
    data_dir = Path(data_dir)
    images_dir = data_dir / 'images'
    labels_path = data_dir / 'labels.json'

    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return [], {}

    labels = {}
    if labels_path.exists():
        with open(labels_path) as f:
            labels = json.load(f)

    # Get image files
    image_files = sorted(images_dir.glob('*.jpg'), key=natural_sort_key)
    image_files += sorted(images_dir.glob('*.png'), key=natural_sort_key)

    # Filter by camera
    if camera == 'left':
        image_files = [f for f in image_files if '_left.' in f.name or '_left_' in f.name]
    elif camera == 'right':
        image_files = [f for f in image_files if '_right.' in f.name or '_right_' in f.name]

    return image_files, labels


# ============================================================
# VIDEO GENERATION
# ============================================================

def generate_video_from_labels(data_dir, output_path, camera='left', fps=15, max_frames=None):
    """Generate review video from labels.json data."""
    image_files, labels = load_labeled_frames(data_dir, camera)

    if not image_files:
        print("ERROR: No images found!")
        return

    if max_frames:
        image_files = image_files[:max_frames]

    print(f"Generating video from {len(image_files)} frames ({camera} camera)...")
    print(f"Labels loaded: {len(labels)}")

    # Get dimensions from first frame
    first_img = cv2.imread(str(image_files[0]))
    h, w = first_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    valid_count = 0
    invalid_count = 0

    for idx, img_path in enumerate(tqdm(image_files, desc="Creating video")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Get label
        label = labels.get(img_path.name, {'x': -1, 'y': -1, 'valid': False})
        valid = label.get('valid', False)

        if valid:
            overlay = overlay_label(img, label['x'], label['y'], True)
            valid_count += 1
        else:
            overlay = overlay_label(img, 0, 0, False)
            invalid_count += 1

        # Add frame number
        cv2.putText(overlay, f"FRAME {idx + 1}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(overlay, f"FRAME {idx + 1}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 1)

        video.write(overlay)

    video.release()

    print(f"\nVideo saved: {output_path}")
    print(f"Frames with ball: {valid_count}")
    print(f"Frames without ball: {invalid_count}")
    print(f"Duration: {len(image_files) / fps:.1f} seconds")


def generate_video_from_labels_stereo(data_dir, output_path, fps=15, max_frames=None):
    """Generate stereo review video from labels.json data (side-by-side)."""
    left_files, labels = load_labeled_frames(data_dir, 'left')
    right_files, _ = load_labeled_frames(data_dir, 'right')

    # Match left/right pairs
    left_dict = {f.name.replace('_left', ''): f for f in left_files}
    right_dict = {f.name.replace('_right', ''): f for f in right_files}

    common = sorted(set(left_dict.keys()) & set(right_dict.keys()))

    if not common:
        print("ERROR: No matching stereo pairs found!")
        return

    if max_frames:
        common = common[:max_frames]

    print(f"Generating stereo video from {len(common)} frame pairs...")

    # Get dimensions
    first_img = cv2.imread(str(left_dict[common[0]]))
    h, w = first_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (w * 2, h))

    for idx, key in enumerate(tqdm(common, desc="Creating video")):
        left_path = left_dict[key]
        right_path = right_dict[key]

        left_img = cv2.imread(str(left_path))
        right_img = cv2.imread(str(right_path))

        if left_img is None or right_img is None:
            continue

        # Get labels
        left_label = labels.get(left_path.name, {'x': -1, 'y': -1, 'valid': False})
        right_label = labels.get(right_path.name, {'x': -1, 'y': -1, 'valid': False})

        left_overlay = overlay_label(left_img, left_label.get('x', -1),
                                      left_label.get('y', -1), left_label.get('valid', False))
        right_overlay = overlay_label(right_img, right_label.get('x', -1),
                                       right_label.get('y', -1), right_label.get('valid', False))

        # Add labels
        cv2.putText(left_overlay, f"LEFT - Frame {idx + 1}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(right_overlay, f"RIGHT - Frame {idx + 1}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        combined = np.hstack([left_overlay, right_overlay])
        video.write(combined)

    video.release()

    print(f"\nVideo saved: {output_path}")
    print(f"Frames: {len(common)} stereo pairs")
    print(f"Duration: {len(common) / fps:.1f} seconds")


def generate_review_video_masks(images_dir, masks_dir, output_path, fps=15):
    """Generate a review video from masks with frame numbers."""
    frames = load_stereo_frames(images_dir, masks_dir)

    if not frames:
        print("ERROR: No frames found!")
        return

    print(f"Generating review video from {len(frames)} frame pairs...")

    # Get dimensions from first frame
    first_img = cv2.imread(str(frames[0]['left_img']))
    h, w = first_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (w * 2, h))

    for idx, frame in enumerate(tqdm(frames, desc="Creating video")):
        left_img = cv2.imread(str(frame['left_img']))
        right_img = cv2.imread(str(frame['right_img']))

        left_mask = cv2.imread(str(frame['left_mask']), cv2.IMREAD_GRAYSCALE) if frame['left_mask'].exists() else None
        right_mask = cv2.imread(str(frame['right_mask']), cv2.IMREAD_GRAYSCALE) if frame['right_mask'].exists() else None

        left_overlay = overlay_mask(left_img, left_mask)
        right_overlay = overlay_mask(right_img, right_mask)

        # Add frame number
        cv2.putText(left_overlay, f"FRAME {idx + 1}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(left_overlay, f"FRAME {idx + 1}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 1)

        cv2.putText(right_overlay, f"FRAME {idx + 1}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(right_overlay, f"FRAME {idx + 1}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 1)

        cv2.putText(left_overlay, "LEFT", (w - 100, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(right_overlay, "RIGHT", (w - 120, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        combined = np.hstack([left_overlay, right_overlay])
        video.write(combined)

    video.release()
    print(f"\nVideo saved: {output_path}")
    print(f"Duration: {len(frames) / fps:.1f} seconds")
    print(f"\nWatch the video, note bad frame numbers, then run:")
    print(f"  python -m ball_detection.segmentation_workflow.review_masks --delete 42,156,789")


# ============================================================
# FRAME DELETION
# ============================================================

def delete_frames_labels(data_dir, frame_numbers, camera='both'):
    """Delete specified frame numbers from labels.json dataset."""
    image_files, labels = load_labeled_frames(data_dir, camera)
    labels_path = Path(data_dir) / 'labels.json'

    if not image_files:
        print("ERROR: No frames found!")
        return

    # Convert to 0-indexed
    indices = [n - 1 for n in frame_numbers if 0 < n <= len(image_files)]

    if not indices:
        print("No valid frame numbers provided.")
        return

    print(f"Deleting {len(indices)} frames...")

    deleted_imgs = 0
    deleted_labels = 0

    for idx in sorted(indices, reverse=True):
        img_path = image_files[idx]

        # Delete image
        if img_path.exists():
            img_path.unlink()
            deleted_imgs += 1
            print(f"  Deleted: {img_path.name}")

        # Remove from labels
        if img_path.name in labels:
            del labels[img_path.name]
            deleted_labels += 1

    # Save updated labels
    with open(labels_path, 'w') as f:
        json.dump(labels, f, indent=2)

    print(f"\nDeleted: {deleted_imgs} images, {deleted_labels} label entries")


def delete_frames_masks(images_dir, masks_dir, frame_numbers):
    """Delete specified frame numbers from mask dataset."""
    frames = load_stereo_frames(images_dir, masks_dir)

    if not frames:
        print("ERROR: No frames found!")
        return

    indices = [n - 1 for n in frame_numbers if 0 < n <= len(frames)]

    if not indices:
        print("No valid frame numbers provided.")
        return

    print(f"Deleting {len(indices)} frames...")

    deleted_imgs = 0
    deleted_masks = 0

    for idx in indices:
        frame = frames[idx]

        for key in ['left_img', 'right_img']:
            if frame[key].exists():
                frame[key].unlink()
                deleted_imgs += 1

        for key in ['left_mask', 'right_mask']:
            if frame[key].exists():
                frame[key].unlink()
                deleted_masks += 1

        print(f"  Deleted frame {idx + 1}: {frame['id']}")

    print(f"\nDeleted: {deleted_imgs} images, {deleted_masks} masks")


# ============================================================
# MASK CORRECTION (for mask mode)
# ============================================================

def correct_frames(images_dir, masks_dir, corrections, mask_radius=20):
    """Correct masks for specified frames."""
    frames = load_stereo_frames(images_dir, masks_dir)
    masks_dir = Path(masks_dir)

    if not frames:
        print("ERROR: No frames found!")
        return

    print(f"Correcting {len(corrections)} masks...")

    for frame_num, camera, x, y in corrections:
        idx = frame_num - 1
        if idx < 0 or idx >= len(frames):
            print(f"  Warning: Frame {frame_num} out of range, skipping")
            continue

        frame = frames[idx]
        img = cv2.imread(str(frame['left_img']))
        h, w = img.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (int(x), int(y)), mask_radius, 255, -1)

        if camera in ['left', 'both']:
            cv2.imwrite(str(frame['left_mask']), mask)
            print(f"  Frame {frame_num} LEFT: ({x}, {y})")

        if camera in ['right', 'both']:
            cv2.imwrite(str(frame['right_mask']), mask)
            print(f"  Frame {frame_num} RIGHT: ({x}, {y})")

    print(f"\nCorrected {len(corrections)} masks")


def parse_corrections(correction_str):
    """Parse correction string: 'frame:camera:x,y;...'"""
    corrections = []
    for part in correction_str.split(';'):
        part = part.strip()
        if not part:
            continue
        try:
            frame_camera, coords = part.rsplit(':', 1)
            if ':' in frame_camera:
                frame_str, camera = frame_camera.split(':')
            else:
                frame_str = frame_camera
                camera = 'both'

            frame_num = int(frame_str)
            x, y = map(float, coords.split(','))
            camera = camera.lower()
            if camera not in ['left', 'right', 'both']:
                camera = 'both'
            corrections.append((frame_num, camera, x, y))
        except ValueError as e:
            print(f"  Warning: Could not parse '{part}': {e}")
    return corrections


# ============================================================
# INTERACTIVE MODE
# ============================================================

class ReviewTool:
    """Interactive tool for reviewing training data."""

    def __init__(self, data_dir=None, images_dir=None, masks_dir=None, mask_radius=20):
        self.mask_radius = mask_radius
        self.use_labels = data_dir is not None

        if self.use_labels:
            self.data_dir = Path(data_dir)
            self.image_files, self.labels = load_labeled_frames(data_dir, 'both')
            self.labels_path = self.data_dir / 'labels.json'
        else:
            self.images_dir = Path(images_dir)
            self.masks_dir = Path(masks_dir)
            self.frames = load_stereo_frames(images_dir, masks_dir)

        self.current_idx = 0
        self.current_camera = 'left'
        self.display_scale = 0.8
        self.pending_changes = {}

        self._print_help()

    def _print_help(self):
        print(f"\nReview Tool (Interactive Mode)")
        print(f"=" * 60)
        if self.use_labels:
            print(f"Data: {self.data_dir}")
            print(f"Found {len(self.image_files)} images")
        else:
            print(f"Images: {self.images_dir}")
            print(f"Masks: {self.masks_dir}")
            print(f"Found {len(self.frames)} stereo frame pairs")
        print(f"\nControls:")
        print("  Click: Set new ball center")
        print("  TAB: Switch left/right camera")
        print("  n/p: Next/previous frame")
        print("  ./,: Jump +10/-10 frames")
        print("  d: Mark for deletion")
        print("  x: Mark 'no ball'")
        print("  u: Undo pending change")
        print("  s: Save all changes")
        print("  q: Quit and save")
        print(f"=" * 60)

    def _get_frame_count(self):
        if self.use_labels:
            return len(self.image_files)
        return len(self.frames)

    def _load_current(self):
        if self.use_labels:
            img_path = self.image_files[self.current_idx]
            img = cv2.imread(str(img_path))
            label = self.labels.get(img_path.name, {'x': -1, 'y': -1, 'valid': False})

            # Check pending changes
            if img_path.name in self.pending_changes:
                change = self.pending_changes[img_path.name]
                if change == 'delete':
                    return img, None, 'MARKED FOR DELETION'
                elif change == 'no_ball':
                    return img, {'x': -1, 'y': -1, 'valid': False}, 'NO BALL'
                elif isinstance(change, dict):
                    return img, change, 'CORRECTED'

            return img, label, None
        else:
            frame = self.frames[self.current_idx]
            if self.current_camera == 'left':
                img_path, mask_path = frame['left_img'], frame['left_mask']
            else:
                img_path, mask_path = frame['right_img'], frame['right_mask']

            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None

            if img_path.name in self.pending_changes:
                change = self.pending_changes[img_path.name]
                if change == 'delete':
                    return img, None, 'MARKED FOR DELETION'
                elif change == 'no_ball':
                    return img, np.zeros_like(mask) if mask is not None else None, 'NO BALL'
                elif isinstance(change, np.ndarray):
                    return img, change, 'CORRECTED'

            return img, mask, None

    def _draw_display(self):
        img, data, status = self._load_current()

        if self.use_labels:
            if data and data.get('valid', False):
                overlay = overlay_label(img, data['x'], data['y'], True)
            else:
                overlay = overlay_label(img, 0, 0, False)
        else:
            overlay = overlay_mask(img, data)

        if status:
            cv2.putText(overlay, status,
                       (overlay.shape[1]//2 - 150, overlay.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Info
        h, w = overlay.shape[:2]
        cv2.putText(overlay, f"Frame {self.current_idx + 1}/{self._get_frame_count()} | Pending: {len(self.pending_changes)}",
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if self.display_scale != 1.0:
            new_w = int(overlay.shape[1] * self.display_scale)
            new_h = int(overlay.shape[0] * self.display_scale)
            overlay = cv2.resize(overlay, (new_w, new_h))

        return overlay

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x = int(x / self.display_scale)
            orig_y = int(y / self.display_scale)

            if self.use_labels:
                img_path = self.image_files[self.current_idx]
                self.pending_changes[img_path.name] = {'x': orig_x, 'y': orig_y, 'valid': True}
                print(f"Corrected: ({orig_x}, {orig_y})")
            else:
                frame = self.frames[self.current_idx]
                img_path = frame['left_img'] if self.current_camera == 'left' else frame['right_img']
                img = cv2.imread(str(img_path))
                mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                cv2.circle(mask, (orig_x, orig_y), self.mask_radius, 255, -1)
                self.pending_changes[img_path.name] = mask
                print(f"Corrected {self.current_camera}: ({orig_x}, {orig_y})")

    def _save_changes(self):
        if not self.pending_changes:
            print("No changes to save.")
            return

        if self.use_labels:
            deleted = 0
            corrected = 0

            for filename, change in self.pending_changes.items():
                img_path = self.data_dir / 'images' / filename

                if change == 'delete':
                    if img_path.exists():
                        img_path.unlink()
                    if filename in self.labels:
                        del self.labels[filename]
                    deleted += 1
                elif change == 'no_ball':
                    self.labels[filename] = {'x': -1, 'y': -1, 'valid': False}
                    corrected += 1
                elif isinstance(change, dict):
                    self.labels[filename] = change
                    corrected += 1

            with open(self.labels_path, 'w') as f:
                json.dump(self.labels, f, indent=2)

            print(f"\nSaved: {deleted} deleted, {corrected} corrected")

            if deleted > 0:
                self.image_files, self.labels = load_labeled_frames(self.data_dir, 'both')
                self.current_idx = min(self.current_idx, len(self.image_files) - 1)
        else:
            deleted = 0
            corrected = 0

            for filename, change in self.pending_changes.items():
                img_path = self.images_dir / filename
                mask_path = self.masks_dir / filename

                if isinstance(change, np.ndarray):
                    cv2.imwrite(str(mask_path), change)
                    corrected += 1
                elif change == 'delete':
                    if img_path.exists():
                        img_path.unlink()
                    if mask_path.exists():
                        mask_path.unlink()
                    deleted += 1
                elif change == 'no_ball':
                    if mask_path.exists():
                        mask_path.unlink()
                    corrected += 1

            print(f"\nSaved: {deleted} deleted, {corrected} corrected")

            if deleted > 0:
                self.frames = load_stereo_frames(self.images_dir, self.masks_dir)
                self.current_idx = min(self.current_idx, len(self.frames) - 1)

        self.pending_changes.clear()

    def run(self):
        window_name = 'Review Tool'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        while True:
            if self._get_frame_count() == 0:
                print("No frames remaining!")
                break

            display = self._draw_display()
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                self._save_changes()
                break
            elif key == ord('n'):
                if self.current_idx < self._get_frame_count() - 1:
                    self.current_idx += 1
            elif key == ord('p'):
                if self.current_idx > 0:
                    self.current_idx -= 1
            elif key == ord('.'):
                self.current_idx = min(self.current_idx + 10, self._get_frame_count() - 1)
            elif key == ord(','):
                self.current_idx = max(self.current_idx - 10, 0)
            elif key == 9:  # TAB
                self.current_camera = 'right' if self.current_camera == 'left' else 'left'
            elif key == ord('d'):
                if self.use_labels:
                    img_path = self.image_files[self.current_idx]
                    self.pending_changes[img_path.name] = 'delete'
                else:
                    frame = self.frames[self.current_idx]
                    img_path = frame['left_img'] if self.current_camera == 'left' else frame['right_img']
                    self.pending_changes[img_path.name] = 'delete'
            elif key == ord('x'):
                if self.use_labels:
                    img_path = self.image_files[self.current_idx]
                    self.pending_changes[img_path.name] = 'no_ball'
                else:
                    frame = self.frames[self.current_idx]
                    img_path = frame['left_img'] if self.current_camera == 'left' else frame['right_img']
                    self.pending_changes[img_path.name] = 'no_ball'
            elif key == ord('u'):
                if self.use_labels:
                    img_path = self.image_files[self.current_idx]
                    if img_path.name in self.pending_changes:
                        del self.pending_changes[img_path.name]
                else:
                    frame = self.frames[self.current_idx]
                    img_path = frame['left_img'] if self.current_camera == 'left' else frame['right_img']
                    if img_path.name in self.pending_changes:
                        del self.pending_changes[img_path.name]
            elif key == ord('s'):
                self._save_changes()

        cv2.destroyAllWindows()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Training Data Review Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate stereo video from labels.json (default)
  python -m ball_detection.segmentation_workflow.review_masks --data-dir ./data

  # Generate single camera video
  python -m ball_detection.segmentation_workflow.review_masks --data-dir ./data --camera left

  # Generate video from masks
  python -m ball_detection.segmentation_workflow.review_masks --images-dir ./images --masks-dir ./masks

  # Delete frames
  python -m ball_detection.segmentation_workflow.review_masks --data-dir ./data --delete 42,156,789

  # Interactive mode
  python -m ball_detection.segmentation_workflow.review_masks --data-dir ./data --interactive
        """
    )

    # Data source (either --data-dir for labels.json, or --images-dir + --masks-dir)
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Data directory with images/ and labels.json')
    parser.add_argument('--images-dir', type=str, default=IMAGES_DIR,
                       help='Images directory (for mask mode)')
    parser.add_argument('--masks-dir', type=str, default=MASKS_DIR,
                       help='Masks directory (for mask mode)')

    # Output
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path')
    parser.add_argument('--fps', type=int, default=FPS,
                       help=f'Video FPS (default: {FPS})')

    # Options
    parser.add_argument('--camera', type=str, choices=['left', 'right'], default=None,
                       help='Single camera only (default: stereo)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process')

    # Actions
    parser.add_argument('--delete', type=str, default=None,
                       help='Comma-separated frame numbers to delete')
    parser.add_argument('--delete-file', type=str, default=None,
                       help='File with frame numbers to delete')
    parser.add_argument('--correct', type=str, default=None,
                       help='Correct masks: "frame:camera:x,y;..."')
    parser.add_argument('--interactive', action='store_true',
                       help='Run interactive GUI mode')
    parser.add_argument('--radius', type=int, default=MASK_RADIUS,
                       help=f'Mask radius for corrections (default: {MASK_RADIUS})')

    args = parser.parse_args()

    print("=" * 60)
    print("TRAINING DATA REVIEW TOOL")
    print("=" * 60)

    use_labels = args.data_dir is not None

    # Determine output path
    if args.output:
        output_path = args.output
    elif use_labels:
        if args.camera:
            output_path = str(Path(args.data_dir) / f'review_{args.camera}.mp4')
        else:
            output_path = str(Path(args.data_dir) / 'review_stereo.mp4')
    else:
        output_path = OUTPUT_VIDEO

    # Handle actions
    if args.delete:
        frame_numbers = [int(n.strip()) for n in args.delete.split(',') if n.strip().isdigit()]
        if use_labels:
            delete_frames_labels(args.data_dir, frame_numbers, args.camera or 'both')
        else:
            delete_frames_masks(args.images_dir, args.masks_dir, frame_numbers)

    elif args.delete_file:
        frame_numbers = []
        with open(args.delete_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.isdigit():
                    frame_numbers.append(int(line))
        if use_labels:
            delete_frames_labels(args.data_dir, frame_numbers, args.camera or 'both')
        else:
            delete_frames_masks(args.images_dir, args.masks_dir, frame_numbers)

    elif args.correct and not use_labels:
        corrections = parse_corrections(args.correct)
        correct_frames(args.images_dir, args.masks_dir, corrections, args.radius)

    elif args.interactive:
        if use_labels:
            tool = ReviewTool(data_dir=args.data_dir, mask_radius=args.radius)
        else:
            tool = ReviewTool(images_dir=args.images_dir, masks_dir=args.masks_dir, mask_radius=args.radius)
        tool.run()

    else:
        # Default: generate video (stereo by default)
        if use_labels:
            if args.camera:
                generate_video_from_labels(args.data_dir, output_path, args.camera, args.fps, args.max_frames)
            else:
                generate_video_from_labels_stereo(args.data_dir, output_path, args.fps, args.max_frames)
        else:
            generate_review_video_masks(args.images_dir, args.masks_dir, output_path, args.fps)


if __name__ == "__main__":
    main()
