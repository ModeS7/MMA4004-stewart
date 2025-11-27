#!/usr/bin/env python3
"""
Mask Review and Correction Tool

Two modes:
1. Video mode (default): Generate review video, then delete frames by number
2. Interactive mode (requires X server): Full GUI with click-to-correct

Usage:
    # Generate review video
    python -m ball_detection.segmentation_workflow.review_masks

    # Delete specific frames (by frame number from video)
    python -m ball_detection.segmentation_workflow.review_masks --delete 42,156,789

    # Delete frames from a file (one number per line)
    python -m ball_detection.segmentation_workflow.review_masks --delete-file bad_frames.txt

    # Interactive mode (requires X server)
    python -m ball_detection.segmentation_workflow.review_masks --interactive
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

# ============================================================
# SETTINGS - Edit these
# ============================================================
IMAGES_DIR = "./ball_detection/data/new_labels/images"
MASKS_DIR = "./ball_detection/data/new_labels/auto_labeled/masks"
OUTPUT_VIDEO = "./ball_detection/segmentation_workflow/mask_review.mp4"
MASK_RADIUS = 20  # Radius for generated circular masks when correcting
FPS = 15  # Video playback speed
# ============================================================


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


def load_frames(images_dir, masks_dir):
    """Load stereo frame pairs."""
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


def generate_review_video(images_dir, masks_dir, output_path, fps=15):
    """Generate a review video with frame numbers for easy identification."""
    frames = load_frames(images_dir, masks_dir)

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
        # Load images and masks
        left_img = cv2.imread(str(frame['left_img']))
        right_img = cv2.imread(str(frame['right_img']))

        left_mask = cv2.imread(str(frame['left_mask']), cv2.IMREAD_GRAYSCALE) if frame['left_mask'].exists() else None
        right_mask = cv2.imread(str(frame['right_mask']), cv2.IMREAD_GRAYSCALE) if frame['right_mask'].exists() else None

        # Create overlays
        left_overlay = overlay_mask(left_img, left_mask)
        right_overlay = overlay_mask(right_img, right_mask)

        # Add frame number (large, visible)
        cv2.putText(left_overlay, f"FRAME {idx + 1}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(left_overlay, f"FRAME {idx + 1}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 1)

        cv2.putText(right_overlay, f"FRAME {idx + 1}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(right_overlay, f"FRAME {idx + 1}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 1)

        # Add camera labels
        cv2.putText(left_overlay, "LEFT", (w - 100, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(right_overlay, "RIGHT", (w - 120, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Combine
        combined = np.hstack([left_overlay, right_overlay])
        video.write(combined)

    video.release()
    print(f"\nVideo saved: {output_path}")
    print(f"Duration: {len(frames) / fps:.1f} seconds")
    print(f"\nWatch the video, note bad frame numbers, then run:")
    print(f"  python -m ball_detection.segmentation_workflow.review_masks --delete 42,156,789")


def delete_frames(images_dir, masks_dir, frame_numbers, delete_both=True):
    """Delete specified frame numbers (1-indexed)."""
    frames = load_frames(images_dir, masks_dir)

    if not frames:
        print("ERROR: No frames found!")
        return

    # Convert to 0-indexed
    indices = [n - 1 for n in frame_numbers if 0 < n <= len(frames)]

    if not indices:
        print("No valid frame numbers provided.")
        return

    print(f"Deleting {len(indices)} frames...")

    deleted_imgs = 0
    deleted_masks = 0

    for idx in indices:
        frame = frames[idx]

        # Delete left
        if frame['left_img'].exists():
            frame['left_img'].unlink()
            deleted_imgs += 1
        if frame['left_mask'].exists():
            frame['left_mask'].unlink()
            deleted_masks += 1

        # Delete right
        if frame['right_img'].exists():
            frame['right_img'].unlink()
            deleted_imgs += 1
        if frame['right_mask'].exists():
            frame['right_mask'].unlink()
            deleted_masks += 1

        print(f"  Deleted frame {idx + 1}: {frame['id']}")

    print(f"\nDeleted: {deleted_imgs} images, {deleted_masks} masks")


def correct_frames(images_dir, masks_dir, corrections, mask_radius=20):
    """
    Correct masks for specified frames.

    corrections: list of (frame_num, camera, x, y) tuples
        - frame_num: 1-indexed frame number
        - camera: 'left', 'right', or 'both'
        - x, y: ball center coordinates
    """
    frames = load_frames(images_dir, masks_dir)
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

        # Get image dimensions
        img = cv2.imread(str(frame['left_img']))
        h, w = img.shape[:2]

        # Generate circular mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (int(x), int(y)), mask_radius, 255, -1)

        # Save to appropriate camera(s)
        if camera in ['left', 'both']:
            cv2.imwrite(str(frame['left_mask']), mask)
            print(f"  Frame {frame_num} LEFT: ({x}, {y})")

        if camera in ['right', 'both']:
            cv2.imwrite(str(frame['right_mask']), mask)
            print(f"  Frame {frame_num} RIGHT: ({x}, {y})")

    print(f"\nCorrected {len(corrections)} masks")


def parse_corrections(correction_str):
    """
    Parse correction string.

    Format: "frame:camera:x,y;frame:camera:x,y;..."
    Examples:
        "42:left:640,360"
        "42:left:640,360;43:right:500,400"
        "42:both:640,360"  (same position for both cameras)
    """
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


def parse_corrections_file(filepath):
    """
    Parse corrections from file.

    Format (one per line):
        frame,camera,x,y
    Examples:
        42,left,640,360
        43,right,500,400
        44,both,600,350
    """
    corrections = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                parts = line.split(',')
                if len(parts) == 4:
                    frame_num = int(parts[0])
                    camera = parts[1].lower()
                    x = float(parts[2])
                    y = float(parts[3])
                elif len(parts) == 3:
                    # No camera specified, assume both
                    frame_num = int(parts[0])
                    camera = 'both'
                    x = float(parts[1])
                    y = float(parts[2])
                else:
                    print(f"  Warning: Could not parse '{line}'")
                    continue

                if camera not in ['left', 'right', 'both']:
                    camera = 'both'
                corrections.append((frame_num, camera, x, y))
            except ValueError as e:
                print(f"  Warning: Could not parse '{line}': {e}")
    return corrections


class MaskReviewTool:
    """Interactive tool for reviewing and correcting segmentation masks (requires X server)."""

    def __init__(self, images_dir, masks_dir, mask_radius=20):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.mask_radius = mask_radius

        # Load image/mask pairs
        self.frames = load_frames(images_dir, masks_dir)

        if not self.frames:
            raise ValueError(f"No image/mask pairs found in {images_dir}")

        # State
        self.current_idx = 0
        self.current_camera = 'left'
        self.display_scale = 0.8
        self.pending_changes = {}  # {filename: 'delete' | 'no_ball' | mask_array}

        print(f"\nMask Review Tool (Interactive Mode)")
        print(f"=" * 60)
        print(f"Images: {self.images_dir}")
        print(f"Masks: {self.masks_dir}")
        print(f"Found {len(self.frames)} stereo frame pairs")
        print(f"\nControls:")
        print("  Click: Set new ball center (generates circular mask)")
        print("  TAB: Switch between left/right camera")
        print("  n/p: Next/previous frame")
        print("  ./,: Jump +10/-10 frames")
        print("  ]/[: Jump +100/-100 frames")
        print("  g: Go to specific frame number")
        print("  d: Delete current frame (image + mask)")
        print("  x: Mark 'no ball' (delete mask only)")
        print("  u: Undo pending change")
        print("  s: Save all changes")
        print("  q: Quit and save")
        print("  +/-: Zoom in/out")
        print(f"=" * 60)
        print()

    def _get_current_paths(self):
        """Get paths for current frame/camera."""
        frame = self.frames[self.current_idx]
        if self.current_camera == 'left':
            return frame['left_img'], frame['left_mask']
        else:
            return frame['right_img'], frame['right_mask']

    def _load_image_and_mask(self, img_path, mask_path):
        """Load image and mask, handling pending changes."""
        img = cv2.imread(str(img_path))

        filename = img_path.name

        # Check for pending changes
        if filename in self.pending_changes:
            change = self.pending_changes[filename]
            if change == 'delete':
                return img, None, 'MARKED FOR DELETION'
            elif change == 'no_ball':
                return img, np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8), 'NO BALL'
            elif isinstance(change, np.ndarray):
                return img, change, 'CORRECTED'

        # Load mask from file
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        return img, mask, None

    def _overlay_mask(self, image, mask):
        """Create image with mask overlay."""
        overlay = image.copy()

        if mask is None:
            return overlay

        # Green overlay for ball region
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 127] = [0, 255, 0]
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)

        # Draw contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        # Draw center of largest blob
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)
                cv2.circle(overlay, (cx, cy), 20, (0, 0, 255), 2)

        return overlay

    def _generate_circular_mask(self, center, img_shape):
        """Generate a circular mask at clicked position."""
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, self.mask_radius, 255, -1)
        return mask

    def _draw_display(self):
        """Create display with both cameras side-by-side."""
        frame = self.frames[self.current_idx]

        # Load left
        left_img, left_mask, left_status = self._load_image_and_mask(
            frame['left_img'], frame['left_mask']
        )
        left_overlay = self._overlay_mask(left_img, left_mask)

        # Load right
        right_img, right_mask, right_status = self._load_image_and_mask(
            frame['right_img'], frame['right_mask']
        )
        right_overlay = self._overlay_mask(right_img, right_mask)

        # Highlight current camera
        if self.current_camera == 'left':
            cv2.rectangle(left_overlay, (0, 0),
                         (left_overlay.shape[1]-1, left_overlay.shape[0]-1),
                         (0, 255, 0), 4)
            current_status = left_status
        else:
            cv2.rectangle(right_overlay, (0, 0),
                         (right_overlay.shape[1]-1, right_overlay.shape[0]-1),
                         (0, 255, 0), 4)
            current_status = right_status

        # Draw status on current camera
        if current_status:
            if self.current_camera == 'left':
                cv2.putText(left_overlay, current_status,
                           (left_overlay.shape[1]//2 - 150, left_overlay.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                cv2.putText(right_overlay, current_status,
                           (right_overlay.shape[1]//2 - 150, right_overlay.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Combine side-by-side
        combined = np.hstack([left_overlay, right_overlay])

        # Info bar
        h, w = combined.shape[:2]
        info_bg = np.zeros((80, w, 3), dtype=np.uint8)

        cv2.putText(info_bg,
                   f"Frame {self.current_idx + 1}/{len(self.frames)} | Camera: {self.current_camera.upper()} | Pending: {len(self.pending_changes)}",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.putText(info_bg,
                   f"LEFT: {frame['left_img'].name} | RIGHT: {frame['right_img'].name}",
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        display = np.vstack([combined, info_bg])

        # Scale
        if self.display_scale != 1.0:
            new_w = int(display.shape[1] * self.display_scale)
            new_h = int(display.shape[0] * self.display_scale)
            display = cv2.resize(display, (new_w, new_h))

        return display

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to correct mask position."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert from display to original coordinates
            orig_x = int(x / self.display_scale)
            orig_y = int(y / self.display_scale)

            # Determine which camera was clicked
            frame = self.frames[self.current_idx]
            left_img = cv2.imread(str(frame['left_img']))
            img_width = left_img.shape[1]

            if orig_x < img_width:
                # Left camera
                self.current_camera = 'left'
                click_x, click_y = orig_x, orig_y
                img_path = frame['left_img']
            else:
                # Right camera
                self.current_camera = 'right'
                click_x = orig_x - img_width
                click_y = orig_y
                img_path = frame['right_img']

            # Generate new circular mask at clicked position
            img = cv2.imread(str(img_path))
            new_mask = self._generate_circular_mask((click_x, click_y), img.shape)
            self.pending_changes[img_path.name] = new_mask

            print(f"Corrected {self.current_camera}: ({click_x}, {click_y})")

    def _save_changes(self):
        """Apply all pending changes."""
        if not self.pending_changes:
            print("No changes to save.")
            return

        deleted = 0
        corrected = 0
        no_ball = 0

        for filename, change in self.pending_changes.items():
            img_path = self.images_dir / filename
            mask_path = self.masks_dir / filename

            if change == 'delete':
                # Delete both image and mask
                if img_path.exists():
                    img_path.unlink()
                if mask_path.exists():
                    mask_path.unlink()
                deleted += 1
                print(f"  Deleted: {filename}")

            elif change == 'no_ball':
                # Delete mask only
                if mask_path.exists():
                    mask_path.unlink()
                no_ball += 1
                print(f"  No ball: {filename}")

            elif isinstance(change, np.ndarray):
                # Save corrected mask
                cv2.imwrite(str(mask_path), change)
                corrected += 1
                print(f"  Corrected: {filename}")

        print(f"\nSaved: {deleted} deleted, {no_ball} no-ball, {corrected} corrected")
        self.pending_changes.clear()

        # Reload frames after deletion
        if deleted > 0:
            self.frames.clear()
            self._load_frames()
            self.current_idx = min(self.current_idx, len(self.frames) - 1)
            if self.current_idx < 0:
                self.current_idx = 0

    def run(self):
        """Run the review GUI."""
        window_name = 'Mask Review Tool'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        while True:
            if not self.frames:
                print("No frames remaining!")
                break

            display = self._draw_display()
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                self._save_changes()
                break

            elif key == ord('n'):
                # Next frame
                if self.current_idx < len(self.frames) - 1:
                    self.current_idx += 1
                    print(f"Frame {self.current_idx + 1}/{len(self.frames)}")

            elif key == ord('p'):
                # Previous frame
                if self.current_idx > 0:
                    self.current_idx -= 1
                    print(f"Frame {self.current_idx + 1}/{len(self.frames)}")

            elif key == ord('.'):
                # Jump +10 frames
                self.current_idx = min(self.current_idx + 10, len(self.frames) - 1)
                print(f"Frame {self.current_idx + 1}/{len(self.frames)}")

            elif key == ord(','):
                # Jump -10 frames
                self.current_idx = max(self.current_idx - 10, 0)
                print(f"Frame {self.current_idx + 1}/{len(self.frames)}")

            elif key == ord(']'):
                # Jump +100 frames
                self.current_idx = min(self.current_idx + 100, len(self.frames) - 1)
                print(f"Frame {self.current_idx + 1}/{len(self.frames)}")

            elif key == ord('['):
                # Jump -100 frames
                self.current_idx = max(self.current_idx - 100, 0)
                print(f"Frame {self.current_idx + 1}/{len(self.frames)}")

            elif key == ord('g'):
                # Go to specific frame
                cv2.destroyWindow(window_name)
                try:
                    frame_input = input(f"Go to frame (1-{len(self.frames)}): ")
                    frame_num = int(frame_input)
                    if 1 <= frame_num <= len(self.frames):
                        self.current_idx = frame_num - 1
                        print(f"Jumped to frame {self.current_idx + 1}")
                    else:
                        print(f"Invalid frame number")
                except ValueError:
                    print("Invalid input")
                cv2.namedWindow(window_name)
                cv2.setMouseCallback(window_name, self._mouse_callback)

            elif key == 9:  # TAB
                # Switch camera
                self.current_camera = 'right' if self.current_camera == 'left' else 'left'
                print(f"Switched to {self.current_camera}")

            elif key == ord('d'):
                # Delete current frame
                img_path, _ = self._get_current_paths()
                self.pending_changes[img_path.name] = 'delete'
                print(f"Marked for deletion: {img_path.name}")

            elif key == ord('x'):
                # Mark no ball
                img_path, _ = self._get_current_paths()
                self.pending_changes[img_path.name] = 'no_ball'
                print(f"Marked no ball: {img_path.name}")

            elif key == ord('u'):
                # Undo pending change
                img_path, _ = self._get_current_paths()
                if img_path.name in self.pending_changes:
                    del self.pending_changes[img_path.name]
                    print(f"Undone: {img_path.name}")

            elif key == ord('s'):
                # Save changes
                print("\nSaving changes...")
                self._save_changes()

            elif key == ord('+') or key == ord('='):
                self.display_scale = min(self.display_scale * 1.2, 2.0)
                print(f"Scale: {self.display_scale:.2f}x")

            elif key == ord('-') or key == ord('_'):
                self.display_scale = max(self.display_scale / 1.2, 0.3)
                print(f"Scale: {self.display_scale:.2f}x")

        cv2.destroyAllWindows()
        print("\nReview complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Mask Review and Correction Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate review video (default)
  python -m ball_detection.segmentation_workflow.review_masks

  # Delete specific frames by number (from video)
  python -m ball_detection.segmentation_workflow.review_masks --delete 42,156,789

  # Delete frames listed in a file (one number per line)
  python -m ball_detection.segmentation_workflow.review_masks --delete-file bad_frames.txt

  # Correct masks by specifying ball position
  python -m ball_detection.segmentation_workflow.review_masks --correct "42:left:640,360;43:right:500,400"

  # Correct from file (format: frame,camera,x,y per line)
  python -m ball_detection.segmentation_workflow.review_masks --correct-file corrections.txt

  # Interactive mode (requires X server/display)
  python -m ball_detection.segmentation_workflow.review_masks --interactive

Correction formats:
  Command line: "frame:camera:x,y" separated by semicolons
    - 42:left:640,360      (correct left camera of frame 42)
    - 42:right:500,400     (correct right camera)
    - 42:both:600,350      (same position for both cameras)

  File format (one per line):
    frame,camera,x,y
    42,left,640,360
    43,right,500,400
        """
    )

    parser.add_argument('--delete', type=str, default=None,
                       help='Comma-separated frame numbers to delete (e.g., 42,156,789)')
    parser.add_argument('--delete-file', type=str, default=None,
                       help='File containing frame numbers to delete (one per line)')
    parser.add_argument('--correct', type=str, default=None,
                       help='Correct masks: "frame:camera:x,y;..." (see examples)')
    parser.add_argument('--correct-file', type=str, default=None,
                       help='File with corrections (frame,camera,x,y per line)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run interactive GUI mode (requires X server)')
    parser.add_argument('--images-dir', type=str, default=IMAGES_DIR,
                       help=f'Images directory (default: {IMAGES_DIR})')
    parser.add_argument('--masks-dir', type=str, default=MASKS_DIR,
                       help=f'Masks directory (default: {MASKS_DIR})')
    parser.add_argument('--output', type=str, default=OUTPUT_VIDEO,
                       help=f'Output video path (default: {OUTPUT_VIDEO})')
    parser.add_argument('--fps', type=int, default=FPS,
                       help=f'Video FPS (default: {FPS})')
    parser.add_argument('--radius', type=int, default=MASK_RADIUS,
                       help=f'Mask radius for corrections (default: {MASK_RADIUS})')

    args = parser.parse_args()

    print("=" * 60)
    print("MASK REVIEW AND CORRECTION TOOL")
    print("=" * 60)

    if args.delete:
        # Delete specified frames
        frame_numbers = [int(n.strip()) for n in args.delete.split(',') if n.strip().isdigit()]
        delete_frames(args.images_dir, args.masks_dir, frame_numbers)

    elif args.delete_file:
        # Delete frames from file
        frame_numbers = []
        with open(args.delete_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.isdigit():
                    frame_numbers.append(int(line))
        delete_frames(args.images_dir, args.masks_dir, frame_numbers)

    elif args.correct:
        # Correct masks from command line
        corrections = parse_corrections(args.correct)
        correct_frames(args.images_dir, args.masks_dir, corrections, args.radius)

    elif args.correct_file:
        # Correct masks from file
        corrections = parse_corrections_file(args.correct_file)
        correct_frames(args.images_dir, args.masks_dir, corrections, args.radius)

    elif args.interactive:
        # Interactive GUI mode
        tool = MaskReviewTool(
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            mask_radius=args.radius
        )
        tool.run()

    else:
        # Default: generate review video
        generate_review_video(args.images_dir, args.masks_dir, args.output, args.fps)


if __name__ == "__main__":
    main()
