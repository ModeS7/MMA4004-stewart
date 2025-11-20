"""
Interactive Labeling GUI for Dual Camera Ball Detection

Labels ball centers in dual camera stereo videos (2560x720).
Automatically splits frames into left (0:1280) and right (1280:2560).
Shows both cameras side-by-side with green border on active camera.

Controls:
    - Click on ball center to label
    - TAB: Switch between left/right camera
    - 'n': Next frame pair
    - 'p': Previous frame pair
    - 's': Save current label
    - 'd': Delete current label
    - 'r': Toggle ROI auto-detection helper
    - '+/-': Zoom in/out
    - 'q': Quit and save all
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path


class BallLabelingTool:
    """Interactive tool for labeling ball centers."""

    def __init__(self, input_path, output_dir=None, crop_size=64):
        self.input_path = Path(input_path)
        self.crop_size = crop_size

        # Setup output directory
        if output_dir is None:
            output_dir = self.input_path.parent / 'labeled_data'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.image_dir = self.output_dir / 'images'
        self.image_dir.mkdir(exist_ok=True)

        self.labels_file = self.output_dir / 'labels.json'

        # Load existing labels if any
        if self.labels_file.exists():
            with open(self.labels_file, 'r') as f:
                self.labels = json.load(f)
            print(f"Loaded {len(self.labels)} existing labels")
        else:
            self.labels = {}

        # Load images or video (always dual camera)
        self.images = []
        self.image_names = []
        self._load_input()

        # Current state
        self.current_idx = 0
        self.current_label = None
        self.show_roi_helper = False
        self.display_scale = 1.0
        self.current_camera = 'left'  # 'left' or 'right'

        # HSV thresholds for red ball ROI helper
        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])

        print(f"\nDual Camera Labeling GUI")
        print(f"Total frames: {len(self.images) // 2}")
        print(f"Output directory: {self.output_dir}")
        print(f"\nControls:")
        print("  Click: Label ball center")
        print("  TAB: Switch between left/right camera")
        print("  n: Next frame pair")
        print("  p: Previous frame pair")
        print("  s: Save label")
        print("  d: Delete label")
        print("  r: Toggle ROI helper")
        print("  +/-: Zoom in/out")
        print("  q: Quit and save\n")

    def _load_input(self):
        """Load dual camera video (2560x720) and split into left/right."""
        if not self.input_path.is_file():
            raise ValueError(f"Expected video file, got: {self.input_path}")

        print(f"Loading dual camera video: {self.input_path}")
        self._extract_frames_from_video()

    def _extract_frames_from_video(self):
        """Extract and split dual camera frames (2560x720 -> left + right)."""
        cap = cv2.VideoCapture(str(self.input_path))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            if w < 2560:
                print(f"Warning: Frame {frame_idx} is {w}x{h}, expected 2560x720")
                print("This doesn't look like a dual camera video!")
                frame_idx += 1
                continue

            # Split 2560x720 into left and right
            left_frame = frame[:, 0:1280]
            right_frame = frame[:, 1280:2560]

            # Save left frame
            left_name = f"frame_{frame_idx:06d}_left.jpg"
            left_path = self.image_dir / left_name
            if not left_path.exists():
                cv2.imwrite(str(left_path), left_frame)
            self.images.append(left_frame)
            self.image_names.append(left_name)

            # Save right frame
            right_name = f"frame_{frame_idx:06d}_right.jpg"
            right_path = self.image_dir / right_name
            if not right_path.exists():
                cv2.imwrite(str(right_path), right_frame)
            self.images.append(right_frame)
            self.image_names.append(right_name)

            frame_idx += 1

        cap.release()
        print(f"Extracted {frame_idx} frames, split into {len(self.images)} images (left + right)")


    def _detect_roi(self, image):
        """Detect red ball ROI using HSV color filtering."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red wraps around in HSV
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Find largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)

            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return (cx, cy), mask

        return None, mask

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert from display coordinates to original image coordinates
            orig_x = int(x / self.display_scale)
            orig_y = int(y / self.display_scale)

            # Determine which camera was clicked
            # Display shows left and right side-by-side (each 1280 wide)
            if orig_x < 1280:
                # Clicked on left camera
                self.current_camera = 'left'
                self.current_idx = (self.current_idx // 2) * 2  # Left index
            else:
                # Clicked on right camera
                self.current_camera = 'right'
                self.current_idx = (self.current_idx // 2) * 2 + 1  # Right index
                orig_x -= 1280  # Adjust x coordinate to right image space

            self.current_label = {'x': orig_x, 'y': orig_y, 'valid': True}
            print(f"Labeled ({self.current_camera}): x={orig_x}, y={orig_y}")

    def _draw_display(self):
        """Create display image with both cameras side-by-side."""
        # Show both left and right frames side by side
        left_idx = (self.current_idx // 2) * 2
        right_idx = left_idx + 1

        if right_idx >= len(self.images):
            # Edge case: incomplete pair, just show what we have
            return self.images[self.current_idx].copy()

        left_img = self.images[left_idx].copy()
        right_img = self.images[right_idx].copy()
        left_name = self.image_names[left_idx]
        right_name = self.image_names[right_idx]

        # Highlight current camera
        if self.current_camera == 'left':
            active_img, active_name = left_img, left_name
            cv2.rectangle(left_img, (0, 0), (left_img.shape[1]-1, left_img.shape[0]-1),
                         (0, 255, 0), 3)
        else:
            active_img, active_name = right_img, right_name
            cv2.rectangle(right_img, (0, 0), (right_img.shape[1]-1, right_img.shape[0]-1),
                         (0, 255, 0), 3)

        # Draw labels and ROI on active camera
        self._draw_labels_on_image(active_img, active_name)

        # Stack horizontally
        img = np.hstack([left_img, right_img])

        # Info text
        h, w = img.shape[:2]
        info_bg = np.zeros((100, w, 3), dtype=np.uint8)

        frame_num = self.current_idx // 2 + 1
        cv2.putText(info_bg, f"Frame {frame_num}/{len(self.images)//2} | Camera: {self.current_camera.upper()}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        left_status = "Saved" if left_name in self.labels else "Unsaved"
        right_status = "Saved" if right_name in self.labels else "Unsaved"
        cv2.putText(info_bg, f"Left: {left_status} | Right: {right_status}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if self.current_label:
            cv2.putText(info_bg, f"Label: ({self.current_label['x']}, {self.current_label['y']})",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        display = np.vstack([img, info_bg])

        # Scale for display
        if self.display_scale != 1.0:
            new_w = int(display.shape[1] * self.display_scale)
            new_h = int(display.shape[0] * self.display_scale)
            display = cv2.resize(display, (new_w, new_h))

        return display

    def _draw_labels_on_image(self, img, img_name):
        """Draw ROI helper and labels on an image."""
        # Draw ROI helper if enabled
        if self.show_roi_helper:
            roi_center, mask = self._detect_roi(img)
            if roi_center is not None:
                cv2.circle(img, roi_center, 5, (0, 255, 255), 2)
                cv2.circle(img, roi_center, self.crop_size//2, (0, 255, 255), 1)

            # Show mask in corner
            mask_resized = cv2.resize(mask, (mask.shape[1]//4, mask.shape[0]//4))
            mask_rgb = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
            h, w = mask_rgb.shape[:2]
            img[:h, :w] = mask_rgb

        # Draw current label
        if self.current_label is not None:
            cx = self.current_label['x']
            cy = self.current_label['y']
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
            cv2.circle(img, (cx, cy), self.crop_size//2, (0, 255, 0), 2)

            # Draw crosshair
            cv2.line(img, (cx-10, cy), (cx+10, cy), (0, 255, 0), 1)
            cv2.line(img, (cx, cy-10), (cx, cy+10), (0, 255, 0), 1)

        # Draw saved label if different from current
        if img_name in self.labels and self.labels[img_name] != self.current_label:
            saved_label = self.labels[img_name]
            sx = saved_label['x']
            sy = saved_label['y']
            cv2.circle(img, (sx, sy), 5, (255, 0, 0), -1)
            cv2.circle(img, (sx, sy), self.crop_size//2, (255, 0, 0), 1)

    def _save_labels(self):
        """Save labels to JSON file."""
        with open(self.labels_file, 'w') as f:
            json.dump(self.labels, f, indent=2)
        print(f"Saved {len(self.labels)} labels to {self.labels_file}")

    def run(self):
        """Run the labeling GUI."""
        window_name = 'Ball Labeling Tool'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        # Load current label if exists
        img_name = self.image_names[self.current_idx]
        if img_name in self.labels:
            self.current_label = self.labels[img_name].copy()

        while True:
            # Display
            display = self._draw_display()
            cv2.imshow(window_name, display)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                # Quit
                print("\nQuitting...")
                self._save_labels()
                break

            elif key == ord('n'):
                # Next frame pair
                if self.current_idx < len(self.images) - 2:
                    self.current_idx = ((self.current_idx // 2) + 1) * 2
                    if self.current_camera == 'right':
                        self.current_idx += 1  # Move to right
                    img_name = self.image_names[self.current_idx]
                    self.current_label = self.labels.get(img_name, None)
                    if self.current_label:
                        self.current_label = self.current_label.copy()
                    print(f"Frame {self.current_idx//2 + 1}/{len(self.images)//2}: {img_name}")

            elif key == ord('p'):
                # Previous frame pair
                if self.current_idx >= 2:
                    self.current_idx = ((self.current_idx // 2) - 1) * 2
                    if self.current_camera == 'right':
                        self.current_idx += 1  # Move to right
                    img_name = self.image_names[self.current_idx]
                    self.current_label = self.labels.get(img_name, None)
                    if self.current_label:
                        self.current_label = self.current_label.copy()
                    print(f"Frame {self.current_idx//2 + 1}/{len(self.images)//2}: {img_name}")

            elif key == 9:  # TAB key
                # Switch between left and right
                if self.current_camera == 'left':
                    self.current_camera = 'right'
                    self.current_idx = (self.current_idx // 2) * 2 + 1
                else:
                    self.current_camera = 'left'
                    self.current_idx = (self.current_idx // 2) * 2

                img_name = self.image_names[self.current_idx]
                self.current_label = self.labels.get(img_name, None)
                if self.current_label:
                    self.current_label = self.current_label.copy()
                print(f"Switched to {self.current_camera}: {img_name}")

            elif key == ord('s'):
                # Save current label
                if self.current_label is not None:
                    img_name = self.image_names[self.current_idx]
                    self.labels[img_name] = self.current_label.copy()
                    print(f"Saved label for {img_name}")
                    self._save_labels()

            elif key == ord('d'):
                # Delete label
                img_name = self.image_names[self.current_idx]
                if img_name in self.labels:
                    del self.labels[img_name]
                    print(f"Deleted label for {img_name}")
                    self._save_labels()
                self.current_label = None

            elif key == ord('r'):
                # Toggle ROI helper
                self.show_roi_helper = not self.show_roi_helper
                print(f"ROI helper: {'ON' if self.show_roi_helper else 'OFF'}")

            elif key == ord('+') or key == ord('='):
                # Zoom in
                self.display_scale = min(self.display_scale * 1.2, 3.0)
                print(f"Scale: {self.display_scale:.2f}x")

            elif key == ord('-') or key == ord('_'):
                # Zoom out
                self.display_scale = max(self.display_scale / 1.2, 0.3)
                print(f"Scale: {self.display_scale:.2f}x")

        cv2.destroyAllWindows()
        print("\nLabeling complete!")
        print(f"Total labels: {len(self.labels)}/{len(self.images)}")


def main():
    parser = argparse.ArgumentParser(
        description='Dual Camera Ball Labeling Tool for 2560x720 stereo videos'
    )

    parser.add_argument('input', type=str,
                        help='Input dual camera video file (2560x720)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for labeled data')
    parser.add_argument('--crop-size', type=int, default=64,
                        help='Crop size for visualization')

    args = parser.parse_args()

    tool = BallLabelingTool(
        input_path=args.input,
        output_dir=args.output,
        crop_size=args.crop_size
    )

    tool.run()


if __name__ == "__main__":
    main()
