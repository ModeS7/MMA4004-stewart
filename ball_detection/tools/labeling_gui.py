"""
Interactive Labeling GUI for Ball Detection

Simple tool to quickly label ball centers in images.
Supports video files and image directories.

Controls:
    - Click on ball center to label
    - 'n': Next image
    - 'p': Previous image
    - 's': Save current label
    - 'd': Delete current label
    - 'q': Quit and save all
    - 'r': Toggle ROI auto-detection helper
    - '+/-': Adjust displayed image size
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

        # Load images or video
        self.images = []
        self.image_names = []
        self._load_input()

        # Current state
        self.current_idx = 0
        self.current_label = None
        self.show_roi_helper = False
        self.display_scale = 1.0

        # HSV thresholds for red ball ROI helper
        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])

        print(f"\nLabeling GUI initialized")
        print(f"Total images: {len(self.images)}")
        print(f"Output directory: {self.output_dir}")
        print(f"\nControls:")
        print("  Click: Label ball center")
        print("  n: Next image")
        print("  p: Previous image")
        print("  s: Save label")
        print("  d: Delete label")
        print("  r: Toggle ROI helper")
        print("  +/-: Zoom in/out")
        print("  q: Quit and save\n")

    def _load_input(self):
        """Load images from video or directory."""
        if self.input_path.is_file():
            # Video file
            print(f"Loading video: {self.input_path}")
            self._extract_frames_from_video()
        elif self.input_path.is_dir():
            # Image directory
            print(f"Loading images from directory: {self.input_path}")
            self._load_images_from_directory()
        else:
            raise ValueError(f"Invalid input path: {self.input_path}")

    def _extract_frames_from_video(self):
        """Extract frames from video file."""
        cap = cv2.VideoCapture(str(self.input_path))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame
            frame_name = f"frame_{frame_idx:06d}.jpg"
            frame_path = self.image_dir / frame_name

            if not frame_path.exists():
                cv2.imwrite(str(frame_path), frame)

            self.images.append(frame)
            self.image_names.append(frame_name)
            frame_idx += 1

        cap.release()
        print(f"Extracted {len(self.images)} frames")

    def _load_images_from_directory(self):
        """Load images from directory."""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        for ext in extensions:
            for img_path in sorted(self.input_path.glob(f'*{ext}')):
                img = cv2.imread(str(img_path))
                if img is not None:
                    self.images.append(img)
                    self.image_names.append(img_path.name)

        print(f"Loaded {len(self.images)} images")

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

            self.current_label = {'x': orig_x, 'y': orig_y, 'valid': True}
            print(f"Labeled: x={orig_x}, y={orig_y}")

    def _draw_display(self):
        """Create display image with overlays."""
        img = self.images[self.current_idx].copy()
        img_name = self.image_names[self.current_idx]

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

        # Add info text
        h, w = img.shape[:2]
        info_bg = np.zeros((80, w, 3), dtype=np.uint8)

        cv2.putText(info_bg, f"Image {self.current_idx+1}/{len(self.images)}: {img_name}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        status = "Saved" if img_name in self.labels else "Unsaved"
        color = (0, 255, 0) if img_name in self.labels else (0, 0, 255)
        cv2.putText(info_bg, f"Status: {status}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        if self.current_label:
            cv2.putText(info_bg, f"Label: ({self.current_label['x']}, {self.current_label['y']})",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Combine
        display = np.vstack([img, info_bg])

        # Scale for display
        if self.display_scale != 1.0:
            new_w = int(display.shape[1] * self.display_scale)
            new_h = int(display.shape[0] * self.display_scale)
            display = cv2.resize(display, (new_w, new_h))

        return display

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
                # Next image
                if self.current_idx < len(self.images) - 1:
                    self.current_idx += 1
                    img_name = self.image_names[self.current_idx]
                    self.current_label = self.labels.get(img_name, None)
                    if self.current_label:
                        self.current_label = self.current_label.copy()
                    print(f"Image {self.current_idx+1}/{len(self.images)}: {img_name}")

            elif key == ord('p'):
                # Previous image
                if self.current_idx > 0:
                    self.current_idx -= 1
                    img_name = self.image_names[self.current_idx]
                    self.current_label = self.labels.get(img_name, None)
                    if self.current_label:
                        self.current_label = self.current_label.copy()
                    print(f"Image {self.current_idx+1}/{len(self.images)}: {img_name}")

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
    parser = argparse.ArgumentParser(description='Interactive Ball Labeling Tool')

    parser.add_argument('input', type=str,
                        help='Input video file or image directory')
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
