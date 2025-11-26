"""
Fast ROI Extraction using Traditional Computer Vision

Uses HSV color filtering to quickly locate red ball and extract crop for CNN.
Optimized for speed (~1-2ms on CPU).
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class RedBallROIExtractor:
    """
    Extract Region of Interest (ROI) around red ball using color filtering.

    This is the first stage of the hybrid pipeline:
    1. HSV color segmentation (fast, CPU)
    2. Contour detection
    3. Extract centered crop for CNN

    Speed: ~1-2ms per frame on modern CPU
    """

    def __init__(self, crop_size=128, min_area=50, downsample_factor=4):
        """
        Initialize ROI extractor.

        Args:
            crop_size: Size of extracted crop (square)
            min_area: Minimum contour area to be considered a ball
        """
        self.crop_size = crop_size
        self.min_area = min_area
        self.downsample_factor = downsample_factor

        # HSV color ranges for red ball
        # Red wraps around in HSV (0-10 and 160-179)
        self.lower_red1 = np.array([0, 230, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 100, 100])
        self.upper_red2 = np.array([179, 255, 255])

        # Morphological kernel for noise removal
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def set_hsv_ranges(self, lower1, upper1, lower2, upper2):
        """
        Manually set HSV ranges for red ball detection.

        Useful for tuning to specific lighting conditions.

        Args:
            lower1, upper1: First HSV range (e.g., [0, 100, 100] to [10, 255, 255])
            lower2, upper2: Second HSV range (e.g., [160, 100, 100] to [180, 255, 255])
        """
        self.lower_red1 = np.array(lower1)
        self.upper_red1 = np.array(upper1)
        self.lower_red2 = np.array(lower2)
        self.upper_red2 = np.array(upper2)
        print(f"Updated HSV ranges:")
        print(f"  Range 1: {self.lower_red1} - {self.upper_red1}")
        print(f"  Range 2: {self.lower_red2} - {self.upper_red2}")

    def extract_roi(self, frame) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """
        Extract ROI crop around red ball.

        Args:
            frame: Input BGR image

        Returns:
            crop: Cropped RGB image of size (crop_size, crop_size, 3), or None if not detected
            center: (x, y) ball center in original frame coordinates, or None
            crop_offset: (x_offset, y_offset) top-left corner of crop in original frame, or None
        """
        # Downsample for fast ROI finding
        scale = self.downsample_factor
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // scale, h // scale), interpolation=cv2.INTER_NEAREST)

        # Convert downsampled frame to HSV
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # Create mask for red color (two ranges)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Single morphological operation for speed
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None, None

        # Find largest contour (assume it's the ball)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Adjust min_area for downsampled image
        if area < self.min_area // (scale * scale):
            return None, None, None

        # Calculate ball center in downsampled image
        M = cv2.moments(largest_contour)
        if M['m00'] == 0:
            return None, None, None

        # Scale coordinates back to full resolution
        cx = int((M['m10'] / M['m00']) * scale)
        cy = int((M['m01'] / M['m00']) * scale)

        # Extract crop from FULL resolution frame
        crop, crop_offset = self._extract_crop(frame, (cx, cy))

        if crop is None:
            return None, None, None

        # Convert to RGB (CNN expects RGB)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        return crop_rgb, (cx, cy), crop_offset

    def _extract_crop(self, frame, center) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """
        Extract centered crop around given point.

        Args:
            frame: Input image
            center: (x, y) center point

        Returns:
            crop: Extracted crop of size (crop_size, crop_size)
            offset: (x_offset, y_offset) top-left corner of crop
        """
        cx, cy = center
        h, w = frame.shape[:2]
        half_size = self.crop_size // 2

        # Calculate crop bounds
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(w, cx + half_size)
        y2 = min(h, cy + half_size)

        # Extract crop
        crop = frame[y1:y2, x1:x2]

        # Pad if crop is too small (near edges)
        if crop.shape[0] < self.crop_size or crop.shape[1] < self.crop_size:
            crop = cv2.resize(crop, (self.crop_size, self.crop_size))
            # Adjust offset for resized crop
            offset = (x1, y1)
        else:
            offset = (x1, y1)

        return crop, offset

    def visualize_detection(self, frame, crop=None, center=None):
        """
        Create visualization of detection for debugging.

        Args:
            frame: Original frame
            crop: Detected crop (optional)
            center: Ball center (optional)

        Returns:
            Visualization image
        """
        vis = frame.copy()

        if center is not None:
            cx, cy = center

            # Draw center point
            cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1)

            # Draw crop region
            half_size = self.crop_size // 2
            cv2.rectangle(vis,
                          (cx - half_size, cy - half_size),
                          (cx + half_size, cy + half_size),
                          (0, 255, 0), 2)

            # Draw crosshair
            cv2.line(vis, (cx - 15, cy), (cx + 15, cy), (0, 255, 0), 1)
            cv2.line(vis, (cx, cy - 15), (cx, cy + 15), (0, 255, 0), 1)

        # Add crop preview in corner if available
        if crop is not None:
            crop_preview = cv2.resize(crop, (100, 100))
            vis[:100, :100] = crop_preview

        return vis


def test_roi_extractor(camera_id=0):
    """
    Test ROI extraction with webcam.

    Args:
        camera_id: Camera device ID
    """
    print("Testing ROI Extractor with webcam")
    print("Press 'q' to quit")

    extractor = RedBallROIExtractor(crop_size=128)

    # Open camera with DirectShow backend (Windows, reduces tearing)
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

    # Reduce buffer size and set resolution
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    while True:
        # Flush old buffered frames
        cap.grab()

        ret, frame = cap.read()
        if not ret:
            break

        # Extract ROI
        crop, center, offset = extractor.extract_roi(frame)

        # Visualize
        vis = extractor.visualize_detection(frame, crop, center)

        # Add info
        if center is not None:
            cv2.putText(vis, f"Ball detected at: {center}",
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "No ball detected",
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('ROI Extractor Test', vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        camera_id = int(sys.argv[1])
    else:
        camera_id = 0

    test_roi_extractor(camera_id)
