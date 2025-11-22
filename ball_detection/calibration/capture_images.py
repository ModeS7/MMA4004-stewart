#!/usr/bin/env python3
"""
Calibration Image Capture Tool

Captures images from camera to help find platform center and calibrate FOV.
Shows crosshairs at current configured center for reference.
"""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ball_detection.utils.camera import create_camera_capture, load_camera_config, apply_camera_settings


def capture_calibration_images(camera_id=1, output_dir="ball_detection/calibration"):
    """
    Capture calibration images with crosshairs showing configured center.

    Args:
        camera_id: Camera device ID
        output_dir: Directory to save calibration images
    """
    print("=" * 70)
    print("CALIBRATION IMAGE CAPTURE")
    print("=" * 70)
    print(f"\nCamera ID: {camera_id}")
    print("\nControls:")
    print("  SPACE: Capture image")
    print("  c: Toggle crosshairs")
    print("  g: Toggle grid")
    print("  q: Quit")
    print("=" * 70 + "\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Open camera
    cap = create_camera_capture(camera_id)
    if cap is None:
        print(f"Error: Could not open camera {camera_id}")
        return

    # Configure camera for stereo
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Load and apply camera settings
    camera_config = load_camera_config()
    apply_camera_settings(cap, camera_config)

    # Get actual resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Camera opened: {width}x{height} @ {fps}fps")

    # Detect stereo camera
    is_stereo = (width > height * 1.5)
    if is_stereo:
        camera_width = width // 2
        print(f"Stereo camera detected, using LEFT camera ({camera_width}x{height})")
    else:
        camera_width = width
        print(f"Mono camera ({camera_width}x{height})")

    # Import calibrated center from config
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.utils import ZEDCameraConfig

    # Use calibrated platform center
    center_x = int(ZEDCameraConfig.CENTER_X)
    center_y = int(ZEDCameraConfig.CENTER_Y)

    # Calculate platform edge distance (300mm diameter = 150mm radius)
    platform_radius_mm = 150.0  # mm
    pixels_to_mm = ZEDCameraConfig.FOV_WIDTH_MM / camera_width
    platform_radius_px = int(platform_radius_mm / pixels_to_mm)

    print(f"\nCalibrated center: ({center_x}, {center_y}) pixels")
    print(f"Platform radius: {platform_radius_mm}mm = {platform_radius_px} pixels")
    print(f"Output directory: {output_path}\n")

    # Display settings
    show_crosshairs = True
    show_grid = True
    capture_count = 0

    # Create window
    cv2.namedWindow('Calibration Capture', cv2.WINDOW_NORMAL)

    while True:
        # Flush old frames
        cap.grab()

        ret, frame = cap.read()
        if not ret:
            print("Error reading from camera")
            break

        # Extract left camera for stereo
        if is_stereo:
            display_frame = frame[:, :camera_width].copy()
        else:
            display_frame = frame.copy()

        # Draw overlays
        if show_grid:
            # Draw grid lines (every 100 pixels)
            color = (100, 100, 100)
            for x in range(0, camera_width, 100):
                cv2.line(display_frame, (x, 0), (x, height), color, 1)
            for y in range(0, height, 100):
                cv2.line(display_frame, (0, y), (camera_width, y), color, 1)

        if show_crosshairs:
            # Draw crosshairs at calibrated platform center
            color = (0, 255, 0)
            thickness = 2

            # Platform circle (300mm diameter)
            cv2.circle(display_frame, (center_x, center_y), platform_radius_px, color, 3)

            # Lines from center to 4 edges of platform
            # Right edge
            cv2.line(display_frame, (center_x, center_y),
                    (center_x + platform_radius_px, center_y), color, thickness)
            # Left edge
            cv2.line(display_frame, (center_x, center_y),
                    (center_x - platform_radius_px, center_y), color, thickness)
            # Top edge
            cv2.line(display_frame, (center_x, center_y),
                    (center_x, center_y - platform_radius_px), color, thickness)
            # Bottom edge
            cv2.line(display_frame, (center_x, center_y),
                    (center_x, center_y + platform_radius_px), color, thickness)

            # Full crosshairs (spanning entire frame)
            cv2.line(display_frame, (0, center_y), (camera_width, center_y), (0, 255, 255), 1)
            cv2.line(display_frame, (center_x, 0), (center_x, height), (0, 255, 255), 1)

            # Center point marker
            cv2.circle(display_frame, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.circle(display_frame, (center_x, center_y), 10, color, 2)

            # Labels
            cv2.putText(display_frame, f"Center: ({center_x}, {center_y})",
                       (center_x + 15, center_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Platform: {platform_radius_mm*2:.0f}mm",
                       (center_x + 15, center_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Instructions
        cv2.putText(display_frame, "SPACE: Capture | C: Crosshairs | G: Grid | Q: Quit",
                   (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show capture count
        if capture_count > 0:
            cv2.putText(display_frame, f"Captured: {capture_count}",
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display
        cv2.imshow('Calibration Capture', display_frame)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            # Capture image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save frame with overlays
            overlay_filename = output_path / f"calibration_{timestamp}_overlay.png"
            cv2.imwrite(str(overlay_filename), display_frame)
            print(f"Saved (with overlay): {overlay_filename}")

            # Save clean frame without overlays
            if is_stereo:
                clean_frame = frame[:, :camera_width]
            else:
                clean_frame = frame

            clean_filename = output_path / f"calibration_{timestamp}_clean.png"
            cv2.imwrite(str(clean_filename), clean_frame)
            print(f"Saved (clean):        {clean_filename}")

            capture_count += 1

            # Flash effect
            flash = np.ones_like(display_frame) * 255
            cv2.imshow('Calibration Capture', flash)
            cv2.waitKey(100)

        elif key == ord('c'):
            show_crosshairs = not show_crosshairs
            print(f"Crosshairs: {'ON' if show_crosshairs else 'OFF'}")

        elif key == ord('g'):
            show_grid = not show_grid
            print(f"Grid: {'ON' if show_grid else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n{'=' * 70}")
    print(f"Captured {capture_count} images")
    print(f"Saved to: {output_path}")
    print(f"{'=' * 70}")

    if capture_count > 0:
        print("\nNext steps:")
        print("1. Open the captured images")
        print("2. Measure the platform center in the image")
        print("3. Compare with crosshair center position")
        print("4. Update ZEDCameraConfig.CENTER_X and CENTER_Y in core/utils.py")
        print("5. Measure platform edges to calculate FOV_WIDTH_MM and FOV_HEIGHT_MM")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Capture calibration images')
    parser.add_argument('--camera', type=int, default=1,
                       help='Camera device ID (default: 1)')
    parser.add_argument('--output', type=str, default='ball_detection/calibration',
                       help='Output directory for images')

    args = parser.parse_args()

    capture_calibration_images(args.camera, args.output)
