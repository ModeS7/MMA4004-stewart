#!/usr/bin/env python3
"""
View Rectified Stereo Images

Displays the original and rectified stereo camera feeds side-by-side
to visualize the effect of stereo rectification.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ball_detection.utils.camera import create_camera_capture

# Camera configuration
CAMERA_INDEX = 0  # ZED stereo camera


def load_stereo_calibration():
    """Load latest stereo calibration data."""
    CALIBRATION_DIRS = [
        Path(__file__).parent.parent / "calibrations",
        Path(r"W:\NTNU\Stereo_cam_test\calibrations")
    ]

    for calib_dir in CALIBRATION_DIRS:
        if not calib_dir.exists():
            continue

        # Find latest stereo calibration files
        p1_files = sorted(calib_dir.glob('stereo_P1_*.csv'), reverse=True)
        map_files = sorted(calib_dir.glob('stereo_left_map1_*.npy'), reverse=True)

        if not p1_files or not map_files:
            continue

        timestamp = p1_files[0].name.replace('stereo_P1_', '').replace('.csv', '')

        try:
            # Load rectification maps
            left_map1 = np.load(calib_dir / f'stereo_left_map1_{timestamp}.npy')
            left_map2 = np.load(calib_dir / f'stereo_left_map2_{timestamp}.npy')
            right_map1 = np.load(calib_dir / f'stereo_right_map1_{timestamp}.npy')
            right_map2 = np.load(calib_dir / f'stereo_right_map2_{timestamp}.npy')

            print(f"Loaded stereo calibration: {timestamp}")
            print(f"Source: {calib_dir}\n")

            return {
                'left_map1': left_map1,
                'left_map2': left_map2,
                'right_map1': right_map1,
                'right_map2': right_map2,
                'timestamp': timestamp
            }

        except Exception as e:
            print(f"Error loading from {calib_dir}: {e}")
            continue

    print("Error: No stereo calibration found!")
    print("\nRun stereo calibration first:")
    print("  python -m ball_detection.calibration.stereo_calibration --calibrate-stereo")
    return None


def main():
    print("=" * 70)
    print("STEREO RECTIFICATION VIEWER")
    print("=" * 70)

    # Load stereo calibration
    print("\n[1/2] Loading stereo calibration...")
    stereo_calib = load_stereo_calibration()
    if not stereo_calib:
        return

    # Open camera
    print("[2/2] Opening camera...")
    cap = create_camera_capture(CAMERA_INDEX)
    if cap is None:
        print(f"Failed to open camera {CAMERA_INDEX}")
        return

    # Configure camera for stereo
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Apply camera settings (same as tune_hsv)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 5)
    cap.set(cv2.CAP_PROP_CONTRAST, 2)
    cap.set(cv2.CAP_PROP_SATURATION, 4)
    cap.set(cv2.CAP_PROP_GAIN, 2)
    cap.set(cv2.CAP_PROP_SHARPNESS, 0)
    cap.set(cv2.CAP_PROP_GAMMA, 102)

    print("\n" + "=" * 70)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frames")
    print("=" * 70 + "\n")

    # Create windows
    cv2.namedWindow('Original (Left | Right)', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Rectified (Left | Right)', cv2.WINDOW_NORMAL)

    frame_count = 0

    while True:
        ret, full_frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_count += 1

        # Split stereo frame
        left_frame = full_frame[:, 0:1280].copy()
        right_frame = full_frame[:, 1280:2560].copy()

        # Apply rectification
        left_rectified = cv2.remap(left_frame, stereo_calib['left_map1'],
                                   stereo_calib['left_map2'], cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_frame, stereo_calib['right_map1'],
                                    stereo_calib['right_map2'], cv2.INTER_LINEAR)

        # Create side-by-side displays
        original_display = np.hstack([left_frame, right_frame])
        rectified_display = np.hstack([left_rectified, right_rectified])

        # Draw horizontal lines on rectified to show alignment
        h = rectified_display.shape[0]
        for y in range(0, h, 50):
            cv2.line(rectified_display, (0, y), (rectified_display.shape[1], y),
                    (0, 255, 0), 1)

        # Add labels
        cv2.putText(original_display, "Original - Left", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(original_display, "Original - Right", (1290, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(rectified_display, "Rectified - Left", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(rectified_display, "Rectified - Right", (1290, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(rectified_display, "Green lines show epipolar alignment", (10, 690),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display
        cv2.imshow('Original (Left | Right)', original_display)
        cv2.imshow('Rectified (Left | Right)', rectified_display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save frames
            timestamp = stereo_calib['timestamp']
            cv2.imwrite(f'original_stereo_{frame_count}.png', original_display)
            cv2.imwrite(f'rectified_stereo_{frame_count}.png', rectified_display)
            print(f"Saved frames: original_stereo_{frame_count}.png, rectified_stereo_{frame_count}.png")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames")


if __name__ == '__main__':
    main()
