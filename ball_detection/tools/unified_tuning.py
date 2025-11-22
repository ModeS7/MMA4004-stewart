#!/usr/bin/env python3
"""
Unified Camera & HSV Tuning Tool

Combined interactive tool for:
1. Camera settings tuning (exposure, brightness, etc.)
2. HSV color range tuning for ball detection

Usage:
    python -m ball_detection.tools.unified_tuning --mode camera
    python -m ball_detection.tools.unified_tuning --mode hsv
    python -m ball_detection.tools.unified_tuning  (interactive menu)
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ball_detection.utils.camera import (
    create_camera_capture,
    load_camera_config,
    save_camera_config,
    apply_camera_settings
)


def nothing(x):
    """Dummy callback for trackbars."""
    pass


# ============================================================
# CAMERA SETTINGS TUNING MODE
# ============================================================

# Camera property mappings
CAMERA_PROPERTIES = {
    'AUTO_EXPOSURE': cv2.CAP_PROP_AUTO_EXPOSURE,
    'EXPOSURE': cv2.CAP_PROP_EXPOSURE,
    'AUTO_WB': cv2.CAP_PROP_AUTO_WB,
    'BRIGHTNESS': cv2.CAP_PROP_BRIGHTNESS,
    'CONTRAST': cv2.CAP_PROP_CONTRAST,
    'SATURATION': cv2.CAP_PROP_SATURATION,
    'GAIN': cv2.CAP_PROP_GAIN,
    'SHARPNESS': cv2.CAP_PROP_SHARPNESS,
    'GAMMA': cv2.CAP_PROP_GAMMA,
}


def get_property_range(name):
    """Get valid range and scale for a camera property."""
    if 'AUTO' in name:
        return (0, 1), 1  # Boolean
    elif name == 'EXPOSURE':
        return (-13, -1), 10  # Log scale
    elif name == 'GAIN':
        return (0, 100), 1
    elif name == 'GAMMA':
        return (0, 500), 1
    else:
        return (0, 255), 1  # Default range


def tune_camera_settings(camera_id=0):
    """
    Interactive camera settings tuning mode.

    Args:
        camera_id: Camera device ID
    """
    print("=" * 70)
    print("CAMERA SETTINGS TUNING MODE")
    print("=" * 70)
    print(f"\nCamera ID: {camera_id}")
    print("\nControls:")
    print("  Adjust trackbars to tune camera settings")
    print("  's' - Save settings to camera_config.json")
    print("  'r' - Reset to defaults")
    print("  'q' - Quit")
    print("=" * 70 + "\n")

    # Open camera
    cap = create_camera_capture(camera_id)
    if cap is None:
        print(f"Error: Could not open camera {camera_id}")
        return

    # Configure for stereo
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Create window and trackbars
    window_name = 'Camera Settings Tuning'
    cv2.namedWindow(window_name)

    # Load existing config or use defaults
    current_config = load_camera_config()

    # Create trackbars for each property
    for name, prop_id in CAMERA_PROPERTIES.items():
        prop_range, scale = get_property_range(name)
        min_val, max_val = prop_range

        # Get current value
        if name in current_config:
            current_value = current_config[name]
        else:
            current_value = cap.get(prop_id)

        # Convert to trackbar range
        trackbar_value = int(current_value * scale)
        trackbar_min = int(min_val * scale)
        trackbar_max = int(max_val * scale)

        cv2.createTrackbar(name, window_name, trackbar_value, trackbar_max, nothing)
        cv2.setTrackbarMin(name, window_name, trackbar_min)

    print("Camera settings tuning started. Adjust track bars to tune...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Read all trackbar values and apply
        settings = {}
        for name, prop_id in CAMERA_PROPERTIES.items():
            prop_range, scale = get_property_range(name)
            trackbar_value = cv2.getTrackbarPos(name, window_name)
            real_value = trackbar_value / scale
            settings[name] = real_value
            cap.set(prop_id, real_value)

        # Display frame
        display = frame.copy()

        # Add current settings text
        y_offset = 30
        cv2.putText(display, "Current Settings:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        y_offset += 25
        for name, value in settings.items():
            text = f"{name}: {value:.2f}" if '.' in str(value) else f"{name}: {value}"
            cv2.putText(display, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20

        cv2.imshow(window_name, display)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save settings
            if save_camera_config(settings):
                print("Settings saved successfully!")
                print("\nSaved settings:")
                for name, value in settings.items():
                    print(f"  {name}: {value}")
        elif key == ord('r'):
            # Reset to defaults
            current_config = load_camera_config()
            for name in CAMERA_PROPERTIES.keys():
                if name in current_config:
                    prop_range, scale = get_property_range(name)
                    trackbar_value = int(current_config[name] * scale)
                    cv2.setTrackbarPos(name, window_name, trackbar_value)
            print("Reset to default settings")

    cap.release()
    cv2.destroyAllWindows()
    print("\nCamera settings tuning complete.")


# ============================================================
# HSV COLOR TUNING MODE
# ============================================================

def tune_hsv_ranges(camera_id=0):
    """
    Interactive HSV color range tuning mode.

    Args:
        camera_id: Camera device ID
    """
    print("=" * 70)
    print("HSV COLOR TUNING MODE")
    print("=" * 70)
    print(f"\nCamera ID: {camera_id}")
    print("\nControls:")
    print("  LEFT CLICK on ball: Auto-suggest HSV range")
    print("  's' - Save HSV values to hsv_config.txt")
    print("  'r' - Reset to defaults")
    print("  'q' - Quit")
    print("\nTip: Click on the red ball to automatically set HSV ranges!")
    print("     Then fine-tune with the trackbars.")
    print("=" * 70 + "\n")

    # Open camera
    cap = create_camera_capture(camera_id)
    if cap is None:
        print(f"Error: Could not open camera {camera_id}")
        return

    # Configure for stereo
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Load and apply camera settings
    print("Loading camera settings...")
    camera_settings = load_camera_config()
    apply_camera_settings(cap, camera_settings)

    # Determine which camera to use
    ret, test_frame = cap.read()
    if not test_frame.shape[1] > 1280:
        # Single camera
        use_left_only = True
        print("Single camera detected")
    else:
        # Stereo camera - ask which side
        print("\nStereo camera detected.")
        choice = input("Tune HSV for [L]eft or [R]ight camera? (default: L): ").strip().lower()
        use_left_only = choice != 'r'
        print(f"Using {'LEFT' if use_left_only else 'RIGHT'} camera")

    # Create windows
    cv2.namedWindow('HSV Tuning')
    cv2.namedWindow('Original')

    # Default HSV ranges for red (two ranges because red wraps around hue)
    cv2.createTrackbar('L1_H', 'HSV Tuning', 0, 179, nothing)
    cv2.createTrackbar('L1_S', 'HSV Tuning', 230, 255, nothing)
    cv2.createTrackbar('L1_V', 'HSV Tuning', 100, 255, nothing)
    cv2.createTrackbar('U1_H', 'HSV Tuning', 10, 179, nothing)
    cv2.createTrackbar('U1_S', 'HSV Tuning', 255, 255, nothing)
    cv2.createTrackbar('U1_V', 'HSV Tuning', 255, 255, nothing)

    cv2.createTrackbar('L2_H', 'HSV Tuning', 160, 179, nothing)
    cv2.createTrackbar('L2_S', 'HSV Tuning', 100, 255, nothing)
    cv2.createTrackbar('L2_V', 'HSV Tuning', 100, 255, nothing)
    cv2.createTrackbar('U2_H', 'HSV Tuning', 179, 179, nothing)
    cv2.createTrackbar('U2_S', 'HSV Tuning', 255, 255, nothing)
    cv2.createTrackbar('U2_V', 'HSV Tuning', 255, 255, nothing)

    # Morphological kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Mouse callback for auto-tuning
    clicked_hsv = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal clicked_hsv
        if event == cv2.EVENT_LBUTTONDOWN:
            hsv_frame = param
            if hsv_frame is not None and 0 <= y < hsv_frame.shape[0] and 0 <= x < hsv_frame.shape[1]:
                clicked_hsv = hsv_frame[y, x]

    hsv_for_callback = None
    cv2.setMouseCallback('Original', mouse_callback, hsv_for_callback)

    print("HSV tuning started. Click on ball or adjust trackbars...\n")

    while True:
        ret, full_frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Select appropriate camera
        if full_frame.shape[1] > 1280:
            if use_left_only:
                frame = full_frame[:, 0:1280]
            else:
                frame = full_frame[:, 1280:2560]
        else:
            frame = full_frame

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_for_callback = hsv

        # Get trackbar values
        lower1 = np.array([
            cv2.getTrackbarPos('L1_H', 'HSV Tuning'),
            cv2.getTrackbarPos('L1_S', 'HSV Tuning'),
            cv2.getTrackbarPos('L1_V', 'HSV Tuning')
        ])
        upper1 = np.array([
            cv2.getTrackbarPos('U1_H', 'HSV Tuning'),
            cv2.getTrackbarPos('U1_S', 'HSV Tuning'),
            cv2.getTrackbarPos('U1_V', 'HSV Tuning')
        ])
        lower2 = np.array([
            cv2.getTrackbarPos('L2_H', 'HSV Tuning'),
            cv2.getTrackbarPos('L2_S', 'HSV Tuning'),
            cv2.getTrackbarPos('L2_V', 'HSV Tuning')
        ])
        upper2 = np.array([
            cv2.getTrackbarPos('U2_H', 'HSV Tuning'),
            cv2.getTrackbarPos('U2_S', 'HSV Tuning'),
            cv2.getTrackbarPos('U2_V', 'HSV Tuning')
        ])

        # Create masks
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Apply morphology
        mask_morphed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_morphed = cv2.morphologyEx(mask_morphed, cv2.MORPH_OPEN, kernel)

        # Stack for display
        mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_morphed_display = cv2.cvtColor(mask_morphed, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([mask_display, mask_morphed_display])

        # Add labels
        cv2.putText(combined, "Raw Mask", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Morphology Applied", (mask.shape[1] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Handle auto-tuning from click
        if clicked_hsv is not None:
            h, s, v = clicked_hsv
            print(f"\nClicked HSV: H={h}, S={s}, V={v}")

            # Auto-suggest ranges
            h_margin = 10
            s_margin = 30
            v_margin = 50

            if h < 10:  # Lower red range
                cv2.setTrackbarPos('L1_H', 'HSV Tuning', max(0, h - h_margin))
                cv2.setTrackbarPos('U1_H', 'HSV Tuning', min(179, h + h_margin))
                cv2.setTrackbarPos('L1_S', 'HSV Tuning', max(0, s - s_margin))
                cv2.setTrackbarPos('U1_S', 'HSV Tuning', min(255, s + s_margin))
                cv2.setTrackbarPos('L1_V', 'HSV Tuning', max(0, v - v_margin))
                cv2.setTrackbarPos('U1_V', 'HSV Tuning', min(255, v + v_margin))
            elif h > 160:  # Upper red range
                cv2.setTrackbarPos('L2_H', 'HSV Tuning', max(0, h - h_margin))
                cv2.setTrackbarPos('U2_H', 'HSV Tuning', min(179, h + h_margin))
                cv2.setTrackbarPos('L2_S', 'HSV Tuning', max(0, s - s_margin))
                cv2.setTrackbarPos('U2_S', 'HSV Tuning', min(255, s + s_margin))
                cv2.setTrackbarPos('L2_V', 'HSV Tuning', max(0, v - v_margin))
                cv2.setTrackbarPos('U2_V', 'HSV Tuning', min(255, v + v_margin))

            print("Auto-tuned HSV ranges based on clicked color")
            clicked_hsv = None

        cv2.imshow('HSV Tuning', combined)
        cv2.imshow('Original', frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save HSV values
            save_path = Path("hsv_config.txt")
            with open(save_path, 'w') as f:
                f.write(f"lower_red1 = np.array([{lower1[0]}, {lower1[1]}, {lower1[2]}])\n")
                f.write(f"upper_red1 = np.array([{upper1[0]}, {upper1[1]}, {upper1[2]}])\n")
                f.write(f"lower_red2 = np.array([{lower2[0]}, {lower2[1]}, {lower2[2]}])\n")
                f.write(f"upper_red2 = np.array([{upper2[0]}, {upper2[1]}, {upper2[2]}])\n")

            print(f"\nHSV values saved to: {save_path}")
            print(f"Range 1: {lower1} - {upper1}")
            print(f"Range 2: {lower2} - {upper2}")
        elif key == ord('r'):
            # Reset to defaults
            cv2.setTrackbarPos('L1_H', 'HSV Tuning', 0)
            cv2.setTrackbarPos('L1_S', 'HSV Tuning', 230)
            cv2.setTrackbarPos('L1_V', 'HSV Tuning', 100)
            cv2.setTrackbarPos('U1_H', 'HSV Tuning', 10)
            cv2.setTrackbarPos('U1_S', 'HSV Tuning', 255)
            cv2.setTrackbarPos('U1_V', 'HSV Tuning', 255)
            cv2.setTrackbarPos('L2_H', 'HSV Tuning', 160)
            cv2.setTrackbarPos('L2_S', 'HSV Tuning', 100)
            cv2.setTrackbarPos('L2_V', 'HSV Tuning', 100)
            cv2.setTrackbarPos('U2_H', 'HSV Tuning', 179)
            cv2.setTrackbarPos('U2_S', 'HSV Tuning', 255)
            cv2.setTrackbarPos('U2_V', 'HSV Tuning', 255)
            print("Reset to default values")

    cap.release()
    cv2.destroyAllWindows()
    print("\nHSV tuning complete.")


# ============================================================
# MAIN MENU
# ============================================================

def show_menu():
    """Interactive menu for mode selection."""
    print("=" * 70)
    print("UNIFIED CAMERA & HSV TUNING TOOL")
    print("=" * 70)
    print("\nSelect tuning mode:")
    print("  1. Camera Settings (exposure, brightness, etc.)")
    print("  2. HSV Color Ranges (for ball detection)")
    print("  q. Quit")
    print("=" * 70)

    while True:
        choice = input("\nEnter choice (1/2/q): ").strip().lower()

        if choice == '1':
            return 'camera'
        elif choice == '2':
            return 'hsv'
        elif choice == 'q':
            return None
        else:
            print("Invalid choice. Please enter 1, 2, or q.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Unified Camera & HSV Tuning Tool')
    parser.add_argument('--mode', choices=['camera', 'hsv'], default=None,
                       help='Tuning mode: camera settings or hsv color ranges')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')

    args = parser.parse_args()

    # Determine mode
    mode = args.mode
    if mode is None:
        mode = show_menu()

    if mode is None:
        print("\nExiting...")
        return

    # Run appropriate tuning mode
    if mode == 'camera':
        tune_camera_settings(args.camera)
    elif mode == 'hsv':
        tune_hsv_ranges(args.camera)


if __name__ == '__main__':
    main()
