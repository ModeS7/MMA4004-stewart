#!/usr/bin/env python3
"""
Camera Configuration Tuning Tool

Interactive tool to configure camera settings and disable automatic features.
Helps achieve consistent lighting/color for reliable ball detection.
"""

import cv2
import numpy as np
import json
from pathlib import Path


def nothing(x):
    """Dummy callback for trackbars."""
    pass


# Camera property mappings (OpenCV constants)
CAMERA_PROPERTIES = {
    'AUTO_EXPOSURE': cv2.CAP_PROP_AUTO_EXPOSURE,
    'EXPOSURE': cv2.CAP_PROP_EXPOSURE,
    'AUTO_WB': cv2.CAP_PROP_AUTO_WB,
    'WB_TEMPERATURE': cv2.CAP_PROP_WB_TEMPERATURE,
    'BRIGHTNESS': cv2.CAP_PROP_BRIGHTNESS,
    'CONTRAST': cv2.CAP_PROP_CONTRAST,
    'SATURATION': cv2.CAP_PROP_SATURATION,
    'GAIN': cv2.CAP_PROP_GAIN,
    'SHARPNESS': cv2.CAP_PROP_SHARPNESS,
    'BACKLIGHT': cv2.CAP_PROP_BACKLIGHT,
    'GAMMA': cv2.CAP_PROP_GAMMA,
}


def list_cameras(max_cameras=10):
    """
    List available cameras.

    Args:
        max_cameras: Maximum number of cameras to check

    Returns:
        List of available camera IDs
    """
    available = []

    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available.append((i, width, height))
            cap.release()

    return available


def get_camera_properties(cap):
    """
    Get all available camera properties and their current values.

    Args:
        cap: OpenCV VideoCapture object

    Returns:
        Dictionary of property name -> (value, min, max, scale)
    """
    properties = {}

    for name, prop_id in CAMERA_PROPERTIES.items():
        value = cap.get(prop_id)

        # Try to determine valid range (camera-specific)
        # Most cameras don't report min/max, so use common ranges
        # Scale: multiplier to convert float to int for trackbar
        if 'AUTO' in name:
            prop_range = (0, 1)  # Boolean: 0=off, 1=on
            scale = 1
        elif name == 'EXPOSURE':
            prop_range = (-13, -1)  # Typical range for exposure (log scale)
            scale = 10  # -13.0 to -1.0 -> -130 to -10
        elif name == 'WB_TEMPERATURE':
            prop_range = (2000, 6500)  # Kelvin (already integers)
            scale = 1
        elif name == 'GAIN':
            prop_range = (0, 100)
            scale = 1
        elif name == 'GAMMA':
            prop_range = (0, 500)  # Gamma * 100
            scale = 1
        else:
            prop_range = (0, 255)  # Default range
            scale = 1

        properties[name] = {
            'value': value,
            'min': prop_range[0],
            'max': prop_range[1],
            'scale': scale,
            'prop_id': prop_id
        }

    return properties


def tune_camera_settings(camera_id=0, width=2560, height=720, fps=60):
    """
    Interactive camera settings tuner with live preview.

    Args:
        camera_id: Camera device ID
        width: Desired frame width
        height: Desired frame height
        fps: Desired frame rate
    """
    print("=" * 70)
    print("CAMERA SETTINGS TUNER")
    print("=" * 70)
    print(f"\nCamera ID: {camera_id}")
    print(f"Requested: {width}x{height} @ {fps}fps")
    print("\nControls:")
    print("  s: Save settings to file")
    print("  r: Reset to defaults")
    print("  d: Disable all AUTO settings")
    print("  q: Quit")
    print("=" * 70 + "\n")

    # Open camera with DirectShow backend (Windows, reduces tearing)
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Fallback to default backend
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

    # Reduce buffer size to minimize tearing
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Set resolution and frame rate
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Try MJPEG codec for better high-res support
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Get actual camera info
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps}fps")

    if actual_width != width or actual_height != height or actual_fps != fps:
        print(f"Warning: Camera did not accept requested resolution/fps")
        print(f"  Requested: {width}x{height} @ {fps}fps")
        print(f"  Actual: {actual_width}x{actual_height} @ {actual_fps}fps")

    # Detect stereo camera
    is_stereo = (actual_width > actual_height * 1.5)
    if is_stereo:
        print(f"Detected stereo camera, showing left camera only")
        display_width = actual_width // 2
    else:
        display_width = actual_width

    # Get available properties
    properties = get_camera_properties(cap)

    print("\nAvailable camera properties:")
    for name, info in properties.items():
        if info['value'] != -1.0:  # -1 means not supported
            print(f"  {name}: {info['value']} (range: {info['min']} - {info['max']})")

    # Create window and trackbars
    cv2.namedWindow('Camera Settings')
    cv2.namedWindow('Preview')

    # Create trackbars for supported properties
    for name, info in properties.items():
        if info['value'] != -1.0:
            # Scale to integer range for trackbar using property-specific scale
            scale = info['scale']
            min_val = int(info['min'] * scale)
            max_val = int(info['max'] * scale)
            current_val = int(info['value'] * scale)

            # Clamp current value to range
            current_val = max(min_val, min(max_val, current_val))

            # Trackbar range is 0 to (max-min), position is (current-min)
            trackbar_range = max_val - min_val
            trackbar_pos = current_val - min_val

            cv2.createTrackbar(name, 'Camera Settings',
                             trackbar_pos, trackbar_range, nothing)

    # Main loop
    settings_changed = False

    while True:
        # Flush old buffered frames to reduce tearing
        cap.grab()

        ret, frame = cap.read()

        if not ret:
            print("Error reading from camera")
            break

        # Extract left camera for stereo
        if is_stereo:
            display_frame = frame[:, :display_width]
        else:
            display_frame = frame

        # Apply current trackbar values to camera
        for name, info in properties.items():
            if info['value'] != -1.0:
                scale = info['scale']
                min_val = int(info['min'] * scale)
                trackbar_val = cv2.getTrackbarPos(name, 'Camera Settings')

                # Convert back to actual value
                actual_val = (trackbar_val + min_val) / scale

                # Set camera property
                cap.set(info['prop_id'], actual_val)

        # Add settings overlay
        overlay = display_frame.copy()

        # Create semi-transparent background for text
        cv2.rectangle(overlay, (0, 0), (400, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

        # Display current settings
        y_offset = 25
        cv2.putText(display_frame, "Current Settings:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 30

        # Show key settings
        important_settings = ['AUTO_EXPOSURE', 'EXPOSURE', 'AUTO_WB',
                             'WB_TEMPERATURE', 'GAIN', 'BRIGHTNESS']

        for name in important_settings:
            if name in properties and properties[name]['value'] != -1.0:
                info = properties[name]
                scale = info['scale']
                min_val = int(info['min'] * scale)
                trackbar_val = cv2.getTrackbarPos(name, 'Camera Settings')
                actual_val = (trackbar_val + min_val) / scale

                # Color code: green if manual (auto off), yellow if auto
                if 'AUTO' in name:
                    color = (0, 255, 0) if actual_val < 0.5 else (0, 255, 255)
                    text = f"{name}: {'OFF' if actual_val < 0.5 else 'ON'}"
                else:
                    color = (255, 255, 255)
                    text = f"{name}: {actual_val:.1f}"

                cv2.putText(display_frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 25

        # Instructions
        cv2.putText(display_frame, "Press 'd' to disable AUTO settings",
                   (10, display_frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(display_frame, "Press 's' to save settings",
                   (10, display_frame.shape[0] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display
        cv2.imshow('Preview', display_frame)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            save_camera_settings(cap, properties, camera_id)
            settings_changed = True
        elif key == ord('d'):
            # Disable all AUTO settings
            print("\nDisabling AUTO settings...")
            for name, info in properties.items():
                if 'AUTO' in name and info['value'] != -1.0:
                    cap.set(info['prop_id'], 0)
                    # Trackbar position 0 = min value (which is 0 for AUTO settings)
                    cv2.setTrackbarPos(name, 'Camera Settings', 0)
                    print(f"  {name} = OFF")
        elif key == ord('r'):
            # Reset to defaults (auto on)
            print("\nResetting to defaults...")
            for name, info in properties.items():
                if 'AUTO' in name and info['value'] != -1.0:
                    cap.set(info['prop_id'], 1)

    cap.release()
    cv2.destroyAllWindows()

    if settings_changed:
        print("\nSettings saved! Apply these in camera_controller.py connect() method.")


def save_camera_settings(cap, properties, camera_id):
    """Save current camera settings to file."""
    settings = {}

    for name, info in properties.items():
        if info['value'] != -1.0:
            current_val = cap.get(info['prop_id'])
            settings[name] = float(current_val)

    # Save to JSON file
    output_file = Path("ball_detection/camera_config.json")

    config = {
        'camera_id': camera_id,
        'settings': settings,
        'notes': 'Apply these settings in camera_controller.py'
    }

    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Camera settings saved to: {output_file}")
    print("=" * 70)

    # Generate code snippet
    print("\nAdd this code to camera_controller.py in connect() method:")
    print("# Apply camera settings (disable AUTO features)")

    for name, value in settings.items():
        if 'AUTO' in name:
            print(f"self.cap.set(cv2.CAP_PROP_{name}, {value:.0f})  # {'OFF' if value < 0.5 else 'ON'}")
        else:
            print(f"self.cap.set(cv2.CAP_PROP_{name}, {value:.1f})")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Tune camera settings')
    parser.add_argument('--camera', type=int, default=None,
                       help='Camera device ID (default: list available cameras)')
    parser.add_argument('--list', action='store_true',
                       help='List available cameras and exit')
    parser.add_argument('--width', type=int, default=2560,
                       help='Desired frame width (default: 2560)')
    parser.add_argument('--height', type=int, default=720,
                       help='Desired frame height (default: 720)')
    parser.add_argument('--fps', type=int, default=60,
                       help='Desired frame rate (default: 60)')

    args = parser.parse_args()

    if args.list or args.camera is None:
        print("Scanning for cameras...\n")
        cameras = list_cameras()

        if not cameras:
            print("No cameras found!")
        else:
            print("Available cameras:")
            for cam_id, width, height in cameras:
                stereo = " (STEREO)" if width > height * 1.5 else ""
                print(f"  Camera {cam_id}: {width}x{height}{stereo}")
            print()

        if not args.list and cameras:
            # Use first available camera
            camera_id = cameras[0][0]
            print(f"Using camera {camera_id}\n")
            tune_camera_settings(camera_id, args.width, args.height, args.fps)
    else:
        tune_camera_settings(args.camera, args.width, args.height, args.fps)
