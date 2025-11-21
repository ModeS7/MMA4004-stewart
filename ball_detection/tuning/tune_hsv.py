#!/usr/bin/env python3
"""
HSV Color Tuning Tool

Interactive tool to tune HSV color ranges for red ball detection.
Uses trackbars to adjust values in real-time with live camera feed.
"""

import cv2
import numpy as np
from pathlib import Path


def nothing(x):
    """Dummy callback for trackbars."""
    pass


def apply_default_camera_settings(cap):
    """Apply default camera settings optimized for ball detection."""
    # Default settings
    settings = {
        'AUTO_EXPOSURE': 0.25,
        'EXPOSURE': -6,
        'AUTO_WB': 1,  # Enable auto white balance
        'BRIGHTNESS': 5,
        'CONTRAST': 2,
        'SATURATION': 4,
        'GAIN': 2,
        'SHARPNESS': 0,
        'GAMMA': 102
    }

    print("Applying default camera settings:")
    for name, value in settings.items():
        prop_id = getattr(cv2, f'CAP_PROP_{name}', None)
        if prop_id is not None:
            cap.set(prop_id, value)
            # Read back to verify
            actual = cap.get(prop_id)
            if 'AUTO' in name:
                print(f"  {name} = {'OFF' if value < 0.5 else 'ON'} (actual: {actual:.1f})")
            else:
                print(f"  {name} = {value:.1f} (actual: {actual:.1f})")


# Global variable for mouse callback
clicked_hsv = None


def mouse_callback(event, x, y, flags, param):
    """Mouse callback to sample HSV values by clicking."""
    global clicked_hsv
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_frame = param
        if hsv_frame is not None and 0 <= y < hsv_frame.shape[0] and 0 <= x < hsv_frame.shape[1]:
            clicked_hsv = hsv_frame[y, x]


def tune_hsv_camera(camera_id=0):
    """
    Tune HSV ranges using live camera feed.

    Args:
        camera_id: Camera device ID
    """
    print("=" * 70)
    print("HSV TUNING TOOL - LIVE CAMERA")
    print("=" * 70)
    print(f"\nCamera ID: {camera_id}")
    print("\nControls:")
    print("  LEFT CLICK on ball: Auto-suggest HSV range")
    print("  s: Save current HSV values to file")
    print("  r: Reset to defaults")
    print("  q: Quit")
    print("\nTip: Click on the red ball to automatically set HSV ranges!")
    print("     Then fine-tune with the trackbars.")
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # Try MJPEG codec for better high-res support
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Get actual camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Camera opened: {width}x{height} @ {fps}fps")

    # Apply default camera settings AFTER resolution is set
    apply_default_camera_settings(cap)
    print()

    # For stereo camera, use only left camera
    if width > height * 1.5:
        print("Detected stereo camera, using left camera only\n")
        use_left_only = True
    else:
        use_left_only = False

    # Create window and trackbars
    cv2.namedWindow('HSV Tuning')
    cv2.namedWindow('Original')

    # Default HSV ranges for red (two ranges because red wraps around hue)
    # Range 1: Lower reds (0-10 degrees)
    cv2.createTrackbar('L1_H', 'HSV Tuning', 0, 179, nothing)
    cv2.createTrackbar('L1_S', 'HSV Tuning', 230, 255, nothing)
    cv2.createTrackbar('L1_V', 'HSV Tuning', 100, 255, nothing)
    cv2.createTrackbar('U1_H', 'HSV Tuning', 10, 179, nothing)
    cv2.createTrackbar('U1_S', 'HSV Tuning', 255, 255, nothing)
    cv2.createTrackbar('U1_V', 'HSV Tuning', 255, 255, nothing)

    # Range 2: Upper reds (160-180 degrees)
    cv2.createTrackbar('L2_H', 'HSV Tuning', 160, 179, nothing)
    cv2.createTrackbar('L2_S', 'HSV Tuning', 100, 255, nothing)
    cv2.createTrackbar('L2_V', 'HSV Tuning', 100, 255, nothing)
    cv2.createTrackbar('U2_H', 'HSV Tuning', 179, 179, nothing)
    cv2.createTrackbar('U2_S', 'HSV Tuning', 255, 255, nothing)
    cv2.createTrackbar('U2_V', 'HSV Tuning', 255, 255, nothing)

    # Morphological kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Set up mouse callback for HSV sampling
    global clicked_hsv
    hsv_for_callback = None
    cv2.setMouseCallback('Original', mouse_callback, hsv_for_callback)

    while True:
        # Flush old buffered frames to reduce tearing
        cap.grab()

        ret, frame = cap.read()

        if not ret:
            print("Error reading from camera")
            break

        # Use left camera only for stereo
        if use_left_only:
            frame = frame[:, :width//2]

        # Convert to HSV for mouse callback
        hsv_for_callback = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.setMouseCallback('Original', mouse_callback, hsv_for_callback)

        # Check if user clicked on a pixel
        if clicked_hsv is not None:
            h, s, v = clicked_hsv
            print(f"\nClicked HSV: H={h}, S={s}, V={v}")

            # Auto-set ranges around clicked color
            # Red in HSV can be 0-10 or 160-179
            if h <= 15 or h >= 160:
                # Lower red range (0-20)
                if h <= 15:
                    cv2.setTrackbarPos('L1_H', 'HSV Tuning', 0)
                    cv2.setTrackbarPos('U1_H', 'HSV Tuning', 20)
                else:
                    cv2.setTrackbarPos('L1_H', 'HSV Tuning', 0)
                    cv2.setTrackbarPos('U1_H', 'HSV Tuning', 15)

                cv2.setTrackbarPos('L1_S', 'HSV Tuning', max(80, s - 80))
                cv2.setTrackbarPos('U1_S', 'HSV Tuning', 255)
                cv2.setTrackbarPos('L1_V', 'HSV Tuning', max(80, v - 100))
                cv2.setTrackbarPos('U1_V', 'HSV Tuning', 255)

                # Upper red range (160-179)
                cv2.setTrackbarPos('L2_H', 'HSV Tuning', 160)
                cv2.setTrackbarPos('U2_H', 'HSV Tuning', 179)
                cv2.setTrackbarPos('L2_S', 'HSV Tuning', max(80, s - 80))
                cv2.setTrackbarPos('U2_S', 'HSV Tuning', 255)
                cv2.setTrackbarPos('L2_V', 'HSV Tuning', max(80, v - 100))
                cv2.setTrackbarPos('U2_V', 'HSV Tuning', 255)

                print(f"Auto-set ranges for RED (H={h})")
                print(f"Range 1: H[0-20], S[{max(80, s-80)}-255], V[{max(80, v-100)}-255]")
                print(f"Range 2: H[160-179], S[{max(80, s-80)}-255], V[{max(80, v-100)}-255]")
            else:
                print(f"Warning: Clicked color (H={h}) doesn't look like red")
                print(f"Red should be H<15 or H>160. You clicked H={h}")
                print(f"Try clicking on a different part of the ball")

            clicked_hsv = None  # Reset

        # Get current trackbar positions
        l1_h = cv2.getTrackbarPos('L1_H', 'HSV Tuning')
        l1_s = cv2.getTrackbarPos('L1_S', 'HSV Tuning')
        l1_v = cv2.getTrackbarPos('L1_V', 'HSV Tuning')
        u1_h = cv2.getTrackbarPos('U1_H', 'HSV Tuning')
        u1_s = cv2.getTrackbarPos('U1_S', 'HSV Tuning')
        u1_v = cv2.getTrackbarPos('U1_V', 'HSV Tuning')

        l2_h = cv2.getTrackbarPos('L2_H', 'HSV Tuning')
        l2_s = cv2.getTrackbarPos('L2_S', 'HSV Tuning')
        l2_v = cv2.getTrackbarPos('L2_V', 'HSV Tuning')
        u2_h = cv2.getTrackbarPos('U2_H', 'HSV Tuning')
        u2_s = cv2.getTrackbarPos('U2_S', 'HSV Tuning')
        u2_v = cv2.getTrackbarPos('U2_V', 'HSV Tuning')

        # Create HSV ranges
        lower_red1 = np.array([l1_h, l1_s, l1_v])
        upper_red1 = np.array([u1_h, u1_s, u1_v])
        lower_red2 = np.array([l2_h, l2_s, l2_v])
        upper_red2 = np.array([u2_h, u2_s, u2_v])

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create masks
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Apply morphological operations
        mask_morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, kernel)

        # Find contours on morphed mask
        contours, _ = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create visualizations
        vis = frame.copy()

        # Show raw mask and morphed mask side by side
        mask_raw_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_morph_vis = cv2.cvtColor(mask_morph, cv2.COLOR_GRAY2BGR)
        mask_vis = np.hstack([mask_raw_vis, mask_morph_vis])

        # Draw contours and info
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > 50:
                # Draw contour
                cv2.drawContours(vis, [largest], -1, (0, 255, 0), 2)

                # Calculate center
                M = cv2.moments(largest)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    # Draw center
                    cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1)
                    cv2.circle(vis, (cx, cy), 10, (0, 255, 0), 2)

                    # Draw info
                    cv2.putText(vis, f"DETECTED - Area: {area:.0f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(vis, f"Click on ball to auto-tune", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(vis, "NO DETECTION - Click on ball!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Add labels to mask visualization
        cv2.putText(mask_raw_vis, "RAW MASK", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(mask_morph_vis, "AFTER MORPHOLOGY", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Add current HSV values to original visualization
        y_offset = frame.shape[0] - 60
        cv2.putText(vis, f"R1: H[{l1_h}-{u1_h}] S[{l1_s}-{u1_s}] V[{l1_v}-{u1_v}]",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        cv2.putText(vis, f"R2: H[{l2_h}-{u2_h}] S[{l2_s}-{u2_s}] V[{l2_v}-{u2_v}]",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display
        cv2.imshow('Original', vis)
        cv2.imshow('HSV Tuning', mask_vis)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            save_hsv_values(lower_red1, upper_red1, lower_red2, upper_red2)
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


def save_hsv_values(lower1, upper1, lower2, upper2):
    """Save HSV values to a file."""
    output_file = Path("ball_detection/hsv_config.txt")

    with open(output_file, 'w') as f:
        f.write("# HSV Configuration for Red Ball Detection\n")
        f.write("# Copy these values into roi_extractor.py\n\n")
        f.write(f"lower_red1 = np.array([{lower1[0]}, {lower1[1]}, {lower1[2]}])\n")
        f.write(f"upper_red1 = np.array([{upper1[0]}, {upper1[1]}, {upper1[2]}])\n")
        f.write(f"lower_red2 = np.array([{lower2[0]}, {lower2[1]}, {lower2[2]}])\n")
        f.write(f"upper_red2 = np.array([{upper2[0]}, {upper2[1]}, {upper2[2]}])\n")

    print("\n" + "=" * 70)
    print("HSV values saved to: ball_detection/hsv_config.txt")
    print("=" * 70)
    print(f"Range 1: {lower1} - {upper1}")
    print(f"Range 2: {lower2} - {upper2}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Tune HSV color ranges for red ball detection')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID (default: 0)')

    args = parser.parse_args()

    tune_hsv_camera(args.camera)
