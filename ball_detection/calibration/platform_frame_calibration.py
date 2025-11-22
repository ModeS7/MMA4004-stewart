#!/usr/bin/env python3
"""
Platform Coordinate Frame Calibration

Calibrates the transformation from camera coordinates to platform coordinates.
Place a physical checkerboard flat on the Stewart platform with one corner at
the desired origin point.

Usage:
    1. Run stereo calibration first (to get P1, P2, rectification maps)
    2. Place checkerboard on platform, align corner with origin
    3. Run: python -m ball_detection.calibration.platform_frame_calibration
    4. Press SPACE to capture and compute transformation
    5. Transformation saved automatically

The checkerboard defines the platform coordinate system:
    - Origin: Corner (0,0) of checkerboard
    - X-axis: Direction along first row of checkerboard
    - Y-axis: Direction along first column of checkerboard
    - Z-axis: Perpendicular to checkerboard (pointing up)
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from ball_detection.utils.coordinate_transform import (
    compute_transformation_from_points,
    save_platform_transform
)

# ============================================================
# SETTINGS - Edit these
# ============================================================
# Chessboard configuration (must match your physical checkerboard)
# IMPORTANT: Use INNER CORNERS, not number of squares!
# If you have 8 rows × 11 columns of SQUARES:
#   Inner corners = (11-1, 8-1) = (10, 7) in format (columns, rows)
CHESSBOARD_SIZE = (8, 6)  # Inner corners (columns, rows) - 8x11 squares
SQUARE_SIZE_MM = 24.88888889  # Size of each square in mm

# Camera configuration
CAMERA_INDEX = 0  # ZED stereo camera

# Calibration directory
CALIBRATION_DIR = Path(__file__).parent / "calibrations"

# Z-offset (optional): Set to platform home height if origin should be (0, 0, home_height)
Z_OFFSET_MM = 0.0  # Change to home_height if needed (e.g., 50.0 for 50mm)
# ============================================================


def create_camera_capture(camera_index):
    """Create VideoCapture object with Windows backends."""
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW]

    for backend in backends:
        try:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                return cap
            cap.release()
        except Exception:
            continue

    cap = cv2.VideoCapture(camera_index)
    return cap if cap.isOpened() else None


def load_stereo_calibration(calib_dir):
    """Load latest stereo calibration data."""
    import glob

    # Find latest stereo calibration files
    p1_files = sorted(calib_dir.glob('stereo_P1_*.csv'), reverse=True)
    p2_files = sorted(calib_dir.glob('stereo_P2_*.csv'), reverse=True)
    map_files = sorted(calib_dir.glob('stereo_left_map1_*.npy'), reverse=True)

    if not p1_files or not p2_files or not map_files:
        print("\nError: No stereo calibration found!")
        print("You need to run stereo calibration first:")
        print("  python -m ball_detection.calibration.stereo_calibration")
        return None

    # Extract timestamp from filename
    timestamp = p1_files[0].name.replace('stereo_P1_', '').replace('.csv', '')

    try:
        # Load projection matrices
        P1 = np.loadtxt(calib_dir / f'stereo_P1_{timestamp}.csv', delimiter=',')
        P2 = np.loadtxt(calib_dir / f'stereo_P2_{timestamp}.csv', delimiter=',')

        # Load rectification maps
        left_map1 = np.load(calib_dir / f'stereo_left_map1_{timestamp}.npy')
        left_map2 = np.load(calib_dir / f'stereo_left_map2_{timestamp}.npy')
        right_map1 = np.load(calib_dir / f'stereo_right_map1_{timestamp}.npy')
        right_map2 = np.load(calib_dir / f'stereo_right_map2_{timestamp}.npy')

        print(f"Loaded stereo calibration: {timestamp}")
        print(f"  Source: {calib_dir}")

        return {
            'P1': P1,
            'P2': P2,
            'left_map1': left_map1,
            'left_map2': left_map2,
            'right_map1': right_map1,
            'right_map2': right_map2,
            'timestamp': timestamp
        }

    except Exception as e:
        print(f"Error loading stereo calibration: {e}")
        return None


def triangulate_3d_point(left_point, right_point, P1, P2):
    """
    Triangulate 3D point from corresponding 2D points.

    Args:
        left_point: (x, y) in left camera
        right_point: (x, y) in right camera
        P1, P2: Projection matrices from stereo calibration

    Returns:
        3D point [x, y, z] in mm, or None if inputs invalid
    """
    if left_point is None or right_point is None:
        return None

    # Convert to homogeneous coordinates
    left_pt = np.array([[left_point[0]], [left_point[1]]], dtype=np.float32)
    right_pt = np.array([[right_point[0]], [right_point[1]]], dtype=np.float32)

    # Triangulate
    points_4d = cv2.triangulatePoints(P1, P2, left_pt, right_pt)

    # Convert from homogeneous to 3D
    points_3d = points_4d[:3] / points_4d[3]

    return points_3d.flatten()


def generate_platform_points(chessboard_size, square_size):
    """
    Generate known 3D positions of checkerboard corners in platform frame.

    Platform coordinate system:
        - Origin at corner (0, 0)
        - X-axis along first row
        - Y-axis along first column
        - Z = 0 (flat on platform)

    Args:
        chessboard_size: (columns, rows) of inner corners
        square_size: Size of each square in mm

    Returns:
        Nx3 array of corner positions in platform frame
    """
    cols, rows = chessboard_size
    points = []

    for row in range(rows):
        for col in range(cols):
            x = col * square_size
            y = row * square_size
            z = 0.0  # Checkerboard is flat on platform
            points.append([x, y, z])

    return np.array(points, dtype=np.float32)


def calibrate_platform_frame():
    """Main platform frame calibration function."""
    print("=" * 60)
    print("PLATFORM COORDINATE FRAME CALIBRATION")
    print("=" * 60)
    print(f"Checkerboard: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} inner corners")
    print(f"Square size: {SQUARE_SIZE_MM} mm")
    print(f"Z offset: {Z_OFFSET_MM} mm")
    print("=" * 60)
    print()

    # Load stereo calibration
    print("[1/4] Loading stereo calibration...")
    stereo_calib = load_stereo_calibration(CALIBRATION_DIR)
    if not stereo_calib:
        return
    print()

    # Open camera
    print("[2/4] Opening camera...")
    cap = create_camera_capture(CAMERA_INDEX)
    if cap is None:
        print(f"Failed to open camera {CAMERA_INDEX}")
        return

    # Configure camera for stereo
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"Camera opened: 2560x720 @ 30fps")
    print()

    # Instructions
    print("[3/4] Instructions:")
    print("  1. Place physical checkerboard flat on Stewart platform")
    print("  2. The ORIGIN corner will be marked with:")
    print("     - Large RED circle")
    print("     - Label 'ORIGIN (0,0,0)'")
    print("     - BLUE arrow showing +X axis direction")
    print("     - GREEN arrow showing +Y axis direction")
    print("  3. Align the ORIGIN corner with your desired platform origin:")
    print("     - Platform center, OR")
    print("     - Platform center at home height")
    print("  4. Ensure checkerboard is perfectly flat (defines XY plane)")
    print("  5. Press SPACE when ready to capture")
    print("  6. Press 'q' to quit")
    print()

    # Generate known platform points
    platform_points = generate_platform_points(CHESSBOARD_SIZE, SQUARE_SIZE_MM)

    # Apply Z offset if specified
    if Z_OFFSET_MM != 0:
        platform_points[:, 2] += Z_OFFSET_MM
        print(f"Applied Z offset: {Z_OFFSET_MM} mm")

    calibrated = False

    try:
        while not calibrated:
            ret, full_frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Split stereo frame
            left_frame = full_frame[:, 0:1280].copy()
            right_frame = full_frame[:, 1280:2560].copy()

            # Apply rectification
            left_rectified = cv2.remap(left_frame, stereo_calib['left_map1'],
                                      stereo_calib['left_map2'], cv2.INTER_LINEAR)
            right_rectified = cv2.remap(right_frame, stereo_calib['right_map1'],
                                       stereo_calib['right_map2'], cv2.INTER_LINEAR)

            # Find checkerboard in both cameras
            gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

            ret_left, corners_left = cv2.findChessboardCorners(
                gray_left, CHESSBOARD_SIZE,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            ret_right, corners_right = cv2.findChessboardCorners(
                gray_right, CHESSBOARD_SIZE,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            # Visualize
            display_left = left_rectified.copy()
            display_right = right_rectified.copy()

            if ret_left and ret_right:
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
                corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

                # Draw checkerboard pattern
                cv2.drawChessboardCorners(display_left, CHESSBOARD_SIZE, corners_left, ret_left)
                cv2.drawChessboardCorners(display_right, CHESSBOARD_SIZE, corners_right, ret_right)

                # Highlight ORIGIN corner (0,0) - first detected corner
                origin_left = tuple(corners_left[0].ravel().astype(int))
                origin_right = tuple(corners_right[0].ravel().astype(int))

                # Draw large red circle at origin
                cv2.circle(display_left, origin_left, 15, (0, 0, 255), 3)
                cv2.circle(display_right, origin_right, 15, (0, 0, 255), 3)

                # Label origin
                cv2.putText(display_left, "ORIGIN (0,0,0)", (origin_left[0] + 20, origin_left[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_right, "ORIGIN (0,0,0)", (origin_right[0] + 20, origin_right[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Draw X-axis direction (to next corner in first row)
                if len(corners_left) > 1:
                    x_axis_pt_left = tuple(corners_left[1].ravel().astype(int))
                    x_axis_pt_right = tuple(corners_right[1].ravel().astype(int))
                    cv2.arrowedLine(display_left, origin_left, x_axis_pt_left, (255, 0, 0), 2, tipLength=0.3)
                    cv2.arrowedLine(display_right, origin_right, x_axis_pt_right, (255, 0, 0), 2, tipLength=0.3)
                    cv2.putText(display_left, "+X", x_axis_pt_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(display_right, "+X", x_axis_pt_right, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Draw Y-axis direction (to next corner in first column)
                cols, rows = CHESSBOARD_SIZE
                if len(corners_left) > cols:
                    y_axis_pt_left = tuple(corners_left[cols].ravel().astype(int))
                    y_axis_pt_right = tuple(corners_right[cols].ravel().astype(int))
                    cv2.arrowedLine(display_left, origin_left, y_axis_pt_left, (0, 255, 0), 2, tipLength=0.3)
                    cv2.arrowedLine(display_right, origin_right, y_axis_pt_right, (0, 255, 0), 2, tipLength=0.3)
                    cv2.putText(display_left, "+Y", y_axis_pt_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_right, "+Y", y_axis_pt_right, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.putText(display_left, "CHECKERBOARD DETECTED - Press SPACE to calibrate",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_right, "CHECKERBOARD DETECTED - Press SPACE to calibrate",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display_left, "Searching for checkerboard...",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_right, "Searching for checkerboard...",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display side-by-side
            combined = np.hstack([display_left, display_right])
            cv2.imshow('Platform Frame Calibration', combined)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' ') and ret_left and ret_right:
                print("[4/4] Computing platform transformation...")

                # Triangulate all corners
                camera_points = []
                for i in range(len(corners_left)):
                    left_pt = corners_left[i][0]
                    right_pt = corners_right[i][0]

                    point_3d = triangulate_3d_point(left_pt, right_pt,
                                                   stereo_calib['P1'],
                                                   stereo_calib['P2'])
                    if point_3d is not None:
                        camera_points.append(point_3d)

                camera_points = np.array(camera_points)

                if len(camera_points) != len(platform_points):
                    print(f"Error: Mismatch in point count")
                    continue

                # Compute transformation
                R, T, rmse = compute_transformation_from_points(camera_points, platform_points)

                print()
                print("Transformation computed successfully!")
                print(f"  RMSE: {rmse:.4f} mm")
                print()
                print("Rotation matrix (R):")
                print(R)
                print()
                print("Translation vector (T):")
                print(T)
                print()

                # Validate by transforming points back
                print("Validation (first 3 corners):")
                print("  Platform → Camera → Platform (should match)")
                for i in range(min(3, len(platform_points))):
                    expected = platform_points[i]
                    camera_pt = camera_points[i]
                    reconstructed = R @ camera_pt + T
                    error = np.linalg.norm(expected - reconstructed)
                    print(f"  Corner {i}: {expected} → {reconstructed} (error: {error:.4f} mm)")

                # Save transformation
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = save_platform_transform(R, T, CALIBRATION_DIR, timestamp)
                print()
                print(f"Transformation saved to: {filepath}")

                calibrated = True

            elif key == ord('q'):
                print("Calibration cancelled")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if calibrated:
        print()
        print("=" * 60)
        print("PLATFORM CALIBRATION COMPLETE!")
        print("=" * 60)
        print()
        print("Your platform coordinate system is now calibrated.")
        print("3D detections will automatically use platform coordinates.")
        print()
        print("Next step:")
        print("  Test 3D detection: python test_stereo_detection.py")
        print("=" * 60)


if __name__ == "__main__":
    calibrate_platform_frame()
