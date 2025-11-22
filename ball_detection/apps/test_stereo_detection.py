#!/usr/bin/env python3
"""
Stereo 3D Ball Detection Test

Simple standalone script to test stereo triangulation with CNN ball detection.
Outputs 3D coordinates in platform reference frame.

Requirements:
    1. Stereo calibration completed (P1, P2, rectification maps)
    2. Platform frame calibration completed (transformation matrix)
    3. ONNX ball detection model exported

Usage:
    python -m ball_detection.apps.test_stereo_detection
"""

import cv2
import numpy as np
from pathlib import Path
import time

from ..core.detector import BallDetector
from ..utils.coordinate_transform import (
    load_platform_transform,
    apply_platform_transform
)

# ============================================================
# SETTINGS - Edit these
# ============================================================
CAMERA_INDEX = 0  # ZED stereo camera
CALIBRATION_DIR = Path(__file__).parent.parent / "calibration" / "calibrations"
MODEL_PATH = Path(__file__).parent.parent / "models" / "best_pixel_error.onnx"

# Detection parameters
CROP_SIZE = 128  # CNN input crop size
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detection
USE_GPU = False  # DirectML GPU acceleration

# Display parameters
SHOW_VISUALIZATION = True  # Show live camera feed with detections
PRINT_COORDINATES = True  # Print 3D coordinates to console
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
    calib_path = Path(calib_dir)

    # Find latest stereo calibration files
    p1_files = sorted(calib_path.glob('stereo_P1_*.csv'), reverse=True)
    p2_files = sorted(calib_path.glob('stereo_P2_*.csv'), reverse=True)
    map_files = sorted(calib_path.glob('stereo_left_map1_*.npy'), reverse=True)

    if not p1_files or not p2_files or not map_files:
        raise FileNotFoundError(
            f"No stereo calibration found in {calib_dir}\n"
            f"Run: python -m ball_detection.calibration.stereo_calibration"
        )

    # Extract timestamp from filename
    timestamp = p1_files[0].name.replace('stereo_P1_', '').replace('.csv', '')

    # Load projection matrices
    P1 = np.loadtxt(calib_path / f'stereo_P1_{timestamp}.csv', delimiter=',')
    P2 = np.loadtxt(calib_path / f'stereo_P2_{timestamp}.csv', delimiter=',')

    # Load rectification maps
    left_map1 = np.load(calib_path / f'stereo_left_map1_{timestamp}.npy')
    left_map2 = np.load(calib_path / f'stereo_left_map2_{timestamp}.npy')
    right_map1 = np.load(calib_path / f'stereo_right_map1_{timestamp}.npy')
    right_map2 = np.load(calib_path / f'stereo_right_map2_{timestamp}.npy')

    return {
        'P1': P1,
        'P2': P2,
        'left_map1': left_map1,
        'left_map2': left_map2,
        'right_map1': right_map1,
        'right_map2': right_map2,
        'timestamp': timestamp
    }


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


def main():
    """Main 3D ball detection loop."""
    print("=" * 60)
    print("STEREO 3D BALL DETECTION TEST")
    print("=" * 60)
    print()

    # Load stereo calibration
    print("[1/4] Loading stereo calibration...")
    try:
        stereo_calib = load_stereo_calibration(CALIBRATION_DIR)
        print(f"  Loaded: {stereo_calib['timestamp']}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    print()

    # Load platform transformation
    print("[2/4] Loading platform transformation...")
    try:
        platform_calib = load_platform_transform(CALIBRATION_DIR)
        R = platform_calib['R']
        T = platform_calib['T']
        print(f"  Loaded: {platform_calib['timestamp']}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Note: Platform calibration is optional for testing camera coordinates")
        print("      Set R=I, T=0 to get camera frame coordinates")
        R = np.eye(3)
        T = np.zeros(3)
    print()

    # Initialize ball detector
    print("[3/4] Loading ball detection model...")
    try:
        detector = BallDetector(
            onnx_model_path=MODEL_PATH,
            use_gpu=USE_GPU,
            crop_size=CROP_SIZE,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        print(f"  Model loaded: {MODEL_PATH}")
        print(f"  GPU: {'Enabled (DirectML)' if USE_GPU else 'Disabled (CPU)'}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure ONNX model is exported:")
        print("  python -m ball_detection.training.export_onnx")
        return
    print()

    # Open camera
    print("[4/4] Opening camera...")
    cap = create_camera_capture(CAMERA_INDEX)
    if cap is None:
        print(f"Failed to open camera {CAMERA_INDEX}")
        return

    # Configure camera for stereo
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Apply camera settings (same as tune_hsv for consistent detection)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 5)
    cap.set(cv2.CAP_PROP_CONTRAST, 2)
    cap.set(cv2.CAP_PROP_SATURATION, 4)
    cap.set(cv2.CAP_PROP_GAIN, 2)
    cap.set(cv2.CAP_PROP_SHARPNESS, 0)
    cap.set(cv2.CAP_PROP_GAMMA, 102)

    print(f"  Camera opened: 2560x720 @ 60fps")
    print()

    print("=" * 60)
    print("LIVE 3D DETECTION - Press 'q' to quit")
    print("=" * 60)
    print()

    # Performance tracking
    frame_count = 0
    detection_times = []
    triangulation_times = []
    fps_start = time.time()

    # Detection statistics
    left_detections = 0
    right_detections = 0
    stereo_pairs = 0

    try:
        while True:
            ret, full_frame = cap.read()
            if not ret:
                print("Failed to capture frame")
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

            # Detect ball in both cameras (separate inferences, no batching)
            det_start = time.time()
            result_left = detector.detect(left_rectified)
            result_right = detector.detect(right_rectified)
            detection_time = (time.time() - det_start) * 1000  # ms
            detection_times.append(detection_time)

            # Track detection statistics
            if result_left:
                left_detections += 1
            if result_right:
                right_detections += 1

            # Triangulate if detected in both cameras
            point_3d_camera = None
            point_3d_platform = None

            if result_left and result_right:
                stereo_pairs += 1

                # Extract sub-pixel coordinates
                x_left, y_left, conf_left = result_left
                x_right, y_right, conf_right = result_right

                # Triangulate (camera coordinates)
                tri_start = time.time()
                point_3d_camera = triangulate_3d_point(
                    (x_left, y_left),
                    (x_right, y_right),
                    stereo_calib['P1'],
                    stereo_calib['P2']
                )
                triangulation_time = (time.time() - tri_start) * 1000  # ms
                triangulation_times.append(triangulation_time)

                # Transform to platform coordinates
                if point_3d_camera is not None:
                    point_3d_platform = apply_platform_transform(point_3d_camera, R, T)

                    # Print coordinates
                    if PRINT_COORDINATES:
                        print(f"3D Position: "
                              f"X={point_3d_platform[0]:7.2f}mm, "
                              f"Y={point_3d_platform[1]:7.2f}mm, "
                              f"Z={point_3d_platform[2]:7.2f}mm "
                              f"[Conf: L={conf_left:.2f}, R={conf_right:.2f}]")

            # Visualization
            if SHOW_VISUALIZATION:
                vis_left = detector.visualize(left_rectified, result_left)
                vis_right = detector.visualize(right_rectified, result_right)

                # Add 3D coordinates on left view
                if point_3d_platform is not None:
                    coord_text = f"3D: ({point_3d_platform[0]:.1f}, {point_3d_platform[1]:.1f}, {point_3d_platform[2]:.1f}) mm"
                    cv2.putText(vis_left, coord_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(vis_left, "3D: WAITING FOR STEREO PAIR", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Add performance info
                if frame_count % 30 == 0 and detection_times:
                    fps = 30 / (time.time() - fps_start)
                    fps_start = time.time()
                    avg_det = np.mean(detection_times[-30:])

                    perf_text = f"FPS: {fps:.1f} | Det: {avg_det:.1f}ms"
                    cv2.putText(vis_left, perf_text, (10, vis_left.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                # Display side-by-side
                combined = np.hstack([vis_left, vis_right])
                cv2.imshow('Stereo 3D Ball Detection', combined)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        # Final statistics
        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)
        print(f"Total frames: {frame_count}")
        print(f"Left detections: {left_detections} ({left_detections/max(frame_count,1)*100:.1f}%)")
        print(f"Right detections: {right_detections} ({right_detections/max(frame_count,1)*100:.1f}%)")
        print(f"Stereo pairs: {stereo_pairs} ({stereo_pairs/max(frame_count,1)*100:.1f}%)")

        if detection_times:
            avg_det = np.mean(detection_times)
            print(f"\nAverage detection time: {avg_det:.2f} ms")

        if triangulation_times:
            avg_tri = np.mean(triangulation_times)
            print(f"Average triangulation time: {avg_tri:.2f} ms")
            total_avg = avg_det + avg_tri
            print(f"Total processing time: {total_avg:.2f} ms")
            print(f"Maximum achievable FPS: {1000/total_avg:.1f}")

        print("=" * 60)


if __name__ == "__main__":
    main()
