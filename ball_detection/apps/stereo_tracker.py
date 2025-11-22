#!/usr/bin/env python3
"""
3D Ball Tracking with CNN Detection + Stereo Triangulation

Combines the new sub-pixel CNN ball detector with existing stereo calibration
for smooth, accurate 3D ball tracking.

Performance:
- Detection: ~6-7ms (dual camera, batched CNN)
- Triangulation: ~1ms
- Total: ~8ms per frame (60 FPS capable with margin)
"""

import cv2
import numpy as np
import os
import glob
import time
from pathlib import Path
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ..core.detector import BallDetector
from ..utils.camera import create_camera_capture
from ..utils.coordinate_transform import (
    load_platform_transform,
    apply_platform_transform
)

# ============================================================
# SETTINGS - Edit these
# ============================================================
# Camera configuration
CAMERA_INDEX = 0  # ZED stereo camera

# Stereo calibration path
CALIBRATION_DIR = Path(__file__).parent.parent / "calibration" / "calibrations"

# 3D plotting configuration
MAX_POINTS = 100  # Maximum number of 3D points to keep in history
# ============================================================


def load_stereo_calibration():
    """Load latest stereo calibration data."""
    calib_dir = CALIBRATION_DIR

    if not calib_dir.exists():
        print("\nError: No stereo calibration found!")
        print("\nYou need to run stereo calibration first:")
        print("  python -m ball_detection.calibration.stereo_calibration --calibrate-individual")
        print("  python -m ball_detection.calibration.stereo_calibration --calibrate-stereo")
        return None

    # Find latest stereo calibration files
    p1_files = sorted(calib_dir.glob('stereo_P1_*.csv'), reverse=True)
    p2_files = sorted(calib_dir.glob('stereo_P2_*.csv'), reverse=True)
    map_files = sorted(calib_dir.glob('stereo_left_map1_*.npy'), reverse=True)

    if not p1_files or not p2_files or not map_files:
        print("\nError: No stereo calibration files found!")
        print("\nYou need to run stereo calibration first:")
        print("  python -m ball_detection.calibration.stereo_calibration --calibrate-individual")
        print("  python -m ball_detection.calibration.stereo_calibration --calibrate-stereo")
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

        print(f"✓ Loaded stereo calibration: {timestamp}")
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
        print(f"\nError loading calibration from {calib_dir}: {e}")
        return None


def triangulate_3d_point(left_point, right_point, P1, P2):
    """
    Triangulate 3D point from corresponding 2D points.

    Now supports sub-pixel accuracy from CNN detector!

    Args:
        left_point: (x, y) in left camera (float for sub-pixel)
        right_point: (x, y) in right camera (float for sub-pixel)
        P1, P2: Projection matrices from stereo calibration

    Returns:
        3D point [x, y, z] in mm, or None if inputs invalid
    """
    if left_point is None or right_point is None:
        return None

    # Convert to homogeneous coordinates (supports sub-pixel)
    left_pt = np.array([[left_point[0]], [left_point[1]]], dtype=np.float32)
    right_pt = np.array([[right_point[0]], [right_point[1]]], dtype=np.float32)

    # Triangulate
    points_4d = cv2.triangulatePoints(P1, P2, left_pt, right_pt)

    # Convert from homogeneous to 3D
    points_3d = points_4d[:3] / points_4d[3]

    return points_3d.flatten()


def update_3d_plot(ax, points_history):
    """Update 3D plot with current points history."""
    ax.clear()

    if not points_history:
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('Live 3D Ball Tracking (No data)')
        return

    # Extract coordinates
    x_coords = [p[0] for p in points_history]
    y_coords = [p[1] for p in points_history]
    z_coords = [p[2] for p in points_history]

    # Plot trajectory line
    if len(points_history) > 1:
        ax.plot(x_coords, y_coords, z_coords, 'b-', alpha=0.6, linewidth=1)

    # Plot recent points with gradient
    n_recent = min(20, len(points_history))
    if n_recent > 0:
        recent_x = x_coords[-n_recent:]
        recent_y = y_coords[-n_recent:]
        recent_z = z_coords[-n_recent:]

        # Color gradient from blue (old) to red (new)
        colors = plt.cm.coolwarm(np.linspace(0, 1, n_recent))
        ax.scatter(recent_x, recent_y, recent_z, c=colors, s=50)

    # Highlight current position
    current = points_history[-1]
    ax.scatter([current[0]], [current[1]], [current[2]],
               c='red', s=100, marker='o')

    # Set labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Live 3D CNN Ball Tracking ({len(points_history)} points)')

    # Auto-scale with padding
    if len(points_history) > 1:
        margin = 10  # mm
        ax.set_xlim([min(x_coords) - margin, max(x_coords) + margin])
        ax.set_ylim([min(y_coords) - margin, max(y_coords) + margin])
        ax.set_zlim([min(z_coords) - margin, max(z_coords) + margin])


def main():
    """Main function for CNN-based 3D ball tracking."""
    print("=" * 60)
    print("3D Ball Tracking: CNN Detection + Stereo Triangulation")
    print("=" * 60)

    # Load stereo calibration
    print("\n[1/4] Loading stereo calibration...")
    stereo_calib = load_stereo_calibration()
    if not stereo_calib:
        return

    # Load platform transformation (optional)
    print("\n[2/4] Loading platform transformation...")
    platform_transform = None
    try:
        platform_calib = load_platform_transform(CALIBRATION_DIR)
        R = platform_calib['R']
        T = platform_calib['T']
        platform_transform = (R, T)
        print(f"✓ Platform transformation loaded: {platform_calib['timestamp']}")
        print(f"  Coordinates will be in platform frame")
    except FileNotFoundError:
        print("  No platform transformation found - using camera coordinates")
        print("  Run platform calibration to get platform coordinates:")
        print("    python -m ball_detection.calibration.platform_frame_calibration")

    # Initialize CNN detector
    print("\n[3/4] Initializing CNN ball detector...")
    # Try relative path (when run from ball_detection dir) and parent path
    model_paths = [
        Path(__file__).parent.parent / "models" / "best_pixel_error.onnx",  # ball_detection/models
        Path("ball_detection/models/best_pixel_error.onnx")  # From root
    ]

    model_path = None
    for path in model_paths:
        if path.exists():
            model_path = path
            break

    if model_path is None:
        print(f"Error: Model not found!")
        print("\nYou need to:")
        print("  1. Collect and label training data")
        print("  2. Train the model (python -m ball_detection.train)")
        print("  3. Export to ONNX (python -m ball_detection.export_onnx)")
        print("\nSee ball_detection/QUICKSTART.md for details")
        return

    detector = BallDetector(
        onnx_model_path=str(model_path),
        use_gpu=False,  # CPU - better performance than DirectML for this model
        crop_size=128,  # Match training crop size
        confidence_threshold=0.5
    )

    # Open camera
    print("\n[4/4] Opening camera...")
    cap = create_camera_capture(CAMERA_INDEX)
    if cap is None:
        print(f"Failed to open camera {CAMERA_INDEX}")
        return

    # Configure camera
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

    print("\n" + "=" * 60)
    print("System ready!")
    print("=" * 60)
    print("\nControls:")
    print("  'q' - Quit")
    print("  'c' - Clear 3D trajectory")
    print("  's' - Print statistics")
    print("  'p' - Toggle 3D plot")
    print("\nStarting tracking...\n")

    # Setup matplotlib (optional)
    show_3d_plot = False
    fig, ax = None, None

    # Initialize points history
    points_history = []

    # Performance tracking
    import time
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

            # Split frame (2560x720 -> two 1280x720)
            left_frame = full_frame[:, 0:1280].copy()
            right_frame = full_frame[:, 1280:2560].copy()

            # Apply stereo rectification
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
            point_3d = None
            if result_left and result_right:
                stereo_pairs += 1

                # Extract sub-pixel coordinates
                x_left, y_left, conf_left = result_left
                x_right, y_right, conf_right = result_right

                # Triangulate (now with sub-pixel accuracy!)
                tri_start = time.time()
                point_3d = triangulate_3d_point(
                    (x_left, y_left),
                    (x_right, y_right),
                    stereo_calib['P1'],
                    stereo_calib['P2']
                )
                triangulation_time = (time.time() - tri_start) * 1000  # ms
                triangulation_times.append(triangulation_time)

                if point_3d is not None:
                    # Transform to platform coordinates if available
                    if platform_transform is not None:
                        R, T = platform_transform
                        point_3d = apply_platform_transform(point_3d, R, T)

                    # Add to history
                    points_history.append(point_3d.copy())

                    # Limit history size
                    if len(points_history) > MAX_POINTS:
                        points_history.pop(0)

            # Visualize detections
            vis_left = detector.visualize(left_rectified, result_left)
            vis_right = detector.visualize(right_rectified, result_right)

            # Add 3D coordinates on left view
            if point_3d is not None:
                frame_type = "Platform" if platform_transform else "Camera"
                coord_text = f"3D ({frame_type}): ({point_3d[0]:.1f}, {point_3d[1]:.1f}, {point_3d[2]:.1f}) mm"
                cv2.putText(vis_left, coord_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(vis_left, "3D: WAITING FOR STEREO PAIR", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Add performance info
            if frame_count % 30 == 0 and detection_times:
                fps = 30 / (time.time() - fps_start)
                fps_start = time.time()
                avg_det = np.mean(detection_times[-30:])

                perf_text = f"FPS: {fps:.1f} | Det: {avg_det:.1f}ms"
                cv2.putText(vis_left, perf_text, (10, vis_left.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Add frame counter
            cv2.putText(vis_left, f"Frame: {frame_count} | 3D Points: {len(points_history)}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display
            combined = np.hstack([vis_left, vis_right])
            cv2.imshow('CNN Stereo Ball Tracking', combined)

            # Update 3D plot if enabled
            if show_3d_plot and frame_count % 3 == 0:  # Update every 3 frames
                if fig is None:
                    plt.ion()
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    plt.show(block=False)

                try:
                    update_3d_plot(ax, points_history)
                    plt.draw()
                    plt.pause(0.001)
                except Exception as e:
                    print(f"Plot error: {e}")

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c'):
                points_history.clear()
                print("✓ Cleared 3D trajectory")
            elif key == ord('p'):
                show_3d_plot = not show_3d_plot
                if not show_3d_plot and fig:
                    plt.close(fig)
                    fig, ax = None, None
                print(f"✓ 3D plot {'enabled' if show_3d_plot else 'disabled'}")
            elif key == ord('s'):
                # Print statistics
                print("\n" + "=" * 60)
                print("STATISTICS")
                print("=" * 60)
                print(f"Frames processed: {frame_count}")
                print(f"Left detections: {left_detections} ({left_detections/frame_count*100:.1f}%)")
                print(f"Right detections: {right_detections} ({right_detections/frame_count*100:.1f}%)")
                print(f"Stereo pairs: {stereo_pairs} ({stereo_pairs/frame_count*100:.1f}%)")
                print(f"3D points collected: {len(points_history)}")

                if detection_times:
                    avg_det = np.mean(detection_times)
                    print(f"\nDetection time: {avg_det:.2f} ± {np.std(detection_times):.2f} ms")
                    print(f"  Min: {np.min(detection_times):.2f} ms")
                    print(f"  Max: {np.max(detection_times):.2f} ms")
                    print(f"  P95: {np.percentile(detection_times, 95):.2f} ms")

                if triangulation_times:
                    avg_tri = np.mean(triangulation_times)
                    print(f"\nTriangulation time: {avg_tri:.2f} ± {np.std(triangulation_times):.2f} ms")

                stats = detector.get_statistics()
                print(f"\nCNN inferences: {stats['cnn_inferences']}")
                print(f"CNN avg time: {stats['cnn_avg_time_ms']:.2f} ms")
                print(f"GPU acceleration: {stats['using_gpu']}")
                print("=" * 60 + "\n")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        if fig:
            plt.close('all')

        # Final statistics
        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)
        print(f"Total frames: {frame_count}")
        print(f"Left detections: {left_detections} ({left_detections/max(frame_count,1)*100:.1f}%)")
        print(f"Right detections: {right_detections} ({right_detections/max(frame_count,1)*100:.1f}%)")
        print(f"Stereo pairs: {stereo_pairs} ({stereo_pairs/max(frame_count,1)*100:.1f}%)")
        print(f"Total 3D points: {len(points_history)}")

        if detection_times:
            avg_det = np.mean(detection_times)
            print(f"\nAverage detection time: {avg_det:.2f} ms")

        if triangulation_times:
            avg_tri = np.mean(triangulation_times)
            print(f"Average triangulation time: {avg_tri:.2f} ms")
            total_avg = avg_det + avg_tri
            print(f"Total processing time: {total_avg:.2f} ms")
            print(f"Maximum achievable FPS: {1000/total_avg:.1f}")

        stats = detector.get_statistics()
        print(f"\nTotal CNN inferences: {stats['cnn_inferences']}")
        print(f"CNN average time: {stats['cnn_avg_time_ms']:.2f} ms")
        print(f"GPU acceleration: {stats['using_gpu']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
