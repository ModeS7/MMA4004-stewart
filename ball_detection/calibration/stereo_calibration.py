#!/usr/bin/env python3
"""
Stereo Camera Calibration for CNN Ball Tracking

Complete calibration pipeline:
1. Individual camera calibration (intrinsics + distortion)
2. Stereo calibration (extrinsics + rectification)
3. Save all parameters for later use

Usage:
    Edit MODE setting below and run: python stereo_calibration.py
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ball_detection.utils.camera import create_camera_capture, load_camera_config, apply_camera_settings

# ============================================================
# SETTINGS - Edit these
# ============================================================
# Calibration mode: "individual", "stereo", or "both"
MODE = "both"  # "individual" = calibrate cameras separately, "stereo" = calibrate stereo pair, "both" = do both

# Chessboard configuration (INNER CORNERS, not squares!)
# If you have 8 rows × 11 columns of SQUARES:
#   Inner corners = (11-1, 8-1) = (10, 7) in format (columns, rows)
CHESSBOARD_SIZE = (10, 7)  # Inner corners (columns, rows) - 8x11 squares
SQUARE_SIZE = 50.0  # mm - size of each checkerboard square

# Camera configuration
CAMERA_INDEX = 0

# Storage (relative to this script's location)
CALIBRATION_DIR = Path(__file__).parent / "calibrations"
# ============================================================

CALIBRATION_DIR.mkdir(exist_ok=True)


def collect_calibration_images(camera_id, num_images=20, is_stereo=False, camera_side='left'):
    """
    Collect calibration images by showing chessboard to camera.

    Args:
        camera_id: Camera device index
        num_images: Number of images to collect
        is_stereo: If True, expects dual camera (2560x720) and splits frame
        camera_side: 'left' or 'right' - which camera to calibrate (for individual calibration)

    Returns:
        List of images with detected chessboard corners
    """
    print(f"\n{'='*60}")
    print(f"Collecting {num_images} calibration images")
    if not is_stereo:
        print(f"Camera: {camera_side.upper()}")
    print(f"{'='*60}")
    print("\nInstructions:")
    print("  1. Hold chessboard in front of camera")
    if not is_stereo:
        print(f"  2. Show checkerboard to {camera_side.upper()} camera ONLY")
        print("  3. Move it to different positions and angles")
    else:
        print("  2. Show checkerboard to BOTH cameras simultaneously")
        print("  3. Move it to different positions and angles")
    print("  4. Press SPACE when chessboard is detected to capture")
    print("  5. Press 'q' to quit early")
    print(f"\nTarget: {num_images} images\n")

    cap = create_camera_capture(camera_id)
    if cap is None:
        print(f"Failed to open camera {camera_id}")
        return None

    # Always configure for stereo (ZED camera outputs 2560x720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    collected_images = []
    collected_count = 0

    # Prepare object points
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    try:
        while collected_count < num_images:
            ret, full_frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Split stereo frame
            left_frame = full_frame[:, 0:1280].copy()
            right_frame = full_frame[:, 1280:2560].copy()

            # For individual calibration, only analyze the selected camera
            if not is_stereo:
                if camera_side.lower() == 'left':
                    active_frame = left_frame
                    inactive_frame = right_frame
                else:  # right
                    active_frame = right_frame
                    inactive_frame = left_frame
            else:
                # For stereo calibration, use full frame
                active_frame = full_frame

            display_active = active_frame.copy()
            gray = cv2.cvtColor(active_frame, cv2.COLOR_BGR2GRAY)

            # Find chessboard corners in ACTIVE camera only
            ret_corners, corners = cv2.findChessboardCorners(
                gray, CHESSBOARD_SIZE,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            # Draw corners if found
            if ret_corners:
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                cv2.drawChessboardCorners(display_active, CHESSBOARD_SIZE, corners_refined, ret_corners)

                # Add status
                cv2.putText(display_active, "CHESSBOARD DETECTED - Press SPACE to capture",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_active, "Searching for chessboard...",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show progress
            cv2.putText(display_active, f"Collected: {collected_count}/{num_images}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display based on mode
            if not is_stereo:
                # Individual calibration: show both cameras side-by-side
                # Mark which one is being calibrated
                display_inactive = inactive_frame.copy()
                cv2.putText(display_inactive, "INACTIVE - Not being calibrated",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

                if camera_side.lower() == 'left':
                    combined_display = np.hstack([display_active, display_inactive])
                    window_title = 'LEFT Camera Calibration (Left=Active, Right=Inactive)'
                else:
                    combined_display = np.hstack([display_inactive, display_active])
                    window_title = 'RIGHT Camera Calibration (Left=Inactive, Right=Active)'

                cv2.imshow(window_title, combined_display)
            else:
                # Stereo calibration: show full frame
                cv2.imshow('Stereo Calibration Image Collection', display_active)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' ') and ret_corners:  # Space to capture
                collected_images.append({
                    'image': gray.copy(),
                    'corners': corners_refined,
                    'objpoints': objp.copy()
                })
                collected_count += 1
                print(f"[OK] Captured image {collected_count}/{num_images}")

            elif key == ord('q'):
                if collected_count >= 10:
                    print(f"\nEarly exit with {collected_count} images (minimum 10 reached)")
                    break
                else:
                    print(f"\nNeed at least 10 images, currently have {collected_count}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if collected_count < 10:
        print(f"\nError: Need at least 10 images, only collected {collected_count}")
        return None

    return collected_images


def calibrate_camera(images_data, image_size):
    """
    Calibrate camera from collected images.

    Args:
        images_data: List of dicts with 'corners' and 'objpoints'
        image_size: (width, height) of images

    Returns:
        Dict with calibration results
    """
    print(f"\nCalibrating camera with {len(images_data)} images...")

    # Prepare data
    objpoints = [img['objpoints'] for img in images_data]
    imgpoints = [img['corners'] for img in images_data]

    # Calibrate
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    if not ret:
        print("Calibration failed!")
        return None

    # Calculate reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                          camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

    mean_error = total_error / len(objpoints)

    print(f"[OK] Calibration successful!")
    print(f"  RMS reprojection error: {mean_error:.4f} pixels")

    return {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'rms_error': mean_error,
        'num_images': len(images_data)
    }


def save_calibration(calib_data, prefix, timestamp=None):
    """Save calibration data to files."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    CALIBRATION_DIR.mkdir(exist_ok=True)

    # Save camera matrix
    np.savetxt(CALIBRATION_DIR / f"{prefix}_camera_matrix_{timestamp}.csv",
               calib_data['camera_matrix'], delimiter=',')

    # Save distortion coefficients
    np.savetxt(CALIBRATION_DIR / f"{prefix}_distortion_{timestamp}.csv",
               calib_data['dist_coeffs'], delimiter=',')

    # Save RMS error
    np.savetxt(CALIBRATION_DIR / f"{prefix}_RMSE_{timestamp}.csv",
               [calib_data['rms_error']], delimiter=',')

    # Save rotation and translation vectors
    rvecs_array = np.array([r.flatten() for r in calib_data['rvecs']])
    tvecs_array = np.array([t.flatten() for t in calib_data['tvecs']])

    np.savetxt(CALIBRATION_DIR / f"{prefix}_rotation_vectors_{timestamp}.csv",
               rvecs_array, delimiter=',')
    np.savetxt(CALIBRATION_DIR / f"{prefix}_translation_vectors_{timestamp}.csv",
               tvecs_array, delimiter=',')

    # Save summary
    with open(CALIBRATION_DIR / f"{prefix}_summary_{timestamp}.txt", 'w') as f:
        f.write(f"Camera Calibration Summary\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Number of images: {calib_data['num_images']}\n")
        f.write(f"RMS reprojection error: {calib_data['rms_error']:.4f} pixels\n\n")
        f.write(f"Camera Matrix:\n{calib_data['camera_matrix']}\n\n")
        f.write(f"Distortion Coefficients:\n{calib_data['dist_coeffs']}\n")

    print(f"\n[OK] Saved calibration to: {CALIBRATION_DIR}/{prefix}_*_{timestamp}.*")


def calibrate_stereo_pair(left_images, right_images, image_size,
                          left_calib, right_calib):
    """
    Perform stereo calibration.

    Args:
        left_images: List of left camera image data
        right_images: List of right camera image data
        image_size: (width, height)
        left_calib: Left camera calibration dict
        right_calib: Right camera calibration dict

    Returns:
        Dict with stereo calibration results
    """
    print(f"\nPerforming stereo calibration with {len(left_images)} image pairs...")

    # Prepare data
    objpoints = [img['objpoints'] for img in left_images]
    imgpoints_left = [img['corners'] for img in left_images]
    imgpoints_right = [img['corners'] for img in right_images]

    # Stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC  # Use individual calibrations

    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        left_calib['camera_matrix'], left_calib['dist_coeffs'],
        right_calib['camera_matrix'], right_calib['dist_coeffs'],
        image_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=flags
    )

    if not ret:
        print("Stereo calibration failed!")
        return None

    print(f"[OK] Stereo calibration successful! RMS error: {ret:.4f}")

    # Stereo rectification
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T,
        alpha=0,  # 0 = crop to valid pixels only
        newImageSize=image_size
    )

    # Compute rectification maps
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1, image_size, cv2.CV_32FC1
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        K2, D2, R2, P2, image_size, cv2.CV_32FC1
    )

    print("[OK] Stereo rectification computed")

    return {
        'K1': K1, 'D1': D1,
        'K2': K2, 'D2': D2,
        'R': R, 'T': T, 'E': E, 'F': F,
        'R1': R1, 'R2': R2,
        'P1': P1, 'P2': P2, 'Q': Q,
        'left_map1': left_map1, 'left_map2': left_map2,
        'right_map1': right_map1, 'right_map2': right_map2,
        'roi_left': roi_left, 'roi_right': roi_right,
        'rms_error': ret,
        'num_pairs': len(objpoints)
    }


def save_stereo_calibration(stereo_calib, timestamp=None):
    """Save stereo calibration data."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    CALIBRATION_DIR.mkdir(exist_ok=True)

    # Save matrices
    np.savetxt(CALIBRATION_DIR / f"stereo_R_{timestamp}.csv", stereo_calib['R'], delimiter=',')
    np.savetxt(CALIBRATION_DIR / f"stereo_T_{timestamp}.csv", stereo_calib['T'], delimiter=',')
    np.savetxt(CALIBRATION_DIR / f"stereo_E_{timestamp}.csv", stereo_calib['E'], delimiter=',')
    np.savetxt(CALIBRATION_DIR / f"stereo_F_{timestamp}.csv", stereo_calib['F'], delimiter=',')
    np.savetxt(CALIBRATION_DIR / f"stereo_R1_{timestamp}.csv", stereo_calib['R1'], delimiter=',')
    np.savetxt(CALIBRATION_DIR / f"stereo_R2_{timestamp}.csv", stereo_calib['R2'], delimiter=',')
    np.savetxt(CALIBRATION_DIR / f"stereo_P1_{timestamp}.csv", stereo_calib['P1'], delimiter=',')
    np.savetxt(CALIBRATION_DIR / f"stereo_P2_{timestamp}.csv", stereo_calib['P2'], delimiter=',')
    np.savetxt(CALIBRATION_DIR / f"stereo_Q_{timestamp}.csv", stereo_calib['Q'], delimiter=',')
    np.savetxt(CALIBRATION_DIR / f"stereo_RMS_{timestamp}.csv", [stereo_calib['rms_error']], delimiter=',')

    # Save rectification maps (binary for speed)
    np.save(CALIBRATION_DIR / f"stereo_left_map1_{timestamp}.npy", stereo_calib['left_map1'])
    np.save(CALIBRATION_DIR / f"stereo_left_map2_{timestamp}.npy", stereo_calib['left_map2'])
    np.save(CALIBRATION_DIR / f"stereo_right_map1_{timestamp}.npy", stereo_calib['right_map1'])
    np.save(CALIBRATION_DIR / f"stereo_right_map2_{timestamp}.npy", stereo_calib['right_map2'])

    # Save summary
    baseline = np.linalg.norm(stereo_calib['T'])
    with open(CALIBRATION_DIR / f"stereo_calibration_summary_{timestamp}.txt", 'w') as f:
        f.write(f"Stereo Calibration Summary\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Number of image pairs: {stereo_calib['num_pairs']}\n")
        f.write(f"RMS error: {stereo_calib['rms_error']:.4f}\n")
        f.write(f"Baseline (camera separation): {baseline:.2f} mm\n\n")
        f.write(f"Rotation Matrix (R):\n{stereo_calib['R']}\n\n")
        f.write(f"Translation Vector (T):\n{stereo_calib['T']}\n\n")
        f.write(f"Projection Matrix P1:\n{stereo_calib['P1']}\n\n")
        f.write(f"Projection Matrix P2:\n{stereo_calib['P2']}\n")

    print(f"\n[OK] Saved stereo calibration to: {CALIBRATION_DIR}/stereo_*_{timestamp}.*")
    print(f"  Baseline: {baseline:.2f} mm")
    print(f"  RMS error: {stereo_calib['rms_error']:.4f}")


def calibrate_individual_cameras():
    """Calibrate both cameras individually."""
    print("=" * 60)
    print("INDIVIDUAL CAMERA CALIBRATION")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Left camera
    print("\n[1/2] LEFT CAMERA")
    left_images = collect_calibration_images(CAMERA_INDEX, num_images=20, is_stereo=False, camera_side='left')
    if not left_images:
        return None

    image_size = (1280, 720)
    left_calib = calibrate_camera(left_images, image_size)
    if not left_calib:
        return None

    save_calibration(left_calib, "left_camera", timestamp)

    input("\nPress ENTER to continue to right camera...")

    # Right camera
    print("\n[2/2] RIGHT CAMERA")
    right_images = collect_calibration_images(CAMERA_INDEX, num_images=20, is_stereo=False, camera_side='right')
    if not right_images:
        return None

    right_calib = calibrate_camera(right_images, image_size)
    if not right_calib:
        return None

    save_calibration(right_calib, "right_camera", timestamp)

    print("\n" + "=" * 60)
    print("[OK] Individual camera calibration complete!")
    print("=" * 60)
    print(f"\nNext step: Run stereo calibration")
    print(f"  python -m ball_detection.stereo_calibration --calibrate-stereo")

    return timestamp


def calibrate_stereo():
    """Calibrate stereo pair."""
    print("=" * 60)
    print("STEREO CALIBRATION")
    print("=" * 60)

    # Load latest individual calibrations
    left_files = sorted(CALIBRATION_DIR.glob("left_camera_camera_matrix_*.csv"), reverse=True)
    right_files = sorted(CALIBRATION_DIR.glob("right_camera_camera_matrix_*.csv"), reverse=True)

    if not left_files or not right_files:
        print("\nError: No individual camera calibrations found!")
        print("Run individual calibration first:")
        print(f"  python -m ball_detection.stereo_calibration --calibrate-individual")
        return

    # Get timestamps
    left_timestamp = left_files[0].name.replace('left_camera_camera_matrix_', '').replace('.csv', '')
    right_timestamp = right_files[0].name.replace('right_camera_camera_matrix_', '').replace('.csv', '')

    # Load calibrations
    left_calib = {
        'camera_matrix': np.loadtxt(CALIBRATION_DIR / f"left_camera_camera_matrix_{left_timestamp}.csv", delimiter=','),
        'dist_coeffs': np.loadtxt(CALIBRATION_DIR / f"left_camera_distortion_{left_timestamp}.csv", delimiter=',')
    }
    right_calib = {
        'camera_matrix': np.loadtxt(CALIBRATION_DIR / f"right_camera_camera_matrix_{right_timestamp}.csv", delimiter=','),
        'dist_coeffs': np.loadtxt(CALIBRATION_DIR / f"right_camera_distortion_{right_timestamp}.csv", delimiter=',')
    }

    print(f"\n[OK] Loaded left camera calibration: {left_timestamp}")
    print(f"[OK] Loaded right camera calibration: {right_timestamp}")

    # Collect stereo image pairs
    print("\n\nCollecting stereo image pairs...")
    print("IMPORTANT: Show the SAME chessboard view to BOTH cameras simultaneously!")

    images = collect_calibration_images(CAMERA_INDEX, num_images=20, is_stereo=True)
    if not images:
        return

    # Split into left and right
    left_images = []
    right_images = []

    for img_data in images:
        full_image = img_data['image']
        left_gray = full_image[:, 0:1280]
        right_gray = full_image[:, 1280:2560]

        # Find corners in both halves
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        ret_left, corners_left = cv2.findChessboardCorners(left_gray, CHESSBOARD_SIZE, None)
        ret_right, corners_right = cv2.findChessboardCorners(right_gray, CHESSBOARD_SIZE, None)

        if ret_left and ret_right:
            corners_left = cv2.cornerSubPix(left_gray, corners_left, (11, 11), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(right_gray, corners_right, (11, 11), (-1, -1), criteria)

            left_images.append({
                'image': left_gray,
                'corners': corners_left,
                'objpoints': img_data['objpoints']
            })
            right_images.append({
                'image': right_gray,
                'corners': corners_right,
                'objpoints': img_data['objpoints']
            })

    print(f"\n[OK] Found {len(left_images)} valid stereo pairs")

    if len(left_images) < 10:
        print("Error: Need at least 10 valid stereo pairs")
        return

    # Perform stereo calibration
    image_size = (1280, 720)
    stereo_calib = calibrate_stereo_pair(left_images, right_images, image_size,
                                         left_calib, right_calib)
    if not stereo_calib:
        return

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_stereo_calibration(stereo_calib, timestamp)

    print("\n" + "=" * 60)
    print("[OK] Stereo calibration complete!")
    print("=" * 60)
    print(f"\nYou can now use the stereo tracker:")
    print(f"  python -m ball_detection.stereo_tracker")


def main():
    print("=" * 60)
    print("STEREO CAMERA CALIBRATION")
    print("=" * 60)
    print(f"Mode: {MODE}")
    print(f"Camera: {CAMERA_INDEX}")
    print(f"Chessboard: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]}")
    print(f"Square size: {SQUARE_SIZE} mm")
    print("=" * 60)

    if MODE == "individual":
        print("\nCalibrating individual cameras...")
        calibrate_individual_cameras()
    elif MODE == "stereo":
        print("\nCalibrating stereo pair...")
        calibrate_stereo()
    elif MODE == "both":
        print("\nCalibrating individual cameras first...")
        calibrate_individual_cameras()
        print("\n\nNow calibrating stereo pair...")
        calibrate_stereo()
    else:
        print(f"\nError: Invalid MODE '{MODE}'")
        print("\nValid modes:")
        print("  'individual' - Calibrate both cameras separately")
        print("  'stereo' - Calibrate stereo pair (requires individual calibrations)")
        print("  'both' - Do both steps sequentially")
        print("\nEdit MODE setting at the top of this file.")
        return

    print("\n" + "=" * 60)
    print("CALIBRATION WORKFLOW COMPLETE")
    print("=" * 60)
    print("\nNext step: Run 3D tracking")
    print("  python -m ball_detection.stereo_tracker")
    print("=" * 60)


if __name__ == "__main__":
    main()
