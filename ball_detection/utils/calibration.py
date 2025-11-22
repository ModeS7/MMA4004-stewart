"""
Stereo Calibration Utilities

Common functions for loading and saving stereo calibration data.
Reduces code duplication across stereo detection applications.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional


def load_stereo_calibration(calib_dir: Path) -> Optional[Dict]:
    """
    Load latest stereo calibration data.

    Args:
        calib_dir: Path to calibration directory

    Returns:
        Dictionary containing:
            - 'P1': 3x4 projection matrix for left camera
            - 'P2': 3x4 projection matrix for right camera
            - 'left_map1', 'left_map2': Rectification maps for left camera
            - 'right_map1', 'right_map2': Rectification maps for right camera
            - 'timestamp': Calibration timestamp string
        Returns None if calibration not found or error occurs

    Example:
        >>> calib_dir = Path("ball_detection/calibration/calibrations")
        >>> calib = load_stereo_calibration(calib_dir)
        >>> if calib:
        >>>     P1, P2 = calib['P1'], calib['P2']
    """
    if not calib_dir.exists():
        print(f"\nError: Calibration directory not found: {calib_dir}")
        print("\nYou need to run stereo calibration first:")
        print("  python -m ball_detection.calibration.stereo_calibration --calibrate-individual")
        print("  python -m ball_detection.calibration.stereo_calibration --calibrate-stereo")
        return None

    # Find latest stereo calibration files
    p1_files = sorted(calib_dir.glob('stereo_P1_*.csv'), reverse=True)
    p2_files = sorted(calib_dir.glob('stereo_P2_*.csv'), reverse=True)
    map_files = sorted(calib_dir.glob('stereo_left_map1_*.npy'), reverse=True)

    if not p1_files or not p2_files or not map_files:
        print(f"\nError: No stereo calibration files found in: {calib_dir}")
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


def load_rectification_maps_only(calib_dir: Path) -> Optional[Dict]:
    """
    Load only rectification maps (lighter weight than full calibration).

    Useful for scripts that only need to rectify images without triangulation.

    Args:
        calib_dir: Path to calibration directory

    Returns:
        Dictionary containing:
            - 'left_map1', 'left_map2': Rectification maps for left camera
            - 'right_map1', 'right_map2': Rectification maps for right camera
            - 'timestamp': Calibration timestamp string
        Returns None if maps not found

    Example:
        >>> calib_dir = Path("ball_detection/calibration/calibrations")
        >>> maps = load_rectification_maps_only(calib_dir)
        >>> if maps:
        >>>     left_rect = cv2.remap(left, maps['left_map1'], maps['left_map2'], cv2.INTER_LINEAR)
    """
    if not calib_dir.exists():
        print(f"\nError: Calibration directory not found: {calib_dir}")
        return None

    # Find latest rectification map files
    map_files = sorted(calib_dir.glob('stereo_left_map1_*.npy'), reverse=True)

    if not map_files:
        print(f"\nError: No rectification maps found in: {calib_dir}")
        return None

    timestamp = map_files[0].name.replace('stereo_left_map1_', '').replace('.npy', '')

    try:
        # Load rectification maps
        left_map1 = np.load(calib_dir / f'stereo_left_map1_{timestamp}.npy')
        left_map2 = np.load(calib_dir / f'stereo_left_map2_{timestamp}.npy')
        right_map1 = np.load(calib_dir / f'stereo_right_map1_{timestamp}.npy')
        right_map2 = np.load(calib_dir / f'stereo_right_map2_{timestamp}.npy')

        return {
            'left_map1': left_map1,
            'left_map2': left_map2,
            'right_map1': right_map1,
            'right_map2': right_map2,
            'timestamp': timestamp
        }

    except Exception as e:
        print(f"\nError loading rectification maps from {calib_dir}: {e}")
        return None
