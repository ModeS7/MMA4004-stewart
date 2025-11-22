"""
Coordinate Transformation Utilities

Transforms 3D points from camera reference frame to platform reference frame.
Uses calibrated rotation and translation matrices.
"""

import numpy as np
from pathlib import Path
import glob


def load_platform_transform(calib_dir="ball_detection/calibrations"):
    """
    Load latest platform transformation matrix.

    Args:
        calib_dir: Directory containing calibration files

    Returns:
        dict with keys:
            'R': 3x3 rotation matrix (camera → platform)
            'T': 3x1 translation vector (camera → platform)
            'timestamp': Calibration timestamp
            'transform_4x4': 4x4 homogeneous transformation matrix

    Raises:
        FileNotFoundError: If no platform calibration found
    """
    calib_path = Path(calib_dir)

    # Find latest platform transformation file
    transform_files = sorted(calib_path.glob('platform_transform_*.csv'), reverse=True)

    if not transform_files:
        raise FileNotFoundError(
            f"No platform transformation found in {calib_dir}\n"
            f"Run: python -m ball_detection.calibration.platform_frame_calibration"
        )

    transform_file = transform_files[0]
    timestamp = transform_file.stem.replace('platform_transform_', '')

    # Load 4x4 homogeneous transformation matrix
    transform_4x4 = np.loadtxt(transform_file, delimiter=',')

    # Extract R (3x3) and T (3x1)
    R = transform_4x4[:3, :3]
    T = transform_4x4[:3, 3]

    return {
        'R': R,
        'T': T,
        'timestamp': timestamp,
        'transform_4x4': transform_4x4
    }


def apply_platform_transform(point_camera, R, T):
    """
    Transform single 3D point from camera frame to platform frame.

    Args:
        point_camera: 3D point in camera coordinates [x, y, z] (numpy array or list)
        R: 3x3 rotation matrix
        T: 3x1 translation vector

    Returns:
        point_platform: 3D point in platform coordinates [x, y, z] (numpy array)

    Example:
        >>> R, T = load_platform_transform()['R'], load_platform_transform()['T']
        >>> point_cam = [12.3, 45.6, 789.0]
        >>> point_plat = apply_platform_transform(point_cam, R, T)
        >>> print(point_plat)  # [x, y, z] in platform frame
    """
    point_camera = np.array(point_camera, dtype=np.float64)

    if point_camera.shape != (3,):
        raise ValueError(f"Point must be shape (3,), got {point_camera.shape}")

    # Apply transformation: P_platform = R @ P_camera + T
    point_platform = R @ point_camera + T

    return point_platform


def transform_batch(points_camera, R, T):
    """
    Transform multiple 3D points from camera frame to platform frame (vectorized).

    Args:
        points_camera: Nx3 array of 3D points in camera coordinates
        R: 3x3 rotation matrix
        T: 3x1 translation vector

    Returns:
        points_platform: Nx3 array of 3D points in platform coordinates

    Example:
        >>> R, T = load_platform_transform()['R'], load_platform_transform()['T']
        >>> points_cam = np.array([[1,2,3], [4,5,6], [7,8,9]])
        >>> points_plat = transform_batch(points_cam, R, T)
    """
    points_camera = np.array(points_camera, dtype=np.float64)

    if points_camera.ndim != 2 or points_camera.shape[1] != 3:
        raise ValueError(f"Points must be shape (N, 3), got {points_camera.shape}")

    # Vectorized transformation: (N, 3) = (N, 3) @ (3, 3).T + (1, 3)
    points_platform = (points_camera @ R.T) + T

    return points_platform


def inverse_transform(point_platform, R, T):
    """
    Transform point from platform frame back to camera frame (inverse).

    Useful for validation and debugging.

    Args:
        point_platform: 3D point in platform coordinates [x, y, z]
        R: 3x3 rotation matrix (camera → platform)
        T: 3x1 translation vector (camera → platform)

    Returns:
        point_camera: 3D point in camera coordinates [x, y, z]

    Math:
        P_camera = R.T @ (P_platform - T)
    """
    point_platform = np.array(point_platform, dtype=np.float64)

    if point_platform.shape != (3,):
        raise ValueError(f"Point must be shape (3,), got {point_platform.shape}")

    # Inverse: P_camera = R^T @ (P_platform - T)
    point_camera = R.T @ (point_platform - T)

    return point_camera


def compute_transformation_from_points(points_camera, points_platform):
    """
    Compute rigid transformation (R, T) from corresponding 3D point pairs.

    Uses SVD-based method (Kabsch algorithm) to find optimal rigid transformation.

    Args:
        points_camera: Nx3 array of points in camera frame
        points_platform: Nx3 array of corresponding points in platform frame

    Returns:
        R: 3x3 rotation matrix
        T: 3x1 translation vector
        rmse: Root mean square error of transformation

    Example:
        >>> # Checkerboard corners in both frames
        >>> camera_pts = np.array([[...], [...], ...])
        >>> platform_pts = np.array([[0, 0, 0], [25, 0, 0], ...])
        >>> R, T, rmse = compute_transformation_from_points(camera_pts, platform_pts)
    """
    points_camera = np.array(points_camera, dtype=np.float64)
    points_platform = np.array(points_platform, dtype=np.float64)

    if points_camera.shape != points_platform.shape:
        raise ValueError("Point arrays must have same shape")

    if points_camera.shape[0] < 3:
        raise ValueError("Need at least 3 point pairs")

    # Compute centroids
    centroid_camera = np.mean(points_camera, axis=0)
    centroid_platform = np.mean(points_platform, axis=0)

    # Center the point sets
    centered_camera = points_camera - centroid_camera
    centered_platform = points_platform - centroid_platform

    # Compute covariance matrix
    H = centered_camera.T @ centered_platform

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation
    R = Vt.T @ U.T

    # Handle reflection case (det(R) = -1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    T = centroid_platform - R @ centroid_camera

    # Compute RMSE
    transformed = (points_camera @ R.T) + T
    errors = np.linalg.norm(transformed - points_platform, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))

    return R, T, rmse


def save_platform_transform(R, T, calib_dir="ball_detection/calibrations", timestamp=None):
    """
    Save platform transformation matrix to file.

    Args:
        R: 3x3 rotation matrix
        T: 3x1 translation vector
        calib_dir: Directory to save calibration
        timestamp: Optional timestamp string (generated if None)

    Returns:
        filepath: Path to saved file
    """
    from datetime import datetime

    calib_path = Path(calib_dir)
    calib_path.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create 4x4 homogeneous transformation matrix
    transform_4x4 = np.eye(4, dtype=np.float64)
    transform_4x4[:3, :3] = R
    transform_4x4[:3, 3] = T

    # Save to CSV
    filepath = calib_path / f"platform_transform_{timestamp}.csv"
    np.savetxt(filepath, transform_4x4, delimiter=',', fmt='%.10f')

    return filepath


if __name__ == "__main__":
    # Test / example usage
    print("Coordinate Transform Utilities Test")
    print("=" * 60)

    # Example: Create sample transformation
    R_example = np.array([
        [0.9, -0.1, 0.0],
        [0.1, 0.9, 0.0],
        [0.0, 0.0, 1.0]
    ])
    T_example = np.array([10.0, 20.0, 30.0])

    # Test single point
    point_camera = np.array([100.0, 200.0, 500.0])
    point_platform = apply_platform_transform(point_camera, R_example, T_example)

    print(f"Camera point: {point_camera}")
    print(f"Platform point: {point_platform}")

    # Test inverse
    point_back = inverse_transform(point_platform, R_example, T_example)
    print(f"Back to camera: {point_back}")
    print(f"Error: {np.linalg.norm(point_camera - point_back):.10f} mm")

    # Test batch
    points_camera = np.array([
        [100, 200, 500],
        [150, 250, 550],
        [200, 300, 600]
    ])
    points_platform = transform_batch(points_camera, R_example, T_example)
    print(f"\nBatch transform: {points_camera.shape[0]} points")
    print(f"First point: {points_camera[0]} → {points_platform[0]}")

    print("\n" + "=" * 60)
    print("Test complete!")
