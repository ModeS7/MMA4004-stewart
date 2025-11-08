#!/usr/bin/env python3
"""
Test platform workspace limits to compare different configurations.

Finds maximum achievable positions and rotations within servo angle limits.
Outputs results in table format for documentation.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.core import StewartPlatformIK
from core.utils import IKZOptimizationConfig


def test_position_limit(ik, axis, direction, max_servo_angle, use_top_surface_offset=True, use_z_optimization=False):
    """
    Find maximum position along an axis in given direction.

    Args:
        ik: StewartPlatformIK instance
        axis: 'x', 'y', or 'z'
        direction: +1 for positive, -1 for negative
        max_servo_angle: Maximum allowed servo angle (deg)
        use_top_surface_offset: Use top surface as reference
        use_z_optimization: Use dynamic Z optimization to extend workspace

    Returns:
        Maximum achievable position (mm) or None if failed
    """
    home_z = ik.home_height_top_surface if use_top_surface_offset else ik.home_height

    # Starting position
    translation = np.array([0.0, 0.0, home_z])
    rotation = np.array([0.0, 0.0, 0.0])

    # Binary search for maximum
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]

    # For Z axis, test relative to home position (no Z optimization for Z itself)
    if axis == 'z':
        low = 0.0
        high = 100.0
        best_valid = 0.0  # At least home position works

        for _ in range(30):
            mid = (low + high) / 2.0
            test_trans = translation.copy()
            test_trans[axis_idx] = home_z + direction * mid

            angles = ik.calculate_servo_angles(test_trans, rotation, use_top_surface_offset)

            if angles is not None and np.max(np.abs(angles)) <= max_servo_angle:
                best_valid = direction * mid
                low = mid
            else:
                high = mid

        return best_valid
    else:
        # For X and Y, test absolute positions
        low = 0.0
        high = 200.0
        best_valid = None

        for _ in range(30):
            mid = (low + high) / 2.0
            test_trans = translation.copy()
            test_trans[axis_idx] = direction * mid

            if use_z_optimization:
                # Use Z optimization to find best Z for this X/Y position
                opt_trans, angles, success = ik.optimize_z_offset(
                    test_trans, rotation,
                    use_top_surface_offset=use_top_surface_offset,
                    z_search_range=IKZOptimizationConfig.Z_SEARCH_RANGE_MM
                )
                if success and angles is not None:
                    angles_valid = np.max(np.abs(angles)) <= max_servo_angle
                else:
                    angles_valid = False
            else:
                angles = ik.calculate_servo_angles(test_trans, rotation, use_top_surface_offset)
                angles_valid = angles is not None and np.max(np.abs(angles)) <= max_servo_angle

            if angles_valid:
                best_valid = direction * mid
                low = mid
            else:
                high = mid

        return best_valid


def test_rotation_limit(ik, axis, direction, max_servo_angle, use_top_surface_offset=True, use_z_optimization=False):
    """
    Find maximum rotation around an axis in given direction.

    Args:
        ik: StewartPlatformIK instance
        axis: 'rx', 'ry', or 'rz' (roll, pitch, yaw)
        direction: +1 for positive, -1 for negative
        max_servo_angle: Maximum allowed servo angle (deg)
        use_top_surface_offset: Use top surface as reference
        use_z_optimization: Use dynamic Z optimization to extend workspace

    Returns:
        Maximum achievable rotation (degrees) or None if failed
    """
    home_z = ik.home_height_top_surface if use_top_surface_offset else ik.home_height

    translation = np.array([0.0, 0.0, home_z])
    rotation = np.array([0.0, 0.0, 0.0])

    # Binary search for maximum rotation
    axis_idx = {'rx': 0, 'ry': 1, 'rz': 2}[axis]
    low = 0.0
    high = 90.0  # Reasonable search range for rotations

    best_valid = None

    for _ in range(30):
        mid = (low + high) / 2.0
        test_rot = rotation.copy()
        test_rot[axis_idx] = direction * mid

        if use_z_optimization:
            # Use Z optimization to find best Z for this rotation
            opt_trans, angles, success = ik.optimize_z_offset(
                translation, test_rot,
                use_top_surface_offset=use_top_surface_offset,
                z_search_range=IKZOptimizationConfig.Z_SEARCH_RANGE_MM
            )
            if success and angles is not None:
                angles_valid = np.max(np.abs(angles)) <= max_servo_angle
            else:
                angles_valid = False
        else:
            angles = ik.calculate_servo_angles(translation, test_rot, use_top_surface_offset)
            angles_valid = angles is not None and np.max(np.abs(angles)) <= max_servo_angle

        if angles_valid:
            best_valid = direction * mid
            low = mid
        else:
            high = mid

    return best_valid


def test_combined_rotation_limit(ik, max_servo_angle, use_top_surface_offset=True):
    """
    Find maximum combined roll+pitch rotation (equal amounts).

    Returns:
        Maximum achievable angle where rx=ry=result
    """
    home_z = ik.home_height_top_surface if use_top_surface_offset else ik.home_height

    translation = np.array([0.0, 0.0, home_z])

    low = 0.0
    high = 60.0

    best_valid = None

    for _ in range(30):
        mid = (low + high) / 2.0
        test_rot = np.array([mid, mid, 0.0])

        angles = ik.calculate_servo_angles(translation, test_rot, use_top_surface_offset)

        if angles is not None and np.max(np.abs(angles)) <= max_servo_angle:
            best_valid = mid
            low = mid
        else:
            high = mid

    return best_valid


def test_platform_limits(platform_params, name, max_servo_angle, use_top_surface_offset=True, use_z_optimization=False):
    """
    Test all workspace limits for a platform configuration.

    Args:
        platform_params: Dict of platform parameters
        name: Name/version of platform (e.g., "V1", "V2")
        max_servo_angle: Maximum servo angle limit (e.g., 40, 70)
        use_top_surface_offset: Use top surface as reference
        use_z_optimization: Use dynamic Z optimization to extend workspace

    Returns:
        Dict with all tested limits
    """
    ik = StewartPlatformIK(**platform_params)

    z_opt_label = " [WITH Z OPTIMIZATION]" if use_z_optimization else ""
    print(f"\nTesting {name} with alpha={max_servo_angle} deg{z_opt_label}")
    print("=" * 70)

    results = {
        'name': name,
        'alpha': max_servo_angle,
        'z_opt': use_z_optimization,
    }

    # Test position limits
    print("\nPosition limits:")
    for axis in ['x', 'y', 'z']:
        pos_max = test_position_limit(ik, axis, +1, max_servo_angle, use_top_surface_offset, use_z_optimization)
        neg_max = test_position_limit(ik, axis, -1, max_servo_angle, use_top_surface_offset, use_z_optimization)

        results[f'{axis}_pos'] = pos_max
        results[f'{axis}_neg'] = neg_max

        print(f"  {axis}: [{neg_max:+6.1f}, {pos_max:+6.1f}] mm" if pos_max and neg_max else f"  {axis}: FAILED")

    # Test rotation limits
    print("\nRotation limits:")
    for axis, name_full in [('rx', 'Roll (phi)'), ('ry', 'Pitch (theta)'), ('rz', 'Yaw (psi)')]:
        pos_max = test_rotation_limit(ik, axis, +1, max_servo_angle, use_top_surface_offset, use_z_optimization)
        neg_max = test_rotation_limit(ik, axis, -1, max_servo_angle, use_top_surface_offset, use_z_optimization)

        results[f'{axis}_pos'] = pos_max
        results[f'{axis}_neg'] = neg_max

        print(f"  {name_full:13s}: [{neg_max:+6.1f}, {pos_max:+6.1f}] deg" if pos_max and neg_max else f"  {name_full}: FAILED")

    # Test combined rotation
    combined = test_combined_rotation_limit(ik, max_servo_angle, use_top_surface_offset)
    results['rx_ry_combined'] = combined
    print(f"  Combined rx=ry: {combined:+6.1f} deg" if combined else "  Combined: FAILED")

    return results


def print_latex_table_row(results):
    """Print LaTeX table rows (2 rows per config: positive then negative)."""
    name = results['name']
    alpha = results['alpha']

    # Get both positive and negative limits
    x_pos = results.get('x_pos', 0) or 0
    x_neg = -(abs(results.get('x_neg', 0) or 0))
    y_pos = results.get('y_pos', 0) or 0
    y_neg = -(abs(results.get('y_neg', 0) or 0))
    z_pos = results.get('z_pos', 0) or 0
    z_neg = -(abs(results.get('z_neg', 0) or 0))
    rx_pos = results.get('rx_pos', 0) or 0
    rx_neg = -(abs(results.get('rx_neg', 0) or 0))
    ry_pos = results.get('ry_pos', 0) or 0
    ry_neg = -(abs(results.get('ry_neg', 0) or 0))
    rz_pos = results.get('rz_pos', 0) or 0
    rz_neg = -(abs(results.get('rz_neg', 0) or 0))

    # Format: 2 rows per configuration
    # Positive row
    print(f"{name} & {alpha} & "
          f"{x_pos:.0f} & {y_pos:.0f} & {z_pos:.0f} & "
          f"{rx_pos:.0f} & {ry_pos:.0f} & {rz_pos:.0f} \\\\")
    # Negative row
    print(f" & & "
          f"{x_neg:.0f} & {y_neg:.0f} & {z_neg:.0f} & "
          f"{rx_neg:.0f} & {ry_neg:.0f} & {rz_neg:.0f} \\\\")


def main():
    """Run workspace limit tests for different platform configurations."""

    # Version 1 parameters (your original platform)
    v1_params = {
        "horn_length": 31.75,
        "rod_length": 145.0,
        "base": 73.025,
        "base_anchors": 36.8893,
        "platform": 67.775,
        "platform_anchors": 12.7,
        "top_surface_offset": 26.0
    }


    # Version 2 parameters (example - modify as needed)
    # These are placeholder values - replace with your actual V2 parameters
    v2_params = {
        "horn_length": 45.3722,
        "rod_length": 205.0,
        "base": 86.6025 + 18.75 + 11,
        "base_anchors": 64.75,
        "platform": 84.0759,
        "platform_anchors": 12.5,
        "top_surface_offset": 38.0
    }

    print("Stewart Platform Workspace Limit Analysis")
    print("=" * 70)
    print("\nTesting workspace limits for different configurations.")
    print("All positions in mm, rotations in degrees.")

    # Store all results
    all_results = []

    # Test V1 with 40° limit
    results = test_platform_limits(v1_params, "V1", 40)
    all_results.append(results)

    # Test V2 with 40° limit
    results = test_platform_limits(v2_params, "V2", 40)
    all_results.append(results)

    # Test V2 with 70° limit and Z optimization
    results = test_platform_limits(v2_params, "V2", 70, use_z_optimization=True)
    all_results.append(results)

    # Print summary table (2 rows per config: positive then negative)
    print("\n" + "=" * 80)
    print("\nSUMMARY TABLE (2 rows per configuration)")
    print("-" * 80)
    print(f"{'Config':<8} {'alpha':>5}     {'x':>6} {'y':>6} {'z':>6}   {'phi':>6} {'theta':>6} {'psi':>6}")
    print(f"{'':8} {'(deg)':>5}     {'(mm)':>6} {'(mm)':>6} {'(mm)':>6}   {'(deg)':>6} {'(deg)':>6} {'(deg)':>6}")
    print("-" * 80)

    for r in all_results:
        x_pos = r.get('x_pos', 0) or 0
        x_neg = -(abs(r.get('x_neg', 0) or 0))
        y_pos = r.get('y_pos', 0) or 0
        y_neg = -(abs(r.get('y_neg', 0) or 0))
        z_pos = r.get('z_pos', 0) or 0
        z_neg = -(abs(r.get('z_neg', 0) or 0))
        rx_pos = r.get('rx_pos', 0) or 0
        rx_neg = -(abs(r.get('rx_neg', 0) or 0))
        ry_pos = r.get('ry_pos', 0) or 0
        ry_neg = -(abs(r.get('ry_neg', 0) or 0))
        rz_pos = r.get('rz_pos', 0) or 0
        rz_neg = -(abs(r.get('rz_neg', 0) or 0))

        # Add marker if Z optimization was used
        name_display = r['name'] + '*' if r.get('z_opt', False) else r['name']

        # Positive row
        print(f"{name_display:<8} {r['alpha']:>5}     "
              f"{x_pos:>6.0f} {y_pos:>6.0f} {z_pos:>6.0f}   "
              f"{rx_pos:>6.0f} {ry_pos:>6.0f} {rz_pos:>6.0f}")
        # Negative row
        print(f"{'':<8} {'':>5}     "
              f"{x_neg:>6.0f} {y_neg:>6.0f} {z_neg:>6.0f}   "
              f"{rx_neg:>6.0f} {ry_neg:>6.0f} {rz_neg:>6.0f}")

    # Print LaTeX table rows
    print("\n" + "=" * 70)
    print("\nLaTeX Table Rows:")
    print("-" * 70)
    print("\\hline")
    for r in all_results:
        print_latex_table_row(r)
        print("\\hline")

    print("\n" + "=" * 80)
    print("\nNotes:")
    print("- Each configuration shows 2 rows: positive limits, then negative limits")
    print("- Negative values shown with minus sign (e.g., -43 means -43mm in that direction)")
    print("- * indicates Z optimization was used to extend workspace")
    print("Test completed successfully!")


if __name__ == "__main__":
    main()
