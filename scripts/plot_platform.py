#!/usr/bin/env python3
"""
Stewart Platform Top Surface Visualization

Plots the top surface of the Stewart platform showing the platform geometry,
anchor points, and coordinate system.

Usage:
    python plot_platform.py
    python plot_platform.py --tilt-x 5 --tilt-y 3  # Show tilted platform
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path

# Add parent directory to path to import core modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.core import StewartPlatformIK


def plot_platform_top_surface(tilt_x=0.0, tilt_y=0.0, output_dir='plots'):
    """
    Plot the top surface of the Stewart platform.

    Args:
        tilt_x: Tilt around X-axis in degrees (roll)
        tilt_y: Tilt around Y-axis in degrees (pitch)
        output_dir: Directory to save plot
    """
    # Initialize platform with default geometry
    platform = StewartPlatformIK()

    print(f"Platform Geometry:")
    print(f"  Platform radius: {platform.platform:.2f} mm")
    print(f"  Platform anchor offset: {platform.platform_anchors_dist:.2f} mm" if hasattr(platform, 'platform_anchors_dist') else "")
    print(f"  Base radius: {platform.base:.2f} mm")
    print(f"  Home height: {platform.home_height:.2f} mm")
    print(f"  Top surface height: {platform.home_height_top_surface:.2f} mm")
    print(f"  Top surface offset: {platform.top_surface_offset:.2f} mm")

    # Get platform anchor points in home position
    platform_anchors = platform.platform_anchors

    # Calculate platform position with requested tilt
    translation = np.array([0.0, 0.0, platform.home_height_top_surface])
    rotation = np.array([tilt_x, tilt_y, 0.0])

    # Transform platform anchors to world coordinates
    quat = platform._euler_to_quaternion(np.radians(rotation))

    # Get anchor points in world frame
    anchor_points_world = []
    for anchor in platform_anchors:
        # Offset anchor down by top_surface_offset to get actual platform frame
        anchor_platform_frame = anchor.copy()
        anchor_platform_frame[2] = -platform.top_surface_offset

        # Rotate and translate to world frame
        anchor_rotated = platform._rotate_vector(anchor_platform_frame, quat)
        anchor_world = translation + anchor_rotated
        anchor_points_world.append(anchor_world)

    anchor_points_world = np.array(anchor_points_world)

    # Create figure with 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot platform outline (connect anchors in a hexagon-like pattern)
    # Platform anchors are arranged in pairs around the circumference
    # Connect: 0-1, 2-3, 4-5 (pairs), and 1-2, 3-4, 5-0 (between pairs)
    platform_outline_indices = [0, 1, 2, 3, 4, 5, 0]
    for i in range(len(platform_outline_indices) - 1):
        idx1 = platform_outline_indices[i]
        idx2 = platform_outline_indices[i + 1]
        ax.plot([anchor_points_world[idx1, 0], anchor_points_world[idx2, 0]],
                [anchor_points_world[idx1, 1], anchor_points_world[idx2, 1]],
                [anchor_points_world[idx1, 2], anchor_points_world[idx2, 2]],
                'b-', linewidth=2, alpha=0.7)

    # Plot anchor points
    ax.scatter(anchor_points_world[:, 0], anchor_points_world[:, 1], anchor_points_world[:, 2],
               c='red', s=100, marker='o', label='Platform Anchors', zorder=5)

    # Label anchor points
    for i, point in enumerate(anchor_points_world):
        ax.text(point[0], point[1], point[2] + 5, f'{i}', fontsize=10, fontweight='bold')

    # Plot platform center
    center = translation
    ax.scatter([center[0]], [center[1]], [center[2]],
               c='green', s=150, marker='X', label='Platform Center', zorder=5)

    # Create mesh grid for platform surface
    # Define a circular platform surface
    theta = np.linspace(0, 2*np.pi, 50)
    r = np.linspace(0, platform.platform + 10, 20)
    THETA, R = np.meshgrid(theta, r)

    # Convert to cartesian in platform frame
    X_platform = R * np.cos(THETA)
    Y_platform = R * np.sin(THETA)
    Z_platform = np.zeros_like(X_platform)

    # Transform surface points to world frame
    surface_points = []
    for i in range(X_platform.shape[0]):
        for j in range(X_platform.shape[1]):
            point_platform = np.array([X_platform[i, j], Y_platform[i, j], 0.0])
            point_rotated = platform._rotate_vector(point_platform, quat)
            point_world = translation + point_rotated
            surface_points.append(point_world)

    surface_points = np.array(surface_points)
    X_world = surface_points[:, 0].reshape(X_platform.shape)
    Y_world = surface_points[:, 1].reshape(Y_platform.shape)
    Z_world = surface_points[:, 2].reshape(Z_platform.shape)

    # Plot platform surface
    ax.plot_surface(X_world, Y_world, Z_world, alpha=0.3, color='cyan',
                   edgecolor='none', label='Platform Surface')

    # Plot coordinate axes at platform center
    axis_length = 40

    # X-axis (red)
    x_axis = platform._rotate_vector(np.array([axis_length, 0, 0]), quat)
    ax.plot([center[0], center[0] + x_axis[0]],
            [center[1], center[1] + x_axis[1]],
            [center[2], center[2] + x_axis[2]],
            'r-', linewidth=3, label='X-axis')
    ax.text(center[0] + x_axis[0], center[1] + x_axis[1], center[2] + x_axis[2],
            'X', fontsize=14, fontweight='bold', color='red')

    # Y-axis (green)
    y_axis = platform._rotate_vector(np.array([0, axis_length, 0]), quat)
    ax.plot([center[0], center[0] + y_axis[0]],
            [center[1], center[1] + y_axis[1]],
            [center[2], center[2] + y_axis[2]],
            'g-', linewidth=3, label='Y-axis')
    ax.text(center[0] + y_axis[0], center[1] + y_axis[1], center[2] + y_axis[2],
            'Y', fontsize=14, fontweight='bold', color='green')

    # Z-axis (blue)
    z_axis = platform._rotate_vector(np.array([0, 0, axis_length]), quat)
    ax.plot([center[0], center[0] + z_axis[0]],
            [center[1], center[1] + z_axis[1]],
            [center[2], center[2] + z_axis[2]],
            'b-', linewidth=3, label='Z-axis')
    ax.text(center[0] + z_axis[0], center[1] + z_axis[1], center[2] + z_axis[2],
            'Z', fontsize=14, fontweight='bold', color='blue')

    # Plot base anchors for reference
    base_anchors = platform.base_anchors
    ax.scatter(base_anchors[:, 0], base_anchors[:, 1], base_anchors[:, 2],
               c='orange', s=80, marker='s', alpha=0.5, label='Base Anchors')

    # Set labels and title
    ax.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (mm)', fontsize=12, fontweight='bold')

    title = f'Stewart Platform Top Surface'
    if tilt_x != 0 or tilt_y != 0:
        title += f'\n(Tilt X: {tilt_x:.1f}°, Tilt Y: {tilt_y:.1f}°)'
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

    # Set equal aspect ratio
    max_range = 100
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, 2*max_range])

    # Set viewing angle
    ax.view_init(elev=30, azim=45)

    # Add legend
    ax.legend(loc='upper left', fontsize=10)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / 'platform_top_surface.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot Stewart Platform top surface')
    parser.add_argument('--tilt-x', type=float, default=0.0,
                       help='Tilt around X-axis in degrees (default: 0.0)')
    parser.add_argument('--tilt-y', type=float, default=0.0,
                       help='Tilt around Y-axis in degrees (default: 0.0)')
    parser.add_argument('--output', type=str, default='plots',
                       help='Output directory for plots (default: plots)')

    args = parser.parse_args()

    plot_platform_top_surface(args.tilt_x, args.tilt_y, args.output)


if __name__ == '__main__':
    main()
