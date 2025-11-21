#!/usr/bin/env python3
"""
Kalman Filter Parameter Tuning Script

Replays recorded data with different Kalman filter parameters to find optimal
R (measurement noise) and Q (process noise) settings.

Usage:
    python kalman_tuning.py <csv_file>
    python kalman_tuning.py  # Uses most recent CSV from data/performance/
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict
from core.control_core import KalmanFilter
from core.utils import BallPhysicsConfig, CAMERA_TYPE


def load_data(csv_file: str) -> pd.DataFrame:
    """Load and validate CSV data."""
    df = pd.read_csv(csv_file)

    # Check required columns
    required = ['elapsed_time', 'ball_x', 'ball_y', 'ball_detected',
                'platform_rx', 'platform_ry']
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Filter to valid detections only
    df_valid = df[df['ball_detected'] == True].copy()

    print(f"Loaded {len(df)} samples ({len(df_valid)} valid detections)")
    print(f"Duration: {df['elapsed_time'].max():.2f}s")
    print(f"Detection rate: {len(df_valid)/len(df)*100:.1f}%\n")

    return df_valid


def calculate_metrics(raw_pos: np.ndarray, filtered_pos: np.ndarray,
                     filtered_vel: np.ndarray, dt: float) -> Dict[str, float]:
    """Calculate performance metrics for filtered estimates."""

    # Position RMSE (how close filtered is to raw)
    pos_error = filtered_pos - raw_pos
    rmse = np.sqrt(np.mean(pos_error**2))

    # Position smoothness (how much filtered position varies)
    pos_diff = np.diff(filtered_pos, axis=0)
    smoothness = np.mean(np.linalg.norm(pos_diff, axis=1))

    # Velocity noise (std of velocity estimates)
    vel_magnitude = np.linalg.norm(filtered_vel, axis=1)
    vel_std = np.std(vel_magnitude)
    vel_mean = np.mean(np.abs(vel_magnitude))

    # Lag estimation (correlation peak offset)
    # Compare filtered to raw position to detect phase lag
    from scipy.signal import correlate
    raw_x_centered = raw_pos[:, 0] - np.mean(raw_pos[:, 0])
    filt_x_centered = filtered_pos[:, 0] - np.mean(filtered_pos[:, 0])

    if len(raw_x_centered) > 10:
        corr = correlate(filt_x_centered, raw_x_centered, mode='same')
        center = len(corr) // 2
        peak_idx = np.argmax(corr)
        lag_samples = peak_idx - center
        lag_ms = abs(lag_samples * dt * 1000)
    else:
        lag_ms = 0.0

    return {
        'rmse_mm': rmse,
        'smoothness_mm': smoothness,
        'vel_std_mm_s': vel_std,
        'vel_mean_mm_s': vel_mean,
        'lag_ms': lag_ms
    }


def replay_kalman(df: pd.DataFrame, R_scale: float, Q_scale: float) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Replay Kalman filter with given parameters."""

    # Calculate average dt
    time_diffs = np.diff(df['elapsed_time'].values)
    avg_dt = np.mean(time_diffs)

    # Initialize Kalman filter
    ball_params = {
        'radius': BallPhysicsConfig.RADIUS_M,
        'mass': BallPhysicsConfig.MASS_KG,
        'gravity': BallPhysicsConfig.GRAVITY_M_S2,
        'mass_factor': BallPhysicsConfig.MASS_FACTOR
    }

    kf = KalmanFilter(
        process_noise_scale=Q_scale,
        measurement_noise_scale=R_scale,
        ball_physics_params=ball_params,
        dt=avg_dt,
        camera_type=CAMERA_TYPE
    )

    # Arrays to store results
    n_samples = len(df)
    filtered_pos = np.zeros((n_samples, 2))
    filtered_vel = np.zeros((n_samples, 2))
    raw_pos = np.zeros((n_samples, 2))

    # Reset filter at first measurement
    first_pos = np.array([df.iloc[0]['ball_x'], df.iloc[0]['ball_y']])
    kf.reset(first_pos)

    # Replay data
    for i, row in df.iterrows():
        idx = i if isinstance(i, int) else list(df.index).index(i)

        # Get platform angles for prediction
        rx_deg = row['platform_rx']
        ry_deg = row['platform_ry']

        # Predict step
        kf.predict([rx_deg, ry_deg])

        # Update step with measurement
        ball_pos_m = [row['ball_x'] / 1000.0, row['ball_y'] / 1000.0]
        kf.update(ball_pos_m, row['elapsed_time'])

        # Store results
        raw_pos[idx] = [row['ball_x'], row['ball_y']]
        filt_x, filt_y = kf.get_position_mm()
        filt_vx, filt_vy = kf.get_velocity_mm_s()
        filtered_pos[idx] = [filt_x, filt_y]
        filtered_vel[idx] = [filt_vx, filt_vy]

    # Calculate metrics
    metrics = calculate_metrics(raw_pos, filtered_pos, filtered_vel, avg_dt)
    metrics['R_scale'] = R_scale
    metrics['Q_scale'] = Q_scale

    return filtered_pos, filtered_vel, metrics


def plot_comparison(df: pd.DataFrame, results: List[Tuple], output_dir: Path):
    """Plot comparison of different parameter sets."""

    time = df['elapsed_time'].values
    raw_x = df['ball_x'].values
    raw_y = df['ball_y'].values

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']

    # Plot X position
    axes[0, 0].plot(time, raw_x, 'k.', alpha=0.3, markersize=2, label='Raw')
    for i, (filt_pos, filt_vel, metrics) in enumerate(results[:6]):
        R = metrics['R_scale']
        Q = metrics['Q_scale']
        axes[0, 0].plot(time, filt_pos[:, 0], colors[i], linewidth=1.5,
                       label=f'R={R}, Q={Q}')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('X Position (mm)')
    axes[0, 0].set_title('X Position: Raw vs Filtered')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot Y position
    axes[0, 1].plot(time, raw_y, 'k.', alpha=0.3, markersize=2, label='Raw')
    for i, (filt_pos, filt_vel, metrics) in enumerate(results[:6]):
        R = metrics['R_scale']
        Q = metrics['Q_scale']
        axes[0, 1].plot(time, filt_pos[:, 1], colors[i], linewidth=1.5,
                       label=f'R={R}, Q={Q}')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Y Position (mm)')
    axes[0, 1].set_title('Y Position: Raw vs Filtered')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot X velocity
    for i, (filt_pos, filt_vel, metrics) in enumerate(results[:6]):
        R = metrics['R_scale']
        Q = metrics['Q_scale']
        axes[1, 0].plot(time, filt_vel[:, 0], colors[i], linewidth=1.5,
                       label=f'R={R}, Q={Q}')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('X Velocity (mm/s)')
    axes[1, 0].set_title('X Velocity Estimates')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot Y velocity
    for i, (filt_pos, filt_vel, metrics) in enumerate(results[:6]):
        R = metrics['R_scale']
        Q = metrics['Q_scale']
        axes[1, 1].plot(time, filt_vel[:, 1], colors[i], linewidth=1.5,
                       label=f'R={R}, Q={Q}')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Y Velocity (mm/s)')
    axes[1, 1].set_title('Y Velocity Estimates')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    # Plot position errors
    for i, (filt_pos, filt_vel, metrics) in enumerate(results[:6]):
        R = metrics['R_scale']
        Q = metrics['Q_scale']
        errors = np.linalg.norm(filt_pos - np.column_stack([raw_x, raw_y]), axis=1)
        axes[2, 0].plot(time, errors, colors[i], linewidth=1.5,
                       label=f'R={R}, Q={Q} (RMSE={metrics["rmse_mm"]:.2f}mm)')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Position Error (mm)')
    axes[2, 0].set_title('Filtered vs Raw Position Error')
    axes[2, 0].legend(fontsize=8)
    axes[2, 0].grid(True, alpha=0.3)

    # Plot velocity magnitude
    for i, (filt_pos, filt_vel, metrics) in enumerate(results[:6]):
        R = metrics['R_scale']
        Q = metrics['Q_scale']
        vel_mag = np.linalg.norm(filt_vel, axis=1)
        axes[2, 1].plot(time, vel_mag, colors[i], linewidth=1.5,
                       label=f'R={R}, Q={Q}')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Velocity Magnitude (mm/s)')
    axes[2, 1].set_title('Velocity Magnitude')
    axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_file = output_dir / f'kalman_tuning_{timestamp}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_file}")

    plt.show()


def plot_metrics_comparison(all_metrics: List[Dict], output_dir: Path):
    """Plot metrics comparison bar charts."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    labels = [f"R={m['R_scale']}, Q={m['Q_scale']}" for m in all_metrics]
    x = np.arange(len(labels))

    # RMSE
    rmse_values = [m['rmse_mm'] for m in all_metrics]
    axes[0, 0].bar(x, rmse_values)
    axes[0, 0].set_ylabel('RMSE (mm)')
    axes[0, 0].set_title('Position RMSE (lower = closer to raw)')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Smoothness
    smooth_values = [m['smoothness_mm'] for m in all_metrics]
    axes[0, 1].bar(x, smooth_values)
    axes[0, 1].set_ylabel('Smoothness (mm/step)')
    axes[0, 1].set_title('Position Smoothness (lower = smoother)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Velocity std
    vel_std_values = [m['vel_std_mm_s'] for m in all_metrics]
    axes[0, 2].bar(x, vel_std_values)
    axes[0, 2].set_ylabel('Velocity Std (mm/s)')
    axes[0, 2].set_title('Velocity Noise (lower = less noisy)')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)

    # Velocity mean
    vel_mean_values = [m['vel_mean_mm_s'] for m in all_metrics]
    axes[1, 0].bar(x, vel_mean_values)
    axes[1, 0].set_ylabel('Velocity Mean (mm/s)')
    axes[1, 0].set_title('Average Velocity Magnitude')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Lag
    lag_values = [m['lag_ms'] for m in all_metrics]
    axes[1, 1].bar(x, lag_values)
    axes[1, 1].set_ylabel('Lag (ms)')
    axes[1, 1].set_title('Phase Lag (lower = more responsive)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    # Score (weighted combination)
    # Lower is better: normalize and weight
    rmse_norm = np.array(rmse_values) / max(rmse_values)
    vel_std_norm = np.array(vel_std_values) / max(vel_std_values)
    lag_norm = np.array(lag_values) / max(lag_values) if max(lag_values) > 0 else np.zeros_like(lag_values)

    # Weighted score: 40% velocity noise, 30% lag, 30% RMSE
    scores = 0.4 * vel_std_norm + 0.3 * lag_norm + 0.3 * rmse_norm

    axes[1, 2].bar(x, scores)
    axes[1, 2].set_ylabel('Combined Score')
    axes[1, 2].set_title('Overall Score (lower = better)')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)

    # Highlight best
    best_idx = np.argmin(scores)
    axes[1, 2].bar(best_idx, scores[best_idx], color='green', alpha=0.7)

    plt.tight_layout()

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_file = output_dir / f'kalman_metrics_{timestamp}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Metrics plot saved to: {plot_file}")

    plt.show()

    return scores


def main():
    """Main tuning workflow."""

    # Get CSV file
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Use most recent file
        data_dir = Path(__file__).parent.parent.parent / 'data' / 'performance'
        csv_files = list(data_dir.glob('performance_*.csv'))

        if not csv_files:
            print("No CSV file found in data/performance/")
            print("Usage: python kalman_tuning.py <path_to_csv_file>")
            sys.exit(1)

        csv_file = max(csv_files, key=lambda p: p.stat().st_mtime)
        print(f"Using most recent file: {csv_file.name}\n")

    # Load data
    df = load_data(csv_file)

    if len(df) < 10:
        print("ERROR: Not enough valid data points for tuning!")
        sys.exit(1)

    # Define parameter combinations to test
    # R_scale: measurement noise (higher = trust measurements less, smoother)
    # Q_scale: process noise (higher = trust model less, faster response)
    param_sets = [
        (0.5, 1.0),   # Low R, normal Q - trust measurements more
        (1.0, 1.0),   # Baseline
        (2.0, 1.0),   # Moderate smoothing
        (5.0, 1.0),   # High smoothing
        (2.0, 0.5),   # Moderate smoothing, trust model more
        (5.0, 0.5),   # High smoothing, trust model more
    ]

    print("=== TESTING PARAMETER COMBINATIONS ===\n")

    results = []
    all_metrics = []

    for R_scale, Q_scale in param_sets:
        print(f"Testing R_scale={R_scale}, Q_scale={Q_scale}...", end=' ')

        try:
            filt_pos, filt_vel, metrics = replay_kalman(df, R_scale, Q_scale)
            results.append((filt_pos, filt_vel, metrics))
            all_metrics.append(metrics)

            print(f"RMSE={metrics['rmse_mm']:.3f}mm, "
                  f"Vel_std={metrics['vel_std_mm_s']:.2f}mm/s, "
                  f"Lag={metrics['lag_ms']:.1f}ms")
        except Exception as e:
            print(f"FAILED: {e}")

    if not results:
        print("\nERROR: All parameter sets failed!")
        sys.exit(1)

    # Print detailed metrics
    print("\n=== DETAILED METRICS ===\n")
    print(f"{'R':>6} {'Q':>6} {'RMSE':>8} {'Smooth':>8} {'VelStd':>8} {'VelMean':>9} {'Lag':>7}")
    print(f"{'Scale':>6} {'Scale':>6} {'(mm)':>8} {'(mm)':>8} {'(mm/s)':>8} {'(mm/s)':>9} {'(ms)':>7}")
    print("-" * 70)

    for m in all_metrics:
        print(f"{m['R_scale']:>6.1f} {m['Q_scale']:>6.1f} "
              f"{m['rmse_mm']:>8.3f} {m['smoothness_mm']:>8.3f} "
              f"{m['vel_std_mm_s']:>8.2f} {m['vel_mean_mm_s']:>9.2f} "
              f"{m['lag_ms']:>7.1f}")

    # Calculate scores
    print("\n=== SCORING (lower is better) ===\n")

    rmse_values = [m['rmse_mm'] for m in all_metrics]
    vel_std_values = [m['vel_std_mm_s'] for m in all_metrics]
    lag_values = [m['lag_ms'] for m in all_metrics]

    rmse_norm = np.array(rmse_values) / max(rmse_values)
    vel_std_norm = np.array(vel_std_values) / max(vel_std_values)
    lag_norm = np.array(lag_values) / max(lag_values) if max(lag_values) > 0 else np.zeros_like(lag_values)

    scores = 0.4 * vel_std_norm + 0.3 * lag_norm + 0.3 * rmse_norm

    for i, m in enumerate(all_metrics):
        print(f"R={m['R_scale']:.1f}, Q={m['Q_scale']:.1f}: Score={scores[i]:.3f}")

    # Find best
    best_idx = np.argmin(scores)
    best = all_metrics[best_idx]

    print(f"\n=== RECOMMENDATION ===")
    print(f"Best parameters: R_scale={best['R_scale']}, Q_scale={best['Q_scale']}")
    print(f"  RMSE: {best['rmse_mm']:.3f}mm")
    print(f"  Velocity std: {best['vel_std_mm_s']:.2f}mm/s")
    print(f"  Lag: {best['lag_ms']:.1f}ms")
    print(f"\nSet in GUI:")
    print(f"  Measurement Noise (R): {best['R_scale']}")
    print(f"  Process Noise (Q): {best['Q_scale']}")

    # Plot results
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'performance'
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_comparison(df, results, output_dir)
    plot_metrics_comparison(all_metrics, output_dir)


if __name__ == "__main__":
    main()
