#!/usr/bin/env python3
"""
Synchronized Performance Comparison Plotting Script

Synchronizes PID and LQR data to the moment when the target starts moving
(transition from static to circle pattern), then compares them on the same graph.

Usage:
    python plot_sync.py --pid data/performance/performance_PID_*.csv --lqr data/performance/performance_LQR_*.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def find_pattern_start(df, movement_threshold=1.0):
    """
    Find the index where the target starts moving (static -> circle transition).

    Args:
        df: DataFrame with target_x and target_y columns
        movement_threshold: Minimum target movement (mm) to detect pattern start

    Returns:
        Index where pattern starts, or 0 if not found
    """
    # Calculate target movement between consecutive samples
    target_dx = df['target_x'].diff().abs()
    target_dy = df['target_y'].diff().abs()
    target_movement = np.sqrt(target_dx**2 + target_dy**2)

    # Find first significant movement
    moving_samples = target_movement > movement_threshold

    if moving_samples.any():
        # Get the first index where movement exceeds threshold
        start_idx = moving_samples.idxmax()
        return start_idx
    else:
        # No movement detected, return start
        return 0


def calculate_metrics(df, settling_threshold=5.0):
    """Calculate performance metrics from data."""
    metrics = {}

    # Max error
    metrics['max_error'] = df['error_magnitude'].max()

    # Mean error
    metrics['mean_error'] = df['error_magnitude'].mean()

    # RMS error
    metrics['rms_error'] = np.sqrt((df['error_magnitude']**2).mean())

    # Settling time (first time error stays below threshold for 1 second)
    settled_idx = None
    sample_rate = len(df) / df['elapsed_time'].iloc[-1] if len(df) > 1 else 100
    window_samples = int(sample_rate)  # 1 second window

    for i in range(len(df) - window_samples):
        if all(df['error_magnitude'].iloc[i:i+window_samples] < settling_threshold):
            settled_idx = i
            break

    if settled_idx is not None:
        metrics['settling_time'] = df['elapsed_time'].iloc[settled_idx]
    else:
        metrics['settling_time'] = None

    # Overshoot (max error in first 2 seconds)
    first_2s = df[df['elapsed_time'] <= 2.0]
    metrics['overshoot'] = first_2s['error_magnitude'].max() if len(first_2s) > 0 else metrics['max_error']

    # Steady-state error (last 5 seconds)
    last_5s = df[df['elapsed_time'] >= df['elapsed_time'].max() - 5.0]
    metrics['steady_state_error'] = last_5s['error_magnitude'].mean() if len(last_5s) > 0 else metrics['mean_error']

    return metrics


def plot_comparison(pid_file, lqr_file, output_dir='plots', cutoff_time=None):
    """Generate synchronized comparison plots for PID vs LQR performance."""
    print(f"Loading data...")
    print(f"  PID: {pid_file}")
    print(f"  LQR: {lqr_file}")

    pid_df = pd.read_csv(pid_file)
    lqr_df = pd.read_csv(lqr_file)

    print(f"\nDetecting pattern transitions...")

    # Find where each dataset transitions from static to moving target
    pid_start_idx = find_pattern_start(pid_df)
    lqr_start_idx = find_pattern_start(lqr_df)

    print(f"  PID pattern starts at index {pid_start_idx} (t={pid_df['elapsed_time'].iloc[pid_start_idx]:.3f}s)")
    print(f"  LQR pattern starts at index {lqr_start_idx} (t={lqr_df['elapsed_time'].iloc[lqr_start_idx]:.3f}s)")

    # Trim both datasets to start from pattern transition
    pid_df = pid_df.iloc[pid_start_idx:].copy()
    lqr_df = lqr_df.iloc[lqr_start_idx:].copy()

    # Reset elapsed_time to start at 0
    pid_df['elapsed_time'] = pid_df['elapsed_time'] - pid_df['elapsed_time'].iloc[0]
    lqr_df['elapsed_time'] = lqr_df['elapsed_time'] - lqr_df['elapsed_time'].iloc[0]

    # Find common time range
    if cutoff_time is not None:
        max_time = cutoff_time
        print(f"\nUsing user-specified cutoff time: {cutoff_time:.3f}s")
    else:
        max_time = min(pid_df['elapsed_time'].max(), lqr_df['elapsed_time'].max())
        print(f"\nUsing automatic cutoff time (min of both datasets)")

    # Trim to common duration
    pid_df = pid_df[pid_df['elapsed_time'] <= max_time].copy()
    lqr_df = lqr_df[lqr_df['elapsed_time'] <= max_time].copy()

    print(f"\nSynchronized data:")
    print(f"  Duration: {max_time:.3f}s")
    print(f"  PID samples: {len(pid_df)}")
    print(f"  LQR samples: {len(lqr_df)}")

    print(f"\nCalculating metrics...")
    pid_metrics = calculate_metrics(pid_df)
    lqr_metrics = calculate_metrics(lqr_df)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create figure with four subplots (2 rows, 2 columns)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Plot 1: Error magnitude over time
    ax1.plot(pid_df['elapsed_time'], pid_df['error_magnitude'],
             label='PID', color='#2E86AB', linewidth=2, alpha=0.9)
    ax1.plot(lqr_df['elapsed_time'], lqr_df['error_magnitude'],
             label='LQR', color='#A23B72', linewidth=2, alpha=0.9)
    ax1.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Error Magnitude (mm)', fontsize=13, fontweight='bold')
    ax1.set_title('Error Magnitude vs Time', fontsize=15, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=12, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_xlim(left=0)
    ax1.tick_params(axis='both', labelsize=11)

    # Plot 2: Top view - Ball and Target Trajectories
    ax2.plot(pid_df['target_x'], pid_df['target_y'],
             label='Target', color='#000000', linewidth=2, alpha=0.5, linestyle='--')
    ax2.plot(pid_df['ball_x'], pid_df['ball_y'],
             label='PID Ball', color='#2E86AB', linewidth=2, alpha=0.7)
    ax2.plot(lqr_df['ball_x'], lqr_df['ball_y'],
             label='LQR Ball', color='#A23B72', linewidth=2, alpha=0.7)
    ax2.scatter([0], [0], color='green', s=150, marker='X', label='Center', zorder=5)
    ax2.set_xlabel('X Position (mm)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Y Position (mm)', fontsize=13, fontweight='bold')
    ax2.set_title('Ball Trajectories (Top View)', fontsize=15, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', fontsize=12, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_aspect('equal', adjustable='box')
    ax2.axhline(y=0, color='#000000', linewidth=1, alpha=0.3)
    ax2.axvline(x=0, color='#000000', linewidth=1, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=11)

    # Plot 3: Error components (X and Y)
    ax3.plot(pid_df['elapsed_time'], pid_df['error_x'],
             label='PID X', color='#2E86AB', linewidth=2, alpha=0.9)
    ax3.plot(pid_df['elapsed_time'], pid_df['error_y'],
             label='PID Y', color='#2E86AB', linewidth=2, alpha=0.9, linestyle='--')
    ax3.plot(lqr_df['elapsed_time'], lqr_df['error_x'],
             label='LQR X', color='#A23B72', linewidth=2, alpha=0.9)
    ax3.plot(lqr_df['elapsed_time'], lqr_df['error_y'],
             label='LQR Y', color='#A23B72', linewidth=2, alpha=0.9, linestyle='--')
    ax3.axhline(y=0, color='#000000', linewidth=1, alpha=0.4)
    ax3.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Error (mm)', fontsize=13, fontweight='bold')
    ax3.set_title('Error Components vs Time', fontsize=15, fontweight='bold', pad=15)
    ax3.legend(loc='upper right', fontsize=12, ncol=2, framealpha=0.95)
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax3.set_xlim(left=0)
    ax3.tick_params(axis='both', labelsize=11)

    # Plot 4: Top view - Error Vectors
    # Sample every N points to avoid cluttering
    sample_rate = max(1, len(pid_df) // 50)
    pid_sample = pid_df.iloc[::sample_rate]
    lqr_sample = lqr_df.iloc[::sample_rate]

    # Plot error vectors from target to ball position
    for i in range(len(pid_sample)):
        row = pid_sample.iloc[i]
        ax4.arrow(row['target_x'], row['target_y'], row['error_x'], row['error_y'],
                 head_width=2, head_length=1, fc='#2E86AB', ec='#2E86AB',
                 alpha=0.3, linewidth=0.5)

    for i in range(len(lqr_sample)):
        row = lqr_sample.iloc[i]
        ax4.arrow(row['target_x'], row['target_y'], row['error_x'], row['error_y'],
                 head_width=2, head_length=1, fc='#A23B72', ec='#A23B72',
                 alpha=0.3, linewidth=0.5)

    # Plot target trajectory
    ax4.plot(pid_df['target_x'], pid_df['target_y'],
             label='Target', color='#000000', linewidth=2, alpha=0.5, linestyle='--')
    ax4.scatter([0], [0], color='green', s=150, marker='X', label='Center', zorder=5)
    ax4.set_xlabel('X Position (mm)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Y Position (mm)', fontsize=13, fontweight='bold')
    ax4.set_title('Error Vectors (Top View)', fontsize=15, fontweight='bold', pad=15)
    ax4.legend(loc='upper right', fontsize=12, framealpha=0.95)
    ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax4.set_aspect('equal', adjustable='box')
    ax4.axhline(y=0, color='#000000', linewidth=1, alpha=0.3)
    ax4.axvline(x=0, color='#000000', linewidth=1, alpha=0.3)
    ax4.tick_params(axis='both', labelsize=11)

    # Add manual legend entries for error vectors
    from matplotlib.patches import FancyArrow
    pid_arrow = FancyArrow(0, 0, 0, 0, color='#2E86AB', alpha=0.3)
    lqr_arrow = FancyArrow(0, 0, 0, 0, color='#A23B72', alpha=0.3)
    handles, labels = ax4.get_legend_handles_labels()
    handles.extend([pid_arrow, lqr_arrow])
    labels.extend(['PID Error', 'LQR Error'])
    ax4.legend(handles, labels, loc='upper right', fontsize=12, framealpha=0.95)

    # Save plot
    output_file = output_path / 'pid_vs_lqr_synchronized.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    # Print metrics summary
    print("\n" + "="*70)
    print("PERFORMANCE METRICS SUMMARY (Synchronized)")
    print("="*70)
    print(f"\n{'Metric':<30} {'PID':>15} {'LQR':>15} {'Winner':>10}")
    print("-"*70)

    # Max Error
    winner = 'PID' if pid_metrics['max_error'] < lqr_metrics['max_error'] else 'LQR'
    print(f"{'Max Error (mm)':<30} {pid_metrics['max_error']:>15.2f} {lqr_metrics['max_error']:>15.2f} {winner:>10}")

    # Mean Error
    winner = 'PID' if pid_metrics['mean_error'] < lqr_metrics['mean_error'] else 'LQR'
    print(f"{'Mean Error (mm)':<30} {pid_metrics['mean_error']:>15.2f} {lqr_metrics['mean_error']:>15.2f} {winner:>10}")

    # RMS Error
    winner = 'PID' if pid_metrics['rms_error'] < lqr_metrics['rms_error'] else 'LQR'
    print(f"{'RMS Error (mm)':<30} {pid_metrics['rms_error']:>15.2f} {lqr_metrics['rms_error']:>15.2f} {winner:>10}")

    # Settling Time
    pid_st = f"{pid_metrics['settling_time']:.2f}" if pid_metrics['settling_time'] else 'N/A'
    lqr_st = f"{lqr_metrics['settling_time']:.2f}" if lqr_metrics['settling_time'] else 'N/A'
    if pid_metrics['settling_time'] and lqr_metrics['settling_time']:
        winner = 'PID' if pid_metrics['settling_time'] < lqr_metrics['settling_time'] else 'LQR'
    else:
        winner = 'N/A'
    print(f"{'Settling Time (s)':<30} {pid_st:>15} {lqr_st:>15} {winner:>10}")

    # Overshoot
    winner = 'PID' if pid_metrics['overshoot'] < lqr_metrics['overshoot'] else 'LQR'
    print(f"{'Overshoot (mm)':<30} {pid_metrics['overshoot']:>15.2f} {lqr_metrics['overshoot']:>15.2f} {winner:>10}")

    # Steady-State Error
    winner = 'PID' if pid_metrics['steady_state_error'] < lqr_metrics['steady_state_error'] else 'LQR'
    print(f"{'Steady-State Error (mm)':<30} {pid_metrics['steady_state_error']:>15.2f} {lqr_metrics['steady_state_error']:>15.2f} {winner:>10}")

    print("="*70)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compare PID vs LQR performance with pattern synchronization')
    parser.add_argument('--pid', type=str, required=True,
                       help='Path to PID performance CSV file')
    parser.add_argument('--lqr', type=str, required=True,
                       help='Path to LQR performance CSV file')
    parser.add_argument('--output', type=str, default='plots',
                       help='Output directory for plots (default: plots)')
    parser.add_argument('--threshold', type=float, default=1.0,
                       help='Movement threshold (mm) to detect pattern start (default: 1.0)')
    parser.add_argument('--cutoff', type=float, default=None,
                       help='Cutoff time (s) for plotting. If not specified, uses minimum of both datasets')

    args = parser.parse_args()

    plot_comparison(args.pid, args.lqr, args.output, cutoff_time=args.cutoff)


if __name__ == '__main__':
    main()
