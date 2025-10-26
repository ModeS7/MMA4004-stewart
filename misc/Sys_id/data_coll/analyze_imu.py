#!/usr/bin/env python3
"""
IMU Data Analysis for Kalman Filter Tuning

Analyzes logged IMU data to calculate noise statistics for Kalman filter tuning.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy import stats


def load_imu_data(file_prefix):
    """Load IMU data from separate CSV files"""
    accel_file = f"{file_prefix}_accel.csv"
    gyro_file = f"{file_prefix}_gyro.csv"

    # Check if files exist
    if not Path(accel_file).exists():
        raise FileNotFoundError(f"Accelerometer file not found: {accel_file}")
    if not Path(gyro_file).exists():
        raise FileNotFoundError(f"Gyroscope file not found: {gyro_file}")

    # Load accelerometer data
    with open(accel_file, 'r') as f:
        skip_rows = 0
        for line in f:
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('"#') or stripped == '':
                skip_rows += 1
            else:
                break
    accel_df = pd.read_csv(accel_file, skiprows=skip_rows)

    # Load gyroscope data
    with open(gyro_file, 'r') as f:
        skip_rows = 0
        for line in f:
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('"#') or stripped == '':
                skip_rows += 1
            else:
                break
    gyro_df = pd.read_csv(gyro_file, skiprows=skip_rows)

    return accel_df, gyro_df


def convert_to_physical_units(accel_df, gyro_df):
    """Convert raw IMU values to physical units

    LSM303 accelerometer: 12-bit resolution, ±2g range
    Sensitivity: 1 mg/LSB = 0.001 g/LSB

    L3GD20 gyroscope: 16-bit resolution, ±250°/s range
    Sensitivity: 8.75 mdps/LSB = 0.00875 °/s/LSB
    """
    # LSM303 accelerometer: 1 mg/LSB (12-bit)
    ACCEL_SENSITIVITY = 1000.0  # LSB/g (1 mg/LSB)

    # L3GD20 gyroscope: 8.75 mdps/LSB
    GYRO_SENSITIVITY = 1.0 / 0.00875  # LSB/(°/s) = 114.29

    # Convert accelerometer to m/s²
    accel_df['ax_ms2'] = (accel_df['ax'] / ACCEL_SENSITIVITY) * 9.81
    accel_df['ay_ms2'] = (accel_df['ay'] / ACCEL_SENSITIVITY) * 9.81
    accel_df['az_ms2'] = (accel_df['az'] / ACCEL_SENSITIVITY) * 9.81

    # Convert gyroscope to rad/s
    gyro_df['gx_rads'] = (gyro_df['gx'] / GYRO_SENSITIVITY) * (np.pi / 180)
    gyro_df['gy_rads'] = (gyro_df['gy'] / GYRO_SENSITIVITY) * (np.pi / 180)
    gyro_df['gz_rads'] = (gyro_df['gz'] / GYRO_SENSITIVITY) * (np.pi / 180)

    return accel_df, gyro_df


def analyze_noise(accel_df, gyro_df):
    """Analyze sensor noise statistics"""
    print("\n" + "="*60)
    print("IMU NOISE ANALYSIS")
    print("="*60)

    # Sample rates
    accel_duration = accel_df['timestamp_pc'].iloc[-1] - accel_df['timestamp_pc'].iloc[0]
    gyro_duration = gyro_df['timestamp_pc'].iloc[-1] - gyro_df['timestamp_pc'].iloc[0]

    accel_rate = len(accel_df) / accel_duration if accel_duration > 0 else 0
    gyro_rate = len(gyro_df) / gyro_duration if gyro_duration > 0 else 0

    print("\nSample Rates:")
    print(f"  Accelerometer: {accel_rate:.2f} Hz ({len(accel_df)} samples)")
    print(f"  Gyroscope:     {gyro_rate:.2f} Hz ({len(gyro_df)} samples)")

    # Accelerometer statistics (in m/s²)
    print("\nAccelerometer (m/s²):")
    print(f"  X-axis: mean={accel_df['ax_ms2'].mean():8.4f}, std={accel_df['ax_ms2'].std():8.4f}")
    print(f"  Y-axis: mean={accel_df['ay_ms2'].mean():8.4f}, std={accel_df['ay_ms2'].std():8.4f}")
    print(f"  Z-axis: mean={accel_df['az_ms2'].mean():8.4f}, std={accel_df['az_ms2'].std():8.4f}")

    # Gyroscope statistics (in rad/s)
    print("\nGyroscope (rad/s):")
    print(f"  X-axis: mean={gyro_df['gx_rads'].mean():8.6f}, std={gyro_df['gx_rads'].std():8.6f}")
    print(f"  Y-axis: mean={gyro_df['gy_rads'].mean():8.6f}, std={gyro_df['gy_rads'].std():8.6f}")
    print(f"  Z-axis: mean={gyro_df['gz_rads'].mean():8.6f}, std={gyro_df['gz_rads'].std():8.6f}")

    # Suggested Kalman filter parameters
    print("\n" + "="*60)
    print("SUGGESTED KALMAN FILTER PARAMETERS")
    print("="*60)

    accel_noise_var = np.mean([accel_df['ax_ms2'].var(), accel_df['ay_ms2'].var(), accel_df['az_ms2'].var()])
    gyro_noise_var = np.mean([gyro_df['gx_rads'].var(), gyro_df['gy_rads'].var(), gyro_df['gz_rads'].var()])

    print("\nMeasurement noise covariance (R):")
    print(f"  Accelerometer variance: {accel_noise_var:.6f} (m/s²)²")
    print(f"  Gyroscope variance:     {gyro_noise_var:.8f} (rad/s)²")

    print("\nFor a diagonal R matrix:")
    print(f"  R_accel = {accel_noise_var:.6f} * I_3  (for accelerometer)")
    print(f"  R_gyro  = {gyro_noise_var:.8f} * I_3  (for gyroscope)")

    # Allan variance for process noise estimation (simplified)
    dt_accel = np.diff(accel_df['timestamp_pc'].values).mean()
    dt_gyro = np.diff(gyro_df['timestamp_pc'].values).mean()

    accel_allan = np.mean([
        np.mean(np.diff(accel_df['ax_ms2'])**2) / (2*dt_accel),
        np.mean(np.diff(accel_df['ay_ms2'])**2) / (2*dt_accel),
        np.mean(np.diff(accel_df['az_ms2'])**2) / (2*dt_accel)
    ])
    gyro_allan = np.mean([
        np.mean(np.diff(gyro_df['gx_rads'])**2) / (2*dt_gyro),
        np.mean(np.diff(gyro_df['gy_rads'])**2) / (2*dt_gyro),
        np.mean(np.diff(gyro_df['gz_rads'])**2) / (2*dt_gyro)
    ])

    print("\nProcess noise covariance (Q) - starting estimate:")
    print(f"  Accelerometer: {accel_allan:.6f} (m/s²)²")
    print(f"  Gyroscope:     {gyro_allan:.8f} (rad/s)²")
    print("\n  Note: Q values should be tuned based on expected dynamics.")
    print("  Start with these values and adjust based on filter performance.")


def test_normality(accel_df, gyro_df):
    """Test if noise distributions are Gaussian"""
    print("\n" + "="*60)
    print("NORMALITY TESTS")
    print("="*60)
    print("\nTesting if noise follows Gaussian distribution...")
    print("(p > 0.05 suggests data is consistent with normal distribution)")

    # Combine data for testing
    test_data = [
        (accel_df, 'ax_ms2', 'Accel X'),
        (accel_df, 'ay_ms2', 'Accel Y'),
        (accel_df, 'az_ms2', 'Accel Z'),
        (gyro_df, 'gx_rads', 'Gyro X'),
        (gyro_df, 'gy_rads', 'Gyro Y'),
        (gyro_df, 'gz_rads', 'Gyro Z')
    ]

    print("\nShapiro-Wilk Test:")
    print("-" * 60)
    for df, axis, label in test_data:
        # Remove mean to test noise distribution
        data = df[axis] - df[axis].mean()

        # Shapiro-Wilk test (samples limited to 5000 for performance)
        sample_size = min(5000, len(data))
        sample = np.random.choice(data, sample_size, replace=False)
        stat, p_value = stats.shapiro(sample)

        # Skewness and kurtosis
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)

        result = "PASS" if p_value > 0.05 else "FAIL"
        print(f"  {label:10s}: p={p_value:.4f} [{result}]  "
              f"skew={skewness:6.3f}  kurtosis={kurtosis:6.3f}")

    print("\n  Interpretation:")
    print("    Skewness: 0 = symmetric, >0 = right tail, <0 = left tail")
    print("    Kurtosis: 0 = normal, >0 = heavy tails, <0 = light tails")
    print("    p-value < 0.05 = significant deviation from normal distribution")


def plot_data(accel_df, gyro_df, output_file=None):
    """Plot IMU data time series"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Time in seconds for each sensor (relative to first accelerometer sample)
    t_start = accel_df['timestamp_pc'].iloc[0]
    t_accel = accel_df['timestamp_pc'] - t_start
    t_gyro = gyro_df['timestamp_pc'] - t_start

    # Accelerometer plot
    axes[0].plot(t_accel, accel_df['ax_ms2'], 'r-', alpha=0.7, label='X', linewidth=0.5)
    axes[0].plot(t_accel, accel_df['ay_ms2'], 'g-', alpha=0.7, label='Y', linewidth=0.5)
    axes[0].plot(t_accel, accel_df['az_ms2'], 'b-', alpha=0.7, label='Z', linewidth=0.5)
    axes[0].set_ylabel('Acceleration (m/s²)')
    axes[0].set_title('Accelerometer Data')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Gyroscope plot
    axes[1].plot(t_gyro, gyro_df['gx_rads'], 'r-', alpha=0.7, label='X', linewidth=0.5)
    axes[1].plot(t_gyro, gyro_df['gy_rads'], 'g-', alpha=0.7, label='Y', linewidth=0.5)
    axes[1].plot(t_gyro, gyro_df['gz_rads'], 'b-', alpha=0.7, label='Z', linewidth=0.5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Angular Velocity (rad/s)')
    axes[1].set_title('Gyroscope Data')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"\nPlot saved to: {output_file}")
    else:
        plt.show()


def plot_noise_histogram(accel_df, gyro_df, output_file=None):
    """Plot histogram of sensor noise"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Accelerometer histograms
    axes[0, 0].hist(accel_df['ax_ms2'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Accel X')
    axes[0, 0].set_xlabel('Acceleration (m/s²)')
    axes[0, 0].axvline(accel_df['ax_ms2'].mean(), color='r', linestyle='--', label='mean')
    axes[0, 0].legend()

    axes[0, 1].hist(accel_df['ay_ms2'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Accel Y')
    axes[0, 1].set_xlabel('Acceleration (m/s²)')
    axes[0, 1].axvline(accel_df['ay_ms2'].mean(), color='r', linestyle='--', label='mean')
    axes[0, 1].legend()

    axes[0, 2].hist(accel_df['az_ms2'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 2].set_title('Accel Z')
    axes[0, 2].set_xlabel('Acceleration (m/s²)')
    axes[0, 2].axvline(accel_df['az_ms2'].mean(), color='r', linestyle='--', label='mean')
    axes[0, 2].legend()

    # Gyroscope histograms
    axes[1, 0].hist(gyro_df['gx_rads'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Gyro X')
    axes[1, 0].set_xlabel('Angular Velocity (rad/s)')
    axes[1, 0].axvline(gyro_df['gx_rads'].mean(), color='r', linestyle='--', label='mean')
    axes[1, 0].legend()

    axes[1, 1].hist(gyro_df['gy_rads'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Gyro Y')
    axes[1, 1].set_xlabel('Angular Velocity (rad/s)')
    axes[1, 1].axvline(gyro_df['gy_rads'].mean(), color='r', linestyle='--', label='mean')
    axes[1, 1].legend()

    axes[1, 2].hist(gyro_df['gz_rads'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 2].set_title('Gyro Z')
    axes[1, 2].set_xlabel('Angular Velocity (rad/s)')
    axes[1, 2].axvline(gyro_df['gz_rads'].mean(), color='r', linestyle='--', label='mean')
    axes[1, 2].legend()

    plt.suptitle('IMU Noise Distribution')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Histogram saved to: {output_file}")
    else:
        plt.show()


def plot_qq(accel_df, gyro_df, output_file=None):
    """Plot Q-Q plots to visually assess normality"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    # Combine data for plotting
    plot_data = [
        (accel_df, 'ax_ms2', 'Accel X'),
        (accel_df, 'ay_ms2', 'Accel Y'),
        (accel_df, 'az_ms2', 'Accel Z'),
        (gyro_df, 'gx_rads', 'Gyro X'),
        (gyro_df, 'gy_rads', 'Gyro Y'),
        (gyro_df, 'gz_rads', 'Gyro Z')
    ]

    for idx, (df, col, title) in enumerate(plot_data):
        # Remove mean to analyze noise distribution
        data = df[col] - df[col].mean()

        # Q-Q plot
        stats.probplot(data, dist="norm", plot=axes[idx])
        axes[idx].set_title(f'{title} Q-Q Plot')
        axes[idx].grid(True, alpha=0.3)

        # Add R^2 value
        theoretical_quantiles = stats.probplot(data, dist="norm")[0][0]
        sample_quantiles = stats.probplot(data, dist="norm")[0][1]
        r_squared = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1]**2
        axes[idx].text(0.05, 0.95, f'R² = {r_squared:.4f}',
                      transform=axes[idx].transAxes,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Q-Q Plots: Comparison to Normal Distribution\n(Points should follow red line if Gaussian)')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Q-Q plots saved to: {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze IMU data for Kalman filter tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_imu.py imu_data_20250125_123456
  python analyze_imu.py imu_data_20250125_123456 --plot
  python analyze_imu.py imu_data_20250125_123456 --plot --save

Note: Expects two files: <prefix>_accel.csv and <prefix>_gyro.csv
        """
    )

    parser.add_argument('file_prefix', type=str,
                        help='Input file prefix from imu_logger.py (without _accel.csv or _gyro.csv)')
    parser.add_argument('--plot', action='store_true',
                        help='Show time series plots')
    parser.add_argument('--histogram', action='store_true',
                        help='Show noise distribution histograms')
    parser.add_argument('--normality', action='store_true',
                        help='Run statistical tests for normality')
    parser.add_argument('--qq', action='store_true',
                        help='Show Q-Q plots for normality assessment')
    parser.add_argument('--save', action='store_true',
                        help='Save plots to files instead of displaying')

    args = parser.parse_args()

    # Load and process data
    print(f"\nLoading data from: {args.file_prefix}_*.csv")
    try:
        accel_df, gyro_df = load_imu_data(args.file_prefix)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    total_samples = len(accel_df) + len(gyro_df)
    print(f"Loaded {total_samples} total samples:")
    print(f"  Accelerometer: {len(accel_df)} samples")
    print(f"  Gyroscope:     {len(gyro_df)} samples")

    accel_df, gyro_df = convert_to_physical_units(accel_df, gyro_df)

    # Analyze noise
    analyze_noise(accel_df, gyro_df)

    # Test normality if requested
    if args.normality:
        test_normality(accel_df, gyro_df)

    # Generate plots if requested
    if args.plot:
        output_file = f"{args.file_prefix}_timeseries.png" if args.save else None
        plot_data(accel_df, gyro_df, output_file)

    if args.histogram:
        output_file = f"{args.file_prefix}_histogram.png" if args.save else None
        plot_noise_histogram(accel_df, gyro_df, output_file)

    if args.qq:
        output_file = f"{args.file_prefix}_qq.png" if args.save else None
        plot_qq(accel_df, gyro_df, output_file)


if __name__ == "__main__":
    main()
