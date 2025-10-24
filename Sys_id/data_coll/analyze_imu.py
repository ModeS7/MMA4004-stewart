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


def load_imu_data(csv_file):
    """Load IMU data from CSV file"""
    # Skip header lines starting with # or quoted comments
    with open(csv_file, 'r') as f:
        skip_rows = 0
        for line in f:
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('"#') or stripped == '':
                skip_rows += 1
            else:
                break

    # Load data
    df = pd.read_csv(csv_file, skiprows=skip_rows)
    return df


def convert_to_physical_units(df):
    """Convert raw IMU values to physical units

    LSM303 accelerometer: 12-bit resolution, ±2g range
    Sensitivity: 1 mg/LSB = 0.001 g/LSB

    L3GD20 gyroscope: 16-bit resolution, ±250°/s range
    Sensitivity: 8.75 mdps/LSB = 0.00875 °/s/LSB

    LSM303 magnetometer: 16-bit resolution, ±1.3 gauss range
    Sensitivity: 1/1100 gauss/LSB = 0.00091 gauss/LSB
    """
    # LSM303 accelerometer: 1 mg/LSB (12-bit)
    ACCEL_SENSITIVITY = 1000.0  # LSB/g (1 mg/LSB)

    # L3GD20 gyroscope: 8.75 mdps/LSB
    GYRO_SENSITIVITY = 1.0 / 0.00875  # LSB/(°/s) = 114.29

    # LSM303 magnetometer: 1/1100 gauss/LSB (at ±1.3 gauss range)
    MAG_SENSITIVITY = 1100.0  # LSB/gauss

    # Convert accelerometer to m/s²
    df['ax_ms2'] = (df['ax'] / ACCEL_SENSITIVITY) * 9.81
    df['ay_ms2'] = (df['ay'] / ACCEL_SENSITIVITY) * 9.81
    df['az_ms2'] = (df['az'] / ACCEL_SENSITIVITY) * 9.81

    # Convert gyroscope to rad/s
    df['gx_rads'] = (df['gx'] / GYRO_SENSITIVITY) * (np.pi / 180)
    df['gy_rads'] = (df['gy'] / GYRO_SENSITIVITY) * (np.pi / 180)
    df['gz_rads'] = (df['gz'] / GYRO_SENSITIVITY) * (np.pi / 180)

    # Convert magnetometer to gauss
    df['mx_gauss'] = df['mx'] / MAG_SENSITIVITY
    df['my_gauss'] = df['my'] / MAG_SENSITIVITY
    df['mz_gauss'] = df['mz'] / MAG_SENSITIVITY

    return df


def analyze_noise(df):
    """Analyze sensor noise statistics"""
    print("\n" + "="*60)
    print("IMU NOISE ANALYSIS")
    print("="*60)

    # Accelerometer statistics (in m/s²)
    print("\nAccelerometer (m/s²):")
    print(f"  X-axis: mean={df['ax_ms2'].mean():8.4f}, std={df['ax_ms2'].std():8.4f}")
    print(f"  Y-axis: mean={df['ay_ms2'].mean():8.4f}, std={df['ay_ms2'].std():8.4f}")
    print(f"  Z-axis: mean={df['az_ms2'].mean():8.4f}, std={df['az_ms2'].std():8.4f}")

    # Gyroscope statistics (in rad/s)
    print("\nGyroscope (rad/s):")
    print(f"  X-axis: mean={df['gx_rads'].mean():8.6f}, std={df['gx_rads'].std():8.6f}")
    print(f"  Y-axis: mean={df['gy_rads'].mean():8.6f}, std={df['gy_rads'].std():8.6f}")
    print(f"  Z-axis: mean={df['gz_rads'].mean():8.6f}, std={df['gz_rads'].std():8.6f}")

    # Magnetometer statistics (in gauss)
    print("\nMagnetometer (gauss):")
    print(f"  X-axis: mean={df['mx_gauss'].mean():8.4f}, std={df['mx_gauss'].std():8.4f}")
    print(f"  Y-axis: mean={df['my_gauss'].mean():8.4f}, std={df['my_gauss'].std():8.4f}")
    print(f"  Z-axis: mean={df['mz_gauss'].mean():8.4f}, std={df['mz_gauss'].std():8.4f}")

    # Suggested Kalman filter parameters
    print("\n" + "="*60)
    print("SUGGESTED KALMAN FILTER PARAMETERS")
    print("="*60)

    accel_noise_var = np.mean([df['ax_ms2'].var(), df['ay_ms2'].var(), df['az_ms2'].var()])
    gyro_noise_var = np.mean([df['gx_rads'].var(), df['gy_rads'].var(), df['gz_rads'].var()])
    mag_noise_var = np.mean([df['mx_gauss'].var(), df['my_gauss'].var(), df['mz_gauss'].var()])

    print("\nMeasurement noise covariance (R):")
    print(f"  Accelerometer variance: {accel_noise_var:.6f} (m/s²)²")
    print(f"  Gyroscope variance:     {gyro_noise_var:.8f} (rad/s)²")
    print(f"  Magnetometer variance:  {mag_noise_var:.6f} (gauss)²")

    print("\nFor a diagonal R matrix:")
    print(f"  R_accel = {accel_noise_var:.6f} * I_3  (for accelerometer)")
    print(f"  R_gyro  = {gyro_noise_var:.8f} * I_3  (for gyroscope)")
    print(f"  R_mag   = {mag_noise_var:.6f} * I_3  (for magnetometer)")

    # Allan variance for process noise estimation (simplified)
    dt = np.diff(df['timestamp_pc'].values).mean()  # Already in seconds
    accel_allan = np.mean([
        np.mean(np.diff(df['ax_ms2'])**2) / (2*dt),
        np.mean(np.diff(df['ay_ms2'])**2) / (2*dt),
        np.mean(np.diff(df['az_ms2'])**2) / (2*dt)
    ])
    gyro_allan = np.mean([
        np.mean(np.diff(df['gx_rads'])**2) / (2*dt),
        np.mean(np.diff(df['gy_rads'])**2) / (2*dt),
        np.mean(np.diff(df['gz_rads'])**2) / (2*dt)
    ])
    mag_allan = np.mean([
        np.mean(np.diff(df['mx_gauss'])**2) / (2*dt),
        np.mean(np.diff(df['my_gauss'])**2) / (2*dt),
        np.mean(np.diff(df['mz_gauss'])**2) / (2*dt)
    ])

    print("\nProcess noise covariance (Q) - starting estimate:")
    print(f"  Accelerometer: {accel_allan:.6f} (m/s²)²")
    print(f"  Gyroscope:     {gyro_allan:.8f} (rad/s)²")
    print(f"  Magnetometer:  {mag_allan:.6f} (gauss)²")
    print("\n  Note: Q values should be tuned based on expected dynamics.")
    print("  Start with these values and adjust based on filter performance.")


def test_normality(df):
    """Test if noise distributions are Gaussian"""
    print("\n" + "="*60)
    print("NORMALITY TESTS")
    print("="*60)
    print("\nTesting if noise follows Gaussian distribution...")
    print("(p > 0.05 suggests data is consistent with normal distribution)")

    axes = ['ax_ms2', 'ay_ms2', 'az_ms2', 'gx_rads', 'gy_rads', 'gz_rads', 'mx_gauss', 'my_gauss', 'mz_gauss']
    labels = ['Accel X', 'Accel Y', 'Accel Z', 'Gyro X', 'Gyro Y', 'Gyro Z', 'Mag X', 'Mag Y', 'Mag Z']

    print("\nShapiro-Wilk Test:")
    print("-" * 60)
    for axis, label in zip(axes, labels):
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


def plot_data(df, output_file=None):
    """Plot IMU data time series"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Time in seconds (timestamp_pc is already in seconds)
    t = df['timestamp_pc'] - df['timestamp_pc'].iloc[0]

    # Accelerometer plot
    axes[0].plot(t, df['ax_ms2'], 'r-', alpha=0.7, label='X', linewidth=0.5)
    axes[0].plot(t, df['ay_ms2'], 'g-', alpha=0.7, label='Y', linewidth=0.5)
    axes[0].plot(t, df['az_ms2'], 'b-', alpha=0.7, label='Z', linewidth=0.5)
    axes[0].set_ylabel('Acceleration (m/s²)')
    axes[0].set_title('Accelerometer Data')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Gyroscope plot
    axes[1].plot(t, df['gx_rads'], 'r-', alpha=0.7, label='X', linewidth=0.5)
    axes[1].plot(t, df['gy_rads'], 'g-', alpha=0.7, label='Y', linewidth=0.5)
    axes[1].plot(t, df['gz_rads'], 'b-', alpha=0.7, label='Z', linewidth=0.5)
    axes[1].set_ylabel('Angular Velocity (rad/s)')
    axes[1].set_title('Gyroscope Data')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Magnetometer plot
    axes[2].plot(t, df['mx_gauss'], 'r-', alpha=0.7, label='X', linewidth=0.5)
    axes[2].plot(t, df['my_gauss'], 'g-', alpha=0.7, label='Y', linewidth=0.5)
    axes[2].plot(t, df['mz_gauss'], 'b-', alpha=0.7, label='Z', linewidth=0.5)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Magnetic Field (gauss)')
    axes[2].set_title('Magnetometer Data')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"\nPlot saved to: {output_file}")
    else:
        plt.show()


def plot_noise_histogram(df, output_file=None):
    """Plot histogram of sensor noise"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # Accelerometer histograms
    axes[0, 0].hist(df['ax_ms2'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Accel X')
    axes[0, 0].set_xlabel('Acceleration (m/s²)')
    axes[0, 0].axvline(df['ax_ms2'].mean(), color='r', linestyle='--', label='mean')
    axes[0, 0].legend()

    axes[0, 1].hist(df['ay_ms2'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Accel Y')
    axes[0, 1].set_xlabel('Acceleration (m/s²)')
    axes[0, 1].axvline(df['ay_ms2'].mean(), color='r', linestyle='--', label='mean')
    axes[0, 1].legend()

    axes[0, 2].hist(df['az_ms2'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 2].set_title('Accel Z')
    axes[0, 2].set_xlabel('Acceleration (m/s²)')
    axes[0, 2].axvline(df['az_ms2'].mean(), color='r', linestyle='--', label='mean')
    axes[0, 2].legend()

    # Gyroscope histograms
    axes[1, 0].hist(df['gx_rads'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Gyro X')
    axes[1, 0].set_xlabel('Angular Velocity (rad/s)')
    axes[1, 0].axvline(df['gx_rads'].mean(), color='r', linestyle='--', label='mean')
    axes[1, 0].legend()

    axes[1, 1].hist(df['gy_rads'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Gyro Y')
    axes[1, 1].set_xlabel('Angular Velocity (rad/s)')
    axes[1, 1].axvline(df['gy_rads'].mean(), color='r', linestyle='--', label='mean')
    axes[1, 1].legend()

    axes[1, 2].hist(df['gz_rads'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 2].set_title('Gyro Z')
    axes[1, 2].set_xlabel('Angular Velocity (rad/s)')
    axes[1, 2].axvline(df['gz_rads'].mean(), color='r', linestyle='--', label='mean')
    axes[1, 2].legend()

    # Magnetometer histograms
    axes[2, 0].hist(df['mx_gauss'], bins=50, alpha=0.7, edgecolor='black')
    axes[2, 0].set_title('Mag X')
    axes[2, 0].set_xlabel('Magnetic Field (gauss)')
    axes[2, 0].axvline(df['mx_gauss'].mean(), color='r', linestyle='--', label='mean')
    axes[2, 0].legend()

    axes[2, 1].hist(df['my_gauss'], bins=50, alpha=0.7, edgecolor='black')
    axes[2, 1].set_title('Mag Y')
    axes[2, 1].set_xlabel('Magnetic Field (gauss)')
    axes[2, 1].axvline(df['my_gauss'].mean(), color='r', linestyle='--', label='mean')
    axes[2, 1].legend()

    axes[2, 2].hist(df['mz_gauss'], bins=50, alpha=0.7, edgecolor='black')
    axes[2, 2].set_title('Mag Z')
    axes[2, 2].set_xlabel('Magnetic Field (gauss)')
    axes[2, 2].axvline(df['mz_gauss'].mean(), color='r', linestyle='--', label='mean')
    axes[2, 2].legend()

    plt.suptitle('IMU Noise Distribution')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Histogram saved to: {output_file}")
    else:
        plt.show()


def plot_qq(df, output_file=None):
    """Plot Q-Q plots to visually assess normality"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    data_cols = ['ax_ms2', 'ay_ms2', 'az_ms2', 'gx_rads', 'gy_rads', 'gz_rads', 'mx_gauss', 'my_gauss', 'mz_gauss']
    titles = ['Accel X', 'Accel Y', 'Accel Z', 'Gyro X', 'Gyro Y', 'Gyro Z', 'Mag X', 'Mag Y', 'Mag Z']

    for idx, (col, title) in enumerate(zip(data_cols, titles)):
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
  python analyze_imu.py data.csv
  python analyze_imu.py data.csv --plot
  python analyze_imu.py data.csv --plot --save
        """
    )

    parser.add_argument('csv_file', type=str,
                        help='Input CSV file from imu_logger.py')
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

    # Check if file exists
    if not Path(args.csv_file).exists():
        print(f"ERROR: File not found: {args.csv_file}")
        return

    # Load and process data
    print(f"\nLoading data from: {args.csv_file}")
    df = load_imu_data(args.csv_file)
    print(f"Loaded {len(df)} samples")

    df = convert_to_physical_units(df)

    # Analyze noise
    analyze_noise(df)

    # Test normality if requested
    if args.normality:
        test_normality(df)

    # Generate plots if requested
    if args.plot:
        output_file = args.csv_file.replace('.csv', '_timeseries.png') if args.save else None
        plot_data(df, output_file)

    if args.histogram:
        output_file = args.csv_file.replace('.csv', '_histogram.png') if args.save else None
        plot_noise_histogram(df, output_file)

    if args.qq:
        output_file = args.csv_file.replace('.csv', '_qq.png') if args.save else None
        plot_qq(df, output_file)


if __name__ == "__main__":
    main()
