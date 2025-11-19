import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def analyze_measurement_noise(csv_file):
    """
    Analyze measurement noise from Pixy camera ball tracking data.
    Returns noise statistics for Kalman filter R matrix.

    Compatible with CSV files from full_c.py performance data collection.
    """
    # Load data
    df = pd.read_csv(csv_file)

    # Filter only valid ball detections
    if 'ball_detected' in df.columns:
        df_detected = df[df['ball_detected'] == True].copy()
        detection_rate = len(df_detected) / len(df) * 100 if len(df) > 0 else 0
    else:
        # Fallback for old CSV files without ball_detected column
        print("WARNING: CSV file doesn't have 'ball_detected' column. Using all data.")
        df_detected = df.copy()
        detection_rate = 100.0

    print(f"Total samples: {len(df)}")
    if 'ball_detected' in df.columns:
        print(f"Ball detected samples: {len(df_detected)}")
        print(f"Detection rate: {detection_rate:.1f}%")
    print(f"Duration: {df['elapsed_time'].max() - df['elapsed_time'].min():.2f} seconds")
    print(f"Sample rate: {len(df) / (df['elapsed_time'].max() - df['elapsed_time'].min()):.1f} Hz")

    if len(df_detected) == 0:
        print("\nERROR: No valid ball detections found in CSV file!")
        print("Make sure the ball is visible to the camera during data collection.")
        return None, None

    if detection_rate < 50.0:
        print(f"\nWARNING: Low detection rate ({detection_rate:.1f}%).")
        print("Results may not be reliable. Ensure good lighting and ball visibility.")

    print()

    # Extract position data (full_c.py uses 'ball_x' and 'ball_y')
    x_data = df_detected['ball_x'].values
    y_data = df_detected['ball_y'].values

    # Calculate statistics
    stats_dict = {
        'ball_x_mm': {
            'mean': np.mean(x_data),
            'std': np.std(x_data, ddof=1),
            'var': np.var(x_data, ddof=1),
            'min': np.min(x_data),
            'max': np.max(x_data),
            'range': np.max(x_data) - np.min(x_data)
        },
        'ball_y_mm': {
            'mean': np.mean(y_data),
            'std': np.std(y_data, ddof=1),
            'var': np.var(y_data, ddof=1),
            'min': np.min(y_data),
            'max': np.max(y_data),
            'range': np.max(y_data) - np.min(y_data)
        }
    }

    print("=== MEASUREMENT NOISE STATISTICS ===\n")
    for axis, stat in stats_dict.items():
        print(f"{axis}:")
        print(f"  Mean: {stat['mean']:.4f} mm")
        print(f"  Std Dev: {stat['std']:.4f} mm")
        print(f"  Variance: {stat['var']:.6f} mm²")
        print(f"  Range: [{stat['min']:.4f}, {stat['max']:.4f}] mm")
        print(f"  Peak-to-peak: {stat['range']:.4f} mm\n")

    # Measurement covariance matrix (R matrix for Kalman filter)
    cov_matrix = np.cov(x_data, y_data)
    print("=== MEASUREMENT COVARIANCE MATRIX (R) ===")
    print(cov_matrix)
    print(f"\nCorrelation coefficient: {np.corrcoef(x_data, y_data)[0, 1]:.4f}\n")

    # Test for normality
    _, p_value_x = stats.normaltest(x_data)
    _, p_value_y = stats.normaltest(y_data)

    print("=== NORMALITY TEST (p > 0.05 suggests Gaussian noise) ===")
    print(f"X-axis p-value: {p_value_x:.4f}")
    print(f"Y-axis p-value: {p_value_y:.4f}\n")

    # Print loop time statistics if available
    if 'loop_time_ms' in df.columns:
        loop_times = df['loop_time_ms'].values
        print("=== LOOP TIME STATISTICS ===")
        print(f"Mean: {np.mean(loop_times):.3f} ms")
        print(f"Std Dev: {np.std(loop_times):.3f} ms")
        print(f"Min: {np.min(loop_times):.3f} ms")
        print(f"Max: {np.max(loop_times):.3f} ms")
        print(f"95th percentile: {np.percentile(loop_times, 95):.3f} ms\n")

    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Time series
    axes[0, 0].plot(df_detected['elapsed_time'], x_data, 'b.-', alpha=0.6, markersize=3)
    axes[0, 0].axhline(stats_dict['ball_x_mm']['mean'], color='r', linestyle='--', label='Mean')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('X Position (mm)')
    axes[0, 0].set_title('X Position Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[1, 0].plot(df_detected['elapsed_time'], y_data, 'g.-', alpha=0.6, markersize=3)
    axes[1, 0].axhline(stats_dict['ball_y_mm']['mean'], color='r', linestyle='--', label='Mean')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Y Position (mm)')
    axes[1, 0].set_title('Y Position Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Histograms with normal fit
    for idx, (data, label, color) in enumerate([(x_data, 'X', 'blue'), (y_data, 'Y', 'green')]):
        ax = axes[idx, 1]
        n, bins, _ = ax.hist(data, bins=30, density=True, alpha=0.6, color=color, edgecolor='black')

        mu, sigma = np.mean(data), np.std(data, ddof=1)
        x_range = np.linspace(data.min(), data.max(), 100)
        ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 'r-', linewidth=2, label='Normal fit')

        ax.set_xlabel(f'{label} Position (mm)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'{label} Position Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Q-Q plots
    for idx, (data, label) in enumerate([(x_data, 'X'), (y_data, 'Y')]):
        ax = axes[idx, 2]
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title(f'{label} Position Q-Q Plot')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot to data/performance directory with timestamp
    from pathlib import Path
    from datetime import datetime
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'performance'
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_file = output_dir / f'noise_analysis_{timestamp}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")

    plt.show()

    # Output for Kalman filter
    print("\n=== FOR KALMAN FILTER IMPLEMENTATION ===")
    print(f"Measurement noise covariance R:")
    print(f"R = [[{cov_matrix[0, 0]:.6f}, {cov_matrix[0, 1]:.6f}],")
    print(f"     [{cov_matrix[1, 0]:.6f}, {cov_matrix[1, 1]:.6f}]]")
    print(f"\nIf assuming independent axes:")
    print(f"R = diag([{stats_dict['ball_x_mm']['var']:.6f}, {stats_dict['ball_y_mm']['var']:.6f}])")

    # Calculate recommended measurement_noise_scale
    baseline_variance = 0.00034  # From KalmanFilter R_base (0.58mm std -> 0.00034 m²)
    avg_measured_variance = (stats_dict['ball_x_mm']['var'] + stats_dict['ball_y_mm']['var']) / 2
    avg_measured_variance_m2 = avg_measured_variance / 1e6  # Convert mm² to m²
    recommended_scale = avg_measured_variance_m2 / baseline_variance

    print(f"\n=== RECOMMENDED KALMAN FILTER TUNING ===")
    print(f"Current baseline R variance: {baseline_variance:.6f} m²")
    print(f"Measured variance (avg): {avg_measured_variance:.6f} mm² = {avg_measured_variance_m2:.6f} m²")
    print(f"Recommended measurement_noise_scale: {recommended_scale:.2f}")

    return stats_dict, cov_matrix


# Run analysis
if __name__ == "__main__":
    import sys

    # Use command line argument or default file
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Default: look for most recent performance CSV file
        from pathlib import Path
        data_dir = Path(__file__).parent.parent.parent / 'data' / 'performance'
        csv_files = list(data_dir.glob('performance_*.csv'))

        if csv_files:
            # Use most recent file
            csv_file = max(csv_files, key=lambda p: p.stat().st_mtime)
            print(f"No file specified. Using most recent: {csv_file.name}\n")
        else:
            print("No CSV file found in data/performance/")
            print("Usage: python noise_measurement.py <path_to_csv_file>")
            sys.exit(1)

    result = analyze_measurement_noise(csv_file)
    if result[0] is None:
        sys.exit(1)