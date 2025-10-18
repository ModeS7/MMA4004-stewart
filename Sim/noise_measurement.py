import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def analyze_measurement_noise(csv_file):
    """
    Analyze measurement noise from Pixy camera ball tracking data.
    Returns noise statistics for Kalman filter R matrix.
    """
    # Load data
    df = pd.read_csv(csv_file)

    # Filter only when ball is detected
    df_detected = df[df['ball_detected'] == True].copy()

    print(f"Total samples: {len(df)}")
    print(f"Ball detected samples: {len(df_detected)}")
    print(f"Detection rate: {len(df_detected) / len(df) * 100:.1f}%")
    print(f"Duration: {df['time'].max() - df['time'].min():.2f} seconds")
    print(f"Sample rate: {len(df) / (df['time'].max() - df['time'].min()):.1f} Hz\n")

    # Extract position data
    x_data = df_detected['ball_x_mm'].values
    y_data = df_detected['ball_y_mm'].values

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

    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Time series
    axes[0, 0].plot(df_detected['time'], x_data, 'b.-', alpha=0.6, markersize=3)
    axes[0, 0].axhline(stats_dict['ball_x_mm']['mean'], color='r', linestyle='--', label='Mean')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('X Position (mm)')
    axes[0, 0].set_title('X Position Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[1, 0].plot(df_detected['time'], y_data, 'g.-', alpha=0.6, markersize=3)
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
    plt.savefig('noise_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Output for Kalman filter
    print("=== FOR KALMAN FILTER IMPLEMENTATION ===")
    print(f"Measurement noise covariance R:")
    print(f"R = [[{cov_matrix[0, 0]:.6f}, {cov_matrix[0, 1]:.6f}],")
    print(f"     [{cov_matrix[1, 0]:.6f}, {cov_matrix[1, 1]:.6f}]]")
    print(f"\nIf assuming independent axes:")
    print(f"R = diag([{stats_dict['ball_x_mm']['var']:.6f}, {stats_dict['ball_y_mm']['var']:.6f}])")

    return stats_dict, cov_matrix


# Run analysis
if __name__ == "__main__":
    stats, R = analyze_measurement_noise('step_response_20251017_180850.csv')