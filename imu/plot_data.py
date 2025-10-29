#!/usr/bin/env python3
"""
Plot IMU data from logged CSV files

Displays accelerometer and gyroscope data over time with physical unit conversion.
Optionally processes through Kalman filter to estimate orientation (like SI.py).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import argparse
from pathlib import Path
from typing import Tuple
from collections import deque
import glob
import sys

# Add parent directory to path to import core modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.core import StewartPlatformIK, FirstOrderServo
from core.utils import SimulationConfig

# Set dark theme
plt.style.use('dark_background')


# ============================================================================
# ORIENTATION KALMAN FILTER (from SI.py)
# ============================================================================

class OrientationKalmanFilter:
    """Extended Kalman Filter for roll and pitch estimation from IMU (like SI.py).

    State vector: [roll, pitch, gyro_bias_x, gyro_bias_y]

    Features:
        - Automatic gravity vector zeroing at initialization
        - Gyroscope bias estimation
        - Removes initial gravity offset to make orientation relative
    """

    def __init__(self, accel_noise=1.0, gyro_noise=1.0, process_noise_angle=0.0, process_noise_bias=0.0):
        # State: [roll, pitch, gyro_bias_x, gyro_bias_y]
        self.state = np.zeros(4)
        self.P = np.eye(4) * 0.1

        # Process noise covariance
        self.Q = np.diag([
            process_noise_angle,
            process_noise_angle,
            process_noise_bias,
            process_noise_bias
        ])

        # Measurement noise covariance
        self.R = np.diag([
            accel_noise ** 2,
            accel_noise ** 2
        ])

        self.initialized = False
        self.initial_accel = None

        # IMU scaling
        self.accel_scale = 0.001 * 9.81  # LSM303: 1mg/LSB -> m/s²
        self.gyro_scale = 0.00875 * np.pi / 180  # L3GD20: 8.75 mdps/LSB -> rad/s

    def initialize(self, accel_raw):
        """Initialize filter state from first accelerometer reading (raw LSB).

        Args:
            accel_raw: Initial acceleration measurement [LSB]
        """
        if not self.initialized:
            # Convert to m/s²
            accel = accel_raw * self.accel_scale
            ax, ay, az = accel

            roll0 = np.arctan2(ay, az)
            pitch0 = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2))

            self.state[0] = roll0
            self.state[1] = pitch0
            self.initial_accel = accel.copy()
            self.initialized = True

    def predict(self, gyro_raw, dt):
        """Prediction step using gyroscope measurements (raw LSB).

        Args:
            gyro_raw: Angular velocity measurement [LSB]
            dt: Time step [s]
        """
        # Convert to rad/s
        gyro = gyro_raw * self.gyro_scale
        gx, gy = gyro[0], gyro[1]

        # Bias-corrected angular velocity
        gx_corrected = gx - self.state[2]
        gy_corrected = gy - self.state[3]

        # State propagation
        self.state[0] += gx_corrected * dt
        self.state[1] += gy_corrected * dt

        # Jacobian of state transition
        F = np.array([
            [1, 0, -dt, 0],
            [0, 1, 0, -dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Covariance propagation
        self.P = F @ self.P @ F.T + self.Q

    def update(self, accel_raw):
        """Update step using accelerometer measurements (raw LSB).

        Args:
            accel_raw: Acceleration measurement [LSB]
        """
        # Convert to m/s²
        accel = accel_raw * self.accel_scale
        ax, ay, az = accel

        # Tilt angles from accelerometer
        roll_meas = np.arctan2(ay, az)
        pitch_meas = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2))

        # Remove initial gravity offset (KEY: makes orientation relative to start)
        if self.initial_accel is not None:
            roll_init = np.arctan2(self.initial_accel[1], self.initial_accel[2])
            pitch_init = np.arctan2(-self.initial_accel[0],
                                    np.sqrt(self.initial_accel[1] ** 2 + self.initial_accel[2] ** 2))
            roll_meas -= roll_init
            pitch_meas -= pitch_init

        z = np.array([roll_meas, pitch_meas])

        # Measurement matrix
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Innovation
        y = z - H @ self.state
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # State and covariance update
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def get_orientation(self):
        """Return current orientation estimate.

        Returns:
            Tuple of (roll, pitch) in radians
        """
        return self.state[0], self.state[1]


# LSM303DLHC Accelerometer: ±2g range, 12-bit resolution
ACCEL_SENSITIVITY = 1000.0  # LSB/g (1 mg/LSB)
ACCEL_SCALE = 0.001 * 9.81  # 1 mg/LSB → m/s²

# L3GD20 Gyroscope: ±250°/s range
GYRO_SENSITIVITY = 1.0 / 0.00875  # LSB/(°/s) = 114.29
GYRO_SCALE = 0.00875 * np.pi / 180  # 8.75 mdps/LSB → rad/s

# Kalman Filter Configuration (tuned with measured IMU noise)
ACCEL_NOISE = 0.0872   # sqrt(0.007602) - measured from analyze_imu.py
GYRO_NOISE = 0.0174    # sqrt(0.00030214) - measured from analyze_imu.py
PROCESS_NOISE_ANGLE = 0.001  # Allow small gyro drift
PROCESS_NOISE_BIAS = 1e-6    # Slow bias adaptation


# Note: Using IMUKalmanFilter from core.control_core (imported above)


# ============================================================================
# DATA LOADING AND CONVERSION
# ============================================================================

def load_imu_data(csv_file):
    """Load IMU data from CSV file

    Args:
        csv_file: Path to CSV file

    Returns:
        DataFrame with IMU data
    """
    df = pd.read_csv(csv_file)
    return df


def convert_to_physical_units(df):
    """Convert raw IMU values to physical units

    Args:
        df: DataFrame with raw IMU data

    Returns:
        DataFrame with added physical unit columns
    """
    # Separate accel and gyro data
    accel_mask = df['type'] == 'A'
    gyro_mask = df['type'] == 'G'

    # Convert accelerometer to m/s² (only for accel rows)
    df.loc[accel_mask, 'ax_ms2'] = (df.loc[accel_mask, 'x'] / ACCEL_SENSITIVITY) * 9.81
    df.loc[accel_mask, 'ay_ms2'] = (df.loc[accel_mask, 'y'] / ACCEL_SENSITIVITY) * 9.81
    df.loc[accel_mask, 'az_ms2'] = (df.loc[accel_mask, 'z'] / ACCEL_SENSITIVITY) * 9.81

    # Convert gyroscope to rad/s (only for gyro rows)
    df.loc[gyro_mask, 'gx_rads'] = (df.loc[gyro_mask, 'x'] / GYRO_SENSITIVITY) * (np.pi / 180)
    df.loc[gyro_mask, 'gy_rads'] = (df.loc[gyro_mask, 'y'] / GYRO_SENSITIVITY) * (np.pi / 180)
    df.loc[gyro_mask, 'gz_rads'] = (df.loc[gyro_mask, 'z'] / GYRO_SENSITIVITY) * (np.pi / 180)

    return df


def print_statistics(df):
    """Print IMU data statistics

    Args:
        df: DataFrame with IMU data
    """
    # Sort by timestamp for accurate duration calculation
    accel_df = df[df['type'] == 'A'].sort_values('timestamp_arduino_us')
    gyro_df = df[df['type'] == 'G'].sort_values('timestamp_arduino_us')

    print("\n" + "="*60)
    print("IMU DATA STATISTICS")
    print("="*60)

    print(f"\nTotal samples: {len(df)}")
    print(f"  Accelerometer: {len(accel_df)}")
    print(f"  Gyroscope: {len(gyro_df)}")

    duration = (df['timestamp_arduino_us'].iloc[-1] - df['timestamp_arduino_us'].iloc[0]) / 1e6
    print(f"\nDuration: {duration:.2f} seconds")

    if len(accel_df) > 0:
        accel_rate = len(accel_df) / duration
        print(f"Accelerometer rate: {accel_rate:.1f} Hz")

    if len(gyro_df) > 0:
        gyro_rate = len(gyro_df) / duration
        print(f"Gyroscope rate: {gyro_rate:.1f} Hz")

    print("\nAccelerometer (m/s²):")
    print(f"  X-axis: mean={accel_df['ax_ms2'].mean():8.4f}, std={accel_df['ax_ms2'].std():8.4f}")
    print(f"  Y-axis: mean={accel_df['ay_ms2'].mean():8.4f}, std={accel_df['ay_ms2'].std():8.4f}")
    print(f"  Z-axis: mean={accel_df['az_ms2'].mean():8.4f}, std={accel_df['az_ms2'].std():8.4f}")

    print("\nGyroscope (rad/s):")
    print(f"  X-axis: mean={gyro_df['gx_rads'].mean():8.6f}, std={gyro_df['gx_rads'].std():8.6f}")
    print(f"  Y-axis: mean={gyro_df['gy_rads'].mean():8.6f}, std={gyro_df['gy_rads'].std():8.6f}")
    print(f"  Z-axis: mean={gyro_df['gz_rads'].mean():8.6f}, std={gyro_df['gz_rads'].std():8.6f}")


def run_kalman_filter(accel_df, gyro_df, all_data, accel_noise, gyro_noise, proc_noise_angle, proc_noise_bias):
    """Run Kalman filter with given parameters (using SI.py approach)

    Args:
        accel_df: Accelerometer dataframe
        gyro_df: Gyroscope dataframe
        all_data: Interleaved sensor data
        accel_noise: Accelerometer measurement noise (m/s²)
        gyro_noise: Gyroscope measurement noise (rad/s)
        proc_noise_angle: Process noise for angle states
        proc_noise_bias: Process noise for bias states

    Returns:
        Tuple of (rx_est, ry_est, bias_x, bias_y, time, platform_rx, platform_ry)
    """
    # Create OrientationKalmanFilter (from SI.py) with custom noise parameters
    kalman = OrientationKalmanFilter(
        accel_noise=accel_noise,
        gyro_noise=gyro_noise,
        process_noise_angle=proc_noise_angle,
        process_noise_bias=proc_noise_bias
    )

    # Initialize from first accelerometer reading
    if len(accel_df) > 0:
        first_accel = accel_df.iloc[0]
        accel_raw = np.array([first_accel['x'], first_accel['y'], first_accel['z']])
        kalman.initialize(accel_raw)
        print(f"Kalman initialized from gravity: rx={np.degrees(kalman.state[0]):.2f}°, ry={np.degrees(kalman.state[1]):.2f}°")

    # Process through Kalman filter
    rx_est = np.zeros(len(accel_df))
    ry_est = np.zeros(len(accel_df))
    bias_x = np.zeros(len(accel_df))
    bias_y = np.zeros(len(accel_df))

    accel_est_idx = 0
    prev_time = 0.0

    for sensor_type, idx, row in all_data:
        current_time = (row['timestamp_arduino_us'] - accel_df.iloc[0]['timestamp_arduino_us']) / 1e6
        dt = current_time - prev_time if prev_time > 0 else 0.001
        dt = max(dt, 0.0001)  # Prevent zero dt

        if sensor_type == 'A':
            # Accelerometer update
            accel_raw = np.array([row['x'], row['y'], row['z']])
            kalman.update(accel_raw)

            # Store estimate (state is [rx, ry, bias_gx, bias_gy] in radians/rad/s)
            rx_est[accel_est_idx] = kalman.state[0]
            ry_est[accel_est_idx] = kalman.state[1]
            bias_x[accel_est_idx] = kalman.state[2]
            bias_y[accel_est_idx] = kalman.state[3]
            accel_est_idx += 1

        else:
            # Gyroscope prediction
            gyro_raw = np.array([row['x'], row['y'], row['z']])
            kalman.predict(gyro_raw, dt)

        prev_time = current_time

    time = (accel_df['timestamp_arduino_us'].values - accel_df['timestamp_arduino_us'].values[0]) / 1e6
    platform_rx = accel_df['platform_rx'].values
    platform_ry = accel_df['platform_ry'].values

    return rx_est, ry_est, bias_x, bias_y, time, platform_rx, platform_ry


def compute_fk_from_servos(df, stewart_ik):
    """Compute forward kinematics from commanded servo angles (no dynamics)

    Uses commanded platform translation from CSV (platform_x, platform_y, platform_z)
    combined with FK rotation from servo angles.

    Args:
        df: DataFrame with servo angle columns (servo0-servo5) and platform position
        stewart_ik: StewartPlatformIK instance

    Returns:
        Arrays of (x, y, z, rx, ry, rz) from FK
    """
    # Use accelerometer timestamps for consistency
    accel_df = df[df['type'] == 'A'].copy().sort_values('timestamp_arduino_us').reset_index(drop=True)

    n_samples = len(accel_df)
    fk_x = np.zeros(n_samples)
    fk_y = np.zeros(n_samples)
    fk_z = np.zeros(n_samples)
    fk_rx = np.zeros(n_samples)
    fk_ry = np.zeros(n_samples)
    fk_rz = np.zeros(n_samples)

    success_count = 0
    fail_count = 0

    # Check if translation columns exist
    has_translation = all(col in accel_df.columns for col in ['platform_x', 'platform_y', 'platform_z'])

    for i in range(n_samples):
        row = accel_df.iloc[i]
        servo_angles = np.array([
            np.radians(row['servo0']),
            np.radians(row['servo1']),
            np.radians(row['servo2']),
            np.radians(row['servo3']),
            np.radians(row['servo4']),
            np.radians(row['servo5'])
        ])

        translation, rotation, success, iterations = stewart_ik.calculate_forward_kinematics(
            servo_angles, use_top_surface_offset=True
        )

        if success:
            # Use commanded translation if available, otherwise use FK translation
            if has_translation:
                fk_x[i] = row['platform_x']
                fk_y[i] = row['platform_y']
                fk_z[i] = row['platform_z']
            else:
                fk_x[i], fk_y[i], fk_z[i] = translation

            fk_rx[i], fk_ry[i], fk_rz[i] = rotation
            success_count += 1
        else:
            # Use NaN for failed FK
            fk_x[i] = fk_y[i] = fk_z[i] = np.nan
            fk_rx[i] = fk_ry[i] = fk_rz[i] = np.nan
            fail_count += 1

    print(f"Commanded FK Results: {success_count} succeeded, {fail_count} failed out of {n_samples} samples")
    if has_translation:
        print(f"  Using commanded translation from CSV (platform_x, platform_y, platform_z)")

    if fail_count > 0 and success_count == 0:
        print("WARNING: All commanded FK computations failed - check servo angle data in CSV")

    return fk_x, fk_y, fk_z, fk_rx, fk_ry, fk_rz


def compute_servo_dynamics_fk(df, stewart_ik):
    """Compute FK from commanded angles processed through FirstOrderServo dynamics

    Uses commanded platform translation from CSV (platform_x, platform_y, platform_z)
    combined with FK rotation from servo angles after dynamics.

    Args:
        df: DataFrame with servo angle columns (servo0-servo5), timestamps, and platform position
        stewart_ik: StewartPlatformIK instance

    Returns:
        Arrays of (x, y, z, rx, ry, rz) from FK with servo dynamics
    """
    # Use accelerometer timestamps for consistency
    accel_df = df[df['type'] == 'A'].copy().sort_values('timestamp_arduino_us').reset_index(drop=True)

    n_samples = len(accel_df)

    # Initialize 6 servo models with measured parameters
    servos = [
        FirstOrderServo(
            K=1.0,
            tau=SimulationConfig.DEFAULT_SERVO_TAU,
            delay=SimulationConfig.DEFAULT_SERVO_DELAY,
            max_velocity=SimulationConfig.DEFAULT_SERVO_MAX_VELOCITY
        )
        for _ in range(6)
    ]

    # Initialize servo positions to first commanded angles
    if n_samples > 0:
        first_row = accel_df.iloc[0]
        for i in range(6):
            servos[i].current_angle = first_row[f'servo{i}']
            servos[i].target_angle = first_row[f'servo{i}']

    # Storage for results
    servo_x = np.zeros(n_samples)
    servo_y = np.zeros(n_samples)
    servo_z = np.zeros(n_samples)
    servo_rx = np.zeros(n_samples)
    servo_ry = np.zeros(n_samples)
    servo_rz = np.zeros(n_samples)

    success_count = 0
    fail_count = 0

    # Check if translation columns exist
    has_translation = all(col in accel_df.columns for col in ['platform_x', 'platform_y', 'platform_z'])

    # Process each timestamp
    for i in range(n_samples):
        row = accel_df.iloc[i]
        current_time = (row['timestamp_arduino_us'] - accel_df.iloc[0]['timestamp_arduino_us']) / 1e6

        # Calculate dt
        if i > 0:
            prev_time = (accel_df.iloc[i-1]['timestamp_arduino_us'] - accel_df.iloc[0]['timestamp_arduino_us']) / 1e6
            dt = current_time - prev_time
        else:
            dt = 0.001  # Initial dt

        # Send commands to servos
        for j in range(6):
            commanded_angle = row[f'servo{j}']
            servos[j].send_command(commanded_angle, current_time)

        # Update servo dynamics
        for servo in servos:
            servo.update(dt, current_time)

        # Get actual servo angles after dynamics
        servo_angles = np.array([np.radians(servo.get_angle()) for servo in servos])

        # Compute FK from actual servo positions
        translation, rotation, success, iterations = stewart_ik.calculate_forward_kinematics(
            servo_angles, use_top_surface_offset=True
        )

        if success:
            # Use commanded translation if available, otherwise use FK translation
            if has_translation:
                servo_x[i] = row['platform_x']
                servo_y[i] = row['platform_y']
                servo_z[i] = row['platform_z']
            else:
                servo_x[i], servo_y[i], servo_z[i] = translation

            servo_rx[i], servo_ry[i], servo_rz[i] = rotation
            success_count += 1
        else:
            servo_x[i] = servo_y[i] = servo_z[i] = np.nan
            servo_rx[i] = servo_ry[i] = servo_rz[i] = np.nan
            fail_count += 1

    print(f"Servo Dynamics FK Results: {success_count} succeeded, {fail_count} failed out of {n_samples} samples")
    if has_translation:
        print(f"  Using commanded translation from CSV (platform_x, platform_y, platform_z)")

    return servo_x, servo_y, servo_z, servo_rx, servo_ry, servo_rz


def process_kalman_filter(df, output_file=None):
    """Interactive platform state comparison: Commanded FK vs Servo FK vs Kalman

    Shows three estimates of platform state:
    - Commanded FK: Direct FK from commanded servo angles (no dynamics)
    - Servo FK: FK from commanded angles processed through first-order servo dynamics
    - Kalman: IMU-based orientation estimate using Extended Kalman Filter

    Args:
        df: DataFrame with IMU data
        output_file: Optional path to save plot
    """
    print("\n" + "="*60)
    print("PLATFORM STATE COMPARISON: COMMANDED vs SERVO vs KALMAN")
    print("="*60)

    # Initialize Stewart platform IK with same parameters as base_simulator
    platform_params = {
        "horn_length": 45.3722,
        "rod_length": 205.0,
        "base": 86.6025 + 18.75 + 11,
        "base_anchors": 64.75 - 45.3722,
        "platform": 84.0759,
        "platform_anchors": 12.5,
        "top_surface_offset": 38.0
    }
    stewart_ik = StewartPlatformIK(**platform_params)
    print(f"Using platform parameters from base_simulator:")

    # Separate and sort accel and gyro data by timestamp
    accel_df = df[df['type'] == 'A'].copy().sort_values('timestamp_arduino_us').reset_index(drop=True)
    gyro_df = df[df['type'] == 'G'].copy().sort_values('timestamp_arduino_us').reset_index(drop=True)

    print(f"\nAccelerometer samples: {len(accel_df)}")
    print(f"Gyroscope samples: {len(gyro_df)}")

    # Compute FK from commanded servo angles (no dynamics)
    print("\nComputing forward kinematics from commanded servo angles...")
    cmd_fk_x, cmd_fk_y, cmd_fk_z, cmd_fk_rx, cmd_fk_ry, cmd_fk_rz = compute_fk_from_servos(df, stewart_ik)

    # Compute FK from servo angles with first-order dynamics
    print("\nComputing forward kinematics with servo dynamics...")
    servo_fk_x, servo_fk_y, servo_fk_z, servo_fk_rx, servo_fk_ry, servo_fk_rz = compute_servo_dynamics_fk(df, stewart_ik)
    print("FK computation complete")

    # Interleave accel and gyro based on timestamps
    print("\nInterleaving sensor data...")
    all_data = []
    accel_idx = 0
    gyro_idx = 0

    while accel_idx < len(accel_df) or gyro_idx < len(gyro_df):
        if accel_idx >= len(accel_df):
            row = gyro_df.iloc[gyro_idx]
            all_data.append(('G', gyro_idx, row))
            gyro_idx += 1
        elif gyro_idx >= len(gyro_df):
            row = accel_df.iloc[accel_idx]
            all_data.append(('A', accel_idx, row))
            accel_idx += 1
        else:
            accel_time = accel_df.iloc[accel_idx]['timestamp_arduino_us']
            gyro_time = gyro_df.iloc[gyro_idx]['timestamp_arduino_us']

            if accel_time <= gyro_time:
                row = accel_df.iloc[accel_idx]
                all_data.append(('A', accel_idx, row))
                accel_idx += 1
            else:
                row = gyro_df.iloc[gyro_idx]
                all_data.append(('G', gyro_idx, row))
                gyro_idx += 1

    print(f"\nInitial Filter Configuration:")
    print(f"  Accel noise: {ACCEL_NOISE:.4f} m/s²")
    print(f"  Gyro noise: {GYRO_NOISE:.6f} rad/s")
    print(f"  Process noise (angle): {PROCESS_NOISE_ANGLE:.6f} rad²")
    print(f"  Process noise (bias): {PROCESS_NOISE_BIAS:.8f} (rad/s)²")
    print(f"\nAdjust sliders to tune filter parameters...")

    # Create figure with 6 subplots for 6 DOF
    fig = plt.figure(figsize=(18, 12), facecolor='#1e1e1e')

    # Create 3x2 grid for 6 DOF plots - use figure fractions to avoid overlap
    gs = fig.add_gridspec(3, 2, left=0.08, right=0.96, bottom=0.22, top=0.96, hspace=0.35, wspace=0.25)

    ax_x = fig.add_subplot(gs[0, 0], facecolor='#1e1e1e')
    ax_y = fig.add_subplot(gs[0, 1], facecolor='#1e1e1e')
    ax_z = fig.add_subplot(gs[1, 0], facecolor='#1e1e1e')
    ax_rx = fig.add_subplot(gs[1, 1], facecolor='#1e1e1e')
    ax_ry = fig.add_subplot(gs[2, 0], facecolor='#1e1e1e')
    ax_rz = fig.add_subplot(gs[2, 1], facecolor='#1e1e1e')

    axes_list = [ax_x, ax_y, ax_z, ax_rx, ax_ry, ax_rz]

    # Create sliders with dark theme styling at bottom
    ax_accel = plt.axes([0.15, 0.12, 0.35, 0.018], facecolor='#2e2e2e')
    slider_accel = Slider(
        ax=ax_accel,
        label='Accel Noise [m/s²]',
        valmin=0.001,
        valmax=0.5,
        valinit=ACCEL_NOISE,
        valstep=0.001,
        color='#4ecdc4',
        track_color='#2e2e2e'
    )
    slider_accel.label.set_color('white')
    slider_accel.valtext.set_color('white')

    ax_gyro = plt.axes([0.55, 0.12, 0.35, 0.018], facecolor='#2e2e2e')
    slider_gyro = Slider(
        ax=ax_gyro,
        label='Gyro Noise [rad/s]',
        valmin=0.001,
        valmax=0.1,
        valinit=GYRO_NOISE,
        valstep=0.0001,
        color='#4ecdc4',
        track_color='#2e2e2e'
    )
    slider_gyro.label.set_color('white')
    slider_gyro.valtext.set_color('white')

    ax_proc_angle = plt.axes([0.15, 0.08, 0.35, 0.018], facecolor='#2e2e2e')
    slider_proc_angle = Slider(
        ax=ax_proc_angle,
        label='Proc Noise Angle [rad²]',
        valmin=0.0,
        valmax=0.01,
        valinit=PROCESS_NOISE_ANGLE,
        valstep=0.0001,
        color='#4ecdc4',
        track_color='#2e2e2e'
    )
    slider_proc_angle.label.set_color('white')
    slider_proc_angle.valtext.set_color('white')

    ax_proc_bias = plt.axes([0.55, 0.08, 0.35, 0.018], facecolor='#2e2e2e')
    slider_proc_bias = Slider(
        ax=ax_proc_bias,
        label='Proc Noise Bias [(rad/s)²]',
        valmin=0.0,
        valmax=1e-4,
        valinit=PROCESS_NOISE_BIAS,
        valstep=1e-7,
        color='#4ecdc4',
        track_color='#2e2e2e'
    )
    slider_proc_bias.label.set_color('white')
    slider_proc_bias.valtext.set_color('white')

    # File loading controls and reset button at bottom
    file_textbox_ax = plt.axes([0.05, 0.03, 0.42, 0.022], facecolor='#2e2e2e')
    file_browse_ax = plt.axes([0.48, 0.03, 0.08, 0.022], facecolor='#2e2e2e')
    file_load_ax = plt.axes([0.57, 0.03, 0.08, 0.022], facecolor='#2e2e2e')
    reset_ax = plt.axes([0.66, 0.03, 0.08, 0.022], facecolor='#2e2e2e')

    # Get directory of current CSV file for default pattern
    csv_dir = Path.cwd() / 'csv'
    file_textbox = TextBox(file_textbox_ax, 'File:', initial=str(csv_dir / '*.csv'), color='#2e2e2e', hovercolor='#3e3e3e')
    file_textbox.label.set_color('white')
    file_textbox.text_disp.set_color('white')

    button_browse = Button(file_browse_ax, 'Browse', color='#2e2e2e', hovercolor='#4ecdc4')
    button_browse.label.set_color('white')

    button_load = Button(file_load_ax, 'Load', color='#2e2e2e', hovercolor='#4ecdc4')
    button_load.label.set_color('white')

    button_reset = Button(reset_ax, 'Reset', color='#2e2e2e', hovercolor='#4ecdc4')
    button_reset.label.set_color('white')

    # Store plot lines and data for updating
    lines = {}
    data_store = {
        'accel_df': accel_df,
        'gyro_df': gyro_df,
        'all_data': all_data,
        'cmd_fk_x': cmd_fk_x,
        'cmd_fk_y': cmd_fk_y,
        'cmd_fk_z': cmd_fk_z,
        'cmd_fk_rx': cmd_fk_rx,
        'cmd_fk_ry': cmd_fk_ry,
        'cmd_fk_rz': cmd_fk_rz,
        'servo_fk_x': servo_fk_x,
        'servo_fk_y': servo_fk_y,
        'servo_fk_z': servo_fk_z,
        'servo_fk_rx': servo_fk_rx,
        'servo_fk_ry': servo_fk_ry,
        'servo_fk_rz': servo_fk_rz,
        'stewart_ik': stewart_ik
    }

    def browse_file(event):
        """Open file dialog to select CSV file"""
        import tkinter as tk
        from tkinter import filedialog

        # Create hidden tkinter root window
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        # Open file dialog
        csv_dir = Path.cwd() / 'csv'
        filename = filedialog.askopenfilename(
            title='Select IMU CSV file',
            initialdir=csv_dir,
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
        )

        if filename:
            file_textbox.set_val(filename)

        root.destroy()

    def load_file(text):
        """Load new CSV file and update all plots"""
        file_pattern = file_textbox.text.strip()

        # Support glob patterns - use most recent file if pattern
        matching_files = sorted(glob.glob(file_pattern))
        if not matching_files:
            print(f"ERROR: No files found matching: {file_pattern}")
            return

        csv_file = matching_files[-1]  # Use most recent
        print(f"\nLoading file: {csv_file}")

        try:
            # Load and process data
            new_df = load_imu_data(csv_file)
            new_df = convert_to_physical_units(new_df)

            # Separate and sort by timestamp
            new_accel_df = new_df[new_df['type'] == 'A'].copy().sort_values('timestamp_arduino_us').reset_index(drop=True)
            new_gyro_df = new_df[new_df['type'] == 'G'].copy().sort_values('timestamp_arduino_us').reset_index(drop=True)

            print(f"Loaded {len(new_accel_df)} accel samples, {len(new_gyro_df)} gyro samples")

            # Interleave accel and gyro based on timestamps
            new_all_data = []
            accel_idx = 0
            gyro_idx = 0

            while accel_idx < len(new_accel_df) or gyro_idx < len(new_gyro_df):
                if accel_idx >= len(new_accel_df):
                    row = new_gyro_df.iloc[gyro_idx]
                    new_all_data.append(('G', gyro_idx, row))
                    gyro_idx += 1
                elif gyro_idx >= len(new_gyro_df):
                    row = new_accel_df.iloc[accel_idx]
                    new_all_data.append(('A', accel_idx, row))
                    accel_idx += 1
                else:
                    accel_time = new_accel_df.iloc[accel_idx]['timestamp_arduino_us']
                    gyro_time = new_gyro_df.iloc[gyro_idx]['timestamp_arduino_us']

                    if accel_time <= gyro_time:
                        row = new_accel_df.iloc[accel_idx]
                        new_all_data.append(('A', accel_idx, row))
                        accel_idx += 1
                    else:
                        row = new_gyro_df.iloc[gyro_idx]
                        new_all_data.append(('G', gyro_idx, row))
                        gyro_idx += 1

            # Compute FK for new data (both commanded and servo dynamics)
            print("Computing forward kinematics...")
            stewart_ik = data_store['stewart_ik']
            new_cmd_fk_x, new_cmd_fk_y, new_cmd_fk_z, new_cmd_fk_rx, new_cmd_fk_ry, new_cmd_fk_rz = compute_fk_from_servos(new_df, stewart_ik)
            new_servo_fk_x, new_servo_fk_y, new_servo_fk_z, new_servo_fk_rx, new_servo_fk_ry, new_servo_fk_rz = compute_servo_dynamics_fk(new_df, stewart_ik)

            # Update data store
            data_store['accel_df'] = new_accel_df
            data_store['gyro_df'] = new_gyro_df
            data_store['all_data'] = new_all_data
            data_store['cmd_fk_x'] = new_cmd_fk_x
            data_store['cmd_fk_y'] = new_cmd_fk_y
            data_store['cmd_fk_z'] = new_cmd_fk_z
            data_store['cmd_fk_rx'] = new_cmd_fk_rx
            data_store['cmd_fk_ry'] = new_cmd_fk_ry
            data_store['cmd_fk_rz'] = new_cmd_fk_rz
            data_store['servo_fk_x'] = new_servo_fk_x
            data_store['servo_fk_y'] = new_servo_fk_y
            data_store['servo_fk_z'] = new_servo_fk_z
            data_store['servo_fk_rx'] = new_servo_fk_rx
            data_store['servo_fk_ry'] = new_servo_fk_ry
            data_store['servo_fk_rz'] = new_servo_fk_rz

            # Clear existing plot lines to force redraw
            lines.clear()

            # Update plots with new data
            update_plot(None)

            print("File loaded successfully")

        except Exception as e:
            print(f"ERROR loading file: {e}")
            import traceback
            traceback.print_exc()

    def update_plot(val):
        """Update plot with new Kalman filter parameters"""
        # Get current slider values
        accel_noise = slider_accel.val
        gyro_noise = slider_gyro.val
        proc_noise_angle = slider_proc_angle.val
        proc_noise_bias = slider_proc_bias.val

        # Get data from store (allows file reloading)
        accel_df = data_store['accel_df']
        gyro_df = data_store['gyro_df']
        all_data = data_store['all_data']
        cmd_fk_x = data_store['cmd_fk_x']
        cmd_fk_y = data_store['cmd_fk_y']
        cmd_fk_z = data_store['cmd_fk_z']
        cmd_fk_rx = data_store['cmd_fk_rx']
        cmd_fk_ry = data_store['cmd_fk_ry']
        cmd_fk_rz = data_store['cmd_fk_rz']
        servo_fk_x = data_store['servo_fk_x']
        servo_fk_y = data_store['servo_fk_y']
        servo_fk_z = data_store['servo_fk_z']
        servo_fk_rx = data_store['servo_fk_rx']
        servo_fk_ry = data_store['servo_fk_ry']
        servo_fk_rz = data_store['servo_fk_rz']

        # Run Kalman filter (only for orientation)
        rx_est, ry_est, bias_x, bias_y, time, platform_rx, platform_ry = run_kalman_filter(
            accel_df, gyro_df, all_data, accel_noise, gyro_noise, proc_noise_angle, proc_noise_bias
        )

        # Update plots
        if 'cmd_fk_x' in lines:
            # Update existing plot data for all 6 DOF
            lines['cmd_fk_x'].set_data(time, cmd_fk_x)
            lines['servo_fk_x'].set_data(time, servo_fk_x)
            lines['cmd_fk_y'].set_data(time, cmd_fk_y)
            lines['servo_fk_y'].set_data(time, servo_fk_y)
            lines['cmd_fk_z'].set_data(time, cmd_fk_z)
            lines['servo_fk_z'].set_data(time, servo_fk_z)

            lines['cmd_fk_rx'].set_data(time, np.degrees(cmd_fk_rx))
            lines['servo_fk_rx'].set_data(time, np.degrees(servo_fk_rx))
            lines['kalman_rx'].set_data(time, np.degrees(rx_est))

            lines['cmd_fk_ry'].set_data(time, np.degrees(cmd_fk_ry))
            lines['servo_fk_ry'].set_data(time, np.degrees(servo_fk_ry))
            lines['kalman_ry'].set_data(time, np.degrees(ry_est))

            lines['cmd_fk_rz'].set_data(time, np.degrees(cmd_fk_rz))
            lines['servo_fk_rz'].set_data(time, np.degrees(servo_fk_rz))

            # Rescale all axes
            for ax in axes_list:
                ax.relim()
                ax.autoscale_view()
        else:
            # Initial plot or after clearing - clear all axes and redraw
            for ax in axes_list:
                ax.clear()
                ax.set_facecolor('#1e1e1e')

            # Plot Position X - Commanded translation (same for both since position is commanded, not from FK)
            lines['cmd_fk_x'], = ax_x.plot(time, cmd_fk_x, '#c77dff', label='Commanded', linewidth=1.5, alpha=0.7, linestyle=':')
            lines['servo_fk_x'], = ax_x.plot(time, servo_fk_x, '#4ecdc4', label='Commanded', linewidth=2, alpha=0.9)
            ax_x.set_ylabel('X Position [mm]', color='white')
            ax_x.set_title('Position X (Commanded)', color='white', fontsize=11)
            ax_x.legend(loc='best', facecolor='#2e2e2e', edgecolor='#555', fontsize=9)
            ax_x.grid(True, alpha=0.2, color='#555')

            # Plot Position Y
            lines['cmd_fk_y'], = ax_y.plot(time, cmd_fk_y, '#c77dff', label='Commanded', linewidth=1.5, alpha=0.7, linestyle=':')
            lines['servo_fk_y'], = ax_y.plot(time, servo_fk_y, '#4ecdc4', label='Commanded', linewidth=2, alpha=0.9)
            ax_y.set_ylabel('Y Position [mm]', color='white')
            ax_y.set_title('Position Y (Commanded)', color='white', fontsize=11)
            ax_y.legend(loc='best', facecolor='#2e2e2e', edgecolor='#555', fontsize=9)
            ax_y.grid(True, alpha=0.2, color='#555')

            # Plot Position Z
            lines['cmd_fk_z'], = ax_z.plot(time, cmd_fk_z, '#c77dff', label='Commanded', linewidth=1.5, alpha=0.7, linestyle=':')
            lines['servo_fk_z'], = ax_z.plot(time, servo_fk_z, '#4ecdc4', label='Commanded', linewidth=2, alpha=0.9)
            ax_z.set_ylabel('Z Position [mm]', color='white')
            ax_z.set_xlabel('Time [s]', color='white')
            ax_z.set_title('Position Z (Commanded)', color='white', fontsize=11)
            ax_z.legend(loc='best', facecolor='#2e2e2e', edgecolor='#555', fontsize=9)
            ax_z.grid(True, alpha=0.2, color='#555')

            # Plot Orientation RX (Roll) - Commanded vs Servo vs Kalman
            lines['cmd_fk_rx'], = ax_rx.plot(time, np.degrees(cmd_fk_rx), '#c77dff', label='Commanded FK', linewidth=1.5, alpha=0.7, linestyle=':')
            lines['servo_fk_rx'], = ax_rx.plot(time, np.degrees(servo_fk_rx), '#4ecdc4', label='Servo FK', linewidth=2, alpha=0.9)
            lines['kalman_rx'], = ax_rx.plot(time, np.degrees(rx_est), '#ff6b6b', label='Kalman (IMU)', linewidth=2, linestyle='--', alpha=0.9)
            ax_rx.set_ylabel('Roll [deg]', color='white')
            ax_rx.set_xlabel('Time [s]', color='white')
            ax_rx.set_title('Orientation RX (Roll)', color='white', fontsize=11)
            ax_rx.legend(loc='best', facecolor='#2e2e2e', edgecolor='#555', fontsize=8)
            ax_rx.grid(True, alpha=0.2, color='#555')

            # Plot Orientation RY (Pitch) - Commanded vs Servo vs Kalman
            lines['cmd_fk_ry'], = ax_ry.plot(time, np.degrees(cmd_fk_ry), '#c77dff', label='Commanded FK', linewidth=1.5, alpha=0.7, linestyle=':')
            lines['servo_fk_ry'], = ax_ry.plot(time, np.degrees(servo_fk_ry), '#4ecdc4', label='Servo FK', linewidth=2, alpha=0.9)
            lines['kalman_ry'], = ax_ry.plot(time, np.degrees(ry_est), '#ff6b6b', label='Kalman (IMU)', linewidth=2, linestyle='--', alpha=0.9)
            ax_ry.set_ylabel('Pitch [deg]', color='white')
            ax_ry.set_xlabel('Time [s]', color='white')
            ax_ry.set_title('Orientation RY (Pitch)', color='white', fontsize=11)
            ax_ry.legend(loc='best', facecolor='#2e2e2e', edgecolor='#555', fontsize=8)
            ax_ry.grid(True, alpha=0.2, color='#555')

            # Plot Orientation RZ (Yaw) - Commanded vs Servo (no Kalman - no magnetometer)
            lines['cmd_fk_rz'], = ax_rz.plot(time, np.degrees(cmd_fk_rz), '#c77dff', label='Commanded FK', linewidth=1.5, alpha=0.7, linestyle=':')
            lines['servo_fk_rz'], = ax_rz.plot(time, np.degrees(servo_fk_rz), '#4ecdc4', label='Servo FK', linewidth=2, alpha=0.9)
            ax_rz.set_ylabel('Yaw [deg]', color='white')
            ax_rz.set_xlabel('Time [s]', color='white')
            ax_rz.set_title('Orientation RZ (Yaw)', color='white', fontsize=11)
            ax_rz.legend(loc='best', facecolor='#2e2e2e', edgecolor='#555', fontsize=9)
            ax_rz.grid(True, alpha=0.2, color='#555')

        fig.canvas.draw_idle()

    def reset(event):
        """Reset sliders to default values"""
        slider_accel.reset()
        slider_gyro.reset()
        slider_proc_angle.reset()
        slider_proc_bias.reset()

    # Connect sliders to update function
    slider_accel.on_changed(update_plot)
    slider_gyro.on_changed(update_plot)
    slider_proc_angle.on_changed(update_plot)
    slider_proc_bias.on_changed(update_plot)
    button_reset.on_clicked(reset)
    button_browse.on_clicked(browse_file)
    button_load.on_clicked(load_file)

    # Initial plot
    update_plot(None)

    plt.tight_layout()
    plt.show()


def plot_gui_selector():
    """Interactive GUI for selecting file and plot type"""
    from matplotlib.widgets import RadioButtons
    import tkinter as tk
    from tkinter import filedialog

    fig = plt.figure(figsize=(12, 8), facecolor='#1e1e1e')
    ax_main = plt.subplot2grid((3, 2), (0, 0), rowspan=3, colspan=1, facecolor='#1e1e1e')
    ax_controls = plt.subplot2grid((3, 2), (0, 1), rowspan=3, facecolor='#1e1e1e')

    ax_main.text(0.5, 0.5, 'Load CSV file to view\n6-DOF Platform State Comparison\n(Commanded FK vs Servo FK vs Kalman)',
                ha='center', va='center', fontsize=13, color='#4ecdc4', transform=ax_main.transAxes)
    ax_main.axis('off')

    # File loading controls with dark theme
    ax_controls.axis('off')
    file_textbox_ax = plt.axes([0.55, 0.85, 0.35, 0.04], facecolor='#2e2e2e')
    file_browse_ax = plt.axes([0.91, 0.85, 0.08, 0.04], facecolor='#2e2e2e')
    file_load_ax = plt.axes([0.55, 0.78, 0.15, 0.05], facecolor='#2e2e2e')

    csv_dir = Path.cwd() / 'csv'
    file_textbox = TextBox(file_textbox_ax, 'File:', initial=str(csv_dir / '*.csv'),
                          color='#2e2e2e', hovercolor='#3e3e3e')
    file_textbox.label.set_color('white')
    file_textbox.text_disp.set_color('white')

    button_browse = Button(file_browse_ax, 'Browse', color='#2e2e2e', hovercolor='#4ecdc4')
    button_browse.label.set_color('white')

    button_load = Button(file_load_ax, 'Load File', color='#2e2e2e', hovercolor='#4ecdc4')
    button_load.label.set_color('white')

    state = {'df': None, 'csv_file': None}

    def browse_file(event):
        """Open file dialog to select CSV file"""
        # Create hidden tkinter root window
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        # Open file dialog
        csv_dir = Path.cwd() / 'csv'
        filename = filedialog.askopenfilename(
            title='Select IMU CSV file',
            initialdir=csv_dir,
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
        )

        if filename:
            file_textbox.set_val(filename)

        root.destroy()

    def load_and_plot(event):
        """Load file and show selected plot"""
        file_pattern = file_textbox.text.strip()

        # Check if it's a specific file or a glob pattern
        if Path(file_pattern).exists() and Path(file_pattern).is_file():
            csv_file = file_pattern
        else:
            # Support glob patterns
            matching_files = sorted(glob.glob(file_pattern))
            if not matching_files:
                print(f"ERROR: No files found matching: {file_pattern}")
                ax_main.clear()
                ax_main.set_facecolor('#1e1e1e')
                ax_main.text(0.5, 0.5, f'No files found:\n{file_pattern}',
                            ha='center', va='center', fontsize=12, color='#ff6b6b',
                            transform=ax_main.transAxes)
                ax_main.axis('off')
                fig.canvas.draw_idle()
                return
            csv_file = matching_files[-1]

        state['csv_file'] = csv_file

        print(f"\nLoading data from: {csv_file}")

        try:
            df = load_imu_data(csv_file)
            df = convert_to_physical_units(df)
            state['df'] = df

            print(f"Loaded {len(df)} samples")
            print_statistics(df)

            # Close this window and open the comprehensive comparison plot
            plt.close(fig)
            process_kalman_filter(df)

        except Exception as e:
            print(f"ERROR loading file: {e}")
            import traceback
            traceback.print_exc()
            ax_main.clear()
            ax_main.set_facecolor('#1e1e1e')
            ax_main.text(0.5, 0.5, f'Error loading file:\n{str(e)}',
                        ha='center', va='center', fontsize=12, color='#ff6b6b',
                        transform=ax_main.transAxes)
            ax_main.axis('off')
            fig.canvas.draw_idle()

    button_browse.on_clicked(browse_file)
    button_load.on_clicked(load_and_plot)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='IMU data visualization: 6-DOF platform state comparison (Commanded FK vs Servo FK vs Kalman)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_data.py                              (open GUI file selector)
  python plot_data.py data.csv                     (6-subplot comparison view)
  python plot_data.py csv/imu_*.csv                (load latest matching file)
  python plot_data.py data.csv --save              (save plot to file)
  python plot_data.py data.csv --stats             (statistics only)

Comparison shows:
  - Commanded FK: Direct forward kinematics from commanded servo angles
  - Servo FK: FK after processing through first-order servo dynamics model
  - Kalman: IMU-based orientation estimate from Extended Kalman Filter
        """
    )

    parser.add_argument('csv_file', type=str, nargs='?',
                        help='Input CSV file from IMU logger (optional - opens GUI if not provided)')
    parser.add_argument('--save', action='store_true',
                        help='Save plot to file instead of displaying')
    parser.add_argument('--stats', action='store_true',
                        help='Print statistics only (no plots)')

    args = parser.parse_args()

    # If no file provided, open GUI selector
    if args.csv_file is None:
        plot_gui_selector()
        return

    # Check if file exists
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"ERROR: File not found: {args.csv_file}")
        return

    # Load and process data
    print(f"\nLoading data from: {args.csv_file}")
    df = load_imu_data(args.csv_file)
    print(f"Loaded {len(df)} samples")

    df = convert_to_physical_units(df)

    # Print statistics
    print_statistics(df)

    if args.stats:
        return

    # Show comprehensive 6-subplot comparison (Commanded vs FK vs Kalman)
    output_file = args.csv_file.replace('.csv', '_comparison.png') if args.save else None
    process_kalman_filter(df, output_file)


if __name__ == "__main__":
    main()
