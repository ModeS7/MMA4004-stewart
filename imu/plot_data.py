#!/usr/bin/env python3
"""
Plot IMU data from logged CSV files

Displays accelerometer and gyroscope data over time with physical unit conversion.
Processes through Kalman filter to estimate orientation.
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import glob
import sys

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSlider, QPushButton, QLabel, QLineEdit, QCheckBox, QFileDialog,
                             QGridLayout, QGroupBox)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

# Add parent directory to path to import core modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.core import StewartPlatformIK, FirstOrderServo
from core.utils import SimulationConfig
from core.control_core import OrientationKalmanFilter, GRAVITY_VECTOR, GRAVITY_MAGNITUDE

# PyQtGraph dark theme configuration
pg.setConfigOption('background', '#1e1e1e')
pg.setConfigOption('foreground', 'w')
pg.setConfigOption('antialias', True)


# ============================================================================
# IMU SENSOR PARAMETERS
# ============================================================================

# LSM303DLHC Accelerometer: ±2g range, 12-bit resolution
ACCEL_SENSITIVITY = 1000.0  # LSB/g (1 mg/LSB)
ACCEL_SCALE = 0.001 * 9.81  # 1 mg/LSB → m/s²

# L3GD20H Gyroscope configuration
# Base sensitivity: 8.75 mdps/LSB (±245 dps range - default, no CTRL_REG4 config in firmware)
# Empirically calibrated multiplier: 6.6x (verified with ±30° step rotations in RX/RY)
# Effective sensitivity: 8.75 × 6.6 = 57.75 mdps/LSB
GYRO_SENSITIVITY = 1.0 / 0.00875  # LSB/(°/s) = 114.29 (±245 dps)
GYRO_SCALE = 0.00875 * np.pi / 180  # 8.75 mdps/LSB → rad/s (multiply by 6.6 empirically calibrated)

# Kalman Filter Configuration (measured from stationary IMU data)
# Accelerometer noise (m/s²): X=0.0686, Y=0.0672, Z=0.0924
# Using RMS of X,Y for tilt measurement: sqrt((0.0686² + 0.0672²)/2) ≈ 0.0679
ACCEL_NOISE = 1.0 #0.0679  # m/s² - RMS of X,Y accelerometer noise

# Gyroscope noise (rad/s) - base measurement (will be scaled by gyro_scale_multiplier)
# Raw measurements: X=0.021750, Y=0.023157, Z=0.006303
# Using RMS of X,Y for roll/pitch rates: sqrt((0.021750² + 0.023157²)/2) ≈ 0.0224
GYRO_NOISE = 0.0224  # rad/s - RMS of X,Y gyroscope noise (base, gets scaled by multiplier)

PROCESS_NOISE_ANGLE = 0.0  # Allow small gyro drift
PROCESS_NOISE_BIAS = 0.0    # Slow bias adaptation

# Measured gyroscope biases (rad/s) - from stationary data (RAW sensor frame)
# These values are automatically scaled by gyro_scale_multiplier and transformed by axis flips/rotations
GYRO_BIAS_X = 0.112679  # rad/s (X-axis mean from stationary data, base value)
GYRO_BIAS_Y = 0.031500  # rad/s (Y-axis mean from stationary data, base value)

# IMU Axis Transformation Configuration
ACCEL_AXIS_FLIP = np.array([-1, 1, -1])  # Axis orientation correction
GYRO_AXIS_FLIP = np.array([-1, 1, -1])

# Frame alignment: 3x3 rotation matrices for 90°/180° corrections
# Set to None if no rotation needed
ACCEL_ROTATION = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
GYRO_ROTATION = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

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


def run_kalman_filter(accel_df, gyro_df, all_data, accel_noise, gyro_noise, proc_noise_angle, proc_noise_bias,
                      accel_axis_flip=None, gyro_axis_flip=None, accel_rotation=None, gyro_rotation=None,
                      enable_accel_updates=True, enable_gyro_predictions=True, gyro_scale_multiplier=1.0):
    """Run Kalman filter with given parameters

    Args:
        accel_df: Accelerometer dataframe
        gyro_df: Gyroscope dataframe
        all_data: Interleaved sensor data
        accel_noise: Accelerometer measurement noise (m/s²)
        gyro_noise: Gyroscope measurement noise (rad/s)
        proc_noise_angle: Process noise for angle states
        proc_noise_bias: Process noise for bias states
        accel_axis_flip: Axis flip for accelerometer
        gyro_axis_flip: Axis flip for gyroscope
        accel_rotation: Rotation matrix for accelerometer
        gyro_rotation: Rotation matrix for gyroscope
        enable_accel_updates: If False, skip accelerometer updates (pure gyro integration)
        enable_gyro_predictions: If False, skip gyro predictions (pure accel tilt)
        gyro_scale_multiplier: Multiplier for gyroscope scale calibration

    Returns:
        Tuple of (rx_est, ry_est, bias_x, bias_y, time, platform_rx, platform_ry)
    """
    # Create OrientationKalmanFilter with custom noise parameters and transformations
    kalman = OrientationKalmanFilter(
        accel_noise=accel_noise,
        gyro_noise=gyro_noise,
        process_noise_angle=proc_noise_angle,
        process_noise_bias=proc_noise_bias,
        accel_axis_flip=accel_axis_flip,
        gyro_axis_flip=gyro_axis_flip,
        accel_rotation=accel_rotation,
        gyro_rotation=gyro_rotation,
        initial_bias_x=GYRO_BIAS_X,
        initial_bias_y=GYRO_BIAS_Y,
        gyro_scale_multiplier=gyro_scale_multiplier
    )

    # Initialize from first accelerometer reading
    if len(accel_df) > 0:
        first_accel = accel_df.iloc[0]
        accel_raw = np.array([first_accel['x'], first_accel['y'], first_accel['z']])
        kalman.initialize(accel_raw)
        print(f"Kalman initialized from gravity: rx={np.degrees(kalman.state[0]):.2f}°, ry={np.degrees(kalman.state[1]):.2f}°")
        print(f"  Initial gyro biases: X={kalman.state[2]:.6f} rad/s, Y={kalman.state[3]:.6f} rad/s")
        print(f"  Measured gravity vector: [{GRAVITY_VECTOR[0]:.4f}, {GRAVITY_VECTOR[1]:.4f}, {GRAVITY_VECTOR[2]:.4f}] m/s² (mag={GRAVITY_MAGNITUDE:.4f})")

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
            # Accelerometer update (skip if disabled)
            accel_raw = np.array([row['x'], row['y'], row['z']])
            if enable_accel_updates:
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
            if enable_gyro_predictions:
                kalman.predict(gyro_raw, dt)
            else:
                # Still predict with zero gyro to propagate covariance
                # This keeps the filter responsive to parameter changes
                kalman.predict(np.zeros(3), dt)

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


class IMUPlotWindow(QMainWindow):
    """PyQt6 window for interactive IMU data visualization"""

    def __init__(self, df, stewart_ik):
        super().__init__()
        self.setWindowTitle("IMU Platform State Comparison")
        self.resize(1800, 1000)

        # Initialize data
        self.stewart_ik = stewart_ik
        self.load_data(df)

        # Debounce timer
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_plots)

        # Setup UI
        self.setup_ui()

        # Update initial rotation status
        self.accel_rotation_label.setText(self._describe_rotation_matrix(self.accel_rotation))
        self.gyro_rotation_label.setText(self._describe_rotation_matrix(self.gyro_rotation))

        # Initial plot
        self.update_plots()

    def load_data(self, df):
        """Load and process IMU data"""
        # Separate and sort accel and gyro data by timestamp
        self.accel_df = df[df['type'] == 'A'].copy().sort_values('timestamp_arduino_us').reset_index(drop=True)
        self.gyro_df = df[df['type'] == 'G'].copy().sort_values('timestamp_arduino_us').reset_index(drop=True)

        print(f"\nAccelerometer samples: {len(self.accel_df)}")
        print(f"Gyroscope samples: {len(self.gyro_df)}")

        # Compute FK from commanded servo angles
        print("\nComputing forward kinematics from commanded servo angles...")
        self.cmd_fk_x, self.cmd_fk_y, self.cmd_fk_z, self.cmd_fk_rx, self.cmd_fk_ry, self.cmd_fk_rz = \
            compute_fk_from_servos(df, self.stewart_ik)

        # Compute FK from servo angles with dynamics
        print("\nComputing forward kinematics with servo dynamics...")
        self.servo_fk_x, self.servo_fk_y, self.servo_fk_z, self.servo_fk_rx, self.servo_fk_ry, self.servo_fk_rz = \
            compute_servo_dynamics_fk(df, self.stewart_ik)
        print("FK computation complete")

        # Interleave accel and gyro based on timestamps
        print("\nInterleaving sensor data...")
        self.all_data = []
        accel_idx = 0
        gyro_idx = 0

        while accel_idx < len(self.accel_df) or gyro_idx < len(self.gyro_df):
            if accel_idx >= len(self.accel_df):
                row = self.gyro_df.iloc[gyro_idx]
                self.all_data.append(('G', gyro_idx, row))
                gyro_idx += 1
            elif gyro_idx >= len(self.gyro_df):
                row = self.accel_df.iloc[accel_idx]
                self.all_data.append(('A', accel_idx, row))
                accel_idx += 1
            else:
                accel_time = self.accel_df.iloc[accel_idx]['timestamp_arduino_us']
                gyro_time = self.gyro_df.iloc[gyro_idx]['timestamp_arduino_us']

                if accel_time <= gyro_time:
                    row = self.accel_df.iloc[accel_idx]
                    self.all_data.append(('A', accel_idx, row))
                    accel_idx += 1
                else:
                    row = self.gyro_df.iloc[gyro_idx]
                    self.all_data.append(('G', gyro_idx, row))
                    gyro_idx += 1

        # IMU transformation state
        self.accel_axis_flip = ACCEL_AXIS_FLIP.copy()
        self.gyro_axis_flip = GYRO_AXIS_FLIP.copy()
        self.accel_rotation = ACCEL_ROTATION.copy() if ACCEL_ROTATION is not None else None
        self.gyro_rotation = GYRO_ROTATION.copy() if GYRO_ROTATION is not None else None

    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create plot grid (3x2 for 6 DOF)
        plot_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(plot_widget, stretch=8)

        # Create plots
        self.plot_x = plot_widget.addPlot(row=0, col=0, title="Position X (Commanded)")
        self.plot_x.setLabel('left', 'X Position', units='mm')
        self.plot_x.setLabel('bottom', 'Time', units='s')
        self.plot_x.addLegend()

        self.plot_y = plot_widget.addPlot(row=0, col=1, title="Position Y (Commanded)")
        self.plot_y.setLabel('left', 'Y Position', units='mm')
        self.plot_y.setLabel('bottom', 'Time', units='s')
        self.plot_y.addLegend()

        self.plot_z = plot_widget.addPlot(row=1, col=0, title="Position Z (Commanded)")
        self.plot_z.setLabel('left', 'Z Position', units='mm')
        self.plot_z.setLabel('bottom', 'Time', units='s')
        self.plot_z.addLegend()

        self.plot_rx = plot_widget.addPlot(row=1, col=1, title="Orientation RX (Roll)")
        self.plot_rx.setLabel('left', 'Roll', units='deg')
        self.plot_rx.setLabel('bottom', 'Time', units='s')
        self.plot_rx.addLegend()

        self.plot_ry = plot_widget.addPlot(row=2, col=0, title="Orientation RY (Pitch)")
        self.plot_ry.setLabel('left', 'Pitch', units='deg')
        self.plot_ry.setLabel('bottom', 'Time', units='s')
        self.plot_ry.addLegend()

        self.plot_rz = plot_widget.addPlot(row=2, col=1, title="Orientation RZ (Yaw)")
        self.plot_rz.setLabel('left', 'Yaw', units='deg')
        self.plot_rz.setLabel('bottom', 'Time', units='s')
        self.plot_rz.addLegend()

        # Controls layout
        controls_layout = QHBoxLayout()
        main_layout.addLayout(controls_layout, stretch=1)

        # File controls
        file_group = QGroupBox("File")
        file_layout = QHBoxLayout()
        file_group.setLayout(file_layout)

        self.file_edit = QLineEdit(str(Path.cwd() / 'csv' / '*.csv'))
        file_layout.addWidget(QLabel("Path:"))
        file_layout.addWidget(self.file_edit, stretch=3)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_btn)

        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self.load_file)
        file_layout.addWidget(load_btn)

        controls_layout.addWidget(file_group, stretch=2)

        # Kalman filter parameters
        params_group = QGroupBox("Kalman Filter Parameters")
        params_layout = QGridLayout()
        params_group.setLayout(params_layout)

        # Accel noise slider (0-1 normalized with scalar)
        params_layout.addWidget(QLabel("Accel Noise [m/s²]:"), 0, 0)
        self.accel_noise_scalar = QLineEdit(f"{ACCEL_NOISE:.6f}")
        self.accel_noise_scalar.setFixedWidth(80)
        self.accel_noise_scalar.editingFinished.connect(self.schedule_update)
        params_layout.addWidget(self.accel_noise_scalar, 0, 1)
        self.accel_noise_slider = QSlider(Qt.Orientation.Horizontal)
        self.accel_noise_slider.setMinimum(0)
        self.accel_noise_slider.setMaximum(1000)
        self.accel_noise_slider.setValue(1000)  # 1.0 default
        self.accel_noise_slider.valueChanged.connect(self.schedule_update)
        params_layout.addWidget(self.accel_noise_slider, 0, 2)
        self.accel_noise_label = QLabel(f"{ACCEL_NOISE:.6f}")
        params_layout.addWidget(self.accel_noise_label, 0, 3)

        # Gyro noise slider (0-1 normalized with scalar)
        params_layout.addWidget(QLabel("Gyro Noise [rad/s]:"), 1, 0)
        self.gyro_noise_scalar = QLineEdit(f"{GYRO_NOISE:.6f}")
        self.gyro_noise_scalar.setFixedWidth(80)
        self.gyro_noise_scalar.editingFinished.connect(self.schedule_update)
        params_layout.addWidget(self.gyro_noise_scalar, 1, 1)
        self.gyro_noise_slider = QSlider(Qt.Orientation.Horizontal)
        self.gyro_noise_slider.setMinimum(0)
        self.gyro_noise_slider.setMaximum(1000)
        self.gyro_noise_slider.setValue(1000)  # 1.0 default
        self.gyro_noise_slider.valueChanged.connect(self.schedule_update)
        params_layout.addWidget(self.gyro_noise_slider, 1, 2)
        self.gyro_noise_label = QLabel(f"{GYRO_NOISE:.6f}")
        params_layout.addWidget(self.gyro_noise_label, 1, 3)

        # Gyro scale multiplier slider (for calibration)
        params_layout.addWidget(QLabel("Gyro Scale Multiplier:"), 2, 0)
        self.gyro_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.gyro_scale_slider.setMinimum(1)
        self.gyro_scale_slider.setMaximum(10000)
        self.gyro_scale_slider.setValue(6600)  # 6.6x empirically calibrated
        self.gyro_scale_slider.valueChanged.connect(self.schedule_update)
        params_layout.addWidget(self.gyro_scale_slider, 2, 2)
        self.gyro_scale_label = QLabel("6.600x")
        params_layout.addWidget(self.gyro_scale_label, 2, 3)

        # Process noise angle slider (0-1 normalized with scalar)
        params_layout.addWidget(QLabel("Proc Noise Angle [rad²]:"), 3, 0)
        self.proc_angle_scalar = QLineEdit("0.001")
        self.proc_angle_scalar.setFixedWidth(80)
        self.proc_angle_scalar.editingFinished.connect(self.schedule_update)
        params_layout.addWidget(self.proc_angle_scalar, 3, 1)
        self.proc_angle_slider = QSlider(Qt.Orientation.Horizontal)
        self.proc_angle_slider.setMinimum(0)
        self.proc_angle_slider.setMaximum(1000)
        self.proc_angle_slider.setValue(0)  # 0.0 default
        self.proc_angle_slider.valueChanged.connect(self.schedule_update)
        params_layout.addWidget(self.proc_angle_slider, 3, 2)
        self.proc_angle_label = QLabel(f"{PROCESS_NOISE_ANGLE:.6f}")
        params_layout.addWidget(self.proc_angle_label, 3, 3)

        # Process noise bias slider (0-1 normalized with scalar)
        params_layout.addWidget(QLabel("Proc Noise Bias [(rad/s)²]:"), 4, 0)
        self.proc_bias_scalar = QLineEdit("0.0001")
        self.proc_bias_scalar.setFixedWidth(80)
        self.proc_bias_scalar.editingFinished.connect(self.schedule_update)
        params_layout.addWidget(self.proc_bias_scalar, 4, 1)
        self.proc_bias_slider = QSlider(Qt.Orientation.Horizontal)
        self.proc_bias_slider.setMinimum(0)
        self.proc_bias_slider.setMaximum(1000)
        self.proc_bias_slider.setValue(0)  # 0.0 default
        self.proc_bias_slider.valueChanged.connect(self.schedule_update)
        params_layout.addWidget(self.proc_bias_slider, 4, 2)
        self.proc_bias_label = QLabel(f"{PROCESS_NOISE_BIAS:.8f}")
        params_layout.addWidget(self.proc_bias_label, 4, 3)

        # Enable/Disable sensor checkboxes
        self.enable_accel_checkbox = QCheckBox("Enable Accelerometer Updates")
        self.enable_accel_checkbox.setChecked(True)
        self.enable_accel_checkbox.stateChanged.connect(self.update_plots)
        params_layout.addWidget(self.enable_accel_checkbox, 5, 0, 1, 2)

        self.enable_gyro_checkbox = QCheckBox("Enable Gyroscope Predictions")
        self.enable_gyro_checkbox.setChecked(True)
        self.enable_gyro_checkbox.stateChanged.connect(self.update_plots)
        params_layout.addWidget(self.enable_gyro_checkbox, 5, 2, 1, 1)

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset_sliders)
        params_layout.addWidget(reset_btn, 6, 0, 1, 3)

        controls_layout.addWidget(params_group, stretch=3)

        # Axis flip and rotation controls
        transform_group = QGroupBox("IMU Transformations")
        transform_layout = QVBoxLayout()
        transform_group.setLayout(transform_layout)

        # Axis flips section
        flip_layout = QHBoxLayout()

        accel_flip_layout = QVBoxLayout()
        accel_flip_layout.addWidget(QLabel("Accelerometer Flips:"))
        self.accel_flip_x = QCheckBox("Flip X")
        self.accel_flip_x.setChecked(bool(self.accel_axis_flip[0] < 0))
        self.accel_flip_x.stateChanged.connect(lambda: self.toggle_accel_flip(0))
        accel_flip_layout.addWidget(self.accel_flip_x)
        self.accel_flip_y = QCheckBox("Flip Y")
        self.accel_flip_y.setChecked(bool(self.accel_axis_flip[1] < 0))
        self.accel_flip_y.stateChanged.connect(lambda: self.toggle_accel_flip(1))
        accel_flip_layout.addWidget(self.accel_flip_y)
        self.accel_flip_z = QCheckBox("Flip Z")
        self.accel_flip_z.setChecked(bool(self.accel_axis_flip[2] < 0))
        self.accel_flip_z.stateChanged.connect(lambda: self.toggle_accel_flip(2))
        accel_flip_layout.addWidget(self.accel_flip_z)
        flip_layout.addLayout(accel_flip_layout)

        gyro_flip_layout = QVBoxLayout()
        gyro_flip_layout.addWidget(QLabel("Gyroscope Flips:"))
        self.gyro_flip_x = QCheckBox("Flip X")
        self.gyro_flip_x.setChecked(bool(self.gyro_axis_flip[0] < 0))
        self.gyro_flip_x.stateChanged.connect(lambda: self.toggle_gyro_flip(0))
        gyro_flip_layout.addWidget(self.gyro_flip_x)
        self.gyro_flip_y = QCheckBox("Flip Y")
        self.gyro_flip_y.setChecked(bool(self.gyro_axis_flip[1] < 0))
        self.gyro_flip_y.stateChanged.connect(lambda: self.toggle_gyro_flip(1))
        gyro_flip_layout.addWidget(self.gyro_flip_y)
        self.gyro_flip_z = QCheckBox("Flip Z")
        self.gyro_flip_z.setChecked(bool(self.gyro_axis_flip[2] < 0))
        self.gyro_flip_z.stateChanged.connect(lambda: self.toggle_gyro_flip(2))
        gyro_flip_layout.addWidget(self.gyro_flip_z)
        flip_layout.addLayout(gyro_flip_layout)

        transform_layout.addLayout(flip_layout)

        # Rotation controls section
        rotation_layout = QGridLayout()
        rotation_layout.addWidget(QLabel("Accelerometer Rotations:"), 0, 0, 1, 3)

        # Accel rotation buttons
        accel_rot_x_neg = QPushButton("X -90°")
        accel_rot_x_neg.clicked.connect(lambda: self.rotate_accel('x', -90))
        rotation_layout.addWidget(accel_rot_x_neg, 1, 0)
        accel_rot_x_pos = QPushButton("X +90°")
        accel_rot_x_pos.clicked.connect(lambda: self.rotate_accel('x', 90))
        rotation_layout.addWidget(accel_rot_x_pos, 1, 1)

        accel_rot_y_neg = QPushButton("Y -90°")
        accel_rot_y_neg.clicked.connect(lambda: self.rotate_accel('y', -90))
        rotation_layout.addWidget(accel_rot_y_neg, 2, 0)
        accel_rot_y_pos = QPushButton("Y +90°")
        accel_rot_y_pos.clicked.connect(lambda: self.rotate_accel('y', 90))
        rotation_layout.addWidget(accel_rot_y_pos, 2, 1)

        accel_rot_z_neg = QPushButton("Z -90°")
        accel_rot_z_neg.clicked.connect(lambda: self.rotate_accel('z', -90))
        rotation_layout.addWidget(accel_rot_z_neg, 3, 0)
        accel_rot_z_pos = QPushButton("Z +90°")
        accel_rot_z_pos.clicked.connect(lambda: self.rotate_accel('z', 90))
        rotation_layout.addWidget(accel_rot_z_pos, 3, 1)

        accel_reset_btn = QPushButton("Reset")
        accel_reset_btn.clicked.connect(lambda: self.reset_accel_rotation())
        rotation_layout.addWidget(accel_reset_btn, 1, 2, 3, 1)

        rotation_layout.addWidget(QLabel("Gyroscope Rotations:"), 0, 3, 1, 3)

        # Gyro rotation buttons
        gyro_rot_x_neg = QPushButton("X -90°")
        gyro_rot_x_neg.clicked.connect(lambda: self.rotate_gyro('x', -90))
        rotation_layout.addWidget(gyro_rot_x_neg, 1, 3)
        gyro_rot_x_pos = QPushButton("X +90°")
        gyro_rot_x_pos.clicked.connect(lambda: self.rotate_gyro('x', 90))
        rotation_layout.addWidget(gyro_rot_x_pos, 1, 4)

        gyro_rot_y_neg = QPushButton("Y -90°")
        gyro_rot_y_neg.clicked.connect(lambda: self.rotate_gyro('y', -90))
        rotation_layout.addWidget(gyro_rot_y_neg, 2, 3)
        gyro_rot_y_pos = QPushButton("Y +90°")
        gyro_rot_y_pos.clicked.connect(lambda: self.rotate_gyro('y', 90))
        rotation_layout.addWidget(gyro_rot_y_pos, 2, 4)

        gyro_rot_z_neg = QPushButton("Z -90°")
        gyro_rot_z_neg.clicked.connect(lambda: self.rotate_gyro('z', -90))
        rotation_layout.addWidget(gyro_rot_z_neg, 3, 3)
        gyro_rot_z_pos = QPushButton("Z +90°")
        gyro_rot_z_pos.clicked.connect(lambda: self.rotate_gyro('z', 90))
        rotation_layout.addWidget(gyro_rot_z_pos, 3, 4)

        gyro_reset_btn = QPushButton("Reset")
        gyro_reset_btn.clicked.connect(lambda: self.reset_gyro_rotation())
        rotation_layout.addWidget(gyro_reset_btn, 1, 5, 3, 1)

        transform_layout.addLayout(rotation_layout)

        # Rotation status displays
        status_layout = QHBoxLayout()

        # Accel rotation status
        accel_status_layout = QVBoxLayout()
        accel_status_layout.addWidget(QLabel("Accelerometer Rotation Status:"))
        self.accel_rotation_label = QLabel("Identity (no rotation)")
        self.accel_rotation_label.setStyleSheet("font-family: monospace; font-size: 9pt;")
        accel_status_layout.addWidget(self.accel_rotation_label)
        status_layout.addLayout(accel_status_layout)

        # Gyro rotation status
        gyro_status_layout = QVBoxLayout()
        gyro_status_layout.addWidget(QLabel("Gyroscope Rotation Status:"))
        self.gyro_rotation_label = QLabel("Identity (no rotation)")
        self.gyro_rotation_label.setStyleSheet("font-family: monospace; font-size: 9pt;")
        gyro_status_layout.addWidget(self.gyro_rotation_label)
        status_layout.addLayout(gyro_status_layout)

        transform_layout.addLayout(status_layout)

        controls_layout.addWidget(transform_group, stretch=2)

    def schedule_update(self):
        """Schedule plot update with debouncing"""
        self.update_timer.stop()
        self.update_timer.start(300)  # 300ms delay

        # Update slider labels immediately (scalar × slider_value)
        try:
            accel_scalar = float(self.accel_noise_scalar.text())
            accel_val = accel_scalar * (self.accel_noise_slider.value() / 1000)
            self.accel_noise_label.setText(f"{accel_val:.6f}")
        except ValueError:
            self.accel_noise_label.setText("Invalid")

        try:
            gyro_scalar = float(self.gyro_noise_scalar.text())
            gyro_val = gyro_scalar * (self.gyro_noise_slider.value() / 1000)
            self.gyro_noise_label.setText(f"{gyro_val:.6f}")
        except ValueError:
            self.gyro_noise_label.setText("Invalid")

        self.gyro_scale_label.setText(f"{self.gyro_scale_slider.value() / 1000:.3f}x")

        try:
            angle_scalar = float(self.proc_angle_scalar.text())
            angle_val = angle_scalar * (self.proc_angle_slider.value() / 1000)
            self.proc_angle_label.setText(f"{angle_val:.6f}")
        except ValueError:
            self.proc_angle_label.setText("Invalid")

        try:
            bias_scalar = float(self.proc_bias_scalar.text())
            bias_val = bias_scalar * (self.proc_bias_slider.value() / 1000)
            self.proc_bias_label.setText(f"{bias_val:.8f}")
        except ValueError:
            self.proc_bias_label.setText("Invalid")

    def update_plots(self):
        """Update plots with current Kalman filter parameters"""
        # Get slider values (scalar × normalized slider value)
        try:
            accel_scalar = float(self.accel_noise_scalar.text())
            accel_noise = accel_scalar * (self.accel_noise_slider.value() / 1000)
        except ValueError:
            accel_noise = ACCEL_NOISE

        try:
            gyro_scalar = float(self.gyro_noise_scalar.text())
            gyro_noise = gyro_scalar * (self.gyro_noise_slider.value() / 1000)
        except ValueError:
            gyro_noise = GYRO_NOISE

        gyro_scale_mult = self.gyro_scale_slider.value() / 1000

        try:
            angle_scalar = float(self.proc_angle_scalar.text())
            proc_noise_angle = angle_scalar * (self.proc_angle_slider.value() / 1000)
        except ValueError:
            proc_noise_angle = PROCESS_NOISE_ANGLE

        try:
            bias_scalar = float(self.proc_bias_scalar.text())
            proc_noise_bias = bias_scalar * (self.proc_bias_slider.value() / 1000)
        except ValueError:
            proc_noise_bias = PROCESS_NOISE_BIAS

        # Scale gyro noise by the scale multiplier (same factor that affects measurements)
        gyro_noise_scaled = gyro_noise * gyro_scale_mult

        # Get checkbox states
        enable_accel = self.enable_accel_checkbox.isChecked()
        enable_gyro = self.enable_gyro_checkbox.isChecked()

        # Run Kalman filter
        rx_est, ry_est, bias_x, bias_y, time, platform_rx, platform_ry = run_kalman_filter(
            self.accel_df, self.gyro_df, self.all_data, accel_noise, gyro_noise_scaled,
            proc_noise_angle, proc_noise_bias,
            self.accel_axis_flip, self.gyro_axis_flip,
            self.accel_rotation, self.gyro_rotation,
            enable_accel, enable_gyro, gyro_scale_mult
        )

        # Clear and update plots
        self.plot_x.clear()
        self.plot_x.plot(time, self.cmd_fk_x, pen=pg.mkPen('#c77dff', width=1.5, style=Qt.PenStyle.DotLine), name='Commanded')
        self.plot_x.plot(time, self.servo_fk_x, pen=pg.mkPen('#4ecdc4', width=2), name='Servo FK')

        self.plot_y.clear()
        self.plot_y.plot(time, self.cmd_fk_y, pen=pg.mkPen('#c77dff', width=1.5, style=Qt.PenStyle.DotLine), name='Commanded')
        self.plot_y.plot(time, self.servo_fk_y, pen=pg.mkPen('#4ecdc4', width=2), name='Servo FK')

        self.plot_z.clear()
        self.plot_z.plot(time, self.cmd_fk_z, pen=pg.mkPen('#c77dff', width=1.5, style=Qt.PenStyle.DotLine), name='Commanded')
        self.plot_z.plot(time, self.servo_fk_z, pen=pg.mkPen('#4ecdc4', width=2), name='Servo FK')

        self.plot_rx.clear()
        self.plot_rx.plot(time, np.degrees(self.cmd_fk_rx), pen=pg.mkPen('#c77dff', width=1.5, style=Qt.PenStyle.DotLine), name='Commanded FK')
        self.plot_rx.plot(time, np.degrees(self.servo_fk_rx), pen=pg.mkPen('#4ecdc4', width=2), name='Servo FK')
        self.plot_rx.plot(time, np.degrees(rx_est), pen=pg.mkPen('#ff6b6b', width=2, style=Qt.PenStyle.DashLine), name='Kalman (IMU)')

        self.plot_ry.clear()
        self.plot_ry.plot(time, np.degrees(self.cmd_fk_ry), pen=pg.mkPen('#c77dff', width=1.5, style=Qt.PenStyle.DotLine), name='Commanded FK')
        self.plot_ry.plot(time, np.degrees(self.servo_fk_ry), pen=pg.mkPen('#4ecdc4', width=2), name='Servo FK')
        self.plot_ry.plot(time, np.degrees(ry_est), pen=pg.mkPen('#ff6b6b', width=2, style=Qt.PenStyle.DashLine), name='Kalman (IMU)')

        self.plot_rz.clear()
        self.plot_rz.plot(time, np.degrees(self.cmd_fk_rz), pen=pg.mkPen('#c77dff', width=1.5, style=Qt.PenStyle.DotLine), name='Commanded FK')
        self.plot_rz.plot(time, np.degrees(self.servo_fk_rz), pen=pg.mkPen('#4ecdc4', width=2), name='Servo FK')

    def toggle_accel_flip(self, axis):
        """Toggle accelerometer axis flip"""
        self.accel_axis_flip[axis] *= -1
        self.update_plots()

    def toggle_gyro_flip(self, axis):
        """Toggle gyroscope axis flip"""
        self.gyro_axis_flip[axis] *= -1
        self.update_plots()

    def _describe_rotation_matrix(self, R):
        """Analyze rotation matrix and return human-readable description"""
        # Check if identity
        if np.allclose(R, np.eye(3), atol=1e-6):
            return "Identity (no rotation)"

        # Format matrix display
        matrix_str = "Matrix:\n"
        for i in range(3):
            row = " ".join([f"{R[i,j]:6.3f}" for j in range(3)])
            matrix_str += f"  [{row}]\n"

        # Try to decompose into simple 90° rotations
        # Check for single-axis 90° rotations
        descriptions = []

        # Check X-axis rotations
        if np.allclose(R[0,:], [1, 0, 0], atol=1e-6):
            if np.allclose(R[1,:], [0, 0, -1], atol=1e-6) and np.allclose(R[2,:], [0, 1, 0], atol=1e-6):
                descriptions.append("X +90°")
            elif np.allclose(R[1,:], [0, 0, 1], atol=1e-6) and np.allclose(R[2,:], [0, -1, 0], atol=1e-6):
                descriptions.append("X -90°")
            elif np.allclose(R[1,:], [0, -1, 0], atol=1e-6) and np.allclose(R[2,:], [0, 0, -1], atol=1e-6):
                descriptions.append("X 180°")

        # Check Y-axis rotations
        if np.allclose(R[1,:], [0, 1, 0], atol=1e-6):
            if np.allclose(R[0,:], [0, 0, 1], atol=1e-6) and np.allclose(R[2,:], [-1, 0, 0], atol=1e-6):
                descriptions.append("Y +90°")
            elif np.allclose(R[0,:], [0, 0, -1], atol=1e-6) and np.allclose(R[2,:], [1, 0, 0], atol=1e-6):
                descriptions.append("Y -90°")
            elif np.allclose(R[0,:], [-1, 0, 0], atol=1e-6) and np.allclose(R[2,:], [0, 0, -1], atol=1e-6):
                descriptions.append("Y 180°")

        # Check Z-axis rotations
        if np.allclose(R[2,:], [0, 0, 1], atol=1e-6):
            if np.allclose(R[0,:], [0, -1, 0], atol=1e-6) and np.allclose(R[1,:], [1, 0, 0], atol=1e-6):
                descriptions.append("Z +90°")
            elif np.allclose(R[0,:], [0, 1, 0], atol=1e-6) and np.allclose(R[1,:], [-1, 0, 0], atol=1e-6):
                descriptions.append("Z -90°")
            elif np.allclose(R[0,:], [-1, 0, 0], atol=1e-6) and np.allclose(R[1,:], [0, -1, 0], atol=1e-6):
                descriptions.append("Z 180°")

        if descriptions:
            return f"Rotation: {', '.join(descriptions)}\n" + matrix_str.rstrip()
        else:
            return f"Complex rotation\n" + matrix_str.rstrip()

    def _get_rotation_matrix(self, axis, angle_deg):
        """Generate rotation matrix for given axis and angle"""
        angle = np.radians(angle_deg)
        c = np.cos(angle)
        s = np.sin(angle)

        if axis == 'x':
            return np.array([
                [1, 0, 0],
                [0, c, -s],
                [0, s, c]
            ])
        elif axis == 'y':
            return np.array([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
            ])
        elif axis == 'z':
            return np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])

    def rotate_accel(self, axis, angle_deg):
        """Apply rotation to accelerometer rotation matrix"""
        rot = self._get_rotation_matrix(axis, angle_deg)
        if self.accel_rotation is not None:
            self.accel_rotation = rot @ self.accel_rotation
        else:
            self.accel_rotation = rot

        # Update status display
        status = self._describe_rotation_matrix(self.accel_rotation)
        self.accel_rotation_label.setText(status)

        print(f"Accel rotated {angle_deg}° around {axis.upper()}-axis")
        print(f"Current accel rotation:\n{status}")
        self.update_plots()

    def rotate_gyro(self, axis, angle_deg):
        """Apply rotation to gyroscope rotation matrix"""
        rot = self._get_rotation_matrix(axis, angle_deg)
        if self.gyro_rotation is not None:
            self.gyro_rotation = rot @ self.gyro_rotation
        else:
            self.gyro_rotation = rot

        # Update status display
        status = self._describe_rotation_matrix(self.gyro_rotation)
        self.gyro_rotation_label.setText(status)

        print(f"Gyro rotated {angle_deg}° around {axis.upper()}-axis")
        print(f"Current gyro rotation:\n{status}")
        self.update_plots()

    def reset_accel_rotation(self):
        """Reset accelerometer rotation to identity"""
        self.accel_rotation = np.eye(3)

        # Update status display
        self.accel_rotation_label.setText("Identity (no rotation)")

        print("Accel rotation reset to identity")
        self.update_plots()

    def reset_gyro_rotation(self):
        """Reset gyroscope rotation to identity"""
        self.gyro_rotation = np.eye(3)

        # Update status display
        self.gyro_rotation_label.setText("Identity (no rotation)")

        print("Gyro rotation reset to identity")
        self.update_plots()

    def reset_sliders(self):
        """Reset sliders to default values"""
        # Reset scalars
        self.accel_noise_scalar.setText(f"{ACCEL_NOISE:.6f}")
        self.gyro_noise_scalar.setText(f"{GYRO_NOISE:.6f}")
        self.proc_angle_scalar.setText("0.001")
        self.proc_bias_scalar.setText("0.0001")

        # Reset sliders to 1.0 (fully scaled)
        self.accel_noise_slider.setValue(1000)
        self.gyro_noise_slider.setValue(1000)
        self.gyro_scale_slider.setValue(6600)  # 6.6x empirically calibrated
        self.proc_angle_slider.setValue(0)
        self.proc_bias_slider.setValue(0)

    def browse_file(self):
        """Open file dialog to select CSV file"""
        csv_dir = Path.cwd() / 'csv'
        filename, _ = QFileDialog.getOpenFileName(
            self,
            'Select IMU CSV file',
            str(csv_dir),
            'CSV files (*.csv);;All files (*.*)'
        )
        if filename:
            self.file_edit.setText(filename)

    def load_file(self):
        """Load new CSV file and update plots"""
        file_pattern = self.file_edit.text().strip()

        # Support glob patterns
        matching_files = sorted(glob.glob(file_pattern))
        if not matching_files:
            print(f"ERROR: No files found matching: {file_pattern}")
            return

        csv_file = matching_files[-1]
        print(f"\nLoading file: {csv_file}")

        try:
            # Load and process data
            df = load_imu_data(csv_file)
            df = convert_to_physical_units(df)

            # Reload data
            self.load_data(df)

            # Update plots
            self.update_plots()

            print("File loaded successfully")

        except Exception as e:
            print(f"ERROR loading file: {e}")
            import traceback
            traceback.print_exc()


def process_kalman_filter(df, output_file=None):
    """Interactive platform state comparison: Commanded FK vs Servo FK vs Kalman

    Shows three estimates of platform state:
    - Commanded FK: Direct FK from commanded servo angles (no dynamics)
    - Servo FK: FK from commanded angles processed through first-order servo dynamics
    - Kalman: IMU-based orientation estimate using Extended Kalman Filter

    Args:
        df: DataFrame with IMU data
        output_file: Optional path to save plot (not used in PyQtGraph version)
    """
    print("\n" + "="*60)
    print("PLATFORM STATE COMPARISON: COMMANDED vs SERVO vs KALMAN")
    print("="*60)

    print(f"\nInitial Filter Configuration:")
    print(f"  Accel noise: {ACCEL_NOISE:.4f} m/s²")
    print(f"  Gyro noise: {GYRO_NOISE:.6f} rad/s")
    print(f"  Process noise (angle): {PROCESS_NOISE_ANGLE:.6f} rad²")
    print(f"  Process noise (bias): {PROCESS_NOISE_BIAS:.8f} (rad/s)²")
    print(f"  Initial gyro biases: X={GYRO_BIAS_X:.6f} rad/s, Y={GYRO_BIAS_Y:.6f} rad/s")
    print(f"  Gravity vector: [{GRAVITY_VECTOR[0]:.4f}, {GRAVITY_VECTOR[1]:.4f}, {GRAVITY_VECTOR[2]:.4f}] m/s² (mag={GRAVITY_MAGNITUDE:.4f})")
    print(f"\nUse sliders to tune filter parameters...")

    # Initialize Stewart platform IK with same parameters as data_logger
    platform_params = {
        "horn_length": 45.3722,
        "rod_length": 205.0,
        "base": 86.6025 + 18.75 + 11,
        "base_anchors": 64.75,
        "platform": 84.0759,
        "platform_anchors": 12.5,
        "top_surface_offset": 38.0
    }
    stewart_ik = StewartPlatformIK(**platform_params)
    print(f"Using platform parameters from data_logger:")

    # Create and show PyQt6 window
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = IMUPlotWindow(df, stewart_ik)
    window.show()

    app.exec()


def plot_gui_selector():
    """Show file dialog to select IMU CSV file"""
    # Create temporary QApplication for file dialog
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    csv_dir = Path.cwd() / 'csv'
    filename, _ = QFileDialog.getOpenFileName(
        None,
        'Select IMU CSV file',
        str(csv_dir),
        'CSV files (*.csv);;All files (*.*)'
    )

    if not filename:
        print("No file selected")
        return

    print(f"\nLoading file: {filename}")

    try:
        # Load and process data
        df = load_imu_data(filename)
        df = convert_to_physical_units(df)

        print(f"Loaded {len(df)} samples")
        print_statistics(df)

        # Show comparison plot
        process_kalman_filter(df)

    except Exception as e:
        print(f"ERROR loading file: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='IMU data visualization: 6-DOF platform state comparison (Commanded FK vs Servo FK vs Kalman)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_data.py                              (open GUI file selector)
  python plot_data.py data.csv                     (6-subplot comparison view)
  python plot_data.py csv/imu_*.csv                (load latest matching file)
  python plot_data.py data.csv --stats             (statistics only)

Comparison shows:
  - Commanded FK: Direct forward kinematics from commanded servo angles
  - Servo FK: FK after processing through first-order servo dynamics model
  - Kalman: IMU-based orientation estimate from Extended Kalman Filter
        """
    )

    parser.add_argument('csv_file', type=str, nargs='?',
                        help='Input CSV file from IMU logger (optional - opens GUI if not provided)')
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

    # Show comparison plot
    process_kalman_filter(df)


if __name__ == "__main__":
    main()
