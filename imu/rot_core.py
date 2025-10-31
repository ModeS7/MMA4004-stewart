#!/usr/bin/env python3
"""
Real-time IMU Orientation Tracking

Live version of plot_data.py - streams IMU data and shows orientation estimates
in real-time with interactive parameter tuning and optional active platform compensation.

Features:
- 10-second startup calibration phase to measure gyroscope bias and gravity vector
- Real-time Kalman filter for roll (RX) and pitch (RY) estimation
- Optional active compensation: sends servo commands to cancel platform rotations
- Live parameter tuning sliders
- Axis transformation controls
- Toggleable servo control for testing at home without platform

Calibration Phase:
- Collects 10 seconds of stationary IMU data at startup
- Calculates gyroscope bias from mean gyro readings
- Establishes gravity vector from mean accelerometer readings
- Initializes Kalman filter with measured values for drift compensation

Compatible Arduino sketches:
- IMU_control.ino (with servo control)
- IMU_standalone.ino (without servos, for home testing)

Usage:
    python rot_core.py --port COM4

    Enable "Send Servo Commands" checkbox to actually control platform.
    Leave unchecked for visualization-only mode (no servos needed).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import serial
import serial.tools.list_ports
import numpy as np
import time
import argparse
import threading
from queue import Queue
from collections import deque

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSlider, QPushButton, QLabel, QLineEdit, QCheckBox, QGridLayout, QGroupBox)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

from core.control_core import OrientationKalmanFilter, apply_imu_transforms
from core.core import StewartPlatformIK
from core.utils import IMUKalmanConfig

# PyQtGraph dark theme
pg.setConfigOption('background', '#1e1e1e')
pg.setConfigOption('foreground', 'w')
pg.setConfigOption('antialias', True)


# Default parameters
ACCEL_NOISE = 1.0
GYRO_NOISE = 0.0224
PROCESS_NOISE_ANGLE = 0.001
PROCESS_NOISE_BIAS = 0.00001
GYRO_BIAS_X = 0.112679
GYRO_BIAS_Y = 0.031500

# Motion detection thresholds
ACCEL_MAGNITUDE_THRESHOLD = 2.0  # m/s² deviation from gravity to reject accel update
GYRO_MAGNITUDE_THRESHOLD = 0.5   # rad/s rotation rate to reject accel update

# Axis transformations (default: no inversions)
ACCEL_AXIS_FLIP = np.array([1, 1, 1])
GYRO_AXIS_FLIP = np.array([1, 1, 1])
ACCEL_ROTATION = np.eye(3)
GYRO_ROTATION = np.eye(3)


class RealtimeOrientationWindow(QMainWindow):
    """Real-time IMU orientation tracking with live parameter tuning"""

    def __init__(self, port, baudrate=2000000):
        super().__init__()
        self.setWindowTitle("Real-time IMU Orientation Tracking")
        self.resize(1600, 900)

        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.running = False

        # Kalman filter
        self.kalman = OrientationKalmanFilter(
            accel_noise=ACCEL_NOISE,
            gyro_noise=GYRO_NOISE,
            process_noise_angle=PROCESS_NOISE_ANGLE,
            process_noise_bias=PROCESS_NOISE_BIAS,
            accel_axis_flip=ACCEL_AXIS_FLIP,
            gyro_axis_flip=GYRO_AXIS_FLIP,
            accel_rotation=ACCEL_ROTATION,
            gyro_rotation=GYRO_ROTATION,
            initial_bias_x=GYRO_BIAS_X,
            initial_bias_y=GYRO_BIAS_Y,
            gyro_scale_multiplier=1.0
        )

        # Data queues
        self.gyro_queue = Queue(maxsize=1000)
        self.accel_queue = Queue(maxsize=2000)
        self.mag_queue = Queue(maxsize=500)

        # Current state
        self.current_rx_imu = 0.0
        self.current_ry_imu = 0.0
        self.current_rx_cmd = 0.0
        self.current_ry_cmd = 0.0
        self.last_update_time = None

        # Statistics
        self.gyro_count = 0
        self.accel_count = 0
        self.mag_count = 0
        self.update_count = 0
        self.servo_command_count = 0
        self.start_time = time.time()

        # Servo control
        self.last_servo_time = 0.0
        self.servo_interval = 1.0 / 100.0  # 100 Hz
        self.compensation_gain = 1.0  # Full compensation

        # Suspension control (active damping)
        self.suspension_enabled = False
        self.suspension_position_gain = 0.5  # Proportional to linear acceleration
        self.suspension_velocity_gain = 0.0  # Damping (integrated accel)
        self.linear_velocity = np.array([0.0, 0.0, 0.0])  # Integrated from acceleration
        self.last_update_time = time.time()

        # Initialization and calibration phases
        self.initializing = True
        self.initialization_duration = 3.0  # 3 seconds to let sensors stabilize
        self.initialization_start_time = None
        self.calibrating = False  # Will start after initialization
        self.calibration_duration = 10.0  # 10 seconds calibration
        self.calibration_start_time = None
        self.calibration_raw_lines = []  # Buffer raw serial lines during calibration
        self.calibration_lock = threading.Lock()  # Thread-safe access to calibration data
        self.calibrated_gravity_vector = None
        self.calibrated_gyro_bias = None
        self.calibrated_mag_offset = None

        # IK for servo commands
        self.ik = StewartPlatformIK()

        # Axis transformations (can be modified via GUI)
        self.accel_axis_flip = ACCEL_AXIS_FLIP.copy()
        self.gyro_axis_flip = GYRO_AXIS_FLIP.copy()
        self.accel_rotation = ACCEL_ROTATION.copy()
        self.gyro_rotation = GYRO_ROTATION.copy()

        # Plot history (30 seconds)
        self.max_history = 600
        self.time_history = deque(maxlen=self.max_history)
        self.rx_imu_history = deque(maxlen=self.max_history)
        self.ry_imu_history = deque(maxlen=self.max_history)
        self.rx_cmd_history = deque(maxlen=self.max_history)
        self.ry_cmd_history = deque(maxlen=self.max_history)
        self.mag_x_history = deque(maxlen=self.max_history)
        self.mag_y_history = deque(maxlen=self.max_history)
        self.mag_z_history = deque(maxlen=self.max_history)
        self.plot_start_time = None

        # Setup UI
        self.setup_ui()

        # Update timer (debounced parameter changes)
        self.param_update_timer = QTimer()
        self.param_update_timer.setSingleShot(True)
        self.param_update_timer.timeout.connect(self.update_filter_parameters)

        # Plot update timer
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plots)
        self.plot_timer.setInterval(50)  # 20 Hz

    def setup_ui(self):
        """Setup GUI"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left: Controls
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls.setMaximumWidth(350)

        # Status
        self.status_label = QLabel("Initializing...")
        controls_layout.addWidget(self.status_label)

        # Kalman parameters
        params_group = QGroupBox("Kalman Filter Parameters")
        params_layout = QGridLayout()
        params_group.setLayout(params_layout)

        # Accel noise
        params_layout.addWidget(QLabel("Accel Noise [m/s²]:"), 0, 0)
        self.accel_noise_scalar = QLineEdit("1.0")
        self.accel_noise_scalar.setMaximumWidth(60)
        self.accel_noise_scalar.editingFinished.connect(self.schedule_param_update)
        params_layout.addWidget(self.accel_noise_scalar, 0, 1)
        self.accel_noise_slider = QSlider(Qt.Orientation.Horizontal)
        self.accel_noise_slider.setMinimum(1)
        self.accel_noise_slider.setMaximum(10000)
        self.accel_noise_slider.setValue(1000)
        self.accel_noise_slider.valueChanged.connect(self.schedule_param_update)
        params_layout.addWidget(self.accel_noise_slider, 0, 2)
        self.accel_noise_label = QLabel("1.000")
        params_layout.addWidget(self.accel_noise_label, 0, 3)

        # Gyro noise
        params_layout.addWidget(QLabel("Gyro Noise [rad/s]:"), 1, 0)
        self.gyro_noise_scalar = QLineEdit("0.0224")
        self.gyro_noise_scalar.setMaximumWidth(60)
        self.gyro_noise_scalar.editingFinished.connect(self.schedule_param_update)
        params_layout.addWidget(self.gyro_noise_scalar, 1, 1)
        self.gyro_noise_slider = QSlider(Qt.Orientation.Horizontal)
        self.gyro_noise_slider.setMinimum(1)
        self.gyro_noise_slider.setMaximum(10000)
        self.gyro_noise_slider.setValue(1000)
        self.gyro_noise_slider.valueChanged.connect(self.schedule_param_update)
        params_layout.addWidget(self.gyro_noise_slider, 1, 2)
        self.gyro_noise_label = QLabel("0.022")
        params_layout.addWidget(self.gyro_noise_label, 1, 3)

        # Gyro scale
        params_layout.addWidget(QLabel("Gyro Scale Mult:"), 2, 0)
        self.gyro_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.gyro_scale_slider.setMinimum(1)
        self.gyro_scale_slider.setMaximum(10000)
        self.gyro_scale_slider.setValue(6600)
        self.gyro_scale_slider.valueChanged.connect(self.schedule_param_update)
        params_layout.addWidget(self.gyro_scale_slider, 2, 2)
        self.gyro_scale_label = QLabel("6.600x")
        params_layout.addWidget(self.gyro_scale_label, 2, 3)

        # Process noise angle
        params_layout.addWidget(QLabel("Proc Noise Angle [rad²]:"), 3, 0)
        self.proc_angle_scalar = QLineEdit("0.001")
        self.proc_angle_scalar.setMaximumWidth(60)
        self.proc_angle_scalar.editingFinished.connect(self.schedule_param_update)
        params_layout.addWidget(self.proc_angle_scalar, 3, 1)
        self.proc_angle_slider = QSlider(Qt.Orientation.Horizontal)
        self.proc_angle_slider.setMinimum(0)
        self.proc_angle_slider.setMaximum(1000)
        self.proc_angle_slider.setValue(0)
        self.proc_angle_slider.valueChanged.connect(self.schedule_param_update)
        params_layout.addWidget(self.proc_angle_slider, 3, 2)
        self.proc_angle_label = QLabel("0.000")
        params_layout.addWidget(self.proc_angle_label, 3, 3)

        # Process noise bias
        params_layout.addWidget(QLabel("Proc Noise Bias [(rad/s)²]:"), 4, 0)
        self.proc_bias_scalar = QLineEdit("0.00001")
        self.proc_bias_scalar.setMaximumWidth(60)
        self.proc_bias_scalar.editingFinished.connect(self.schedule_param_update)
        params_layout.addWidget(self.proc_bias_scalar, 4, 1)
        self.proc_bias_slider = QSlider(Qt.Orientation.Horizontal)
        self.proc_bias_slider.setMinimum(0)
        self.proc_bias_slider.setMaximum(1000)
        self.proc_bias_slider.setValue(0)
        self.proc_bias_slider.valueChanged.connect(self.schedule_param_update)
        params_layout.addWidget(self.proc_bias_slider, 4, 2)
        self.proc_bias_label = QLabel("0.000")
        params_layout.addWidget(self.proc_bias_label, 4, 3)

        # Enable/disable
        self.enable_accel_checkbox = QCheckBox("Enable Accel Updates")
        self.enable_accel_checkbox.setChecked(True)
        params_layout.addWidget(self.enable_accel_checkbox, 5, 0, 1, 2)

        self.enable_gyro_checkbox = QCheckBox("Enable Gyro Predictions")
        self.enable_gyro_checkbox.setChecked(True)
        params_layout.addWidget(self.enable_gyro_checkbox, 5, 2, 1, 2)

        # Enable servo commands
        self.enable_servo_checkbox = QCheckBox("Send Servo Commands")
        self.enable_servo_checkbox.setChecked(False)  # Default OFF for safety
        params_layout.addWidget(self.enable_servo_checkbox, 6, 0, 1, 4)

        controls_layout.addWidget(params_group)

        # Motion detection group
        motion_group = QGroupBox("Motion Detection (Impact Rejection)")
        motion_layout = QGridLayout()
        motion_group.setLayout(motion_layout)

        # Enable motion detection
        self.enable_motion_detection_checkbox = QCheckBox("Enable Motion Detection")
        self.enable_motion_detection_checkbox.setChecked(False)  # Default OFF to not affect basic operation
        self.enable_motion_detection_checkbox.stateChanged.connect(self.toggle_motion_detection)
        motion_layout.addWidget(self.enable_motion_detection_checkbox, 0, 0, 1, 4)

        # Accel magnitude threshold
        motion_layout.addWidget(QLabel("Accel Threshold [m/s²]:"), 1, 0)
        self.accel_threshold_input = QLineEdit(str(IMUKalmanConfig.DEFAULT_ACCEL_THRESHOLD))
        self.accel_threshold_input.setMaximumWidth(60)
        self.accel_threshold_input.editingFinished.connect(self.schedule_param_update)
        motion_layout.addWidget(self.accel_threshold_input, 1, 1)
        self.accel_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.accel_threshold_slider.setMinimum(0)
        self.accel_threshold_slider.setMaximum(10000)
        self.accel_threshold_slider.setValue(int(IMUKalmanConfig.DEFAULT_ACCEL_THRESHOLD * 1000))
        self.accel_threshold_slider.valueChanged.connect(self.schedule_param_update)
        motion_layout.addWidget(self.accel_threshold_slider, 1, 2)
        self.accel_threshold_label = QLabel(f"{IMUKalmanConfig.DEFAULT_ACCEL_THRESHOLD:.2f}")
        motion_layout.addWidget(self.accel_threshold_label, 1, 3)

        # Gyro magnitude threshold
        motion_layout.addWidget(QLabel("Gyro Threshold [rad/s]:"), 2, 0)
        self.gyro_threshold_input = QLineEdit("0.5")
        self.gyro_threshold_input.setMaximumWidth(60)
        self.gyro_threshold_input.editingFinished.connect(self.schedule_param_update)
        motion_layout.addWidget(self.gyro_threshold_input, 2, 1)
        self.gyro_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.gyro_threshold_slider.setMinimum(0)
        self.gyro_threshold_slider.setMaximum(5000)
        self.gyro_threshold_slider.setValue(500)
        self.gyro_threshold_slider.valueChanged.connect(self.schedule_param_update)
        motion_layout.addWidget(self.gyro_threshold_slider, 2, 2)
        self.gyro_threshold_label = QLabel("0.50")
        motion_layout.addWidget(self.gyro_threshold_label, 2, 3)

        # Magnetometer backup
        self.enable_magnetometer_checkbox = QCheckBox("Use Magnetometer Backup")
        self.enable_magnetometer_checkbox.setChecked(False)
        self.enable_magnetometer_checkbox.stateChanged.connect(self.toggle_magnetometer)
        motion_layout.addWidget(self.enable_magnetometer_checkbox, 4, 0, 1, 2)

        # Calibrate magnetometer offset button
        self.calibrate_mag_button = QPushButton("Calibrate Mag")
        self.calibrate_mag_button.clicked.connect(self.calibrate_magnetometer_offset)
        motion_layout.addWidget(self.calibrate_mag_button, 4, 2, 1, 2)

        # Rejection statistics
        self.rejection_stats_label = QLabel("Rejected: 0 / 0 (0.0%) | Mag: 0")
        motion_layout.addWidget(self.rejection_stats_label, 5, 0, 1, 4)

        controls_layout.addWidget(motion_group)

        # Axis transformation controls
        transform_group = QGroupBox("IMU Axis Transformations")
        transform_layout = QVBoxLayout()
        transform_group.setLayout(transform_layout)

        # Accel flips
        accel_flip_layout = QHBoxLayout()
        accel_flip_layout.addWidget(QLabel("Accel Flip:"))
        self.accel_flip_x = QCheckBox("X")
        self.accel_flip_x.setChecked(False)
        self.accel_flip_x.stateChanged.connect(self.update_axis_flips)
        accel_flip_layout.addWidget(self.accel_flip_x)
        self.accel_flip_y = QCheckBox("Y")
        self.accel_flip_y.setChecked(False)
        self.accel_flip_y.stateChanged.connect(self.update_axis_flips)
        accel_flip_layout.addWidget(self.accel_flip_y)
        self.accel_flip_z = QCheckBox("Z")
        self.accel_flip_z.setChecked(False)
        self.accel_flip_z.stateChanged.connect(self.update_axis_flips)
        accel_flip_layout.addWidget(self.accel_flip_z)
        transform_layout.addLayout(accel_flip_layout)

        # Gyro flips
        gyro_flip_layout = QHBoxLayout()
        gyro_flip_layout.addWidget(QLabel("Gyro Flip:"))
        self.gyro_flip_x = QCheckBox("X")
        self.gyro_flip_x.setChecked(False)
        self.gyro_flip_x.stateChanged.connect(self.update_axis_flips)
        gyro_flip_layout.addWidget(self.gyro_flip_x)
        self.gyro_flip_y = QCheckBox("Y")
        self.gyro_flip_y.setChecked(False)
        self.gyro_flip_y.stateChanged.connect(self.update_axis_flips)
        gyro_flip_layout.addWidget(self.gyro_flip_y)
        self.gyro_flip_z = QCheckBox("Z")
        self.gyro_flip_z.setChecked(False)
        self.gyro_flip_z.stateChanged.connect(self.update_axis_flips)
        gyro_flip_layout.addWidget(self.gyro_flip_z)
        transform_layout.addLayout(gyro_flip_layout)

        # Reset button
        reset_btn = QPushButton("Reset Transformations")
        reset_btn.clicked.connect(self.reset_transformations)
        transform_layout.addWidget(reset_btn)

        controls_layout.addWidget(transform_group)

        # Suspension control group
        suspension_group = QGroupBox("Active Suspension Control")
        suspension_layout = QGridLayout()
        suspension_group.setLayout(suspension_layout)

        # Enable suspension
        self.enable_suspension_checkbox = QCheckBox("Enable Suspension")
        self.enable_suspension_checkbox.setChecked(False)
        self.enable_suspension_checkbox.stateChanged.connect(self.toggle_suspension)
        suspension_layout.addWidget(self.enable_suspension_checkbox, 0, 0, 1, 4)

        # Position gain (accel feedback)
        suspension_layout.addWidget(QLabel("Accel Gain [mm/(m/s²)]:"), 1, 0)
        self.suspension_pos_gain_input = QLineEdit("0.5")
        self.suspension_pos_gain_input.setMaximumWidth(60)
        self.suspension_pos_gain_input.editingFinished.connect(self.update_suspension_gains)
        suspension_layout.addWidget(self.suspension_pos_gain_input, 1, 1)
        self.suspension_pos_slider = QSlider(Qt.Orientation.Horizontal)
        self.suspension_pos_slider.setMinimum(0)
        self.suspension_pos_slider.setMaximum(5000)
        self.suspension_pos_slider.setValue(500)
        self.suspension_pos_slider.valueChanged.connect(self.update_suspension_gains)
        suspension_layout.addWidget(self.suspension_pos_slider, 1, 2)
        self.suspension_pos_label = QLabel("0.50")
        suspension_layout.addWidget(self.suspension_pos_label, 1, 3)

        # Velocity gain (damping)
        suspension_layout.addWidget(QLabel("Damping Gain [mm/(m/s)]:"), 2, 0)
        self.suspension_vel_gain_input = QLineEdit("0.0")
        self.suspension_vel_gain_input.setMaximumWidth(60)
        self.suspension_vel_gain_input.editingFinished.connect(self.update_suspension_gains)
        suspension_layout.addWidget(self.suspension_vel_gain_input, 2, 1)
        self.suspension_vel_slider = QSlider(Qt.Orientation.Horizontal)
        self.suspension_vel_slider.setMinimum(0)
        self.suspension_vel_slider.setMaximum(5000)
        self.suspension_vel_slider.setValue(0)
        self.suspension_vel_slider.valueChanged.connect(self.update_suspension_gains)
        suspension_layout.addWidget(self.suspension_vel_slider, 2, 2)
        self.suspension_vel_label = QLabel("0.00")
        suspension_layout.addWidget(self.suspension_vel_label, 2, 3)

        controls_layout.addWidget(suspension_group)
        controls_layout.addStretch()

        main_layout.addWidget(controls)

        # Right: Plots
        self.plot_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.plot_widget)

        # Roll plot
        self.plot_rx = self.plot_widget.addPlot(row=0, col=0, title="Roll (RX)")
        self.plot_rx.setLabel('left', 'Angle', units='°')
        self.plot_rx.setLabel('bottom', 'Time', units='s')
        self.plot_rx.showGrid(x=True, y=True, alpha=0.3)
        self.plot_rx.addLegend()
        self.curve_rx_imu = self.plot_rx.plot(pen=pg.mkPen('#ff6b6b', width=2), name='IMU')
        self.curve_rx_cmd = self.plot_rx.plot(pen=pg.mkPen('#4ecdc4', width=2, style=Qt.PenStyle.DashLine), name='Commanded')

        # Pitch plot
        self.plot_ry = self.plot_widget.addPlot(row=1, col=0, title="Pitch (RY)")
        self.plot_ry.setLabel('left', 'Angle', units='°')
        self.plot_ry.setLabel('bottom', 'Time', units='s')
        self.plot_ry.showGrid(x=True, y=True, alpha=0.3)
        self.plot_ry.addLegend()
        self.curve_ry_imu = self.plot_ry.plot(pen=pg.mkPen('#ff6b6b', width=2), name='IMU')
        self.curve_ry_cmd = self.plot_ry.plot(pen=pg.mkPen('#4ecdc4', width=2, style=Qt.PenStyle.DashLine), name='Commanded')

        # Magnetometer plot
        self.plot_mag = self.plot_widget.addPlot(row=2, col=0, title="Magnetometer (Raw)")
        self.plot_mag.setLabel('left', 'Value', units='LSB')
        self.plot_mag.setLabel('bottom', 'Time', units='s')
        self.plot_mag.showGrid(x=True, y=True, alpha=0.3)
        self.plot_mag.addLegend()
        self.curve_mag_x = self.plot_mag.plot(pen=pg.mkPen('#ff6b6b', width=2), name='X')
        self.curve_mag_y = self.plot_mag.plot(pen=pg.mkPen('#4ecdc4', width=2), name='Y')
        self.curve_mag_z = self.plot_mag.plot(pen=pg.mkPen('#95e1d3', width=2), name='Z')

    def schedule_param_update(self):
        """Schedule parameter update with debouncing"""
        self.param_update_timer.start(300)  # 300ms debounce
        self.update_param_labels()

    def update_param_labels(self):
        """Update parameter display labels"""
        try:
            accel_scalar = float(self.accel_noise_scalar.text())
            accel_val = accel_scalar * (self.accel_noise_slider.value() / 1000)
            self.accel_noise_label.setText(f"{accel_val:.3f}")
        except ValueError:
            self.accel_noise_label.setText("Invalid")

        try:
            gyro_scalar = float(self.gyro_noise_scalar.text())
            gyro_val = gyro_scalar * (self.gyro_noise_slider.value() / 1000)
            self.gyro_noise_label.setText(f"{gyro_val:.3f}")
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
            self.proc_bias_label.setText(f"{bias_val:.6f}")
        except ValueError:
            self.proc_bias_label.setText("Invalid")

    def update_axis_flips(self):
        """Update axis flip arrays and recreate filter"""
        self.accel_axis_flip = np.array([
            -1 if self.accel_flip_x.isChecked() else 1,
            -1 if self.accel_flip_y.isChecked() else 1,
            -1 if self.accel_flip_z.isChecked() else 1
        ])
        self.gyro_axis_flip = np.array([
            -1 if self.gyro_flip_x.isChecked() else 1,
            -1 if self.gyro_flip_y.isChecked() else 1,
            -1 if self.gyro_flip_z.isChecked() else 1
        ])
        self.schedule_param_update()

    def reset_transformations(self):
        """Reset all axis transformations to default"""
        self.accel_flip_x.setChecked(False)
        self.accel_flip_y.setChecked(False)
        self.accel_flip_z.setChecked(False)
        self.gyro_flip_x.setChecked(False)
        self.gyro_flip_y.setChecked(False)
        self.gyro_flip_z.setChecked(False)
        self.accel_rotation = ACCEL_ROTATION.copy()
        self.gyro_rotation = GYRO_ROTATION.copy()
        self.schedule_param_update()

    def toggle_suspension(self):
        """Enable/disable suspension control"""
        self.suspension_enabled = self.enable_suspension_checkbox.isChecked()
        if self.suspension_enabled:
            # Reset velocity integrator when enabling
            self.linear_velocity = np.array([0.0, 0.0, 0.0])
            self.last_update_time = time.time()
            print("Suspension control ENABLED")
        else:
            print("Suspension control DISABLED")

    def toggle_motion_detection(self):
        """Enable/disable motion detection (impact rejection)"""
        enabled = self.enable_motion_detection_checkbox.isChecked()
        if hasattr(self, 'kalman'):
            self.kalman.enable_rejection = enabled
            if enabled:
                print("Motion detection enabled: accelerometer updates rejected during impacts")
            else:
                print("Motion detection disabled: all accelerometer updates accepted")
        else:
            print("Kalman filter not yet initialized")

    def toggle_magnetometer(self):
        """Enable/disable magnetometer backup during accel rejection"""
        enabled = self.enable_magnetometer_checkbox.isChecked()
        if hasattr(self, 'kalman'):
            self.kalman.use_magnetometer = enabled
            if enabled:
                print("Magnetometer backup ENABLED - will use mag tilt when accel rejected")
                print(f"Current mag offset: {self.kalman.mag_offset}")
            else:
                print("Magnetometer backup DISABLED - gyro-only during rejection")
        else:
            print("Kalman filter not yet initialized")

    def calibrate_magnetometer_offset(self):
        """Calibrate magnetometer hard-iron offset from current reading"""
        if not hasattr(self, 'current_mag_x'):
            print("No magnetometer data available yet")
            return

        # Use current mag reading as offset (assumes sensor is level and stationary)
        mag_raw = np.array([self.current_mag_x, self.current_mag_y, self.current_mag_z])

        if hasattr(self, 'kalman'):
            self.kalman.mag_offset = mag_raw
            print(f"Magnetometer offset calibrated: [{mag_raw[0]:.1f}, {mag_raw[1]:.1f}, {mag_raw[2]:.1f}]")
            print("Keep sensor level during calibration for best results")
        else:
            print("Kalman filter not yet initialized")

    def update_suspension_gains(self):
        """Update suspension gains from UI"""
        try:
            # Position gain (accel feedback)
            scalar_pos = float(self.suspension_pos_gain_input.text())
            slider_pos = self.suspension_pos_slider.value() / 1000
            self.suspension_position_gain = scalar_pos * slider_pos
            self.suspension_pos_label.setText(f"{self.suspension_position_gain:.2f}")

            # Velocity gain (damping)
            scalar_vel = float(self.suspension_vel_gain_input.text())
            slider_vel = self.suspension_vel_slider.value() / 1000
            self.suspension_velocity_gain = scalar_vel * slider_vel
            self.suspension_vel_label.setText(f"{self.suspension_velocity_gain:.2f}")

        except ValueError:
            pass

    def update_filter_parameters(self):
        """Recreate Kalman filter with new parameters"""
        try:
            accel_noise = float(self.accel_noise_scalar.text()) * (self.accel_noise_slider.value() / 1000)
            gyro_noise = float(self.gyro_noise_scalar.text()) * (self.gyro_noise_slider.value() / 1000)
            gyro_scale_mult = self.gyro_scale_slider.value() / 1000
            proc_noise_angle = float(self.proc_angle_scalar.text()) * (self.proc_angle_slider.value() / 1000)
            proc_noise_bias = float(self.proc_bias_scalar.text()) * (self.proc_bias_slider.value() / 1000)

            # Motion detection thresholds
            accel_threshold = float(self.accel_threshold_input.text()) * (self.accel_threshold_slider.value() / 1000)
            gyro_threshold = float(self.gyro_threshold_input.text()) * (self.gyro_threshold_slider.value() / 1000)

            # Update labels
            self.accel_noise_label.setText(f"{accel_noise:.3f}")
            self.gyro_noise_label.setText(f"{gyro_noise * gyro_scale_mult:.3f}")
            self.gyro_scale_label.setText(f"{gyro_scale_mult:.3f}x")
            self.proc_angle_label.setText(f"{proc_noise_angle:.3f}")
            self.proc_bias_label.setText(f"{proc_noise_bias:.5f}")
            self.accel_threshold_label.setText(f"{accel_threshold:.2f}")
            self.gyro_threshold_label.setText(f"{gyro_threshold:.2f}")

            # Scale gyro noise by multiplier
            gyro_noise_scaled = gyro_noise * gyro_scale_mult

            # Recreate filter with current axis transformations
            self.kalman = OrientationKalmanFilter(
                accel_noise=accel_noise,
                gyro_noise=gyro_noise_scaled,
                process_noise_angle=proc_noise_angle,
                process_noise_bias=proc_noise_bias,
                accel_axis_flip=self.accel_axis_flip,
                gyro_axis_flip=self.gyro_axis_flip,
                accel_rotation=self.accel_rotation,
                gyro_rotation=self.gyro_rotation,
                initial_bias_x=GYRO_BIAS_X,
                initial_bias_y=GYRO_BIAS_Y,
                gyro_scale_multiplier=gyro_scale_mult,
                accel_magnitude_threshold=accel_threshold,
                gyro_magnitude_threshold=gyro_threshold
            )

            # Apply current motion detection enable/disable state
            self.kalman.enable_rejection = self.enable_motion_detection_checkbox.isChecked()

            print(f"Filter updated: accel={accel_noise:.3f}, gyro={gyro_noise_scaled:.3f}, "
                  f"proc_angle={proc_noise_angle:.6f}, proc_bias={proc_noise_bias:.6f}, scale={gyro_scale_mult:.2f}x, "
                  f"accel_thresh={accel_threshold:.2f}, gyro_thresh={gyro_threshold:.2f}, "
                  f"rejection={'ON' if self.kalman.enable_rejection else 'OFF'}")

        except ValueError as e:
            print(f"Parameter update error: {e}")

    def update_plots(self):
        """Update plots with current data"""
        if self.plot_start_time is None:
            self.plot_start_time = time.time()

        elapsed = time.time() - self.plot_start_time

        # Update rejection statistics display
        if hasattr(self, 'kalman') and self.kalman.initialized:
            rejected, total, rate, mag_updates = self.kalman.get_rejection_stats()
            self.rejection_stats_label.setText(f"Rejected: {rejected} / {total} ({rate:.1f}%) | Mag: {mag_updates}")

        # Add current data
        self.time_history.append(elapsed)
        self.rx_imu_history.append(self.current_rx_imu)
        self.ry_imu_history.append(self.current_ry_imu)
        self.rx_cmd_history.append(self.current_rx_cmd)
        self.ry_cmd_history.append(self.current_ry_cmd)

        # Add magnetometer data (or zeros if not available)
        if hasattr(self, 'current_mag_x'):
            self.mag_x_history.append(self.current_mag_x)
            self.mag_y_history.append(self.current_mag_y)
            self.mag_z_history.append(self.current_mag_z)
        else:
            self.mag_x_history.append(0)
            self.mag_y_history.append(0)
            self.mag_z_history.append(0)

        # Update status
        uptime = time.time() - self.start_time
        gyro_hz = self.gyro_count / uptime if uptime > 0 else 0
        accel_hz = self.accel_count / uptime if uptime > 0 else 0
        mag_hz = self.mag_count / uptime if uptime > 0 else 0
        update_hz = self.update_count / uptime if uptime > 0 else 0
        servo_hz = self.servo_command_count / uptime if uptime > 0 else 0

        # Check if still initializing
        if self.initializing:
            if self.initialization_start_time is None:
                self.initialization_start_time = time.time()
            initialization_remaining = self.initialization_duration - (time.time() - self.initialization_start_time)

            if initialization_remaining <= 0:
                # Initialization complete - start calibration
                self.initializing = False
                self.calibrating = True
                self.calibration_start_time = time.time()
                print("\nInitialization complete - Starting calibration...")
            else:
                # Show initialization progress
                self.status_label.setText(
                    f"INITIALIZING... {initialization_remaining:.1f}s remaining (sensors stabilizing)"
                )

        # Check if still calibrating
        elif self.calibrating:
            if self.calibration_start_time is None:
                self.calibration_start_time = time.time()
            calibration_remaining = self.calibration_duration - (time.time() - self.calibration_start_time)

            if calibration_remaining <= 0:
                # Calibration complete - process data
                self.process_calibration_data()
                self.calibrating = False
                print("\nCalibration complete")
            else:
                # Show calibration progress
                # Count buffered lines (thread-safe)
                with self.calibration_lock:
                    total_lines = len(self.calibration_raw_lines)
                self.status_label.setText(
                    f"CALIBRATING... {calibration_remaining:.1f}s remaining | "
                    f"Buffered: {total_lines} lines"
                )
        else:
            servo_status = "[ON]" if self.enable_servo_checkbox.isChecked() else "[OFF]"
            self.status_label.setText(
                f"IMU: rx={self.current_rx_imu:+6.2f}° ry={self.current_ry_imu:+6.2f}° | "
                f"Comp: rx={self.current_rx_cmd:+6.2f}° ry={self.current_ry_cmd:+6.2f}° | "
                f"Servos:{servo_status} | "
                f"Gyro:{gyro_hz:.0f}Hz Accel:{accel_hz:.0f}Hz Mag:{mag_hz:.0f}Hz"
            )

        # Update plots
        if len(self.time_history) > 0:
            time_array = np.array(self.time_history)
            time_relative = time_array - time_array[0]

            self.curve_rx_imu.setData(time_relative, np.array(self.rx_imu_history))
            self.curve_rx_cmd.setData(time_relative, np.array(self.rx_cmd_history))
            self.curve_ry_imu.setData(time_relative, np.array(self.ry_imu_history))
            self.curve_ry_cmd.setData(time_relative, np.array(self.ry_cmd_history))

            # Update magnetometer plot
            self.curve_mag_x.setData(time_relative, np.array(self.mag_x_history))
            self.curve_mag_y.setData(time_relative, np.array(self.mag_y_history))
            self.curve_mag_z.setData(time_relative, np.array(self.mag_z_history))

    def connect(self):
        """Connect to Arduino/Teensy"""
        try:
            print(f"Connecting to {self.port}...")
            # Increase buffer sizes for high-speed data
            self.serial = serial.Serial(
                self.port,
                self.baudrate,
                timeout=0.01,
                write_timeout=0.01
            )
            # Set large OS-level buffers (Windows)
            try:
                self.serial.set_buffer_size(rx_size=65536, tx_size=65536)
            except:
                pass  # May not be supported on all platforms
            time.sleep(2.5)

            # Clear startup messages (don't print them)
            start_wait = time.time()
            init_found = False
            while time.time() - start_wait < 2.0:
                if self.serial.in_waiting:
                    try:
                        line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                        # Only check if we got READY or INIT message
                        if line.startswith("READY:") or line.startswith("INIT:"):
                            init_found = True
                    except:
                        pass

            if init_found:
                print("Connected successfully")
            else:
                print("Connected (no startup message received)")

            return True

        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def disconnect(self):
        """Disconnect"""
        self.running = False
        if self.serial and self.serial.is_open:
            self.serial.close()

    def _read_loop(self):
        """Read IMU data"""
        buffer = ""
        calibration_batch = []  # Batch calibration lines to reduce lock contention

        while self.running:
            try:
                # Read all available data
                bytes_available = self.serial.in_waiting
                if bytes_available > 0:
                    # Read in large chunks for speed
                    chunk = self.serial.read(bytes_available).decode('utf-8', errors='ignore')
                    buffer += chunk

                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()

                        if not line:
                            continue

                        # During calibration: ONLY buffer raw lines (skip all parsing)
                        if self.calibrating:
                            if line.startswith("A:") or line.startswith("G:") or line.startswith("M:"):
                                calibration_batch.append(line)
                                # Batch append every 100 lines to reduce lock overhead
                                if len(calibration_batch) >= 100:
                                    with self.calibration_lock:
                                        self.calibration_raw_lines.extend(calibration_batch)
                                    calibration_batch = []
                            continue  # Skip all other processing during calibration

                        # Normal operation: parse and queue data
                        if line.startswith("A:"):
                            parts = line[2:].split(',')
                            if len(parts) == 4:
                                try:
                                    timestamp_us = int(parts[0])
                                    ax, ay, az = int(parts[1]), int(parts[2]), int(parts[3])
                                    accel_data = np.array([ax, ay, az])

                                    if not self.accel_queue.full():
                                        self.accel_queue.put((timestamp_us, accel_data))
                                        self.accel_count += 1
                                except ValueError:
                                    pass

                        elif line.startswith("G:"):
                            parts = line[2:].split(',')
                            if len(parts) == 4:
                                try:
                                    timestamp_us = int(parts[0])
                                    gx, gy, gz = int(parts[1]), int(parts[2]), int(parts[3])
                                    gyro_data = np.array([gx, gy, gz])

                                    if not self.gyro_queue.full():
                                        self.gyro_queue.put((timestamp_us, gyro_data))
                                        self.gyro_count += 1
                                except ValueError:
                                    pass

                        elif line.startswith("M:"):
                            parts = line[2:].split(',')
                            if len(parts) == 4:
                                try:
                                    timestamp_us = int(parts[0])
                                    mx, my, mz = int(parts[1]), int(parts[2]), int(parts[3])
                                    mag_data = np.array([mx, my, mz])

                                    if not self.mag_queue.full():
                                        self.mag_queue.put((timestamp_us, mag_data))
                                        self.mag_count += 1
                                except ValueError:
                                    pass

                        elif line.startswith("SERVO:"):
                            # Parse commanded servo angles
                            # Format: SERVO:timestamp,s0,s1,s2,s3,s4,s5
                            parts = line[6:].split(',')
                            if len(parts) == 7:
                                try:
                                    # For now, just placeholder
                                    pass
                                except ValueError:
                                    pass
                else:
                    # Flush any remaining calibration batch when no data available
                    if calibration_batch:
                        with self.calibration_lock:
                            self.calibration_raw_lines.extend(calibration_batch)
                        calibration_batch = []
                    # Adjust sleep duration based on calibration state
                    if self.calibrating:
                        time.sleep(0.00001)  # 10 microseconds for fast calibration reading
                    else:
                        time.sleep(0.001)  # 1 millisecond during normal operation

            except Exception as e:
                if self.running:
                    print(f"Read error: {e}")
                break

        # Flush final batch on exit
        if calibration_batch:
            with self.calibration_lock:
                self.calibration_raw_lines.extend(calibration_batch)

    def _control_loop(self):
        """Process IMU data through filter"""
        latest_accel = None
        latest_mag = None

        while self.running:
            try:
                # During initialization or calibration: drain queues and skip filter processing
                # (Serial thread writes directly to calibration_samples during calibration)
                if self.initializing or self.calibrating:
                    # Drain all queues (keep latest for display)
                    while not self.accel_queue.empty():
                        _, accel_raw = self.accel_queue.get_nowait()
                        latest_accel = accel_raw
                    while not self.gyro_queue.empty():
                        self.gyro_queue.get_nowait()
                    while not self.mag_queue.empty():
                        _, mag_raw = self.mag_queue.get_nowait()
                        self.current_mag_x = mag_raw[0]
                        self.current_mag_y = mag_raw[1]
                        self.current_mag_z = mag_raw[2]

                    time.sleep(0.001)
                    continue  # Skip filter processing during initialization/calibration

                # Get checkbox states
                enable_accel = self.enable_accel_checkbox.isChecked()
                enable_gyro = self.enable_gyro_checkbox.isChecked()

                # Initialize latest sensor data
                latest_mag = None

                # Normal operation: Process magnetometer queue
                while not self.mag_queue.empty():
                    timestamp_us, mag_raw = self.mag_queue.get_nowait()
                    latest_mag = mag_raw
                    self.current_mag_x = mag_raw[0]
                    self.current_mag_y = mag_raw[1]
                    self.current_mag_z = mag_raw[2]

                # Process accel queue (get latest accel for each gyro sample)
                while not self.accel_queue.empty():
                    _, latest_accel = self.accel_queue.get_nowait()

                # Process gyro - this drives the filter updates
                while not self.gyro_queue.empty():
                    timestamp_us, gyro_raw = self.gyro_queue.get_nowait()

                    if latest_accel is not None:
                        # Calculate dt
                        current_time = time.time()
                        if self.last_update_time is not None:
                            dt = current_time - self.last_update_time
                            dt = max(dt, 0.0001)
                        else:
                            dt = 0.001
                        self.last_update_time = current_time

                        # Initialize if needed
                        if not self.kalman.initialized:
                            self.kalman.initialize(latest_accel, self.calibrated_gravity_vector)

                        # Predict
                        if enable_gyro:
                            self.kalman.predict(gyro_raw, dt)
                        else:
                            self.kalman.predict(np.zeros(3), dt)

                        # Update (pass magnetometer data for backup during impacts)
                        if enable_accel:
                            self.kalman.update(latest_accel, latest_mag)

                        # Store latest accel for suspension control
                        self.latest_accel = latest_accel

                        # Get orientation
                        roll, pitch = self.kalman.get_orientation()
                        self.current_rx_imu = np.degrees(roll)
                        self.current_ry_imu = np.degrees(pitch)
                        self.update_count += 1

                        # Send compensating servo commands (rate limited)
                        current_time = time.time()
                        if current_time - self.last_servo_time >= self.servo_interval:
                            self.send_compensation()
                            self.last_servo_time = current_time

                time.sleep(0.001)

            except Exception as e:
                if self.running:
                    print(f"Control loop error: {e}")
                    import traceback
                    traceback.print_exc()
                break

    def run(self):
        """Start real-time tracking"""
        self.running = True
        self.start_time = time.time()

        # Start threads
        read_thread = threading.Thread(target=self._read_loop, daemon=True)
        control_thread = threading.Thread(target=self._control_loop, daemon=True)

        read_thread.start()
        control_thread.start()

        # Start plot timer
        self.plot_timer.start()

        print("IMU tracking started - Initializing sensors (3s) then calibrating (10s)...")

    def process_calibration_data(self):
        """Process collected calibration data to extract biases and gravity vector"""
        # Thread-safe copy of raw buffered lines
        with self.calibration_lock:
            raw_lines = self.calibration_raw_lines.copy()

        if len(raw_lines) == 0:
            print("Warning: No calibration data collected")
            return

        print(f"\nParsing {len(raw_lines)} calibration lines...")

        # Diagnostic: check first and last timestamps
        if len(raw_lines) > 0:
            first_line = raw_lines[0]
            last_line = raw_lines[-1]
            try:
                first_ts = int(first_line.split(',')[0].split(':')[1])
                last_ts = int(last_line.split(',')[0].split(':')[1])
                duration_us = last_ts - first_ts
                duration_s = duration_us / 1e6
                print(f"Time span: {duration_s:.2f}s (expected ~10s)")
                print(f"Average rate: {len(raw_lines)/duration_s:.0f} lines/s (expected ~2364 lines/s)")
            except:
                pass

        # Parse all buffered lines
        calibration_samples = []
        for line in raw_lines:
            try:
                if line.startswith("A:"):
                    parts = line[2:].split(',')
                    if len(parts) == 4:
                        timestamp_us = int(parts[0])
                        ax, ay, az = int(parts[1]), int(parts[2]), int(parts[3])
                        calibration_samples.append(('A', timestamp_us, np.array([ax, ay, az])))

                elif line.startswith("G:"):
                    parts = line[2:].split(',')
                    if len(parts) == 4:
                        timestamp_us = int(parts[0])
                        gx, gy, gz = int(parts[1]), int(parts[2]), int(parts[3])
                        calibration_samples.append(('G', timestamp_us, np.array([gx, gy, gz])))

                elif line.startswith("M:"):
                    parts = line[2:].split(',')
                    if len(parts) == 4:
                        timestamp_us = int(parts[0])
                        mx, my, mz = int(parts[1]), int(parts[2]), int(parts[3])
                        calibration_samples.append(('M', timestamp_us, np.array([mx, my, mz])))
            except (ValueError, IndexError):
                pass  # Skip malformed lines

        if len(calibration_samples) == 0:
            print("Warning: No valid calibration samples parsed")
            return

        # Sort all samples by timestamp (like data_logger.py)
        calibration_samples.sort(key=lambda x: x[1])

        # Separate by sensor type
        accel_samples = []
        gyro_samples = []
        mag_samples = []

        for sample_type, timestamp_us, data in calibration_samples:
            if sample_type == 'A':
                accel_samples.append(data)
            elif sample_type == 'G':
                gyro_samples.append(data)
            elif sample_type == 'M':
                mag_samples.append(data)

        if len(accel_samples) == 0 or len(gyro_samples) == 0:
            print("Warning: Missing accelerometer or gyroscope data")
            return

        # Convert to arrays
        accel_samples = np.array(accel_samples)
        gyro_samples = np.array(gyro_samples)

        # Calculate mean raw values
        accel_mean_raw = np.mean(accel_samples, axis=0)
        gyro_mean_raw = np.mean(gyro_samples, axis=0)

        # Convert to physical units (same scaling as in OrientationKalmanFilter)
        accel_scale = 0.001 * 9.81  # LSM303: 1mg/LSB -> m/s²
        gyro_scale_mult = self.gyro_scale_slider.value() / 1000
        gyro_scale = 0.00875 * np.pi / 180 * gyro_scale_mult  # L3GD20: 8.75 mdps/LSB -> rad/s

        # Apply axis transformations
        from core.control_core import apply_imu_transforms
        accel_mean_transformed = apply_imu_transforms(accel_mean_raw, self.accel_axis_flip,
                                                       self.accel_rotation, accel_scale)
        gyro_mean_transformed = apply_imu_transforms(gyro_mean_raw, self.gyro_axis_flip,
                                                      self.gyro_rotation, gyro_scale)

        # Store calibrated values
        self.calibrated_gravity_vector = accel_mean_transformed
        self.calibrated_gyro_bias = gyro_mean_transformed

        # Process magnetometer data if available
        if len(mag_samples) > 0:
            mag_samples_array = np.array(mag_samples)
            mag_mean_raw = np.mean(mag_samples_array, axis=0)
            self.calibrated_mag_offset = mag_mean_raw

        # Print calibration results
        print("\n" + "="*60)
        print("CALIBRATION RESULTS")
        print("="*60)
        print(f"Samples collected: Accel={len(accel_samples)}, Gyro={len(gyro_samples)}", end="")
        if len(mag_samples) > 0:
            print(f", Mag={len(mag_samples)}")
        else:
            print()

        print(f"\nGravity vector [m/s²]: [{accel_mean_transformed[0]:.4f}, "
              f"{accel_mean_transformed[1]:.4f}, {accel_mean_transformed[2]:.4f}]")
        print(f"Gravity magnitude: {np.linalg.norm(accel_mean_transformed):.4f} m/s²")

        # Check if IMU was level during calibration
        tilt_x = np.arctan2(accel_mean_transformed[1], accel_mean_transformed[2])
        tilt_y = np.arctan2(-accel_mean_transformed[0], np.sqrt(accel_mean_transformed[1]**2 + accel_mean_transformed[2]**2))
        tilt_x_deg = np.degrees(tilt_x)
        tilt_y_deg = np.degrees(tilt_y)
        max_tilt = max(abs(tilt_x_deg), abs(tilt_y_deg))

        if max_tilt > 5.0:
            print(f"\nWarning: IMU was tilted during calibration")
            print(f"Tilt: RX={tilt_x_deg:.1f}°, RY={tilt_y_deg:.1f}°")
            print(f"This orientation will be used as the zero reference point")
        else:
            print(f"IMU level check: RX={tilt_x_deg:.1f}°, RY={tilt_y_deg:.1f}°")
        print(f"\nGyroscope bias [rad/s]: [{gyro_mean_transformed[0]:.6f}, "
              f"{gyro_mean_transformed[1]:.6f}, {gyro_mean_transformed[2]:.6f}]")
        print(f"Gyroscope bias [°/s]: [{np.degrees(gyro_mean_transformed[0]):.4f}, "
              f"{np.degrees(gyro_mean_transformed[1]):.4f}, {np.degrees(gyro_mean_transformed[2]):.4f}]")

        if self.calibrated_mag_offset is not None:
            print(f"\nMagnetometer offset [LSB]: [{self.calibrated_mag_offset[0]:.1f}, "
                  f"{self.calibrated_mag_offset[1]:.1f}, {self.calibrated_mag_offset[2]:.1f}]")

        print("="*60 + "\n")

        # Recreate Kalman filter with calibrated values
        try:
            accel_noise = float(self.accel_noise_scalar.text()) * (self.accel_noise_slider.value() / 1000)
            gyro_noise = float(self.gyro_noise_scalar.text()) * (self.gyro_noise_slider.value() / 1000)
            gyro_scale_mult = self.gyro_scale_slider.value() / 1000
            proc_noise_angle = float(self.proc_angle_scalar.text()) * (self.proc_angle_slider.value() / 1000)
            proc_noise_bias = float(self.proc_bias_scalar.text()) * (self.proc_bias_slider.value() / 1000)
            gyro_noise_scaled = gyro_noise * gyro_scale_mult

            # Motion detection thresholds
            accel_threshold = float(self.accel_threshold_input.text()) * (self.accel_threshold_slider.value() / 1000)
            gyro_threshold = float(self.gyro_threshold_input.text()) * (self.gyro_threshold_slider.value() / 1000)

            # Create new filter with calibrated biases
            self.kalman = OrientationKalmanFilter(
                accel_noise=accel_noise,
                gyro_noise=gyro_noise_scaled,
                process_noise_angle=proc_noise_angle,
                process_noise_bias=proc_noise_bias,
                accel_axis_flip=self.accel_axis_flip,
                gyro_axis_flip=self.gyro_axis_flip,
                accel_rotation=self.accel_rotation,
                gyro_rotation=self.gyro_rotation,
                initial_bias_x=gyro_mean_transformed[0],  # Use calibrated bias
                initial_bias_y=gyro_mean_transformed[1],  # Use calibrated bias
                gyro_scale_multiplier=gyro_scale_mult,
                accel_magnitude_threshold=accel_threshold,
                gyro_magnitude_threshold=gyro_threshold
            )

            # Apply current motion detection enable/disable state
            self.kalman.enable_rejection = self.enable_motion_detection_checkbox.isChecked()

            print(f"Kalman filter reinitialized with calibrated values (rejection={'ON' if self.kalman.enable_rejection else 'OFF'})\n")

        except Exception as e:
            print(f"Error reinitializing filter: {e}")

    def send_compensation(self):
        """Calculate and optionally send compensating servo commands to cancel rotation"""
        # Don't send commands during calibration
        if self.calibrating:
            return

        try:
            # Compensation angles (negative to cancel rotation)
            rx_compensate = -self.current_rx_imu * self.compensation_gain
            ry_compensate = -self.current_ry_imu * self.compensation_gain

            # Limit compensation
            max_angle = 15.0
            rx_compensate = np.clip(rx_compensate, -max_angle, max_angle)
            ry_compensate = np.clip(ry_compensate, -max_angle, max_angle)

            # Store commanded angles for plotting (always calculate, even if not sending)
            self.current_rx_cmd = rx_compensate
            self.current_ry_cmd = ry_compensate

            # Only compute IK and send if servo commands are enabled
            if self.enable_servo_checkbox.isChecked():
                # Calculate suspension compensation from linear acceleration
                translation_compensation = np.array([0.0, 0.0, 0.0])

                if self.suspension_enabled and hasattr(self, 'latest_accel'):
                    # Get linear acceleration (gravity removed)
                    linear_accel = self.kalman.get_linear_acceleration(self.latest_accel)

                    # Time step for integration
                    current_time = time.time()
                    dt = current_time - self.last_update_time
                    self.last_update_time = current_time

                    # Integrate acceleration to get velocity (simple Euler)
                    self.linear_velocity += linear_accel * dt

                    # Apply decay to velocity (prevents drift)
                    self.linear_velocity *= 0.95

                    # Suspension control law: move opposite to acceleration (like damper)
                    # Position term: proportional to acceleration
                    # Velocity term: damping based on integrated velocity
                    translation_compensation = (
                        -self.suspension_position_gain * linear_accel * 1000.0 +  # m/s² to mm
                        -self.suspension_velocity_gain * self.linear_velocity * 1000.0  # m/s to mm
                    )

                    # Limit translation compensation (±20mm)
                    translation_compensation = np.clip(translation_compensation, -20.0, 20.0)

                # Compute IK with translation compensation
                translation = np.array([
                    translation_compensation[0],
                    translation_compensation[1],
                    self.ik.home_height_top_surface + translation_compensation[2]
                ])
                rotation = np.array([rx_compensate, ry_compensate, 0.0])

                servo_angles = self.ik.calculate_servo_angles(translation, rotation, use_top_surface_offset=True)

                if servo_angles is not None:
                    # Send command
                    cmd = ','.join([f"{angle:.2f}" for angle in servo_angles]) + '\n'

                    if self.serial and self.serial.is_open:
                        self.serial.write(cmd.encode('utf-8'))
                        self.servo_command_count += 1

        except Exception as e:
            print(f"Compensation error: {e}")

    def closeEvent(self, event):
        """Handle window close"""
        print("\nClosing...")
        self.disconnect()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description='Real-time IMU orientation tracking')
    parser.add_argument('--port', type=str, required=True, help='Serial port')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = RealtimeOrientationWindow(args.port)
    window.show()

    if window.connect():
        window.run()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
