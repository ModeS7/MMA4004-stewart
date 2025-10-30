#!/usr/bin/env python3
"""
Real-time IMU Orientation Tracking

Live version of plot_data.py - streams IMU data and shows orientation estimates
in real-time with interactive parameter tuning and active platform compensation.

Features:
- 10-second startup calibration phase to measure gyroscope bias and gravity vector
- Real-time Kalman filter for roll (RX) and pitch (RY) estimation
- Active compensation: sends servo commands to cancel platform rotations
- Live parameter tuning sliders
- Axis transformation controls

Calibration Phase:
- Collects 10 seconds of stationary IMU data at startup
- Calculates gyroscope bias from mean gyro readings
- Establishes gravity vector from mean accelerometer readings
- Initializes Kalman filter with measured values for drift compensation

Usage:
    python rot_core.py --port COM4
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

from core.control_core import OrientationKalmanFilter, apply_imu_transforms, GRAVITY_VECTOR, GRAVITY_MAGNITUDE
from core.core import StewartPlatformIK

# PyQtGraph dark theme
pg.setConfigOption('background', '#1e1e1e')
pg.setConfigOption('foreground', 'w')
pg.setConfigOption('antialias', True)


# Default parameters
ACCEL_NOISE = 1.0
GYRO_NOISE = 0.0224
PROCESS_NOISE_ANGLE = 0.0
PROCESS_NOISE_BIAS = 0.0
GYRO_BIAS_X = 0.112679
GYRO_BIAS_Y = 0.031500

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
            gyro_scale_multiplier=6.6
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

        # Calibration phase
        self.calibrating = True
        self.calibration_duration = 10.0  # 10 seconds calibration
        self.calibration_start_time = None
        self.calibration_accel_samples = []
        self.calibration_gyro_samples = []
        self.calibration_mag_samples = []
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

        controls_layout.addWidget(params_group)

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

    def update_filter_parameters(self):
        """Recreate Kalman filter with new parameters"""
        try:
            accel_noise = float(self.accel_noise_scalar.text()) * (self.accel_noise_slider.value() / 1000)
            gyro_noise = float(self.gyro_noise_scalar.text()) * (self.gyro_noise_slider.value() / 1000)
            gyro_scale_mult = self.gyro_scale_slider.value() / 1000
            proc_noise_angle = float(self.proc_angle_scalar.text()) * (self.proc_angle_slider.value() / 1000)
            proc_noise_bias = float(self.proc_bias_scalar.text()) * (self.proc_bias_slider.value() / 1000)

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
                gyro_scale_multiplier=gyro_scale_mult
            )

            print(f"Filter updated: accel={accel_noise:.3f}, gyro={gyro_noise_scaled:.3f}, "
                  f"proc_angle={proc_noise_angle:.6f}, proc_bias={proc_noise_bias:.6f}, scale={gyro_scale_mult:.2f}x")

        except ValueError as e:
            print(f"Parameter update error: {e}")

    def update_plots(self):
        """Update plots with current data"""
        if self.plot_start_time is None:
            self.plot_start_time = time.time()

        elapsed = time.time() - self.plot_start_time

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

        # Check if still calibrating
        if self.calibrating:
            if self.calibration_start_time is None:
                self.calibration_start_time = time.time()
            calibration_remaining = self.calibration_duration - (time.time() - self.calibration_start_time)

            if calibration_remaining <= 0:
                # Calibration complete - process data
                self.process_calibration_data()
                self.calibrating = False
                print("\nCalibration complete - Starting active compensation")
            else:
                # Show calibration progress
                accel_samples = len(self.calibration_accel_samples)
                gyro_samples = len(self.calibration_gyro_samples)
                mag_samples = len(self.calibration_mag_samples)
                self.status_label.setText(
                    f"CALIBRATING... {calibration_remaining:.1f}s remaining | "
                    f"Samples: Accel={accel_samples} Gyro={gyro_samples} Mag={mag_samples}"
                )
        else:
            self.status_label.setText(
                f"IMU: rx={self.current_rx_imu:+6.2f}° ry={self.current_ry_imu:+6.2f}° | "
                f"Comp: rx={self.current_rx_cmd:+6.2f}° ry={self.current_ry_cmd:+6.2f}° | "
                f"Gyro:{gyro_hz:.0f}Hz Accel:{accel_hz:.0f}Hz Mag:{mag_hz:.0f}Hz Servo:{servo_hz:.0f}Hz"
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
        """Connect to Teensy"""
        try:
            print(f"Connecting to {self.port} at {self.baudrate} baud...")
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            time.sleep(2.5)

            # Read startup
            start_wait = time.time()
            while time.time() - start_wait < 2.0:
                if self.serial.in_waiting:
                    try:
                        line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                        if line and not line.startswith("A:") and not line.startswith("G:"):
                            print(f"  Teensy: {line}")
                    except:
                        pass

            print("Connected successfully\n")
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

        while self.running:
            try:
                if self.serial.in_waiting > 0:
                    chunk = self.serial.read(self.serial.in_waiting).decode('utf-8', errors='ignore')
                    buffer += chunk

                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()

                        if not line:
                            continue

                        if line.startswith("A:"):
                            parts = line[2:].split(',')
                            if len(parts) == 4:
                                try:
                                    timestamp_us = int(parts[0])
                                    ax, ay, az = int(parts[1]), int(parts[2]), int(parts[3])
                                    if not self.accel_queue.full():
                                        self.accel_queue.put((timestamp_us, np.array([ax, ay, az])))
                                        self.accel_count += 1
                                except ValueError:
                                    pass

                        elif line.startswith("G:"):
                            parts = line[2:].split(',')
                            if len(parts) == 4:
                                try:
                                    timestamp_us = int(parts[0])
                                    gx, gy, gz = int(parts[1]), int(parts[2]), int(parts[3])
                                    if not self.gyro_queue.full():
                                        self.gyro_queue.put((timestamp_us, np.array([gx, gy, gz])))
                                        self.gyro_count += 1
                                except ValueError:
                                    pass

                        elif line.startswith("M:"):
                            parts = line[2:].split(',')
                            if len(parts) == 4:
                                try:
                                    timestamp_us = int(parts[0])
                                    mx, my, mz = int(parts[1]), int(parts[2]), int(parts[3])
                                    if not self.mag_queue.full():
                                        self.mag_queue.put((timestamp_us, np.array([mx, my, mz])))
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
                    time.sleep(0.0001)

            except Exception as e:
                if self.running:
                    print(f"Read error: {e}")
                break

    def _control_loop(self):
        """Process IMU data through filter"""
        latest_accel = None
        latest_mag = None
        enable_accel = self.enable_accel_checkbox.isChecked()
        enable_gyro = self.enable_gyro_checkbox.isChecked()

        while self.running:
            try:
                # During calibration: collect ALL samples from all queues independently
                if self.calibrating:
                    # Collect all accelerometer samples
                    while not self.accel_queue.empty():
                        _, accel_raw = self.accel_queue.get_nowait()
                        self.calibration_accel_samples.append(accel_raw.copy())
                        latest_accel = accel_raw

                    # Collect all gyroscope samples
                    while not self.gyro_queue.empty():
                        _, gyro_raw = self.gyro_queue.get_nowait()
                        self.calibration_gyro_samples.append(gyro_raw.copy())

                    # Collect all magnetometer samples
                    while not self.mag_queue.empty():
                        _, mag_raw = self.mag_queue.get_nowait()
                        self.calibration_mag_samples.append(mag_raw.copy())
                        self.current_mag_x = mag_raw[0]
                        self.current_mag_y = mag_raw[1]
                        self.current_mag_z = mag_raw[2]

                    time.sleep(0.001)
                    continue  # Skip filter processing during calibration

                # Normal operation: Process magnetometer queue (for display only)
                while not self.mag_queue.empty():
                    timestamp_us, mag_raw = self.mag_queue.get_nowait()
                    latest_mag = mag_raw
                    self.current_mag_x = mag_raw[0]
                    self.current_mag_y = mag_raw[1]
                    self.current_mag_z = mag_raw[2]

                # Process gyro
                while not self.gyro_queue.empty():
                    timestamp_us, gyro_raw = self.gyro_queue.get_nowait()

                    # Get latest accel
                    while not self.accel_queue.empty():
                        _, latest_accel = self.accel_queue.get_nowait()

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
                            self.kalman.initialize(latest_accel)

                        # Predict
                        if enable_gyro:
                            self.kalman.predict(gyro_raw, dt)
                        else:
                            self.kalman.predict(np.zeros(3), dt)

                        # Update
                        if enable_accel:
                            self.kalman.update(latest_accel)

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

        print("Real-time orientation tracking started")

    def process_calibration_data(self):
        """Process collected calibration data to extract biases and gravity vector"""
        if len(self.calibration_accel_samples) == 0 or len(self.calibration_gyro_samples) == 0:
            print("WARNING: No calibration data collected!")
            return

        # Convert lists to arrays
        accel_samples = np.array(self.calibration_accel_samples)
        gyro_samples = np.array(self.calibration_gyro_samples)

        # Calculate mean raw values
        accel_mean_raw = np.mean(accel_samples, axis=0)
        gyro_mean_raw = np.mean(gyro_samples, axis=0)

        # Convert to physical units (same scaling as in OrientationKalmanFilter)
        accel_scale = 0.001 * 9.81  # LSM303: 1mg/LSB -> m/s²
        gyro_scale = 0.00875 * np.pi / 180 * 6.6  # L3GD20: 8.75 mdps/LSB * 6.6x -> rad/s

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
        if len(self.calibration_mag_samples) > 0:
            mag_samples = np.array(self.calibration_mag_samples)
            mag_mean_raw = np.mean(mag_samples, axis=0)
            self.calibrated_mag_offset = mag_mean_raw

        # Print calibration results
        print("\n" + "="*60)
        print("CALIBRATION RESULTS")
        print("="*60)
        print(f"Samples collected: Accel={len(accel_samples)}, Gyro={len(gyro_samples)}", end="")
        if len(self.calibration_mag_samples) > 0:
            print(f", Mag={len(self.calibration_mag_samples)}")
        else:
            print()

        print(f"\nGravity vector [m/s²]: [{accel_mean_transformed[0]:.4f}, "
              f"{accel_mean_transformed[1]:.4f}, {accel_mean_transformed[2]:.4f}]")
        print(f"Gravity magnitude: {np.linalg.norm(accel_mean_transformed):.4f} m/s²")
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
                gyro_scale_multiplier=gyro_scale_mult
            )

            # Override the gravity vector in the filter
            self.kalman.gravity_initial = self.calibrated_gravity_vector.copy()

            print("Kalman filter reinitialized with calibrated values\n")

        except Exception as e:
            print(f"Error reinitializing filter: {e}")

    def send_compensation(self):
        """Send compensating servo commands to cancel rotation"""
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

            # Store commanded angles for plotting
            self.current_rx_cmd = rx_compensate
            self.current_ry_cmd = ry_compensate

            # Compute IK with NO translation (x=0, y=0, z=0)
            translation = np.array([0.0, 0.0, self.ik.home_height_top_surface])
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
