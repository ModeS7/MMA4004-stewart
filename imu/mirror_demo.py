#!/usr/bin/env python3
"""
IMU Mirror Demo - Platform mimics IMU 6DOF pose (Suspension Mode)

Platform follows IMU orientation and reacts to movements:
- Orientation: Kalman filter fusing gyro + accelerometer (stable tracking)
- Position: Double integration with high-pass filter (suspension effect)
  * Reacts to sudden movements/accelerations
  * Naturally drifts back to center (0,0,0)

Usage:
    python imu/mirror_demo.py --port COM3
    python imu/mirror_demo.py --port COM3 --invert-imu --remap-gyro
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import serial
import serial.tools.list_ports
import numpy as np
import time
import argparse
import threading
from queue import Queue, Empty
from collections import deque

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import QTimer
import pyqtgraph as pg

from core.control_core import IMUKalmanFilter
from core.core import StewartPlatformIK


class IMUMirrorController(QMainWindow):
    """
    IMU mirror controller: platform mimics IMU 6DOF pose.

    Features:
    - Orientation estimation: Dual-rate Kalman filter (gyro ~759 Hz, accel ~1265 Hz)
    - Position estimation: Double integration of linear acceleration
    - Real-time servo control with IK
    - Gyro bias calibration mode
    - Real-time plotting of all IMU data and estimates
    """

    def __init__(self, port, baudrate=2000000, invert_imu=False, remap_gyro=False, no_servos=False, rotation_threshold=20.0):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.connected = False
        self.invert_imu = invert_imu  # Invert IMU axes for upside-down mounting
        self.remap_gyro = remap_gyro  # Swap gyro X/Y axes (for sensor misalignment)
        self.no_servos = no_servos    # Skip servo commands (IMU tracking only)
        self.rotation_threshold = rotation_threshold  # Gyro magnitude threshold for rotation suppression

        # IMU Kalman filter with measured noise parameters
        self.kalman = IMUKalmanFilter(
            gyro_scale=0.00875,   # L3GD20: 0.00875 deg/s per raw unit (250 dps mode)
            accel_scale=0.001,    # LSM303: 0.001 g per raw unit (2g mode)
            dt_gyro=1.0/768.06,   # Measured gyro rate: 768.06 Hz
            dynamic_accel_threshold=2.0,  # m/s² threshold for quasi-static detection
            # Noise std from analyze_imu.py (uses measured defaults if not specified)
            accel_noise_std=[0.0701, 0.0651, 0.0889],  # m/s²
            gyro_noise_std=[0.017550, 0.017716, 0.004335]  # rad/s
        )

        # Platform inverse kinematics
        self.ik = StewartPlatformIK()

        # Communication threads
        self.read_thread = None
        self.control_thread = None
        self.running = False

        # Data queues (larger sizes to prevent drops during calibration)
        self.accel_queue = Queue(maxsize=2000)  # ~1.5s buffer at 1340Hz
        self.gyro_queue = Queue(maxsize=1000)   # ~1.3s buffer at 760Hz
        self.command_queue = Queue(maxsize=20)

        # Calibration
        self.calibration_mode = False
        self.calibration_gyro_samples = []
        self.calibration_accel_samples = []
        self.calibration_duration = 3.0

        # Statistics
        self.gyro_count = 0  # Total gyro samples received from serial
        self.accel_count = 0  # Total accel samples received from serial
        self.gyro_processed_count = 0  # Gyro samples processed by Kalman filter
        self.accel_processed_count = 0  # Accel samples processed by Kalman filter
        self.gyro_dropped_count = 0  # Gyro samples dropped (queue full)
        self.accel_dropped_count = 0  # Accel samples dropped (queue full)
        self.update_count = 0
        self.rejected_update_count = 0
        self.servo_command_count = 0
        self.start_time = time.time()

        # Current state
        self.current_rx = 0.0
        self.current_ry = 0.0
        self.current_pos_x = 0.0  # mm
        self.current_pos_y = 0.0  # mm
        self.current_pos_z = 0.0  # mm (relative to home height)
        self.current_vel_x = 0.0  # mm/s
        self.current_vel_y = 0.0  # mm/s
        self.current_vel_z = 0.0  # mm/s
        self.last_accel_mag = 0.0

        # Position integration
        self.last_position_update_time = None

        # Measured gravity (will be calibrated at startup)
        self.gravity_magnitude = 9.81  # m/s² (updated during calibration)

        # Accelerometer biases (measured during calibration when stationary)
        self.accel_bias_x = 0.0  # m/s²
        self.accel_bias_y = 0.0  # m/s²
        self.accel_bias_z = 0.0  # m/s²

        # Servo command rate limiting
        self.last_servo_command_time = 0.0
        self.servo_command_interval = 1.0 / 200.0  # 200 Hz max servo command rate

        # Plot data history (last 30 seconds at 20 Hz update rate)
        self.max_history = 600  # 30 seconds * 20 Hz = 600 samples
        self.time_history = deque(maxlen=self.max_history)
        self.rx_history = deque(maxlen=self.max_history)
        self.ry_history = deque(maxlen=self.max_history)
        self.pos_x_history = deque(maxlen=self.max_history)
        self.pos_y_history = deque(maxlen=self.max_history)
        self.pos_z_history = deque(maxlen=self.max_history)
        self.accel_x_history = deque(maxlen=self.max_history)
        self.accel_y_history = deque(maxlen=self.max_history)
        self.accel_z_history = deque(maxlen=self.max_history)
        self.gyro_x_history = deque(maxlen=self.max_history)
        self.gyro_y_history = deque(maxlen=self.max_history)
        self.gyro_z_history = deque(maxlen=self.max_history)
        self.plot_start_time = None

        # Latest raw sensor values (for plotting)
        self.latest_accel = [0.0, 0.0, 0.0]
        self.latest_gyro = [0.0, 0.0, 0.0]

        # Rotation detection (to suppress position updates during rotation)
        self.gyro_magnitude = 0.0  # Current gyroscope magnitude (deg/s)
        # rotation_threshold set in constructor parameter
        self.is_rotating = False

        # Setup UI
        self.setup_ui()

        # Plot update timer
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plots)
        self.plot_timer.setInterval(50)  # 20 Hz update rate

    def setup_ui(self):
        """Setup PyQtGraph UI with real-time plots"""
        self.setWindowTitle("IMU Mirror Demo - Real-time Tracking")
        self.resize(1400, 900)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Status label
        self.status_label = QLabel("Initializing...")
        layout.addWidget(self.status_label)

        # Create plot widget
        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget)

        # Row 0: Orientation (RX, RY)
        self.plot_rx = self.plot_widget.addPlot(row=0, col=0, title="Roll (RX)")
        self.plot_rx.setLabel('left', 'Angle', units='degrees')
        self.plot_rx.setLabel('bottom', 'Time', units='s')
        self.plot_rx.showGrid(x=True, y=True, alpha=0.3)
        self.curve_rx = self.plot_rx.plot(pen=pg.mkPen('c', width=2))

        self.plot_ry = self.plot_widget.addPlot(row=0, col=1, title="Pitch (RY)")
        self.plot_ry.setLabel('left', 'Angle', units='degrees')
        self.plot_ry.setLabel('bottom', 'Time', units='s')
        self.plot_ry.showGrid(x=True, y=True, alpha=0.3)
        self.curve_ry = self.plot_ry.plot(pen=pg.mkPen('c', width=2))

        # Row 1: Position (X, Y, Z)
        self.plot_x = self.plot_widget.addPlot(row=1, col=0, title="Position X")
        self.plot_x.setLabel('left', 'Position', units='mm')
        self.plot_x.setLabel('bottom', 'Time', units='s')
        self.plot_x.showGrid(x=True, y=True, alpha=0.3)
        self.curve_x = self.plot_x.plot(pen=pg.mkPen('r', width=2))

        self.plot_y = self.plot_widget.addPlot(row=1, col=1, title="Position Y")
        self.plot_y.setLabel('left', 'Position', units='mm')
        self.plot_y.setLabel('bottom', 'Time', units='s')
        self.plot_y.showGrid(x=True, y=True, alpha=0.3)
        self.curve_y = self.plot_y.plot(pen=pg.mkPen('g', width=2))

        self.plot_z = self.plot_widget.addPlot(row=1, col=2, title="Position Z")
        self.plot_z.setLabel('left', 'Position', units='mm')
        self.plot_z.setLabel('bottom', 'Time', units='s')
        self.plot_z.showGrid(x=True, y=True, alpha=0.3)
        self.curve_z = self.plot_z.plot(pen=pg.mkPen('b', width=2))

        # Row 2: Accelerometer (ax, ay, az)
        self.plot_accel = self.plot_widget.addPlot(row=2, col=0, colspan=2, title="Accelerometer")
        self.plot_accel.setLabel('left', 'Acceleration', units='g')
        self.plot_accel.setLabel('bottom', 'Time', units='s')
        self.plot_accel.addLegend()
        self.plot_accel.showGrid(x=True, y=True, alpha=0.3)
        self.curve_accel_x = self.plot_accel.plot(pen=pg.mkPen('r', width=2), name='AX')
        self.curve_accel_y = self.plot_accel.plot(pen=pg.mkPen('g', width=2), name='AY')
        self.curve_accel_z = self.plot_accel.plot(pen=pg.mkPen('b', width=2), name='AZ')

        # Row 2: Gyroscope (gx, gy, gz)
        self.plot_gyro = self.plot_widget.addPlot(row=2, col=2, title="Gyroscope")
        self.plot_gyro.setLabel('left', 'Angular velocity', units='deg/s')
        self.plot_gyro.setLabel('bottom', 'Time', units='s')
        self.plot_gyro.addLegend()
        self.plot_gyro.showGrid(x=True, y=True, alpha=0.3)
        self.curve_gyro_x = self.plot_gyro.plot(pen=pg.mkPen('r', width=2), name='GX')
        self.curve_gyro_y = self.plot_gyro.plot(pen=pg.mkPen('g', width=2), name='GY')
        self.curve_gyro_z = self.plot_gyro.plot(pen=pg.mkPen('b', width=2), name='GZ')

    def update_plots(self):
        """Update all plots with current data"""
        if self.plot_start_time is None:
            self.plot_start_time = time.time()

        elapsed = time.time() - self.plot_start_time

        # Update status label with rotation suppression indicator
        rot_indicator = " [ROTATION SUPPRESSED]" if self.is_rotating else ""
        status_text = (f"Running - Pos: ({self.current_pos_x:.1f}, {self.current_pos_y:.1f}, {self.current_pos_z:.1f})mm | "
                       f"Rot: ({self.current_rx:.1f}, {self.current_ry:.1f})° | "
                       f"Gyro: {self.gyro_magnitude:.1f}°/s{rot_indicator}")
        self.status_label.setText(status_text)

        # Add current data to history
        self.time_history.append(elapsed)
        self.rx_history.append(self.current_rx)
        self.ry_history.append(self.current_ry)
        self.pos_x_history.append(self.current_pos_x)
        self.pos_y_history.append(self.current_pos_y)
        self.pos_z_history.append(self.current_pos_z)
        self.accel_x_history.append(self.latest_accel[0])
        self.accel_y_history.append(self.latest_accel[1])
        self.accel_z_history.append(self.latest_accel[2])
        self.gyro_x_history.append(self.latest_gyro[0])
        self.gyro_y_history.append(self.latest_gyro[1])
        self.gyro_z_history.append(self.latest_gyro[2])

        # Update plots with rolling 30-second window
        if len(self.time_history) > 0:
            time_array = np.array(self.time_history)

            # Make time relative to oldest sample (rolling 30s window)
            time_relative = time_array - time_array[0]

            # Orientation
            self.curve_rx.setData(time_relative, np.array(self.rx_history))
            self.curve_ry.setData(time_relative, np.array(self.ry_history))

            # Position
            self.curve_x.setData(time_relative, np.array(self.pos_x_history))
            self.curve_y.setData(time_relative, np.array(self.pos_y_history))
            self.curve_z.setData(time_relative, np.array(self.pos_z_history))

            # Accelerometer
            self.curve_accel_x.setData(time_relative, np.array(self.accel_x_history))
            self.curve_accel_y.setData(time_relative, np.array(self.accel_y_history))
            self.curve_accel_z.setData(time_relative, np.array(self.accel_z_history))

            # Gyroscope
            self.curve_gyro_x.setData(time_relative, np.array(self.gyro_x_history))
            self.curve_gyro_y.setData(time_relative, np.array(self.gyro_y_history))
            self.curve_gyro_z.setData(time_relative, np.array(self.gyro_z_history))

    def closeEvent(self, event):
        """Handle window close"""
        print("\nClosing application...")
        self.disconnect()
        event.accept()

    def connect(self):
        """Connect to Teensy running IMU_control.ino"""
        try:
            print(f"Connecting to {self.port} at {self.baudrate} baud...")
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            time.sleep(2.5)  # Wait for Teensy reset

            # Read startup messages
            startup_messages = []
            start_wait = time.time()
            while time.time() - start_wait < 2.0:
                if self.serial.in_waiting:
                    try:
                        line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            startup_messages.append(line)
                            # Only print non-IMU data messages
                            if not line.startswith("A:") and not line.startswith("G:"):
                                print(f"  Teensy: {line}")
                    except:
                        pass

            self.connected = True
            print("Connected successfully\n")
            return True

        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def disconnect(self):
        """Disconnect from Teensy"""
        self.running = False

        if self.read_thread:
            self.read_thread.join(timeout=1.0)
        if self.control_thread:
            self.control_thread.join(timeout=1.0)

        if self.serial and self.serial.is_open:
            self.serial.close()

        self.connected = False
        print("\nDisconnected")

    def _read_loop(self):
        """Read IMU data from serial port"""
        buffer = ""

        while self.running:
            try:
                # Read all available data in chunks
                if self.serial.in_waiting > 0:
                    chunk = self.serial.read(self.serial.in_waiting).decode('utf-8', errors='ignore')
                    buffer += chunk

                    # Process all complete lines in buffer
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()

                        if not line:
                            continue

                        if line.startswith("A:"):
                            # Accelerometer: A:timestamp_us,ax,ay,az
                            parts = line[2:].split(',')
                            if len(parts) == 4:
                                try:
                                    timestamp_us = int(parts[0])
                                    ax = int(parts[1])
                                    ay = int(parts[2])
                                    az = int(parts[3])

                                    # Invert axes if IMU is mounted upside down
                                    if self.invert_imu:
                                        ax = -ax
                                        ay = -ay
                                        az = -az

                                    # Store latest values for plotting (convert to g)
                                    self.latest_accel = [
                                        ax * 0.001,  # LSM303: 1mg/digit
                                        ay * 0.001,
                                        az * 0.001
                                    ]

                                    if not self.accel_queue.full():
                                        self.accel_queue.put((timestamp_us, ax, ay, az))
                                        self.accel_count += 1
                                    else:
                                        self.accel_dropped_count += 1

                                except ValueError:
                                    pass

                        elif line.startswith("G:"):
                            # Gyroscope: G:timestamp_us,gx,gy,gz
                            parts = line[2:].split(',')
                            if len(parts) == 4:
                                try:
                                    timestamp_us = int(parts[0])
                                    gx = int(parts[1])
                                    gy = int(parts[2])
                                    gz = int(parts[3])

                                    # Invert axes if IMU is mounted upside down
                                    if self.invert_imu:
                                        gx = -gx
                                        gy = -gy
                                        gz = -gz

                                    # Remap gyro axes (swap X/Y for sensor misalignment)
                                    if self.remap_gyro:
                                        gx, gy = gy, gx

                                    # Store latest values for plotting (convert to deg/s)
                                    gyro_scale_deg = 0.00875  # L3GD20: 8.75 mdps/digit
                                    self.latest_gyro = [
                                        gx * gyro_scale_deg,
                                        gy * gyro_scale_deg,
                                        gz * gyro_scale_deg
                                    ]

                                    # Update gyroscope magnitude for rotation detection
                                    self.gyro_magnitude = np.sqrt(
                                        (gx * gyro_scale_deg)**2 +
                                        (gy * gyro_scale_deg)**2 +
                                        (gz * gyro_scale_deg)**2
                                    )
                                    self.is_rotating = self.gyro_magnitude > self.rotation_threshold

                                    if not self.gyro_queue.full():
                                        self.gyro_queue.put((timestamp_us, gx, gy, gz))
                                        self.gyro_count += 1
                                    else:
                                        self.gyro_dropped_count += 1

                                except ValueError:
                                    pass

                        elif line.startswith("READY:"):
                            print(f"  {line}")
                        elif line.startswith("RATE:"):
                            pass  # Suppress rate reports
                else:
                    # No data available, small sleep
                    time.sleep(0.0001)

            except Exception as e:
                if self.running:
                    print(f"Read error: {e}")
                    import traceback
                    traceback.print_exc()
                break

    def _control_loop(self):
        """Main control loop: process IMU data and send servo commands"""
        last_status_time = time.time()
        status_interval = 1.0  # Print status every second

        # Latest accelerometer reading
        latest_accel = None
        gyro_processed = False

        print("DEBUG: Control loop started")
        loop_count = 0

        while self.running:
            try:
                # Debug first few loops
                loop_count += 1
                if loop_count <= 5:
                    print(f"\nDEBUG Loop {loop_count}: gyro_queue={self.gyro_queue.qsize()}, accel_queue={self.accel_queue.qsize()}")

                # Process gyro samples (prediction)
                gyro_count = 0
                # During calibration, drain queue completely to avoid drops
                max_samples = 1000 if self.calibration_mode else 50
                while not self.gyro_queue.empty() and gyro_count < max_samples:
                    timestamp_us, gx, gy, gz = self.gyro_queue.get_nowait()

                    if self.calibration_mode:
                        self.calibration_gyro_samples.append([gx, gy, gz])
                    else:
                        # Run Kalman prediction
                        state = self.kalman.predict([gx, gy, gz])
                        self.current_rx, self.current_ry = state[0], state[1]
                        self.gyro_processed_count += 1
                        gyro_processed = True
                    gyro_count += 1

                if loop_count <= 5 and gyro_count > 0:
                    print(f"  Processed {gyro_count} gyro samples, current angle: rx={self.current_rx:.2f}°, ry={self.current_ry:.2f}°")

                # Process accelerometer samples (keep only latest)
                accel_count = 0
                # During calibration, drain queue completely to avoid drops
                max_samples = 1000 if self.calibration_mode else 50
                while not self.accel_queue.empty() and accel_count < max_samples:
                    timestamp_us, ax, ay, az = self.accel_queue.get_nowait()
                    if self.calibration_mode:
                        self.calibration_accel_samples.append([ax, ay, az])
                    else:
                        self.accel_processed_count += 1
                    latest_accel = [ax, ay, az]  # Keep only latest
                    accel_count += 1

                if loop_count <= 5 and accel_count > 0:
                    print(f"  Processed {accel_count} accel samples")

                # Update with latest accelerometer reading
                if latest_accel is not None and not self.calibration_mode:
                    updated, state = self.kalman.update(latest_accel)
                    if updated:
                        self.update_count += 1
                        self.current_rx, self.current_ry = state[0], state[1]
                    else:
                        self.rejected_update_count += 1

                    self.last_accel_mag = self.kalman.last_accel_magnitude

                    # Estimate position (simplified - just decay to zero)
                    self.update_position(latest_accel)
                    latest_accel = None

                # Send servo commands if state changed (rate limited)
                if gyro_processed and not self.calibration_mode:
                    current_time = time.time()
                    if current_time - self.last_servo_command_time >= self.servo_command_interval:
                        # Handle upside-down IMU mounting: invert orientation if angles are near ±180°
                        rx_cmd = self.current_rx
                        ry_cmd = self.current_ry

                        # Detect upside-down: if angles > 90°, flip them
                        if abs(rx_cmd) > 90 or abs(ry_cmd) > 90:
                            # Invert orientation for upside-down mounting
                            if rx_cmd > 0:
                                rx_cmd = rx_cmd - 180.0
                            else:
                                rx_cmd = rx_cmd + 180.0

                            if ry_cmd > 0:
                                ry_cmd = ry_cmd - 180.0
                            else:
                                ry_cmd = ry_cmd + 180.0

                        self.send_servo_command(rx_cmd, ry_cmd,
                                               self.current_pos_x, self.current_pos_y, -self.current_pos_z)
                        self.last_servo_command_time = current_time
                    gyro_processed = False

                # Print status periodically (skip during calibration for max speed)
                if not self.calibration_mode and time.time() - last_status_time >= status_interval:
                    self.print_status()
                    last_status_time = time.time()

                # No sleep - run as fast as possible to process all data

            except Exception as e:
                if self.running:
                    print(f"Control loop error: {e}")
                    import traceback
                    traceback.print_exc()
                break

    def update_position(self, accel_raw):
        """
        Update position estimate from accelerometer (suspension mode).
        Reacts to movements but drifts back to center.

        Suppresses position updates during rotation to avoid false position
        estimates from centripetal/tangential acceleration.

        Args:
            accel_raw: [ax, ay, az] raw accelerometer values
        """
        # Suppress position updates during rotation (centripetal acceleration creates false readings)
        if self.is_rotating:
            # During rotation: only apply decay, no new acceleration integration
            decay_vel = 0.80
            decay_pos = 0.95
            self.current_vel_x *= decay_vel
            self.current_vel_y *= decay_vel
            self.current_vel_z *= decay_vel
            self.current_pos_x *= decay_pos
            self.current_pos_y *= decay_pos
            self.current_pos_z *= decay_pos
            return

        # Convert to m/s²
        ax_ms2 = accel_raw[0] * 0.001 * 9.81
        ay_ms2 = accel_raw[1] * 0.001 * 9.81
        az_ms2 = accel_raw[2] * 0.001 * 9.81

        # Subtract accelerometer biases (measured during calibration when stationary)
        # Only subtract X and Y biases (true sensor offsets)
        # DON'T subtract Z bias here - it includes gravity, which we remove after rotation
        ax_ms2 -= self.accel_bias_x
        ay_ms2 -= self.accel_bias_y
        # az_ms2 -= self.accel_bias_z  # Skip Z - gravity handled separately

        # Get current orientation (radians)
        rx_rad = self.current_rx * (np.pi / 180.0)
        ry_rad = self.current_ry * (np.pi / 180.0)

        # Transform accelerometer from body frame to world frame (simplified)
        cos_rx = np.cos(rx_rad)
        sin_rx = np.sin(rx_rad)
        cos_ry = np.cos(ry_rad)
        sin_ry = np.sin(ry_rad)

        # Rotation matrix: R_y(ry) * R_x(rx)
        ax_world = cos_ry * ax_ms2 + sin_ry * az_ms2
        ay_world = sin_rx * sin_ry * ax_ms2 + cos_rx * ay_ms2 - sin_rx * cos_ry * az_ms2
        az_world = -sin_ry * ax_ms2 + sin_rx * cos_ry * ay_ms2 + cos_rx * cos_ry * az_ms2

        # Subtract gravity (use measured value from calibration)
        # When upside down (|rx| > 90° or |ry| > 90°), gravity direction is different
        g = self.gravity_magnitude

        # Detect upside-down orientation
        upside_down = abs(rx_rad) > np.pi/2 or abs(ry_rad) > np.pi/2

        if upside_down:
            # Upside down: gravity is in +Z world direction (IMU is flipped)
            linear_ax = ax_world
            linear_ay = ay_world
            linear_az = az_world + g  # Gravity is now positive Z
        else:
            # Right-side up: gravity is in -Z world direction
            linear_ax = ax_world
            linear_ay = ay_world
            linear_az = az_world - g  # Remove gravity to get linear acceleration

        # Time step
        current_time = time.time()
        if self.last_position_update_time is not None:
            dt = current_time - self.last_position_update_time
            dt = min(dt, 0.01)  # Limit dt to prevent instability

            # Integrate acceleration to velocity (m/s)
            self.current_vel_x += linear_ax * dt
            self.current_vel_y += linear_ay * dt
            self.current_vel_z += linear_az * dt

            # Integrate velocity to position (m -> mm)
            self.current_pos_x += self.current_vel_x * dt * 1000.0
            self.current_pos_y += self.current_vel_y * dt * 1000.0
            self.current_pos_z += self.current_vel_z * dt * 1000.0

        # Aggressive high-pass filter: drift back to zero (suspension effect)
        decay_vel = 0.80   # Velocity decays to 80% per update (faster return)
        decay_pos = 0.95   # Position decays to 95% per update (faster return)

        self.current_vel_x *= decay_vel
        self.current_vel_y *= decay_vel
        self.current_vel_z *= decay_vel
        self.current_pos_x *= decay_pos
        self.current_pos_y *= decay_pos
        self.current_pos_z *= decay_pos

        # Limit position to safe range (prevent IK failure and excessive motion)
        max_pos = 25.0  # mm (user constraint)
        self.current_pos_x = np.clip(self.current_pos_x, -max_pos, max_pos)
        self.current_pos_y = np.clip(self.current_pos_y, -max_pos, max_pos)
        self.current_pos_z = np.clip(self.current_pos_z, -max_pos, max_pos)

        self.last_position_update_time = current_time

    def send_servo_command(self, rx_deg, ry_deg, pos_x_mm=0.0, pos_y_mm=0.0, pos_z_mm=0.0):
        """
        Compute servo angles for given pose and send to Teensy.

        Args:
            rx_deg: Roll angle in degrees
            ry_deg: Pitch angle in degrees
            pos_x_mm: X position in mm
            pos_y_mm: Y position in mm
            pos_z_mm: Z position in mm (relative to home height)
        """
        # Skip servo commands if no_servos flag is enabled
        if self.no_servos:
            return

        try:
            # Apply motion constraints (user limits)
            max_angle = 12.0  # degrees
            rx_deg = np.clip(rx_deg, -max_angle, max_angle)
            ry_deg = np.clip(ry_deg, -max_angle, max_angle)

            max_pos = 25.0  # mm
            pos_x_mm = np.clip(pos_x_mm, -max_pos, max_pos)
            pos_y_mm = np.clip(pos_y_mm, -max_pos, max_pos)
            pos_z_mm = np.clip(pos_z_mm, -max_pos, max_pos)

            # Platform pose: match IMU position + orientation
            # Z position is relative to home height
            translation = np.array([pos_x_mm, pos_y_mm, self.ik.home_height_top_surface + pos_z_mm])
            rotation = np.array([rx_deg, ry_deg, 0.0])

            # Compute inverse kinematics
            servo_angles = self.ik.calculate_servo_angles(translation, rotation, use_top_surface_offset=True)

            if servo_angles is not None:
                # Format command: angle0,angle1,angle2,angle3,angle4,angle5
                cmd = ','.join([f"{angle:.2f}" for angle in servo_angles])
                cmd += '\n'

                # Send command (non-blocking)
                if self.serial and self.serial.is_open:
                    self.serial.write(cmd.encode('utf-8'))
                    self.servo_command_count += 1

                    # Debug: print first few commands
                    if self.servo_command_count <= 5:
                        print(f"\nDEBUG: Sent command #{self.servo_command_count}: {cmd.strip()}")
                        print(f"       rx={rx_deg:.2f}°, ry={ry_deg:.2f}°, pos=({pos_x_mm:.1f}, {pos_y_mm:.1f}, {pos_z_mm:.1f})")
                else:
                    if self.servo_command_count == 0:
                        print("\nDEBUG: Serial not open!")
            else:
                # IK failed - angles out of range
                if self.servo_command_count <= 5:
                    print(f"\nDEBUG: IK failed for rx={rx_deg:.2f}°, ry={ry_deg:.2f}° (out of range)")

        except Exception as e:
            if self.servo_command_count <= 5:
                print(f"\nServo command error: {e}")
                import traceback
                traceback.print_exc()

    def calibrate_gyro(self, duration=10.0):
        """
        Calibrate IMU: measure gyro biases and initial orientation from gravity.

        IMU must be stationary during calibration.

        Args:
            duration: Calibration duration in seconds
        """
        print(f"\n{'='*60}")
        print(f"CALIBRATION: Collecting IMU data for {duration} seconds...")
        print(f"Keep IMU STATIONARY in desired starting orientation!")
        print(f"{'='*60}\n")

        # Update status label
        self.status_label.setText(f"Calibrating - Keep IMU still for {duration:.0f} seconds...")
        QApplication.processEvents()

        self.calibration_mode = True
        self.calibration_gyro_samples = []
        self.calibration_accel_samples = []
        self.calibration_duration = duration

        # Wait for samples
        start_time = time.time()
        last_print = start_time
        while time.time() - start_time < duration:
            time.sleep(0.1)

            # Print progress every 0.5 seconds
            if time.time() - last_print >= 0.5:
                gyro_count = len(self.calibration_gyro_samples)
                accel_count = len(self.calibration_accel_samples)
                elapsed = time.time() - start_time
                remaining = duration - elapsed
                progress = (elapsed / duration) * 100

                # Progress bar
                bar_length = 40
                filled = int(bar_length * elapsed / duration)
                bar = '█' * filled + '░' * (bar_length - filled)

                status_text = f"Calibrating: [{bar}] {progress:5.1f}% | Gyro:{gyro_count} Accel:{accel_count}"
                print(f"  [{bar}] {progress:5.1f}% ({elapsed:.1f}s / {duration:.1f}s) | "
                      f"Gyro: {gyro_count:5d} samples | Accel: {accel_count:5d} samples", end='\r')

                # Update status label
                self.status_label.setText(status_text)
                QApplication.processEvents()

                last_print = time.time()

        print(f"\n\nCalibration data collected:")
        print(f"  Gyro samples:  {len(self.calibration_gyro_samples)}")
        print(f"  Accel samples: {len(self.calibration_accel_samples)}")

        # Validate sample count
        if len(self.calibration_gyro_samples) < 100 or len(self.calibration_accel_samples) < 100:
            print("\nERROR: Not enough samples for calibration!")
            print("Make sure IMU is connected and sending data.\n")
            self.calibration_mode = False
            return

        # Convert to arrays
        gyro_array = np.array(self.calibration_gyro_samples)
        accel_array = np.array(self.calibration_accel_samples)

        # === GYRO BIAS CALIBRATION ===
        # Compute mean gyro values (raw units)
        gyro_mean_raw = np.mean(gyro_array, axis=0)

        # Convert to rad/s
        gyro_scale = self.kalman.gyro_scale
        bias_gx = gyro_mean_raw[0] * gyro_scale
        bias_gy = gyro_mean_raw[1] * gyro_scale
        bias_gz = gyro_mean_raw[2] * gyro_scale

        print(f"\nGyro bias calibration:")
        print(f"  bias_gx = {bias_gx * 180/np.pi:+7.3f} °/s ({gyro_mean_raw[0]:+8.1f} raw)")
        print(f"  bias_gy = {bias_gy * 180/np.pi:+7.3f} °/s ({gyro_mean_raw[1]:+8.1f} raw)")
        print(f"  bias_gz = {bias_gz * 180/np.pi:+7.3f} °/s ({gyro_mean_raw[2]:+8.1f} raw)")

        # === INITIAL ORIENTATION FROM GRAVITY ===
        # Compute mean accelerometer values (raw units)
        accel_mean_raw = np.mean(accel_array, axis=0)

        # Convert to m/s²
        accel_scale = self.kalman.accel_scale
        ax_mean = accel_mean_raw[0] * accel_scale
        ay_mean = accel_mean_raw[1] * accel_scale
        az_mean = accel_mean_raw[2] * accel_scale

        # Gravity direction gives initial orientation
        # rx = atan2(-ay, az)  (roll around X)
        # ry = atan2(ax, az)   (pitch around Y)
        rx_init_rad = np.arctan2(-ay_mean, az_mean)
        ry_init_rad = np.arctan2(ax_mean, az_mean)

        rx_init_deg = rx_init_rad * (180.0 / np.pi)
        ry_init_deg = ry_init_rad * (180.0 / np.pi)

        # Compute gravity magnitude from each sample, then take mean
        # This is more robust than computing magnitude from mean vector
        accel_array_ms2 = accel_array * accel_scale
        accel_magnitudes = np.sqrt(np.sum(accel_array_ms2**2, axis=1))
        gravity_mean = np.mean(accel_magnitudes)
        gravity_std = np.std(accel_magnitudes)

        # Store measured gravity magnitude
        self.gravity_magnitude = gravity_mean

        # Store accelerometer biases (for position tracking)
        # These are the mean values when stationary - we'll subtract them to get linear acceleration
        self.accel_bias_x = ax_mean
        self.accel_bias_y = ay_mean
        self.accel_bias_z = az_mean

        print(f"\nInitial orientation from gravity:")
        print(f"  Accel mean: ax={ax_mean:+6.2f} ay={ay_mean:+6.2f} az={az_mean:+6.2f} m/s²")
        print(f"  Measured gravity: {gravity_mean:.3f} ± {gravity_std:.3f} m/s² (nominal 9.81 m/s²)")
        print(f"  rx = {rx_init_deg:+7.2f}° (roll)")
        print(f"  ry = {ry_init_deg:+7.2f}° (pitch)")
        print(f"\nAccelerometer bias compensation:")
        print(f"  bias_x = {ax_mean:+6.3f} m/s² (applied)")
        print(f"  bias_y = {ay_mean:+6.3f} m/s² (applied)")
        print(f"  bias_z = {az_mean:+6.3f} m/s² (not applied - gravity handled separately)")

        # Warn if not quasi-static
        if abs(gravity_mean - 9.81) > 1.0:
            print(f"\n  WARNING: Acceleration magnitude differs significantly from nominal gravity!")
            print(f"           IMU may have been moving during calibration.")
            print(f"           Recommend re-running calibration.")

        if gravity_std > 0.5:
            print(f"\n  WARNING: High acceleration variance ({gravity_std:.3f} m/s²)!")
            print(f"           IMU may have been moving during calibration.")

        # === UPDATE KALMAN FILTER ===
        # Initialize state: [rx, ry, bias_gx, bias_gy, bias_gz]
        self.kalman.x[0] = rx_init_rad
        self.kalman.x[1] = ry_init_rad
        self.kalman.x[2] = bias_gx
        self.kalman.x[3] = bias_gy
        self.kalman.x[4] = bias_gz

        # Reset covariance (low uncertainty after calibration)
        self.kalman.P = np.diag([
            0.001,  # rx variance (rad²)
            0.001,  # ry variance (rad²)
            (0.01 * gyro_scale) ** 2,  # bias_gx variance
            (0.01 * gyro_scale) ** 2,  # bias_gy variance
            (0.01 * gyro_scale) ** 2   # bias_gz variance
        ])

        # Update current state
        self.current_rx = rx_init_deg
        self.current_ry = ry_init_deg

        print(f"\nKalman filter initialized with calibrated values.")
        print(f"{'='*60}")
        print(f"Calibration complete!\n")

        self.calibration_mode = False
        self.calibration_gyro_samples = []
        self.calibration_accel_samples = []

    def print_status(self):
        """Print current status with sample processing verification"""
        elapsed = time.time() - self.start_time

        # Sample rates (received from serial)
        gyro_hz = self.gyro_count / elapsed if elapsed > 0 else 0
        accel_hz = self.accel_count / elapsed if elapsed > 0 else 0
        servo_hz = self.servo_command_count / elapsed if elapsed > 0 else 0

        # Processing efficiency
        total_gyro = self.gyro_count + self.gyro_dropped_count
        total_accel = self.accel_count + self.accel_dropped_count
        gyro_efficiency = (self.gyro_processed_count / self.gyro_count * 100) if self.gyro_count > 0 else 0
        accel_efficiency = (self.accel_processed_count / self.accel_count * 100) if self.accel_count > 0 else 0

        total_updates = self.update_count + self.rejected_update_count
        update_rate = self.update_count / total_updates if total_updates > 0 else 0

        # Get gyro biases (in rad/s)
        bias_gx, bias_gy, bias_gz = self.kalman.get_gyro_biases()
        # Convert to deg/s for display
        bias_gx_deg = bias_gx * (180.0 / np.pi)
        bias_gy_deg = bias_gy * (180.0 / np.pi)
        bias_gz_deg = bias_gz * (180.0 / np.pi)

        # Rotation suppression indicator
        rot_status = "ROT-SUPPRESS" if self.is_rotating else ""

        print(f"\r[{elapsed:6.1f}s] "
              f"Pos:x={self.current_pos_x:+6.1f}mm y={self.current_pos_y:+6.1f}mm z={self.current_pos_z:+6.1f}mm | "
              f"Rot:rx={self.current_rx:+6.2f}° ry={self.current_ry:+6.2f}° | "
              f"GyroMag:{self.gyro_magnitude:5.1f}°/s {rot_status:13s} | "
              f"Gyro:{gyro_hz:4.0f}Hz({gyro_efficiency:3.0f}%) Accel:{accel_hz:4.0f}Hz({accel_efficiency:3.0f}%) "
              f"KF:{update_rate*100:3.0f}% Servo:{servo_hz:3.0f}Hz",
              end='')

    def run(self, calibrate=True):
        """
        Run IMU mirror demo.

        Args:
            calibrate: If True, run calibration before starting (default: True)
        """
        if not self.connected:
            print("ERROR: Not connected")
            return

        # Start threads
        self.running = True

        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()

        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

        # Wait for initial data
        print("Waiting for IMU data...")
        time.sleep(1.0)

        # Automatic calibration (can be disabled with --no-calibrate flag)
        if calibrate:
            self.calibrate_gyro(duration=10.0)

        # Reset statistics
        self.start_time = time.time()
        self.gyro_count = 0
        self.accel_count = 0
        self.gyro_processed_count = 0
        self.accel_processed_count = 0
        self.gyro_dropped_count = 0
        self.accel_dropped_count = 0
        self.update_count = 0
        self.rejected_update_count = 0
        self.servo_command_count = 0

        print("IMU Mirror Demo running...")
        print("Platform will mimic IMU 6DOF pose (position + orientation)")
        print("Motion constraints: ±12° angle, ±25mm position")
        print("Move the IMU and watch the platform follow!")
        print("Close window to stop\n")

        # Update status label
        self.status_label.setText("Running - Move the IMU to see real-time tracking")

        # Start plot timer
        self.plot_timer.start()

        # Qt event loop will handle everything - no sleep loop needed


def list_ports():
    """List available serial ports"""
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found")
        return

    print("\nAvailable serial ports:")
    for i, port in enumerate(ports, 1):
        print(f"  {i}. {port.device} - {port.description}")


def main():
    parser = argparse.ArgumentParser(
        description='IMU Mirror Demo - Platform mimics IMU 6DOF pose (position + orientation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mirror_demo.py --list
  python mirror_demo.py --port COM3
  python mirror_demo.py --port COM3 --invert-imu
  python mirror_demo.py --port COM3 --no-calibrate

Note: Upload IMU_control.ino to Teensy first.
      Automatic calibration runs at startup (10 seconds, keep IMU still).
      Use --invert-imu if IMU is mounted upside down.
      Platform will follow IMU position and orientation in real-time.
      Move/tilt the IMU and watch the platform mimic its motion!
        """
    )

    parser.add_argument('--list', action='store_true',
                        help='List available serial ports and exit')
    parser.add_argument('--port', type=str,
                        help='Serial port (e.g., COM3, /dev/ttyACM0)')
    parser.add_argument('--invert-imu', action='store_true',
                        help='Invert IMU axes for upside-down mounting')
    parser.add_argument('--remap-gyro', action='store_true',
                        help='Swap gyro X/Y axes (for Adafruit 10DOF sensor misalignment)')
    parser.add_argument('--no-servos', action='store_true',
                        help='Skip servo commands (IMU tracking only, no platform control)')
    parser.add_argument('--rotation-threshold', type=float, default=20.0,
                        help='Gyroscope magnitude threshold (deg/s) for rotation suppression (default: 20.0)')
    parser.add_argument('--no-calibrate', action='store_true',
                        help='Skip automatic calibration on startup')

    args = parser.parse_args()

    # List ports and exit
    if args.list:
        list_ports()
        return

    # Check if port specified
    if not args.port:
        print("ERROR: --port is required")
        print("\nUse --list to see available ports")
        list_ports()
        return

    # Create Qt application
    app = QApplication(sys.argv)

    # Create controller and show window
    controller = IMUMirrorController(
        args.port,
        invert_imu=args.invert_imu,
        remap_gyro=args.remap_gyro,
        no_servos=args.no_servos,
        rotation_threshold=args.rotation_threshold
    )
    controller.show()

    if args.invert_imu:
        print("IMU axis inversion enabled (upside-down mounting)")
    if args.remap_gyro:
        print("Gyro axis remapping enabled (X/Y swap for Adafruit 10DOF)")
    if args.no_servos:
        print("Servo commands disabled (IMU tracking only)")
    if args.rotation_threshold != 20.0:
        print(f"Rotation suppression threshold: {args.rotation_threshold}°/s")

    # Connect and start
    if controller.connect():
        # Calibrate by default unless --no-calibrate flag is used
        if not args.no_calibrate:
            controller.calibrate_gyro(duration=10.0)

        # Start the run
        controller.run()

        # Start Qt event loop
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
