#!/usr/bin/env python3
"""
IMU Calibration Tool

Uses Stewart platform to move IMU to known positions/orientations for calibration.
Records IMU data (accel, gyro) and servo angles to CSV for analysis.

Features:
- 6DOF sliders (x, y, z, rx, ry, rz)
- Real-time IK calculation
- Records IMU sensor data (A: and G: lines)
- Records servo angles (SERVO: lines)
- Saves synchronized CSV file
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import serial
import serial.tools.list_ports
import numpy as np
import time
import csv
import threading
from queue import Queue, Empty
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QSlider, QLineEdit, QComboBox,
    QGroupBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QDoubleValidator

from core.core import StewartPlatformIK


class IMUCalibrationTool(QMainWindow):
    """IMU calibration tool with 6DOF control and data logging."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMU Calibration Tool - 6DOF Platform Control")
        self.resize(900, 800)

        # Serial connection
        self.serial_port = None
        self.connected = False
        self.read_thread = None
        self.running = False

        # Data queues for high-rate IMU data
        self.accel_queue = Queue(maxsize=2000)
        self.gyro_queue = Queue(maxsize=1500)
        self.servo_queue = Queue(maxsize=100)

        # Stewart Platform IK
        self.platform_params = {
            "horn_length": 45.3722,             #31.75
            "rod_length": 205.0,                #145.0
            "base": 86.6025 + 18.75 + 11,       #73.025
            "base_anchors": 64.75,              #36.8893
            "platform": 84.0759,                #67.775
            "platform_anchors": 12.5,           #12.7
            "top_surface_offset": 38.0
        }
        self.ik = StewartPlatformIK(**self.platform_params)

        # 6DOF values
        self.dof_values = {
            'x': 0.0,
            'y': 0.0,
            'z': self.ik.home_height_top_surface,
            'rx': 0.0,
            'ry': 0.0,
            'rz': 0.0
        }

        # Servo angles
        self.current_servo_angles = [0.0] * 6

        # IMU data
        self.latest_accel = [0.0, 0.0, 0.0]
        self.latest_gyro = [0.0, 0.0, 0.0]

        # Logging
        self.logging = False
        self.csv_file = None
        self.csv_writer = None
        self.log_count = 0

        # Colors
        self.colors = {
            'bg': '#1e1e1e',
            'panel_bg': '#2d2d2d',
            'widget_bg': '#3d3d3d',
            'fg': '#e0e0e0',
            'highlight': '#007acc',
            'button_bg': '#0e639c',
            'border': '#555555',
            'success': '#4ec9b0',
            'warning': '#ce9178',
            'error': '#f44747'
        }

        self.setup_dark_theme()
        self.init_ui()

        # Update timer for display/logging (from queues)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.process_data_queues)
        self.update_timer.start(20)  # 50Hz display update

    def setup_dark_theme(self):
        """Apply dark theme."""
        stylesheet = f"""
            QMainWindow, QWidget {{
                background-color: {self.colors['bg']};
                color: {self.colors['fg']};
            }}
            QGroupBox {{
                background-color: {self.colors['panel_bg']};
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: {self.colors['highlight']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
            QPushButton {{
                background-color: {self.colors['button_bg']};
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 3px;
                font-size: 10pt;
            }}
            QPushButton:hover {{
                background-color: {self.colors['highlight']};
            }}
            QPushButton:disabled {{
                background-color: {self.colors['widget_bg']};
                color: {self.colors['border']};
            }}
            QSlider::groove:horizontal {{
                background: {self.colors['widget_bg']};
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {self.colors['highlight']};
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }}
            QLineEdit, QComboBox {{
                background-color: {self.colors['widget_bg']};
                color: {self.colors['fg']};
                border: 1px solid {self.colors['border']};
                padding: 4px;
                border-radius: 3px;
            }}
            QLabel {{
                color: {self.colors['fg']};
            }}
        """
        self.setStyleSheet(stylesheet)

    def init_ui(self):
        """Initialize user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Connection panel
        main_layout.addWidget(self.create_connection_panel())

        # 6DOF sliders
        main_layout.addWidget(self.create_dof_controls())

        # IMU data display
        main_layout.addWidget(self.create_imu_display())

        # Logging controls
        main_layout.addWidget(self.create_logging_controls())

        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont("Consolas", 9))
        main_layout.addWidget(self.status_label)

        main_layout.addStretch()

    def create_connection_panel(self):
        """Create connection panel."""
        group = QGroupBox("Serial Connection")
        layout = QHBoxLayout()

        layout.addWidget(QLabel("Port:"))

        self.port_combo = QComboBox()
        layout.addWidget(self.port_combo)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_ports)
        layout.addWidget(refresh_btn)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_serial)
        layout.addWidget(self.connect_btn)

        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_serial)
        self.disconnect_btn.setEnabled(False)
        layout.addWidget(self.disconnect_btn)

        self.connection_status = QLabel("Not connected")
        layout.addWidget(self.connection_status)

        layout.addStretch()
        group.setLayout(layout)

        self.refresh_ports()
        return group

    def create_dof_controls(self):
        """Create 6DOF slider controls."""
        group = QGroupBox("6DOF Platform Control")
        layout = QGridLayout()

        self.dof_sliders = {}
        self.dof_labels = {}
        self.dof_inputs = {}

        # Translation (mm)
        dof_configs = [
            ('x', -50, 50, 'mm'),
            ('y', -50, 50, 'mm'),
            ('z', int(self.ik.home_height_top_surface - 50), int(self.ik.home_height_top_surface + 50), 'mm'),
            ('rx', -30, 30, '°'),
            ('ry', -30, 30, '°'),
            ('rz', -30, 30, '°')
        ]

        for idx, (dof, min_val, max_val, unit) in enumerate(dof_configs):
            # Label
            label = QLabel(f"{dof.upper()}:")
            label.setMinimumWidth(40)
            layout.addWidget(label, idx, 0)

            # Slider
            slider = QSlider(Qt.Orientation.Horizontal)
            if dof in ['x', 'y']:
                slider.setMinimum(int(min_val * 10))
                slider.setMaximum(int(max_val * 10))
                slider.setValue(0)
            elif dof == 'z':
                slider.setMinimum(min_val)
                slider.setMaximum(max_val)
                slider.setValue(int(self.dof_values['z']))
            else:  # Rotations
                slider.setMinimum(int(min_val * 10))
                slider.setMaximum(int(max_val * 10))
                slider.setValue(0)

            slider.valueChanged.connect(lambda val, d=dof: self.on_dof_change(d, val))
            layout.addWidget(slider, idx, 1)
            self.dof_sliders[dof] = slider

            # Manual input box
            input_box = QLineEdit()
            input_box.setMaximumWidth(70)
            input_box.setAlignment(Qt.AlignmentFlag.AlignRight)
            if dof == 'z':
                input_box.setText(f"{self.dof_values['z']:.1f}")
            else:
                input_box.setText("0.0")
            validator = QDoubleValidator()
            validator.setDecimals(1)
            validator.setRange(min_val, max_val)
            input_box.setValidator(validator)
            input_box.returnPressed.connect(lambda d=dof, box=input_box: self.on_manual_input(d, box))
            layout.addWidget(input_box, idx, 2)
            self.dof_inputs[dof] = input_box

            # Value label
            value_label = QLabel(unit)
            value_label.setMinimumWidth(30)
            value_label.setFont(QFont("Consolas", 10))
            value_label.setStyleSheet(f"color: {self.colors['success']};")
            layout.addWidget(value_label, idx, 3)
            self.dof_labels[dof] = value_label

        # Reset button
        reset_btn = QPushButton("Reset to Home")
        reset_btn.clicked.connect(self.reset_to_home)
        layout.addWidget(reset_btn, 6, 0, 1, 4)

        group.setLayout(layout)
        return group

    def create_imu_display(self):
        """Create IMU data display."""
        group = QGroupBox("IMU Data (Real-time)")
        layout = QGridLayout()

        self.accel_labels = []
        self.gyro_labels = []

        # Accelerometer
        layout.addWidget(QLabel("Accel (raw):"), 0, 0)
        for i, axis in enumerate(['X', 'Y', 'Z']):
            lbl = QLabel(f"{axis}: 0")
            lbl.setFont(QFont("Consolas", 9))
            layout.addWidget(lbl, 0, i+1)
            self.accel_labels.append(lbl)

        # Gyroscope
        layout.addWidget(QLabel("Gyro (raw):"), 1, 0)
        for i, axis in enumerate(['X', 'Y', 'Z']):
            lbl = QLabel(f"{axis}: 0")
            lbl.setFont(QFont("Consolas", 9))
            layout.addWidget(lbl, 1, i+1)
            self.gyro_labels.append(lbl)

        # Servo angles
        layout.addWidget(QLabel("Servo angles (°):"), 2, 0)
        self.servo_angle_label = QLabel("0.0, 0.0, 0.0, 0.0, 0.0, 0.0")
        self.servo_angle_label.setFont(QFont("Consolas", 9))
        layout.addWidget(self.servo_angle_label, 2, 1, 1, 3)

        group.setLayout(layout)
        return group

    def create_logging_controls(self):
        """Create logging controls."""
        group = QGroupBox("Data Logging")
        layout = QHBoxLayout()

        self.log_btn = QPushButton("Start Logging")
        self.log_btn.clicked.connect(self.toggle_logging)
        layout.addWidget(self.log_btn)

        self.log_status = QLabel("Not logging")
        layout.addWidget(self.log_status)

        self.log_count_label = QLabel("Samples: 0")
        layout.addWidget(self.log_count_label)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def refresh_ports(self):
        """Refresh COM ports."""
        ports = list(serial.tools.list_ports.comports())
        self.port_combo.clear()
        for port in ports:
            self.port_combo.addItem(f"{port.device} - {port.description}")
        if not ports:
            self.port_combo.addItem("No ports found")
            self.connect_btn.setEnabled(False)
        else:
            self.connect_btn.setEnabled(True)

    def connect_serial(self):
        """Connect to Teensy."""
        if self.port_combo.count() == 0:
            return

        port_text = self.port_combo.currentText()
        port_name = port_text.split(" - ")[0]

        try:
            self.serial_port = serial.Serial(port_name, baudrate=2000000, timeout=0.1)
            time.sleep(2.5)

            # Clear startup messages
            time.sleep(0.5)
            self.serial_port.reset_input_buffer()

            # Start read thread for high-rate data capture
            self.running = True
            self.read_thread = threading.Thread(target=self._serial_read_thread, daemon=True)
            self.read_thread.start()

            self.connected = True
            self.connection_status.setText("Connected")
            self.connection_status.setStyleSheet(f"color: {self.colors['success']};")
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.status_label.setText("Ready - move sliders to control platform")

        except Exception as e:
            self.status_label.setText(f"Connection failed: {str(e)}")
            self.connection_status.setStyleSheet(f"color: {self.colors['error']};")

    def disconnect_serial(self):
        """Disconnect from Teensy."""
        if self.logging:
            self.toggle_logging()

        # Stop read thread
        self.running = False
        if self.read_thread:
            self.read_thread.join(timeout=1.0)

        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()

        self.connected = False
        self.connection_status.setText("Not connected")
        self.connection_status.setStyleSheet(f"color: {self.colors['border']};")
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)

    def on_dof_change(self, dof, value):
        """Handle DOF slider change."""
        if dof in ['x', 'y', 'rx', 'ry', 'rz']:
            real_value = value / 10.0
        else:  # z
            real_value = float(value)

        self.dof_values[dof] = real_value

        # Update manual input box
        self.dof_inputs[dof].setText(f"{real_value:.1f}")

        # Calculate and send servo angles
        self.update_servo_angles()

    def on_manual_input(self, dof, input_box):
        """Handle manual input from text box."""
        try:
            value = float(input_box.text())

            # Update slider (scaled for x, y, rx, ry, rz)
            if dof in ['x', 'y', 'rx', 'ry', 'rz']:
                self.dof_sliders[dof].setValue(int(value * 10))
            else:  # z
                self.dof_sliders[dof].setValue(int(value))

        except ValueError:
            # Restore previous value if invalid
            input_box.setText(f"{self.dof_values[dof]:.1f}")

    def reset_to_home(self):
        """Reset platform to home position."""
        self.dof_values['x'] = 0.0
        self.dof_values['y'] = 0.0
        self.dof_values['z'] = self.ik.home_height_top_surface
        self.dof_values['rx'] = 0.0
        self.dof_values['ry'] = 0.0
        self.dof_values['rz'] = 0.0

        # Update sliders (which will trigger on_dof_change and update input boxes)
        self.dof_sliders['x'].setValue(0)
        self.dof_sliders['y'].setValue(0)
        self.dof_sliders['z'].setValue(int(self.dof_values['z']))
        self.dof_sliders['rx'].setValue(0)
        self.dof_sliders['ry'].setValue(0)
        self.dof_sliders['rz'].setValue(0)

    def update_servo_angles(self):
        """Calculate IK and send servo angles."""
        if not self.connected:
            return

        translation = np.array([
            self.dof_values['x'],
            self.dof_values['y'],
            self.dof_values['z']
        ])

        # Rotation in degrees (IK converts to radians internally)
        rotation = np.array([
            self.dof_values['rx'],
            self.dof_values['ry'],
            self.dof_values['rz']
        ])

        angles = self.ik.calculate_servo_angles(translation, rotation, use_top_surface_offset=True)

        if angles is not None:
            self.current_servo_angles = angles

            # Send to Teensy
            command = ",".join([f"{a:.2f}" for a in angles]) + "\n"
            try:
                self.serial_port.write(command.encode('utf-8'))
                self.serial_port.flush()
            except Exception as e:
                self.status_label.setText(f"Send failed: {str(e)}")

    def _serial_read_thread(self):
        """Background thread for continuous high-rate serial reading."""
        while self.running:
            try:
                if self.serial_port and self.serial_port.is_open and self.serial_port.in_waiting:
                    line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()

                    if line.startswith("A:"):
                        # Accelerometer data
                        parts = line[2:].split(',')
                        if len(parts) == 4:
                            try:
                                timestamp = int(parts[0])
                                data = [int(parts[1]), int(parts[2]), int(parts[3])]
                                if not self.accel_queue.full():
                                    self.accel_queue.put((timestamp, data))
                            except ValueError:
                                pass

                    elif line.startswith("G:"):
                        # Gyroscope data
                        parts = line[2:].split(',')
                        if len(parts) == 4:
                            try:
                                timestamp = int(parts[0])
                                data = [int(parts[1]), int(parts[2]), int(parts[3])]
                                if not self.gyro_queue.full():
                                    self.gyro_queue.put((timestamp, data))
                            except ValueError:
                                pass

                    elif line.startswith("SERVO:"):
                        # Servo angle feedback with timestamp
                        parts = line[6:].split(',')
                        if len(parts) == 7:  # timestamp + 6 angles
                            try:
                                timestamp = int(parts[0])
                                angles = [float(parts[i]) for i in range(1, 7)]
                                if not self.servo_queue.full():
                                    self.servo_queue.put((timestamp, angles))
                            except ValueError:
                                pass

                else:
                    time.sleep(0.0001)  # 100μs sleep when no data

            except Exception as e:
                if self.running:
                    print(f"Serial read error: {e}")
                time.sleep(0.001)

    def process_data_queues(self):
        """Process queued data for display and logging."""
        # Process accelerometer queue
        while not self.accel_queue.empty():
            try:
                timestamp, data = self.accel_queue.get_nowait()
                self.latest_accel = data
                self.update_accel_display()
                if self.logging:
                    self.log_data('A', timestamp, data)
            except Empty:
                break

        # Process gyroscope queue
        while not self.gyro_queue.empty():
            try:
                timestamp, data = self.gyro_queue.get_nowait()
                self.latest_gyro = data
                self.update_gyro_display()
                if self.logging:
                    self.log_data('G', timestamp, data)
            except Empty:
                break

        # Process servo queue (only latest for display, log all if recording)
        try:
            while not self.servo_queue.empty():
                timestamp, angles = self.servo_queue.get_nowait()
                self.current_servo_angles = angles
                if self.logging:
                    self.log_data('SERVO', timestamp, angles)
            self.update_servo_display()
        except Empty:
            pass

    def update_accel_display(self):
        """Update accelerometer display."""
        for i in range(3):
            self.accel_labels[i].setText(f"{'XYZ'[i]}: {self.latest_accel[i]}")

    def update_gyro_display(self):
        """Update gyroscope display."""
        for i in range(3):
            self.gyro_labels[i].setText(f"{'XYZ'[i]}: {self.latest_gyro[i]}")

    def update_servo_display(self):
        """Update servo angle display."""
        angles_str = ", ".join([f"{a:.1f}" for a in self.current_servo_angles])
        self.servo_angle_label.setText(angles_str)

    def toggle_logging(self):
        """Toggle data logging."""
        if not self.logging:
            # Start logging
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"imu_calibration_{timestamp}.csv"

            try:
                self.csv_file = open(filename, 'w', newline='')
                self.csv_writer = csv.writer(self.csv_file)

                # Header
                self.csv_writer.writerow([
                    'timestamp_arduino_us', 'type',
                    'x', 'y', 'z',
                    'platform_x', 'platform_y', 'platform_z',
                    'platform_rx', 'platform_ry', 'platform_rz',
                    'servo0', 'servo1', 'servo2', 'servo3', 'servo4', 'servo5'
                ])

                self.logging = True
                self.log_count = 0
                self.log_btn.setText("Stop Logging")
                self.log_status.setText(f"Logging to: {filename}")
                self.log_status.setStyleSheet(f"color: {self.colors['success']};")

            except Exception as e:
                self.status_label.setText(f"Failed to start logging: {str(e)}")

        else:
            # Stop logging
            if self.csv_file:
                self.csv_file.close()

            self.logging = False
            self.log_btn.setText("Start Logging")
            self.log_status.setText(f"Stopped - logged {self.log_count} samples")
            self.log_status.setStyleSheet(f"color: {self.colors['fg']};")

    def log_data(self, data_type, timestamp, values):
        """Log data to CSV."""
        if not self.logging or not self.csv_writer:
            return

        # Handle different data types
        if data_type == 'SERVO':
            # SERVO type: values is already 6 servo angles
            self.csv_writer.writerow([
                timestamp,
                data_type,
                '', '', '',  # Empty x, y, z for SERVO type
                self.dof_values['x'], self.dof_values['y'], self.dof_values['z'],
                self.dof_values['rx'], self.dof_values['ry'], self.dof_values['rz'],
                values[0], values[1], values[2], values[3], values[4], values[5]
            ])
        else:
            # A or G type: values is 3-element IMU data (x, y, z)
            self.csv_writer.writerow([
                timestamp,
                data_type,
                values[0], values[1], values[2],
                self.dof_values['x'], self.dof_values['y'], self.dof_values['z'],
                self.dof_values['rx'], self.dof_values['ry'], self.dof_values['rz'],
                self.current_servo_angles[0], self.current_servo_angles[1],
                self.current_servo_angles[2], self.current_servo_angles[3],
                self.current_servo_angles[4], self.current_servo_angles[5]
            ])

        self.log_count += 1
        self.log_count_label.setText(f"Samples: {self.log_count}")

    def closeEvent(self, event):
        """Handle window close."""
        if self.logging:
            self.toggle_logging()
        self.disconnect_serial()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = IMUCalibrationTool()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
