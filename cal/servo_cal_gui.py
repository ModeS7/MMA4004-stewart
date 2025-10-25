#!/usr/bin/env python3
"""
Servo Calibration GUI

High-precision servo control for Stewart Platform calibration.
Sends angles directly to Maestro via Teensy passthrough (no calibration offsets).

Features:
- 6 individual servo sliders with manual input
- "All servos" control with absolute/offset mode
- High resolution sliders for precise adjustment
- Neutral position button
- COM port auto-detection for Teensy
"""

import sys
import serial
import serial.tools.list_ports
import time
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QSlider, QLineEdit,
    QComboBox, QGroupBox, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QDoubleValidator


class ServoCalibrationGUI(QMainWindow):
    """Main calibration GUI window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stewart Platform - Servo Calibration")
        self.resize(800, 900)

        # Serial connection
        self.serial_port = None
        self.connected = False

        # Servo angles (degrees)
        self.servo_angles = [0.0] * 6

        # All servos mode
        self.all_servos_mode = "absolute"  # "absolute" or "offset"

        # Colors (dark theme)
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

        # Update timer for status
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(100)

    def setup_dark_theme(self):
        """Apply dark theme stylesheet."""
        stylesheet = f"""
            QMainWindow {{
                background-color: {self.colors['bg']};
            }}
            QWidget {{
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
            QLabel {{
                color: {self.colors['fg']};
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
            QPushButton:pressed {{
                background-color: #005a9e;
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
            QLineEdit {{
                background-color: {self.colors['widget_bg']};
                color: {self.colors['fg']};
                border: 1px solid {self.colors['border']};
                padding: 4px;
                border-radius: 3px;
            }}
            QComboBox {{
                background-color: {self.colors['widget_bg']};
                color: {self.colors['fg']};
                border: 1px solid {self.colors['border']};
                padding: 4px;
                border-radius: 3px;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {self.colors['fg']};
                margin-right: 5px;
            }}
            QRadioButton {{
                color: {self.colors['fg']};
                spacing: 8px;
            }}
            QRadioButton::indicator {{
                width: 16px;
                height: 16px;
                border: 2px solid {self.colors['border']};
                border-radius: 8px;
                background-color: {self.colors['widget_bg']};
            }}
            QRadioButton::indicator:checked {{
                background-color: {self.colors['highlight']};
                border-color: {self.colors['highlight']};
            }}
        """
        self.setStyleSheet(stylesheet)

    def init_ui(self):
        """Initialize user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)

        # Connection panel
        main_layout.addWidget(self.create_connection_panel())

        # Individual servo controls
        main_layout.addWidget(self.create_servo_controls())

        # All servos control
        main_layout.addWidget(self.create_all_servos_control())

        # Action buttons
        main_layout.addWidget(self.create_action_buttons())

        # Status display
        main_layout.addWidget(self.create_status_panel())

        main_layout.addStretch()

    def create_connection_panel(self):
        """Create serial connection panel."""
        group = QGroupBox("Serial Connection")
        layout = QVBoxLayout()

        # Port selection
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("Teensy Port:"))

        self.port_combo = QComboBox()
        port_layout.addWidget(self.port_combo)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_ports)
        refresh_btn.setMaximumWidth(100)
        port_layout.addWidget(refresh_btn)

        port_layout.addStretch()
        layout.addLayout(port_layout)

        # Connection buttons
        btn_layout = QHBoxLayout()

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_serial)
        self.connect_btn.setMaximumWidth(120)
        btn_layout.addWidget(self.connect_btn)

        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_serial)
        self.disconnect_btn.setEnabled(False)
        self.disconnect_btn.setMaximumWidth(120)
        btn_layout.addWidget(self.disconnect_btn)

        btn_layout.addStretch()

        self.connection_status = QLabel("Not connected")
        self.connection_status.setStyleSheet(f"color: {self.colors['border']};")
        btn_layout.addWidget(self.connection_status)

        layout.addLayout(btn_layout)
        group.setLayout(layout)

        # Initial port refresh
        self.refresh_ports()

        return group

    def create_servo_controls(self):
        """Create individual servo control sliders."""
        group = QGroupBox("Individual Servo Control")
        layout = QVBoxLayout()

        self.servo_sliders = []
        self.servo_inputs = []
        self.servo_value_labels = []

        for i in range(6):
            servo_layout = self.create_servo_slider(i)
            layout.addLayout(servo_layout)

        group.setLayout(layout)
        return group

    def create_servo_slider(self, servo_id):
        """Create a single servo slider with manual input."""
        layout = QHBoxLayout()

        # Label
        label = QLabel(f"Servo {servo_id}:")
        label.setMinimumWidth(70)
        label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        layout.addWidget(label)

        # Slider (high resolution: 0.01 degree steps)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(-4500)  # -45.00 degrees
        slider.setMaximum(4500)   # +45.00 degrees
        slider.setValue(0)
        slider.setTickInterval(500)  # Ticks every 5 degrees
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.valueChanged.connect(lambda val, sid=servo_id: self.on_slider_change(sid, val))
        layout.addWidget(slider)
        self.servo_sliders.append(slider)

        # Value display
        value_label = QLabel("0.00°")
        value_label.setMinimumWidth(70)
        value_label.setFont(QFont("Consolas", 10))
        value_label.setStyleSheet(f"color: {self.colors['success']};")
        layout.addWidget(value_label)
        self.servo_value_labels.append(value_label)

        # Manual input
        input_field = QLineEdit()
        input_field.setPlaceholderText("Manual")
        input_field.setMaximumWidth(80)
        input_field.setValidator(QDoubleValidator(-45.0, 45.0, 2))
        input_field.returnPressed.connect(lambda sid=servo_id: self.on_manual_input(sid))
        layout.addWidget(input_field)
        self.servo_inputs.append(input_field)

        return layout

    def create_all_servos_control(self):
        """Create all servos control with mode selection."""
        group = QGroupBox("All Servos Control")
        layout = QVBoxLayout()

        # Mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))

        self.mode_button_group = QButtonGroup()

        self.absolute_radio = QRadioButton("Absolute Position")
        self.absolute_radio.setChecked(True)
        self.absolute_radio.toggled.connect(self.on_mode_change)
        self.mode_button_group.addButton(self.absolute_radio)
        mode_layout.addWidget(self.absolute_radio)

        self.offset_radio = QRadioButton("Offset from Current")
        self.offset_radio.toggled.connect(self.on_mode_change)
        self.mode_button_group.addButton(self.offset_radio)
        mode_layout.addWidget(self.offset_radio)

        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # All servos slider
        slider_layout = QHBoxLayout()

        label = QLabel("All Servos:")
        label.setMinimumWidth(70)
        label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        slider_layout.addWidget(label)

        self.all_servos_slider = QSlider(Qt.Orientation.Horizontal)
        self.all_servos_slider.setMinimum(-4500)
        self.all_servos_slider.setMaximum(4500)
        self.all_servos_slider.setValue(0)
        self.all_servos_slider.setTickInterval(500)
        self.all_servos_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.all_servos_slider.valueChanged.connect(self.on_all_servos_slider_change)
        slider_layout.addWidget(self.all_servos_slider)

        self.all_servos_value_label = QLabel("0.00°")
        self.all_servos_value_label.setMinimumWidth(70)
        self.all_servos_value_label.setFont(QFont("Consolas", 10))
        self.all_servos_value_label.setStyleSheet(f"color: {self.colors['success']};")
        slider_layout.addWidget(self.all_servos_value_label)

        self.all_servos_input = QLineEdit()
        self.all_servos_input.setPlaceholderText("Manual")
        self.all_servos_input.setMaximumWidth(80)
        self.all_servos_input.setValidator(QDoubleValidator(-45.0, 45.0, 2))
        self.all_servos_input.returnPressed.connect(self.on_all_servos_manual_input)
        slider_layout.addWidget(self.all_servos_input)

        layout.addLayout(slider_layout)
        group.setLayout(layout)
        return group

    def create_action_buttons(self):
        """Create action buttons."""
        group = QGroupBox("Actions")
        layout = QHBoxLayout()

        neutral_btn = QPushButton("Neutral Position (0°)")
        neutral_btn.clicked.connect(self.set_neutral)
        neutral_btn.setMinimumHeight(40)
        layout.addWidget(neutral_btn)

        send_btn = QPushButton("Send Current Angles")
        send_btn.clicked.connect(self.send_servo_angles)
        send_btn.setMinimumHeight(40)
        layout.addWidget(send_btn)

        group.setLayout(layout)
        return group

    def create_status_panel(self):
        """Create status display panel."""
        group = QGroupBox("Status")
        layout = QVBoxLayout()

        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont("Consolas", 9))
        layout.addWidget(self.status_label)

        self.angles_display = QLabel("Current: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]")
        self.angles_display.setFont(QFont("Consolas", 9))
        layout.addWidget(self.angles_display)

        group.setLayout(layout)
        return group

    def refresh_ports(self):
        """Refresh available COM ports."""
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
            # Open serial port - 200000 baud (match control_v1.ino)
            self.serial_port = serial.Serial(port_name, baudrate=200000, timeout=0.1)
            time.sleep(2)  # Wait for Teensy reset
            self.connected = True

            # Clear startup messages
            time.sleep(0.5)
            self.serial_port.reset_input_buffer()

            self.connection_status.setText("Connected")
            self.connection_status.setStyleSheet(f"color: {self.colors['success']};")
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.status_label.setText("Ready")

        except Exception as e:
            self.status_label.setText(f"Connection failed: {str(e)}")
            self.connection_status.setStyleSheet(f"color: {self.colors['error']};")

    def disconnect_serial(self):
        """Disconnect from Teensy."""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()

        self.connected = False
        self.connection_status.setText("Not connected")
        self.connection_status.setStyleSheet(f"color: {self.colors['border']};")
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.status_label.setText("Disconnected")

    def on_slider_change(self, servo_id, value):
        """Handle individual servo slider change."""
        angle = value / 100.0  # Convert to degrees (0.01 resolution)
        self.servo_angles[servo_id] = angle
        self.servo_value_labels[servo_id].setText(f"{angle:.2f}°")
        self.update_angles_display()
        self.send_servo_angles()

    def on_manual_input(self, servo_id):
        """Handle manual angle input."""
        try:
            text = self.servo_inputs[servo_id].text()
            if text:
                angle = float(text)
                angle = max(-45.0, min(45.0, angle))  # Clamp for safety
                self.servo_angles[servo_id] = angle
                self.servo_sliders[servo_id].setValue(int(angle * 100))  # 0.01 resolution
                self.servo_inputs[servo_id].clear()
                self.send_servo_angles()
        except ValueError:
            pass

    def on_mode_change(self):
        """Handle all servos mode change."""
        if self.absolute_radio.isChecked():
            self.all_servos_mode = "absolute"
        else:
            self.all_servos_mode = "offset"

        # Reset slider to 0 when switching modes
        self.all_servos_slider.setValue(0)

    def on_all_servos_slider_change(self, value):
        """Handle all servos slider change."""
        angle = value / 100.0  # 0.01 resolution
        self.all_servos_value_label.setText(f"{angle:.2f}°")

        if self.all_servos_mode == "absolute":
            # Set all servos to the same angle
            for i in range(6):
                self.servo_angles[i] = angle
                self.servo_sliders[i].setValue(value)
        else:
            # Apply offset to all servos
            # Only apply when slider is released or use a button
            pass

    def on_all_servos_manual_input(self):
        """Handle all servos manual input."""
        try:
            text = self.all_servos_input.text()
            if text:
                angle = float(text)
                angle = max(-45.0, min(45.0, angle))

                if self.all_servos_mode == "absolute":
                    # Set all to same angle
                    for i in range(6):
                        self.servo_angles[i] = angle
                        self.servo_sliders[i].setValue(int(angle * 100))  # 0.01 resolution
                else:
                    # Apply offset
                    for i in range(6):
                        new_angle = self.servo_angles[i] + angle
                        new_angle = max(-45.0, min(45.0, new_angle))
                        self.servo_angles[i] = new_angle
                        self.servo_sliders[i].setValue(int(new_angle * 100))  # 0.01 resolution

                self.all_servos_input.clear()
                self.all_servos_slider.setValue(0)
                self.send_servo_angles()
        except ValueError:
            pass

    def set_neutral(self):
        """Set all servos to neutral position (0 degrees)."""
        for i in range(6):
            self.servo_angles[i] = 0.0
            self.servo_sliders[i].setValue(0)

        self.all_servos_slider.setValue(0)
        self.send_servo_angles()
        self.status_label.setText("Set to neutral position")

    def send_servo_angles(self):
        """Send current servo angles to Teensy."""
        if not self.connected or not self.serial_port:
            return

        try:
            # Format: "angle0,angle1,angle2,angle3,angle4,angle5\n"
            command = ",".join([f"{angle:.2f}" for angle in self.servo_angles]) + "\n"

            # Send command (non-blocking like your working code)
            self.serial_port.write(command.encode('utf-8'))
            self.serial_port.flush()

            # Status update happens via update_status() timer

        except Exception as e:
            self.status_label.setText(f"Send failed: {str(e)}")

    def update_angles_display(self):
        """Update angles display."""
        angles_str = "[" + ", ".join([f"{a:.2f}" for a in self.servo_angles]) + "]"
        self.angles_display.setText(f"Current: {angles_str}")

    def update_status(self):
        """Update status periodically - read Arduino responses."""
        if not self.connected or not self.serial_port:
            return

        try:
            # Read any pending responses (non-blocking)
            while self.serial_port.in_waiting:
                line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    if line.startswith("ACK:"):
                        # Successful command (match control_v1 format)
                        self.status_label.setText("OK")
                    elif line.startswith("ERROR:"):
                        # Error from Arduino
                        self.status_label.setText(line.replace("ERROR:", "Error: "))
                    elif line.startswith("READY:") or line.startswith("FORMAT:"):
                        # Startup messages - ignore
                        pass
                    else:
                        # Other messages
                        self.status_label.setText(line)
        except Exception as e:
            pass  # Ignore errors in status updates

    def closeEvent(self, event):
        """Handle window close event."""
        self.disconnect_serial()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = ServoCalibrationGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
