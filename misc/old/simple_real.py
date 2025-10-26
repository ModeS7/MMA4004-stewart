#!/usr/bin/env python3
"""
Stewart Platform Hardware Controller

Real-time control interface for physical Stewart platform.
Communicates with Arduino/Maestro servo controller via serial.
Uses shared IK solver from stewart_platform_core.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import serial
import serial.tools.list_ports
import time
import threading
from queue import Queue, Empty

from core import StewartPlatformIK


class MaestroSerialInterface:
    """
    Serial communication interface for Pololu Maestro servo controller.
    Handles connection, command transmission, and response parsing.
    """

    def __init__(self):
        self.serial_port = None
        self.is_connected = False
        self.response_queue = Queue()
        self.read_thread = None
        self.running = False

    def list_ports(self):
        """List available serial ports."""
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]

    def connect(self, port, baudrate=115200, timeout=1.0):
        """
        Establish serial connection.

        Args:
            port: Serial port path (e.g., '/dev/ttyACM0', 'COM3')
            baudrate: Communication speed
            timeout: Read timeout in seconds

        Returns:
            bool: Connection success status
        """
        try:
            self.serial_port = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=timeout,
                write_timeout=timeout
            )
            time.sleep(2.0)  # Allow Arduino to reset

            # Flush input buffer
            self.serial_port.reset_input_buffer()

            self.is_connected = True
            self.running = True

            # Start read thread
            self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.read_thread.start()

            return True

        except serial.SerialException as e:
            print(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Close serial connection."""
        self.running = False
        if self.read_thread:
            self.read_thread.join(timeout=1.0)

        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()

        self.is_connected = False

    def _read_loop(self):
        """Background thread for reading serial responses."""
        while self.running and self.serial_port and self.serial_port.is_open:
            try:
                if self.serial_port.in_waiting > 0:
                    line = self.serial_port.readline().decode('utf-8').strip()
                    if line:
                        self.response_queue.put(line)
            except Exception as e:
                print(f"Read error: {e}")
                break

    def send_angles(self, angles):
        """
        Send servo angles to controller.

        Args:
            angles: Array of 6 servo angles in degrees

        Returns:
            bool: Transmission success status
        """
        if not self.is_connected or not self.serial_port:
            return False

        try:
            # Format: "theta0,theta1,theta2,theta3,theta4,theta5\n"
            command = ','.join(f"{angle:.2f}" for angle in angles) + '\n'
            self.serial_port.write(command.encode('utf-8'))
            return True

        except Exception as e:
            print(f"Send error: {e}")
            return False

    def set_speed(self, speed):
        """
        Set servo movement speed.

        Args:
            speed: Speed value 0-255 (0=unlimited)

        Returns:
            bool: Transmission success status
        """
        if not self.is_connected or not self.serial_port:
            return False

        try:
            command = f"SPD:{speed}\n"
            self.serial_port.write(command.encode('utf-8'))
            return True

        except Exception as e:
            print(f"Send error: {e}")
            return False

    def set_acceleration(self, acceleration):
        """
        Set servo acceleration.

        Args:
            acceleration: Acceleration value 0-255 (0=unlimited)

        Returns:
            bool: Transmission success status
        """
        if not self.is_connected or not self.serial_port:
            return False

        try:
            command = f"ACC:{acceleration}\n"
            self.serial_port.write(command.encode('utf-8'))
            return True

        except Exception as e:
            print(f"Send error: {e}")
            return False

    def get_response(self, timeout=0.1):
        """
        Get next response from queue.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            str or None: Response message
        """
        try:
            return self.response_queue.get(timeout=timeout)
        except Empty:
            return None


class HardwareControllerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stewart Platform Hardware Controller")
        self.root.geometry("800x900")

        # Initialize platform IK
        self.platform_params = {
            "horn_length": 31.75,
            "rod_length": 145.0,
            "base": 73.025,
            "base_anchors": 36.8893,
            "platform": 67.775,
            "platform_anchors": 12.7,
            "top_surface_offset": 26.0
        }
        self.ik = StewartPlatformIK(**self.platform_params)

        # Initialize hardware interface
        self.maestro = MaestroSerialInterface()

        # Control state
        self.control_enabled = False
        self.use_top_surface_offset = tk.BooleanVar(value=True)
        self.auto_send = tk.BooleanVar(value=False)

        self.dof_values = {
            'x': 0.0, 'y': 0.0, 'z': self.ik.home_height_top_surface,
            'rx': 0.0, 'ry': 0.0, 'rz': 0.0
        }

        self.dof_config = {
            'x': (-30.0, 30.0, 0.1, 0.0, "X Position (mm)"),
            'y': (-30.0, 30.0, 0.1, 0.0, "Y Position (mm)"),
            'z': (self.ik.home_height_top_surface - 30,
                  self.ik.home_height_top_surface + 30,
                  0.1, self.ik.home_height_top_surface, "Z Height (mm)"),
            'rx': (-15.0, 15.0, 0.1, 0.0, "Roll (°)"),
            'ry': (-15.0, 15.0, 0.1, 0.0, "Pitch (°)"),
            'rz': (-15.0, 15.0, 0.1, 0.0, "Yaw (°)")
        }

        self.sliders = {}
        self.value_labels = {}
        self.current_angles = np.zeros(6)

        # Response monitoring
        self.monitor_responses = True
        self.response_thread = None

        self.create_widgets()

        # Protocol for window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # Connection control
        conn_frame = ttk.LabelFrame(self.root, text="Serial Connection", padding=10)
        conn_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(conn_frame, text="Port:").grid(row=0, column=0, padx=5, sticky='w')

        self.port_combo = ttk.Combobox(conn_frame, width=20, state='readonly')
        self.port_combo.grid(row=0, column=1, padx=5)

        ttk.Button(conn_frame, text="Refresh Ports",
                   command=self.refresh_ports).grid(row=0, column=2, padx=5)

        self.connect_btn = ttk.Button(conn_frame, text="Connect",
                                      command=self.connect_serial)
        self.connect_btn.grid(row=0, column=3, padx=5)

        self.disconnect_btn = ttk.Button(conn_frame, text="Disconnect",
                                         command=self.disconnect_serial,
                                         state='disabled')
        self.disconnect_btn.grid(row=0, column=4, padx=5)

        self.status_label = ttk.Label(conn_frame, text="Not Connected",
                                      foreground='red', font=('TkDefaultFont', 9, 'bold'))
        self.status_label.grid(row=0, column=5, padx=10)

        # Servo parameters
        servo_params_frame = ttk.LabelFrame(self.root, text="Servo Parameters", padding=10)
        servo_params_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(servo_params_frame, text="Speed (0-255):").grid(
            row=0, column=0, padx=5, sticky='w')
        self.speed_var = tk.IntVar(value=20)
        speed_spin = ttk.Spinbox(servo_params_frame, from_=0, to=255,
                                 textvariable=self.speed_var, width=10)
        speed_spin.grid(row=0, column=1, padx=5)

        ttk.Button(servo_params_frame, text="Set Speed",
                   command=self.set_speed).grid(row=0, column=2, padx=10)

        ttk.Label(servo_params_frame, text="Acceleration (0-255):").grid(
            row=0, column=3, padx=5, sticky='w')
        self.accel_var = tk.IntVar(value=20)
        accel_spin = ttk.Spinbox(servo_params_frame, from_=0, to=255,
                                 textvariable=self.accel_var, width=10)
        accel_spin.grid(row=0, column=4, padx=5)

        ttk.Button(servo_params_frame, text="Set Acceleration",
                   command=self.set_acceleration).grid(row=0, column=5, padx=10)

        # Configuration
        config_frame = ttk.LabelFrame(self.root, text="Configuration", padding=10)
        config_frame.pack(fill='x', padx=10, pady=5)

        ttk.Checkbutton(config_frame, text="Use Top Surface Offset",
                        variable=self.use_top_surface_offset,
                        command=self.on_offset_toggle).pack(anchor='w')

        ttk.Checkbutton(config_frame, text="Auto-send on slider change",
                        variable=self.auto_send).pack(anchor='w')

        # Sliders for DOF control
        sliders_frame = ttk.LabelFrame(self.root, text="Platform Pose (6 DOF)", padding=10)
        sliders_frame.pack(fill='both', expand=True, padx=10, pady=5)

        for idx, (dof, (min_val, max_val, res, default, label)) in enumerate(self.dof_config.items()):
            ttk.Label(sliders_frame, text=label).grid(row=idx, column=0, sticky='w', pady=5)

            slider = ttk.Scale(sliders_frame, from_=min_val, to=max_val, orient='horizontal',
                               command=lambda val, d=dof: self.on_slider_change(d, val))
            slider.grid(row=idx, column=1, sticky='ew', padx=10, pady=5)
            self.sliders[dof] = slider

            value_label = ttk.Label(sliders_frame, text=f"{default:.2f}", width=10)
            value_label.grid(row=idx, column=2, pady=5)
            self.value_labels[dof] = value_label
            slider.set(default)

        sliders_frame.columnconfigure(1, weight=1)

        # Control buttons
        control_frame = ttk.LabelFrame(self.root, text="Control", padding=10)
        control_frame.pack(fill='x', padx=10, pady=5)

        self.send_btn = ttk.Button(control_frame, text="Send Pose",
                                   command=self.send_pose, state='disabled')
        self.send_btn.pack(side='left', padx=5)

        self.home_btn = ttk.Button(control_frame, text="Go Home",
                                   command=self.go_home, state='disabled')
        self.home_btn.pack(side='left', padx=5)

        ttk.Button(control_frame, text="Emergency Stop",
                   command=self.emergency_stop).pack(side='left', padx=5)

        # Servo angles display
        angles_frame = ttk.LabelFrame(self.root, text="Calculated Servo Angles", padding=10)
        angles_frame.pack(fill='x', padx=10, pady=5)

        self.angle_labels = []
        servo_names = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
        for i in range(6):
            label = ttk.Label(angles_frame, text=f"{servo_names[i]}: 0.00°",
                              font=('Courier', 9))
            label.grid(row=i // 3, column=i % 3, padx=10, pady=2, sticky='w')
            self.angle_labels.append(label)

        # Log
        log_frame = ttk.LabelFrame(self.root, text="System Log", padding=10)
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, font=('Courier', 8))
        self.log_text.pack(fill='both', expand=True)

        # Initial port refresh
        self.refresh_ports()

    def log(self, message, level="INFO"):
        """Add timestamped message to log."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {level}: {message}\n")
        self.log_text.see(tk.END)

    def refresh_ports(self):
        """Refresh list of available serial ports."""
        ports = self.maestro.list_ports()
        self.port_combo['values'] = ports
        if ports:
            self.port_combo.current(0)
            self.log(f"Found {len(ports)} port(s)")
        else:
            self.log("No serial ports found", "WARN")

    def connect_serial(self):
        """Establish serial connection."""
        port = self.port_combo.get()
        if not port:
            messagebox.showerror("Error", "Please select a serial port")
            return

        self.log(f"Connecting to {port}...")

        if self.maestro.connect(port):
            self.status_label.config(text="Connected", foreground='green')
            self.connect_btn.config(state='disabled')
            self.disconnect_btn.config(state='normal')
            self.send_btn.config(state='normal')
            self.home_btn.config(state='normal')
            self.control_enabled = True

            # Start response monitor
            self.monitor_responses = True
            self.response_thread = threading.Thread(target=self.monitor_response_queue,
                                                    daemon=True)
            self.response_thread.start()

            self.log("Connected successfully")

            # Set initial servo parameters
            self.set_speed()
            self.set_acceleration()

        else:
            self.log("Connection failed", "ERROR")
            messagebox.showerror("Error", "Failed to connect to serial port")

    def disconnect_serial(self):
        """Close serial connection."""
        self.monitor_responses = False
        self.maestro.disconnect()

        self.status_label.config(text="Not Connected", foreground='red')
        self.connect_btn.config(state='normal')
        self.disconnect_btn.config(state='disabled')
        self.send_btn.config(state='disabled')
        self.home_btn.config(state='disabled')
        self.control_enabled = False

        self.log("Disconnected")

    def monitor_response_queue(self):
        """Monitor serial responses in background thread."""
        while self.monitor_responses:
            response = self.maestro.get_response(timeout=0.1)
            if response:
                if response == "OK":
                    self.log("Command acknowledged", "DEBUG")
                elif response.startswith("ERROR"):
                    self.log(response, "ERROR")
                else:
                    self.log(f"Response: {response}")

            time.sleep(0.05)

    def set_speed(self):
        """Send speed parameter to controller."""
        if not self.control_enabled:
            return

        speed = self.speed_var.get()
        if self.maestro.set_speed(speed):
            self.log(f"Speed set to {speed}")

    def set_acceleration(self):
        """Send acceleration parameter to controller."""
        if not self.control_enabled:
            return

        accel = self.accel_var.get()
        if self.maestro.set_acceleration(accel):
            self.log(f"Acceleration set to {accel}")

    def on_offset_toggle(self):
        """Handle offset toggle."""
        enabled = self.use_top_surface_offset.get()
        home_z = self.ik.home_height_top_surface if enabled else self.ik.home_height

        self.sliders['z'].config(from_=home_z - 30, to=home_z + 30)
        self.dof_values['z'] = home_z
        self.sliders['z'].set(home_z)
        self.value_labels['z'].config(text=f"{home_z:.2f}")

        self.log(f"Offset mode: {'Top Surface' if enabled else 'Anchor Center'}")

    def on_slider_change(self, dof, value):
        """Handle slider changes."""
        val = float(value)
        self.dof_values[dof] = val
        self.value_labels[dof].config(text=f"{val:.2f}")

        # Calculate and display IK
        self.calculate_ik()

        # Auto-send if enabled and connected
        if self.auto_send.get() and self.control_enabled:
            self.send_pose()

    def calculate_ik(self):
        """Calculate inverse kinematics and update display."""
        translation = np.array([self.dof_values['x'], self.dof_values['y'],
                               self.dof_values['z']])
        rotation = np.array([self.dof_values['rx'], self.dof_values['ry'],
                            self.dof_values['rz']])

        angles = self.ik.calculate_servo_angles(
            translation, rotation,
            use_top_surface_offset=self.use_top_surface_offset.get()
        )

        if angles is not None:
            self.current_angles = angles
            for i in range(6):
                self.angle_labels[i].config(text=f"S{i + 1}: {angles[i]:6.2f}°")
        else:
            for i in range(6):
                self.angle_labels[i].config(text=f"S{i + 1}: ERROR")
            self.log("IK calculation failed (pose unreachable)", "WARN")

    def send_pose(self):
        """Send current pose to hardware."""
        if not self.control_enabled:
            messagebox.showwarning("Warning", "Not connected to hardware")
            return

        # Validate angles before sending
        if np.any(np.abs(self.current_angles) > 40):
            self.log("Pose exceeds safe angle limits", "ERROR")
            messagebox.showerror("Error", "Pose exceeds safe servo angle limits (±40°)")
            return

        if self.maestro.send_angles(self.current_angles):
            self.log(f"Sent angles: {self.current_angles}")
        else:
            self.log("Failed to send command", "ERROR")

    def go_home(self):
        """Move platform to home position."""
        if not self.control_enabled:
            return

        # Reset sliders to home
        for dof, (_, _, _, default, _) in self.dof_config.items():
            if dof == 'z':
                home_z = (self.ik.home_height_top_surface if self.use_top_surface_offset.get()
                          else self.ik.home_height)
                self.sliders[dof].set(home_z)
            else:
                self.sliders[dof].set(default)

        self.log("Moving to home position")
        self.send_pose()

    def emergency_stop(self):
        """Emergency stop - go to home position."""
        if self.control_enabled:
            self.go_home()
            self.log("EMERGENCY STOP - Returning to home", "WARN")
        else:
            self.log("Not connected to hardware", "WARN")

    def on_closing(self):
        """Handle window close event."""
        if self.control_enabled:
            response = messagebox.askyesno(
                "Disconnect",
                "Platform is still connected. Return to home position before closing?"
            )
            if response:
                self.go_home()
                time.sleep(1.0)  # Allow time for movement

        self.disconnect_serial()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = HardwareControllerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()