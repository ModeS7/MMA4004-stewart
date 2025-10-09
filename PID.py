#!/usr/bin/env python3
"""
Stewart Platform Ball Balance Controller with PID
Single Teensy with Pixy2 + Maestro
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import serial
import serial.tools.list_ports
import numpy as np
import time
import threading
import queue
from collections import deque
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class PIDController:
    """PID controller with anti-windup and derivative filtering."""

    def __init__(self, kp=0.0, ki=0.0, kd=0.0, output_limits=(-15, 15),
                 integral_limits=(-10, 10), derivative_filter_alpha=0.2):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.integral_limits = integral_limits
        self.derivative_filter_alpha = derivative_filter_alpha

        self.reset()

    def reset(self):
        """Reset PID state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_filtered_derivative = 0.0
        self.last_time = None
        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0

    def update(self, error, dt=None):
        """Calculate PID output.

        Args:
            error: Current error value
            dt: Time delta in seconds (optional, will be calculated if None)

        Returns:
            PID output value (clamped to output_limits)
        """
        # Handle timing
        current_time = time.time()
        if self.last_time is None:
            self.last_time = current_time
            self.prev_error = error
            return 0.0

        if dt is None:
            dt = current_time - self.last_time

        if dt <= 0:
            return self.p_term + self.i_term + self.d_term  # Return last output

        # Proportional term
        self.p_term = self.kp * error

        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, self.integral_limits[0], self.integral_limits[1])
        self.i_term = self.ki * self.integral

        # Derivative term with low-pass filter
        raw_derivative = (error - self.prev_error) / dt
        filtered_derivative = (self.derivative_filter_alpha * raw_derivative +
                               (1 - self.derivative_filter_alpha) * self.prev_filtered_derivative)
        self.d_term = self.kd * filtered_derivative

        # Calculate output
        output = self.p_term + self.i_term + self.d_term
        output = np.clip(output, self.output_limits[0], self.output_limits[1])

        # Update state
        self.prev_error = error
        self.prev_filtered_derivative = filtered_derivative
        self.last_time = current_time

        return output

    def set_gains(self, kp=None, ki=None, kd=None):
        """Update PID gains."""
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd


class StewartPlatformIK:
    """Inverse kinematics - simplified from existing code."""

    def __init__(self, horn_length=31.75, rod_length=145.0, base=73.025,
                 base_anchors=36.8893, platform=67.775, platform_anchors=12.7,
                 top_surface_offset=26.0):
        self.horn_length = horn_length
        self.rod_length = rod_length
        self.base = base
        self.base_anchors = base_anchors
        self.platform = platform
        self.platform_anchors = platform_anchors
        self.top_surface_offset = top_surface_offset

        # Calculate anchor positions
        base_angels = np.array([-np.pi / 2, np.pi / 6, np.pi * 5 / 6])
        platform_angels = np.array([-np.pi * 5 / 6, -np.pi / 6, np.pi / 2])

        self.base_anchors = self._calculate_coordinates(self.base, self.base_anchors, base_angels)
        platform_anchors_temp = self._calculate_coordinates(self.platform, self.platform_anchors, platform_angels)
        self.platform_anchors = np.roll(platform_anchors_temp, shift=-1, axis=0)

        self.beta_angles = self._calculate_beta_angles()

        # Calculate home height
        base_pos = self.base_anchors[0]
        platform_pos = self.platform_anchors[0]

        horn_end_x = base_pos[0] + self.horn_length * np.cos(self.beta_angles[0])
        horn_end_y = base_pos[1] + self.horn_length * np.sin(self.beta_angles[0])

        dx = platform_pos[0] - horn_end_x
        dy = platform_pos[1] - horn_end_y
        horiz_dist_sq = dx ** 2 + dy ** 2

        self.home_height = np.sqrt(self.rod_length ** 2 - horiz_dist_sq)
        self.home_height_top_surface = self.home_height + self.top_surface_offset

    def _calculate_coordinates(self, l, d, phi):
        angels = np.array([-np.pi / 2, np.pi / 2])
        xy = np.zeros((6, 3))
        for i in range(len(phi)):
            for j in range(len(angels)):
                x = l * np.cos(phi[i]) + d * np.cos(phi[i] + angels[j])
                y = l * np.sin(phi[i]) + d * np.sin(phi[i] + angels[j])
                xy[i * 2 + j] = np.array([x, y, 0])
        return xy

    def _calculate_beta_angles(self):
        beta_angles = np.zeros(6)
        beta_angles[0] = 0
        beta_angles[1] = np.pi

        dx_23 = self.base_anchors[3, 0] - self.base_anchors[2, 0]
        dy_23 = self.base_anchors[3, 1] - self.base_anchors[2, 1]
        angle_23 = np.arctan2(dy_23, dx_23)
        beta_angles[2] = angle_23
        beta_angles[3] = angle_23 + np.pi

        dx_54 = self.base_anchors[4, 0] - self.base_anchors[5, 0]
        dy_54 = self.base_anchors[4, 1] - self.base_anchors[5, 1]
        angle_54 = np.arctan2(dy_54, dx_54)
        beta_angles[5] = angle_54
        beta_angles[4] = angle_54 + np.pi

        return beta_angles

    def calculate_servo_angles(self, translation, rotation, use_top_surface_offset=True):
        """Calculate servo angles for desired pose."""
        quat = self._euler_to_quaternion(np.radians(rotation))

        if use_top_surface_offset:
            offset_platform_frame = np.array([0, 0, -self.top_surface_offset])
            offset_world_frame = self._rotate_vector(offset_platform_frame, quat)
            anchor_center_translation = translation + offset_world_frame
        else:
            anchor_center_translation = translation

        angles = np.zeros(6)

        for k in range(6):
            p_world = anchor_center_translation + self._rotate_vector(self.platform_anchors[k], quat)
            leg = p_world - self.base_anchors[k]
            leg_length_sq = np.dot(leg, leg)

            e_k = 2 * self.horn_length * leg[2]
            f_k = 2 * self.horn_length * (
                    np.cos(self.beta_angles[k]) * leg[0] +
                    np.sin(self.beta_angles[k]) * leg[1]
            )
            g_k = leg_length_sq - (self.rod_length ** 2 - self.horn_length ** 2)

            sqrt_term = e_k ** 2 + f_k ** 2
            if sqrt_term < 1e-6:
                return None

            ratio = g_k / np.sqrt(sqrt_term)
            if abs(ratio) > 1.0:
                return None

            alpha_k = np.arcsin(ratio) - np.arctan2(f_k, e_k)
            angles[k] = np.degrees(alpha_k)

            if abs(angles[k]) > 40:
                return None

        return -angles

    def _euler_to_quaternion(self, euler):
        rx, ry, rz = euler
        cy, sy = np.cos(rz * 0.5), np.sin(rz * 0.5)
        cp, sp = np.cos(ry * 0.5), np.sin(ry * 0.5)
        cr, sr = np.cos(rx * 0.5), np.sin(rx * 0.5)

        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        ])

    def _rotate_vector(self, v, q):
        w, x, y, z = q
        vx, vy, vz = v

        return np.array([
            vx * (w * w + x * x - y * y - z * z) + vy * (2 * x * y - 2 * w * z) + vz * (2 * x * z + 2 * w * y),
            vx * (2 * x * y + 2 * w * z) + vy * (w * w - x * x + y * y - z * z) + vz * (2 * y * z - 2 * w * x),
            vx * (2 * x * z - 2 * w * y) + vy * (2 * y * z + 2 * w * x) + vz * (w * w - x * x - y * y + z * z)
        ])


class BallBalanceGUI:
    """Main GUI for ball balance control with PID tuning."""

    def __init__(self, root):
        self.root = root
        self.root.title("Stewart Platform Ball Balance Controller")
        self.root.geometry("1000x900")

        # Initialize IK
        self.ik = StewartPlatformIK()

        # Initialize PID controllers
        self.pid_x = PIDController(output_limits=(-15, 15))
        self.pid_y = PIDController(output_limits=(-15, 15))

        # Serial communication
        self.serial_conn = None
        self.is_connected = False
        self.serial_thread = None
        self.data_queue = queue.Queue()
        self.running = False

        # Ball state
        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_detected = False
        self.error_x = 0.0
        self.error_y = 0.0
        self.last_ball_time = 0

        # Setpoint
        self.setpoint_x = 158.0
        self.setpoint_y = 104.0

        # Control state
        self.control_enabled = False
        self.platform_roll = 0.0
        self.platform_pitch = 0.0
        self.platform_z = self.ik.home_height_top_surface

        # History for plotting
        self.history_length = 200
        self.time_history = deque(maxlen=self.history_length)
        self.error_x_history = deque(maxlen=self.history_length)
        self.error_y_history = deque(maxlen=self.history_length)

        # PID gain variables (base value 0-10, multiplier)
        self.pid_vars = {
            'x_kp_base': tk.DoubleVar(value=5.0),
            'x_kp_mult': tk.DoubleVar(value=-2.0),  # 10^-2 = 0.01
            'x_ki_base': tk.DoubleVar(value=5.0),
            'x_ki_mult': tk.DoubleVar(value=-3.0),  # 10^-3 = 0.001
            'x_kd_base': tk.DoubleVar(value=5.0),
            'x_kd_mult': tk.DoubleVar(value=-2.0),
            'y_kp_base': tk.DoubleVar(value=5.0),
            'y_kp_mult': tk.DoubleVar(value=-2.0),
            'y_ki_base': tk.DoubleVar(value=5.0),
            'y_ki_mult': tk.DoubleVar(value=-3.0),
            'y_kd_base': tk.DoubleVar(value=5.0),
            'y_kd_mult': tk.DoubleVar(value=-2.0),
        }

        # Update PID gains from initial values
        self.update_pid_gains()

        self.create_widgets()
        self.start_update_loop()

    def create_widgets(self):
        """Create all GUI widgets."""

        # Connection frame
        conn_frame = ttk.LabelFrame(self.root, text="Connection", padding=10)
        conn_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(conn_frame, text="Port:").grid(row=0, column=0, padx=5)
        self.port_combo = ttk.Combobox(conn_frame, width=20, state='readonly')
        self.port_combo.grid(row=0, column=1, padx=5)
        self.refresh_ports()

        ttk.Button(conn_frame, text="Refresh", command=self.refresh_ports).grid(row=0, column=2, padx=5)
        self.connect_btn = ttk.Button(conn_frame, text="Connect", command=self.toggle_connection)
        self.connect_btn.grid(row=0, column=3, padx=5)
        self.status_label = ttk.Label(conn_frame, text="● Disconnected", foreground="red")
        self.status_label.grid(row=0, column=4, padx=5)

        # Ball position frame
        ball_frame = ttk.LabelFrame(self.root, text="Ball State", padding=10)
        ball_frame.pack(fill='x', padx=10, pady=5)

        self.ball_pos_label = ttk.Label(ball_frame, text="Position: X=0.00  Y=0.00  ✗ Not Detected",
                                        font=('TkDefaultFont', 10))
        self.ball_pos_label.grid(row=0, column=0, columnspan=2, sticky='w', pady=2)

        self.ball_error_label = ttk.Label(ball_frame, text="Error: X=0.00  Y=0.00",
                                          font=('TkDefaultFont', 10))
        self.ball_error_label.grid(row=1, column=0, columnspan=2, sticky='w', pady=2)

        ttk.Label(ball_frame, text="Setpoint X:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.setpoint_x_entry = ttk.Entry(ball_frame, width=10)
        self.setpoint_x_entry.insert(0, "158.0")
        self.setpoint_x_entry.grid(row=2, column=1, sticky='w', pady=2)

        ttk.Label(ball_frame, text="Setpoint Y:").grid(row=3, column=0, sticky='w', padx=5, pady=2)
        self.setpoint_y_entry = ttk.Entry(ball_frame, width=10)
        self.setpoint_y_entry.insert(0, "104.0")
        self.setpoint_y_entry.grid(row=3, column=1, sticky='w', pady=2)

        ttk.Button(ball_frame, text="Update Setpoint", command=self.update_setpoint).grid(
            row=2, column=2, rowspan=2, padx=10)

        # Control frame
        control_frame = ttk.LabelFrame(self.root, text="Control", padding=10)
        control_frame.pack(fill='x', padx=10, pady=5)

        self.enable_btn = ttk.Button(control_frame, text="Enable Control", command=self.toggle_control)
        self.enable_btn.grid(row=0, column=0, padx=5)

        ttk.Button(control_frame, text="Emergency Stop", command=self.emergency_stop).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Reset Platform", command=self.reset_platform).grid(row=0, column=2, padx=5)

        self.control_status_label = ttk.Label(control_frame, text="● Control Disabled", foreground="orange")
        self.control_status_label.grid(row=0, column=3, padx=10)

        # PID Tuning frame
        pid_frame = ttk.LabelFrame(self.root, text="PID Tuning", padding=10)
        pid_frame.pack(fill='both', expand=False, padx=10, pady=5)

        # X-axis PID
        ttk.Label(pid_frame, text="X-Axis (Roll)", font=('TkDefaultFont', 10, 'bold')).grid(
            row=0, column=0, columnspan=5, sticky='w', pady=5)

        self.create_pid_row(pid_frame, 1, "Kp", 'x_kp')
        self.create_pid_row(pid_frame, 2, "Ki", 'x_ki')
        self.create_pid_row(pid_frame, 3, "Kd", 'x_kd')

        ttk.Separator(pid_frame, orient='horizontal').grid(row=4, column=0, columnspan=5, sticky='ew', pady=10)

        # Y-axis PID
        ttk.Label(pid_frame, text="Y-Axis (Pitch)", font=('TkDefaultFont', 10, 'bold')).grid(
            row=5, column=0, columnspan=5, sticky='w', pady=5)

        self.create_pid_row(pid_frame, 6, "Kp", 'y_kp')
        self.create_pid_row(pid_frame, 7, "Ki", 'y_ki')
        self.create_pid_row(pid_frame, 8, "Kd", 'y_kd')

        # Platform state frame
        state_frame = ttk.LabelFrame(self.root, text="Platform State", padding=10)
        state_frame.pack(fill='x', padx=10, pady=5)

        self.platform_state_label = ttk.Label(state_frame,
                                              text="Roll=0.0° Pitch=0.0° Z=169.2mm",
                                              font=('Courier', 10))
        self.platform_state_label.grid(row=0, column=0, sticky='w', pady=2)

        self.pid_output_label = ttk.Label(state_frame,
                                          text="PID Out: X_roll=0.0° Y_pitch=0.0°",
                                          font=('Courier', 10))
        self.pid_output_label.grid(row=1, column=0, sticky='w', pady=2)

        self.pid_terms_x_label = ttk.Label(state_frame,
                                           text="X-axis: P=0.0  I=0.0  D=0.0",
                                           font=('Courier', 9))
        self.pid_terms_x_label.grid(row=2, column=0, sticky='w', pady=2)

        self.pid_terms_y_label = ttk.Label(state_frame,
                                           text="Y-axis: P=0.0  I=0.0  D=0.0",
                                           font=('Courier', 9))
        self.pid_terms_y_label.grid(row=3, column=0, sticky='w', pady=2)

        # Plot frame
        plot_frame = ttk.LabelFrame(self.root, text="Error History", padding=10)
        plot_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.fig = Figure(figsize=(8, 3), dpi=80)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Error (pixels)')
        self.ax.set_title('Ball Position Error')
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Log frame
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=10)
        log_frame.pack(fill='both', expand=False, padx=10, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, font=('Courier', 8))
        self.log_text.pack(fill='both', expand=True)

    def create_pid_row(self, parent, row, label, prefix):
        """Create a row of PID tuning controls."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=5)

        # Base slider (0-10)
        base_var = self.pid_vars[f'{prefix}_base']
        base_slider = ttk.Scale(parent, from_=0, to=10, orient='horizontal',
                                variable=base_var, command=lambda v: self.on_pid_change())
        base_slider.grid(row=row, column=1, sticky='ew', padx=5)

        ttk.Label(parent, text="×").grid(row=row, column=2, padx=2)

        # Multiplier slider (10^-7 to 10^2)
        mult_var = self.pid_vars[f'{prefix}_mult']
        mult_slider = ttk.Scale(parent, from_=-7, to=2, orient='horizontal',
                                variable=mult_var, command=lambda v: self.on_pid_change())
        mult_slider.grid(row=row, column=3, sticky='ew', padx=5)

        # Result label
        result_label = ttk.Label(parent, text="= 0.000", width=12, font=('Courier', 9))
        result_label.grid(row=row, column=4, padx=5)

        setattr(self, f'{prefix}_result_label', result_label)

        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(3, weight=1)

    def on_pid_change(self):
        """Called when any PID slider changes."""
        self.update_pid_gains()

    def update_pid_gains(self):
        """Calculate and apply PID gains from sliders."""
        # X-axis
        kp_x = self.pid_vars['x_kp_base'].get() * (10 ** self.pid_vars['x_kp_mult'].get())
        ki_x = self.pid_vars['x_ki_base'].get() * (10 ** self.pid_vars['x_ki_mult'].get())
        kd_x = self.pid_vars['x_kd_base'].get() * (10 ** self.pid_vars['x_kd_mult'].get())

        self.pid_x.set_gains(kp=kp_x, ki=ki_x, kd=kd_x)

        self.x_kp_result_label.config(text=f"= {kp_x:.6f}")
        self.x_ki_result_label.config(text=f"= {ki_x:.6f}")
        self.x_kd_result_label.config(text=f"= {kd_x:.6f}")

        # Y-axis
        kp_y = self.pid_vars['y_kp_base'].get() * (10 ** self.pid_vars['y_kp_mult'].get())
        ki_y = self.pid_vars['y_ki_base'].get() * (10 ** self.pid_vars['y_ki_mult'].get())
        kd_y = self.pid_vars['y_kd_base'].get() * (10 ** self.pid_vars['y_kd_mult'].get())

        self.pid_y.set_gains(kp=kp_y, ki=ki_y, kd=kd_y)

        self.y_kp_result_label.config(text=f"= {kp_y:.6f}")
        self.y_ki_result_label.config(text=f"= {ki_y:.6f}")
        self.y_kd_result_label.config(text=f"= {kd_y:.6f}")

    def log(self, message):
        """Add message to log."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def refresh_ports(self):
        """Refresh available serial ports."""
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo['values'] = ports
        if ports:
            self.port_combo.current(0)

    def toggle_connection(self):
        """Connect or disconnect from Teensy."""
        if not self.is_connected:
            self.connect()
        else:
            self.disconnect()

    def connect(self):
        """Connect to Teensy."""
        port = self.port_combo.get()
        if not port:
            messagebox.showerror("Error", "No port selected")
            return

        try:
            self.serial_conn = serial.Serial(port, 115200, timeout=0.1)
            time.sleep(2)  # Wait for Arduino reset

            self.is_connected = True
            self.running = True

            # Start serial reader thread
            self.serial_thread = threading.Thread(target=self.serial_reader, daemon=True)
            self.serial_thread.start()

            self.connect_btn.config(text="Disconnect")
            self.status_label.config(text="● Connected", foreground="green")
            self.log(f"Connected to {port}")

        except Exception as e:
            messagebox.showerror("Connection Error", str(e))
            self.log(f"Connection error: {e}")

    def disconnect(self):
        """Disconnect from Teensy."""
        self.running = False
        self.control_enabled = False

        if self.serial_thread:
            self.serial_thread.join(timeout=1)

        if self.serial_conn:
            self.serial_conn.close()
            self.serial_conn = None

        self.is_connected = False
        self.connect_btn.config(text="Connect")
        self.status_label.config(text="● Disconnected", foreground="red")
        self.log("Disconnected")

    def serial_reader(self):
        """Read data from Teensy in background thread."""
        while self.running:
            try:
                if self.serial_conn and self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()

                    if line.startswith("BALL:"):
                        # Parse ball data: timestamp,x,y,detected,error_x,error_y
                        parts = line[5:].split(',')
                        if len(parts) == 6:
                            data = {
                                'timestamp': float(parts[0]),
                                'x': float(parts[1]),
                                'y': float(parts[2]),
                                'detected': int(parts[3]) == 1,
                                'error_x': float(parts[4]),
                                'error_y': float(parts[5])
                            }
                            self.data_queue.put(('ball', data))

                    elif line.startswith("ACK:") or line.startswith("ERROR:"):
                        self.data_queue.put(('log', line))

                    elif line.startswith("READY:") or line.startswith("INIT:"):
                        self.data_queue.put(('log', line))

                time.sleep(0.001)  # Small delay

            except Exception as e:
                self.data_queue.put(('log', f"Serial error: {e}"))
                break

    def process_data_queue(self):
        """Process incoming data from serial thread."""
        try:
            while not self.data_queue.empty():
                msg_type, data = self.data_queue.get_nowait()

                if msg_type == 'ball':
                    self.update_ball_state(data)
                elif msg_type == 'log':
                    self.log(data)

        except queue.Empty:
            pass

    def update_ball_state(self, data):
        """Update ball state from received data."""
        self.ball_x = data['x']
        self.ball_y = data['y']
        self.ball_detected = data['detected']

        # Calculate error from setpoint
        if self.ball_detected:
            self.error_x = self.ball_x - self.setpoint_x
            self.error_y = self.setpoint_y - self.ball_y  # Y is inverted
            self.last_ball_time = time.time()
        else:
            self.error_x = 0.0
            self.error_y = 0.0

        # Update display
        detected_str = "✓ Detected" if self.ball_detected else "✗ Not Detected"
        self.ball_pos_label.config(text=f"Position: X={self.ball_x:.2f}  Y={self.ball_y:.2f}  {detected_str}")
        self.ball_error_label.config(text=f"Error: X={self.error_x:.2f}  Y={self.error_y:.2f}")

        # Add to history
        current_time = time.time()
        if len(self.time_history) == 0:
            self.time_history.append(0)
        else:
            self.time_history.append(current_time - self.last_ball_time + self.time_history[-1])

        self.error_x_history.append(self.error_x)
        self.error_y_history.append(self.error_y)

    def update_setpoint(self):
        """Update setpoint from entry fields."""
        try:
            self.setpoint_x = float(self.setpoint_x_entry.get())
            self.setpoint_y = float(self.setpoint_y_entry.get())
            self.log(f"Setpoint updated: X={self.setpoint_x:.1f}, Y={self.setpoint_y:.1f}")
        except ValueError:
            messagebox.showerror("Error", "Invalid setpoint values")

    def toggle_control(self):
        """Enable or disable control."""
        if not self.is_connected:
            messagebox.showwarning("Not Connected", "Please connect to Teensy first")
            return

        self.control_enabled = not self.control_enabled

        if self.control_enabled:
            self.pid_x.reset()
            self.pid_y.reset()
            self.enable_btn.config(text="Disable Control")
            self.control_status_label.config(text="● Control Enabled", foreground="green")
            self.log("Control ENABLED")
        else:
            self.enable_btn.config(text="Enable Control")
            self.control_status_label.config(text="● Control Disabled", foreground="orange")
            self.log("Control DISABLED")
            self.reset_platform()

    def emergency_stop(self):
        """Emergency stop - disable control and return to home."""
        self.control_enabled = False
        self.enable_btn.config(text="Enable Control")
        self.control_status_label.config(text="● Control Disabled", foreground="orange")
        self.pid_x.reset()
        self.pid_y.reset()
        self.reset_platform()
        self.log("EMERGENCY STOP")

    def reset_platform(self):
        """Return platform to home position."""
        self.platform_roll = 0.0
        self.platform_pitch = 0.0
        self.platform_z = self.ik.home_height_top_surface
        self.send_platform_pose()
        self.log("Platform reset to home")

    def control_loop(self):
        """Main control loop - calculate PID and send to platform."""
        if not self.control_enabled or not self.ball_detected:
            return

        # Check for ball timeout
        if time.time() - self.last_ball_time > 1.0:
            self.log("Ball lost - holding position")
            return

        # Calculate PID outputs
        roll_output = self.pid_x.update(self.error_x)  # X error → Roll
        pitch_output = self.pid_y.update(self.error_y)  # Y error → Pitch

        self.platform_roll = roll_output
        self.platform_pitch = pitch_output

        # Update display
        self.pid_output_label.config(text=f"PID Out: X_roll={roll_output:.2f}° Y_pitch={pitch_output:.2f}°")
        self.pid_terms_x_label.config(
            text=f"X-axis: P={self.pid_x.p_term:.2f}  I={self.pid_x.i_term:.2f}  D={self.pid_x.d_term:.2f}")
        self.pid_terms_y_label.config(
            text=f"Y-axis: P={self.pid_y.p_term:.2f}  I={self.pid_y.i_term:.2f}  D={self.pid_y.d_term:.2f}")

        # Send to platform
        self.send_platform_pose()

    def send_platform_pose(self):
        """Calculate IK and send servo angles to Teensy."""
        translation = np.array([0, 0, self.platform_z])
        rotation = np.array([self.platform_pitch, self.platform_roll, 0])

        angles = self.ik.calculate_servo_angles(translation, rotation, use_top_surface_offset=True)

        if angles is not None:
            # Send to Teensy
            if self.serial_conn and self.is_connected:
                command = ",".join([f"{angle:.3f}" for angle in angles]) + "\n"
                try:
                    self.serial_conn.write(command.encode())
                except Exception as e:
                    self.log(f"Send error: {e}")

            # Update display
            self.platform_state_label.config(
                text=f"Roll={self.platform_roll:.2f}° Pitch={self.platform_pitch:.2f}° Z={self.platform_z:.2f}mm")
        else:
            self.log("ERROR: IK failed (unreachable pose)")
            self.emergency_stop()

    def update_plot(self):
        """Update error history plot."""
        if len(self.time_history) > 1:
            self.ax.clear()
            self.ax.plot(list(self.time_history), list(self.error_x_history), 'b-', label='Error X', linewidth=1.5)
            self.ax.plot(list(self.time_history), list(self.error_y_history), 'r-', label='Error Y', linewidth=1.5)
            self.ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Error (pixels)')
            self.ax.set_title('Ball Position Error')
            self.ax.legend(loc='upper right')
            self.ax.grid(True, alpha=0.3)
            self.canvas.draw()

    def start_update_loop(self):
        """Start the main GUI update loop."""
        self.update_loop()

    def update_loop(self):
        """Main update loop - runs at ~50Hz."""
        # Process incoming data
        self.process_data_queue()

        # Run control loop if enabled
        if self.control_enabled:
            self.control_loop()

        # Update plot every 500ms
        if hasattr(self, '_last_plot_update'):
            if time.time() - self._last_plot_update > 0.5:
                self.update_plot()
                self._last_plot_update = time.time()
        else:
            self._last_plot_update = time.time()

        # Schedule next update
        self.root.after(20, self.update_loop)  # 50Hz

    def cleanup(self):
        """Cleanup on exit."""
        self.running = False
        self.control_enabled = False

        if self.serial_thread:
            self.serial_thread.join(timeout=1)

        if self.serial_conn:
            self.serial_conn.close()


def main():
    root = tk.Tk()
    app = BallBalanceGUI(root)

    def on_closing():
        app.cleanup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()