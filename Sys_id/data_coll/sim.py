#!/usr/bin/env python3
"""
Stewart Platform Automated Experiments with IMU Logging
Optimized: Theta sent only on change, IMU at full speed
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import serial
import serial.tools.list_ports
import numpy as np
import time
import threading
import csv
from datetime import datetime
from queue import Queue, Empty
import os


class StewartPlatformIK:
    """Inverse kinematics using Robert Eisele's method."""

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

        base_angels = np.array([-np.pi / 2, np.pi / 6, np.pi * 5 / 6])
        platform_angels = np.array([-np.pi * 5 / 6, -np.pi / 6, np.pi / 2])

        self.base_anchors = self.claculate_home_coordinates(self.base, self.base_anchors, base_angels)
        platform_anchors_out_of_fase = self.claculate_home_coordinates(self.platform, self.platform_anchors,
                                                                       platform_angels)
        self.platform_anchors = np.roll(platform_anchors_out_of_fase, shift=-1, axis=0)

        self.beta_angles = self._calculate_beta_angles()

        base_pos = self.base_anchors[0]
        platform_pos = self.platform_anchors[0]

        horn_end_x = base_pos[0] + self.horn_length * np.cos(self.beta_angles[0])
        horn_end_y = base_pos[1] + self.horn_length * np.sin(self.beta_angles[0])

        dx = platform_pos[0] - horn_end_x
        dy = platform_pos[1] - horn_end_y
        horiz_dist_sq = dx ** 2 + dy ** 2

        self.home_height = np.sqrt(self.rod_length ** 2 - horiz_dist_sq)
        self.home_height_top_surface = self.home_height + self.top_surface_offset

    def claculate_home_coordinates(self, l, d, phi):
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

    def calculate_servo_angles(self, translation: np.ndarray, rotation: np.ndarray,
                               use_top_surface_offset: bool = True):
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

    def _euler_to_quaternion(self, euler: np.ndarray) -> np.ndarray:
        rx, ry, rz = euler
        cy = np.cos(rz * 0.5)
        sy = np.sin(rz * 0.5)
        cp = np.cos(ry * 0.5)
        sp = np.sin(ry * 0.5)
        cr = np.cos(rx * 0.5)
        sr = np.sin(rx * 0.5)

        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        ])

    def _rotate_vector(self, v: np.ndarray, q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        vx, vy, vz = v

        return np.array([
            vx * (w * w + x * x - y * y - z * z) + vy * (2 * x * y - 2 * w * z) + vz * (2 * x * z + 2 * w * y),
            vx * (2 * x * y + 2 * w * z) + vy * (w * w - x * x + y * y - z * z) + vz * (2 * y * z - 2 * w * x),
            vx * (2 * x * z - 2 * w * y) + vy * (2 * y * z + 2 * w * x) + vz * (w * w - x * x - y * y + z * z)
        ])


class IMUReader(threading.Thread):
    """Background thread for continuous IMU data reading."""

    def __init__(self, serial_conn, data_queue, log_callback):
        super().__init__(daemon=True)
        self.serial_conn = serial_conn
        self.data_queue = data_queue
        self.log_callback = log_callback
        self.running = False
        self.debug_counter = 0
        self.parse_errors = 0

    def run(self):
        self.running = True
        self.log_callback("IMU Reader thread started")

        while self.running:
            try:
                if self.serial_conn and self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()

                    if not line:
                        continue

                    if self.debug_counter < 10:
                        self.log_callback(f"RX: {line}")
                        self.debug_counter += 1

                    if line.startswith("IMU,"):
                        parts = line.split(',')
                        # Format: IMU,imu_micros,cmd_micros,ax,ay,az,gx,gy,gz
                        if len(parts) == 9:
                            try:
                                timestamp_us = int(parts[1])
                                command_time_us = int(parts[2])
                                accel = [int(parts[3]), int(parts[4]), int(parts[5])]
                                gyro = [int(parts[6]), int(parts[7]), int(parts[8])]

                                self.data_queue.put({
                                    'type': 'imu',
                                    'timestamp_us': timestamp_us,
                                    'command_time_us': command_time_us,
                                    'timestamp_pc': time.time(),
                                    'accel': accel,
                                    'gyro': gyro
                                })

                                self.parse_errors = 0

                            except (ValueError, IndexError) as e:
                                if self.parse_errors < 5:
                                    self.log_callback(f"IMU parse error: {e}")
                                    self.parse_errors += 1
                        else:
                            if self.parse_errors < 5:
                                self.log_callback(f"Wrong IMU format: got {len(parts)} parts, expected 9")
                                self.parse_errors += 1

                    elif line.startswith("CMD,"):
                        parts = line.split(',')
                        # Format: CMD,cmd_us,theta0,theta1,theta2,theta3,theta4,theta5
                        if len(parts) == 8:
                            try:
                                command_time_us = int(parts[1])
                                theta = [float(parts[2]), float(parts[3]), float(parts[4]),
                                         float(parts[5]), float(parts[6]), float(parts[7])]

                                self.data_queue.put({
                                    'type': 'cmd',
                                    'command_time_us': command_time_us,
                                    'timestamp_pc': time.time(),
                                    'theta': theta
                                })

                            except (ValueError, IndexError) as e:
                                if self.parse_errors < 5:
                                    self.log_callback(f"CMD parse error: {e}")
                                    self.parse_errors += 1

                    elif line.startswith("IMU_"):
                        self.log_callback(line)
                    elif line == "READY":
                        self.log_callback("Teensy ready")
                    elif line == "OK" or line == "OK_SPD":
                        pass
                    else:
                        if self.debug_counter < 20:
                            self.log_callback(f"Unknown: {line}")
                            self.debug_counter += 1

            except Exception as e:
                if self.parse_errors < 5:
                    self.log_callback(f"IMU thread error: {e}")
                    self.parse_errors += 1

            time.sleep(0.0001)

    def stop(self):
        self.running = False
        self.log_callback("IMU Reader thread stopped")


class ExperimentRunner:
    """Manages automated experiment sequences."""

    @staticmethod
    def step_response(axis, amplitude, duration=2.0):
        return {
            'name': f'Step_{axis}_{amplitude:+.1f}deg',
            'type': 'step',
            'axis': axis,
            'amplitude': amplitude,
            'duration': duration
        }

    @staticmethod
    def sine_wave(axis, amplitude, frequency, duration=5.0):
        return {
            'name': f'Sine_{axis}_A{amplitude:.1f}deg_F{frequency:.2f}Hz',
            'type': 'sine',
            'axis': axis,
            'amplitude': amplitude,
            'frequency': frequency,
            'duration': duration
        }

    @staticmethod
    def get_default_experiments():
        """Simple experiment sequence with large angle steps."""
        experiments = []

        # Step responses for RX and RY - ±15 degrees
        for axis in ['rx', 'ry']:
            experiments.append(ExperimentRunner.step_response(axis, 15.0, 3.0))
            experiments.append(ExperimentRunner.step_response(axis, -15.0, 3.0))

        return experiments


class StewartControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stewart Platform - Automated Experiments")
        self.root.geometry("700x850")

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
        self.serial_conn = None
        self.is_connected = False

        self.imu_queue = Queue(maxsize=10000)
        self.imu_reader = None
        self.imu_samples_received = 0

        self.csv_file = None
        self.csv_writer = None
        self.is_recording = False
        self.csv_rows_written = 0

        self.is_running_experiments = False
        self.current_experiment = None
        self.experiment_thread = None

        self.use_top_surface_offset = tk.BooleanVar(value=True)

        self.current_angles = np.zeros(6)

        # Track latest theta values from CMD messages
        self.latest_theta = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.latest_command_time_us = 0

        self.create_widgets()

    def create_widgets(self):
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
        self.status_label = ttk.Label(conn_frame, text="Disconnected", foreground="red")
        self.status_label.grid(row=0, column=4, padx=5)

        # Experiment control frame
        exp_frame = ttk.LabelFrame(self.root, text="Automated Experiments", padding=10)
        exp_frame.pack(fill='x', padx=10, pady=5)

        self.exp_status_label = ttk.Label(exp_frame, text="Ready to run experiments", foreground="blue")
        self.exp_status_label.pack(pady=5)

        btn_container = ttk.Frame(exp_frame)
        btn_container.pack(fill='x', pady=5)

        self.run_exp_btn = ttk.Button(btn_container, text="Run All Experiments",
                                      command=self.start_experiments, state='disabled')
        self.run_exp_btn.pack(side='left', padx=5)

        self.stop_exp_btn = ttk.Button(btn_container, text="Stop Experiments",
                                       command=self.stop_experiments, state='disabled')
        self.stop_exp_btn.pack(side='left', padx=5)

        ttk.Label(exp_frame, text="Output folder:").pack(anchor='w', pady=(10, 0))

        folder_frame = ttk.Frame(exp_frame)
        folder_frame.pack(fill='x', pady=2)

        self.output_folder_var = tk.StringVar(value=os.path.join(os.getcwd(), "experiments"))
        ttk.Entry(folder_frame, textvariable=self.output_folder_var, width=50).pack(side='left', fill='x', expand=True)
        ttk.Button(folder_frame, text="Browse", command=self.browse_folder).pack(side='left', padx=5)

        # Manual control frame
        manual_frame = ttk.LabelFrame(self.root, text="Manual Control", padding=10)
        manual_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(manual_frame, text="Home Position", command=self.home_position).pack(side='left', padx=5)
        ttk.Button(manual_frame, text="Emergency Stop", command=self.emergency_stop).pack(side='left', padx=5)

        # Status frame
        status_frame = ttk.LabelFrame(self.root, text="Statistics", padding=10)
        status_frame.pack(fill='x', padx=10, pady=5)

        self.stats_label = ttk.Label(status_frame, text="IMU samples: 0 | CSV rows: 0", font=('Courier', 9))
        self.stats_label.pack()

        # Servo angles display
        angles_frame = ttk.LabelFrame(self.root, text="Current Servo Angles", padding=10)
        angles_frame.pack(fill='x', padx=10, pady=5)

        self.angle_labels = []
        servo_names = ['θ1', 'θ2', 'θ3', 'θ4', 'θ5', 'θ6']
        for i in range(6):
            label = ttk.Label(angles_frame, text=f"{servo_names[i]}: 0.00°", font=('Courier', 9))
            label.grid(row=i // 3, column=i % 3, padx=10, pady=2, sticky='w')
            self.angle_labels.append(label)

        # IMU data display
        imu_frame = ttk.LabelFrame(self.root, text="IMU Data (Last Sample)", padding=10)
        imu_frame.pack(fill='x', padx=10, pady=5)

        self.imu_label = ttk.Label(imu_frame, text="Accel: [-, -, -] | Gyro: [-, -, -]",
                                   font=('Courier', 9))
        self.imu_label.pack()

        # Debug log
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=10)
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, font=('Courier', 8))
        self.log_text.pack(fill='both', expand=True)

        self.update_displays()

    def browse_folder(self):
        folder = filedialog.askdirectory(initialdir=self.output_folder_var.get())
        if folder:
            self.output_folder_var.set(folder)

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def refresh_ports(self):
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo['values'] = ports
        if ports:
            self.port_combo.current(0)

    def toggle_connection(self):
        if not self.is_connected:
            self.connect()
        else:
            self.disconnect()

    def connect(self):
        port = self.port_combo.get()
        if not port:
            messagebox.showerror("Error", "No port selected")
            return

        try:
            self.log(f"Connecting to {port}...")
            self.serial_conn = serial.Serial(port, 2000000, timeout=0.5)
            self.log("Serial port opened, waiting for Teensy startup...")

            time.sleep(2)

            # Read startup messages
            startup_lines = []
            start_time = time.time()
            while time.time() - start_time < 1.0:
                if self.serial_conn.in_waiting:
                    try:
                        line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            startup_lines.append(line)
                            self.log(f"Teensy: {line}")
                    except:
                        break
                else:
                    time.sleep(0.1)

            self.log("Setting servo speed to maximum...")
            self.serial_conn.write(b"SPD:0\n")
            time.sleep(0.1)

            self.imu_reader = IMUReader(self.serial_conn, self.imu_queue, self.log)
            self.imu_reader.start()

            self.is_connected = True
            self.connect_btn.config(text="Disconnect")
            self.status_label.config(text=f"Connected to {port}", foreground="green")
            self.run_exp_btn.config(state='normal')

            self.log("Connection established successfully")
            self.log("Waiting for IMU data...")

            time.sleep(0.5)

            self.home_position()

        except Exception as e:
            messagebox.showerror("Connection Error", str(e))
            self.log(f"Connection error: {e}")

            if self.serial_conn:
                try:
                    self.serial_conn.close()
                except:
                    pass
                self.serial_conn = None

    def disconnect(self):
        if self.imu_reader:
            self.imu_reader.stop()
            time.sleep(0.2)
            self.imu_reader = None

        if self.csv_file:
            self.stop_recording()

        if self.serial_conn:
            self.serial_conn.close()
            self.serial_conn = None

        self.is_connected = False
        self.connect_btn.config(text="Connect")
        self.status_label.config(text="Disconnected", foreground="red")
        self.run_exp_btn.config(state='disabled')
        self.log("Disconnected")

    def update_displays(self):
        try:
            while not self.imu_queue.empty():
                data = self.imu_queue.get_nowait()

                if data['type'] == 'imu':
                    self.imu_samples_received += 1

                    accel = data['accel']
                    gyro = data['gyro']

                    accel_str = ', '.join([f"{a:6d}" for a in accel])
                    gyro_str = ', '.join([f"{g:6d}" for g in gyro])

                    self.imu_label.config(text=f"Accel: [{accel_str}] | Gyro: [{gyro_str}]")

                    if self.is_recording and self.csv_writer:
                        self.write_csv_row(data)

                elif data['type'] == 'cmd':
                    # Update latest theta values
                    self.latest_theta = data['theta']
                    self.latest_command_time_us = data['command_time_us']

                    # Update angle display
                    for i in range(6):
                        self.angle_labels[i].config(text=f"θ{i + 1}: {self.latest_theta[i]:6.2f}°")

        except Empty:
            pass

        self.stats_label.config(
            text=f"IMU samples: {self.imu_samples_received} | CSV rows: {self.csv_rows_written} | Queue: {self.imu_queue.qsize()}"
        )

        self.root.after(50, self.update_displays)

    def start_recording(self, filename):
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            self.csv_file = open(filename, 'w', newline='', buffering=1)
            self.csv_writer = csv.writer(self.csv_file)

            self.csv_writer.writerow([
                'timestamp_us', 'command_time_us', 'timestamp_pc',
                'theta0', 'theta1', 'theta2', 'theta3', 'theta4', 'theta5',
                'accel_x', 'accel_y', 'accel_z',
                'gyro_x', 'gyro_y', 'gyro_z'
            ])

            self.is_recording = True
            self.csv_rows_written = 0
            self.log(f"Recording started: {os.path.basename(filename)}")

        except Exception as e:
            self.log(f"ERROR starting recording: {e}")
            messagebox.showerror("Recording Error", str(e))

    def stop_recording(self):
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            self.log(f"Recording stopped. Rows written: {self.csv_rows_written}")
        self.is_recording = False

    def write_csv_row(self, imu_data):
        try:
            # Merge IMU data with latest theta values
            row = [
                imu_data['timestamp_us'],
                imu_data['command_time_us'],
                imu_data['timestamp_pc'],
                *self.latest_theta,
                *imu_data['accel'],
                *imu_data['gyro']
            ]
            self.csv_writer.writerow(row)
            self.csv_rows_written += 1
        except Exception as e:
            self.log(f"ERROR writing CSV: {e}")

    def send_position(self, x, y, z, rx, ry, rz):
        translation = np.array([x, y, z])
        rotation = np.array([rx, ry, rz])

        angles = self.ik.calculate_servo_angles(translation, rotation,
                                                use_top_surface_offset=self.use_top_surface_offset.get())

        if angles is not None:
            self.current_angles = angles

            if self.is_connected and self.serial_conn:
                command = ",".join([f"{angle:.3f}" for angle in angles]) + "\n"
                self.serial_conn.write(command.encode())
                return True
        else:
            self.log("ERROR: Unreachable position")
            return False

    def home_position(self):
        home_z = self.ik.home_height_top_surface if self.use_top_surface_offset.get() else self.ik.home_height
        self.send_position(0, 0, home_z, 0, 0, 0)
        self.log("Moving to home position")

    def emergency_stop(self):
        self.stop_experiments()
        self.home_position()
        self.log("EMERGENCY STOP - Returning to home")

    def start_experiments(self):
        if self.is_running_experiments:
            return

        self.is_running_experiments = True
        self.run_exp_btn.config(state='disabled')
        self.stop_exp_btn.config(state='normal')
        self.connect_btn.config(state='disabled')

        self.experiment_thread = threading.Thread(target=self.run_experiments, daemon=True)
        self.experiment_thread.start()

    def stop_experiments(self):
        self.is_running_experiments = False
        if self.experiment_thread:
            self.experiment_thread.join(timeout=1)

        self.run_exp_btn.config(state='normal')
        self.stop_exp_btn.config(state='disabled')
        self.connect_btn.config(state='normal')
        self.exp_status_label.config(text="Experiments stopped")

    def run_experiments(self):
        experiments = ExperimentRunner.get_default_experiments()
        output_folder = self.output_folder_var.get()

        self.log(f"Starting {len(experiments)} experiments")
        self.log(f"Output folder: {output_folder}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for idx, exp in enumerate(experiments):
            if not self.is_running_experiments:
                break

            self.current_experiment = exp
            self.exp_status_label.config(
                text=f"Running {idx + 1}/{len(experiments)}: {exp['name']}"
            )
            self.log(f"=== Experiment {idx + 1}/{len(experiments)}: {exp['name']} ===")

            filename = os.path.join(output_folder, f"{timestamp}_{exp['name']}.csv")

            home_z = self.ik.home_height_top_surface if self.use_top_surface_offset.get() else self.ik.home_height
            self.send_position(0, 0, home_z, 0, 0, 0)
            time.sleep(1.5)

            while not self.imu_queue.empty():
                try:
                    self.imu_queue.get_nowait()
                except Empty:
                    break

            self.start_recording(filename)
            time.sleep(0.2)

            if exp['type'] == 'step':
                self.run_step_experiment(exp)
            elif exp['type'] == 'sine':
                self.run_sine_experiment(exp)

            time.sleep(0.5)

            self.stop_recording()

            self.send_position(0, 0, home_z, 0, 0, 0)
            time.sleep(1.0)

        self.is_running_experiments = False
        self.run_exp_btn.config(state='normal')
        self.stop_exp_btn.config(state='disabled')
        self.connect_btn.config(state='normal')
        self.exp_status_label.config(text="All experiments completed!")
        self.log("=== ALL EXPERIMENTS COMPLETED ===")

    def run_step_experiment(self, exp):
        axis = exp['axis']
        amplitude = exp['amplitude']
        duration = exp['duration']

        home_z = self.ik.home_height_top_surface if self.use_top_surface_offset.get() else self.ik.home_height

        pos = {'x': 0, 'y': 0, 'z': home_z, 'rx': 0, 'ry': 0, 'rz': 0}
        pos[axis] = amplitude

        self.send_position(pos['x'], pos['y'], pos['z'], pos['rx'], pos['ry'], pos['rz'])
        time.sleep(duration)

    def run_sine_experiment(self, exp):
        axis = exp['axis']
        amplitude = exp['amplitude']
        frequency = exp['frequency']
        duration = exp['duration']

        home_z = self.ik.home_height_top_surface if self.use_top_surface_offset.get() else self.ik.home_height

        start_time = time.time()
        dt = 0.02

        while time.time() - start_time < duration and self.is_running_experiments:
            t = time.time() - start_time
            value = amplitude * np.sin(2 * np.pi * frequency * t)

            pos = {'x': 0, 'y': 0, 'z': home_z, 'rx': 0, 'ry': 0, 'rz': 0}
            pos[axis] = value

            self.send_position(pos['x'], pos['y'], pos['z'], pos['rx'], pos['ry'], pos['rz'])
            time.sleep(dt)

    def cleanup(self):
        self.stop_experiments()
        self.disconnect()


def main():
    root = tk.Tk()
    app = StewartControlGUI(root)

    def on_closing():
        app.cleanup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()