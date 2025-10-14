#!/usr/bin/env python3
"""
Stewart Platform Z-Axis Step Delay Measurement - GUI
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import serial
import serial.tools.list_ports
import numpy as np
import time
import threading
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class StewartPlatformIK:
    """Inverse kinematics - copied from sim.py"""

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


class DelayMeasurementGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stewart Platform - Delay Measurement")
        self.root.geometry("900x700")

        self.ik = StewartPlatformIK()
        self.serial_conn = None
        self.is_connected = False
        self.is_measuring = False

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

        # Measurement control frame
        measure_frame = ttk.LabelFrame(self.root, text="Delay Measurement", padding=10)
        measure_frame.pack(fill='x', padx=10, pady=5)

        # RY rotation input
        input_frame = ttk.Frame(measure_frame)
        input_frame.pack(pady=5)

        ttk.Label(input_frame, text="RY rotation (deg):").pack(side='left', padx=5)
        self.ry_step_var = tk.StringVar(value="15.0")
        ttk.Entry(input_frame, textvariable=self.ry_step_var, width=10).pack(side='left', padx=5)

        # Measure button
        self.measure_btn = ttk.Button(measure_frame, text="Run Measurement",
                                      command=self.start_measurement, state='disabled')
        self.measure_btn.pack(pady=5)

        # Plot frame
        plot_frame = ttk.LabelFrame(self.root, text="Z-Axis Acceleration", padding=10)
        plot_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Manual controls
        manual_frame = ttk.LabelFrame(self.root, text="Manual Control", padding=10)
        manual_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(manual_frame, text="Home Position", command=self.home_position).pack(side='left', padx=5)

        # Log frame
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=10)
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, font=('Courier', 8))
        self.log_text.pack(fill='both', expand=True)

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
            self.serial_conn = serial.Serial(port, 2000000, timeout=0.1)
            time.sleep(2)

            # Read startup messages
            while self.serial_conn.in_waiting:
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    self.log(f"Teensy: {line}")

            # Set maximum speed
            self.log("Setting servo speed to maximum...")
            self.serial_conn.write(b"SPD:0\n")
            time.sleep(0.1)

            self.is_connected = True
            self.connect_btn.config(text="Disconnect")
            self.status_label.config(text=f"Connected to {port}", foreground="green")
            self.measure_btn.config(state='normal')

            self.log("Connection established")
            self.home_position()

        except Exception as e:
            messagebox.showerror("Connection Error", str(e))
            self.log(f"Connection error: {e}")

    def disconnect(self):
        if self.serial_conn:
            self.serial_conn.close()
            self.serial_conn = None

        self.is_connected = False
        self.connect_btn.config(text="Connect")
        self.status_label.config(text="Disconnected", foreground="red")
        self.measure_btn.config(state='disabled')
        self.log("Disconnected")

    def send_position(self, x, y, z, rx, ry, rz):
        translation = np.array([x, y, z])
        rotation = np.array([rx, ry, rz])

        angles = self.ik.calculate_servo_angles(translation, rotation, use_top_surface_offset=True)

        if angles is None:
            self.log("ERROR: Position unreachable")
            return None

        command = ",".join([f"{angle:.3f}" for angle in angles]) + "\n"
        send_time = time.time()
        self.serial_conn.write(command.encode())
        return send_time

    def home_position(self):
        home_z = self.ik.home_height_top_surface
        self.send_position(0, 0, home_z, 0, 0, 0)
        self.log("Moving to home position")

    def start_measurement(self):
        if self.is_measuring:
            return

        try:
            ry_step = float(self.ry_step_var.get())
            if abs(ry_step) > 20:
                messagebox.showerror("Error", "RY rotation must be between -20 and 20 degrees")
                return
        except ValueError:
            messagebox.showerror("Error", "Invalid rotation value")
            return

        self.is_measuring = True
        self.measure_btn.config(state='disabled')
        self.connect_btn.config(state='disabled')

        thread = threading.Thread(target=self.run_measurement, args=(ry_step,), daemon=True)
        thread.start()

    def run_measurement(self, ry_step):
        try:
            home_z = self.ik.home_height_top_surface

            self.log("=" * 50)
            self.log(f"Starting measurement with RY = {ry_step:+.1f} deg")

            # Go to home and stabilize
            self.log(f"Moving to home...")
            self.send_position(0, 0, home_z, 0, 0, 0)
            time.sleep(4.0)

            self.log("Stabilizing...")
            time.sleep(2.0)

            # Clear serial buffer
            while self.serial_conn.in_waiting:
                self.serial_conn.readline()

            # Send rotation command
            self.log(f"Sending RY rotation: {ry_step:+.1f} deg")
            command_time = self.send_position(0, 0, home_z, 0, ry_step, 0)

            if command_time is None:
                self.log("ERROR: Failed to send position")
                return

            # Collect 400ms of IMU data
            self.log("Collecting 400ms of data...")
            samples = self.read_imu_samples(duration=0.4)
            self.log(f"Collected {len(samples)} samples")

            if len(samples) < 20:
                self.log("ERROR: Insufficient IMU samples")
                return

            # Plot the data
            self.plot_data(samples, command_time)

            # Return to home
            time.sleep(0.5)
            self.send_position(0, 0, home_z, 0, 0, 0)

        except Exception as e:
            self.log(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_measuring = False
            self.measure_btn.config(state='normal')
            self.connect_btn.config(state='normal')

    def plot_data(self, samples, command_time):
        """Plot Z-axis acceleration with command impulse"""
        timestamps = np.array([s['timestamp_pc'] for s in samples])
        accel_z = np.array([s['accel'][2] for s in samples])

        # Convert to relative time (ms)
        t_rel = (timestamps - command_time) * 1000

        # Clear and plot
        self.ax.clear()
        self.ax.plot(t_rel, accel_z, 'b-', linewidth=1, label='Z acceleration')
        self.ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='RY command sent')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('Time from command (ms)')
        self.ax.set_ylabel('Z Acceleration (raw)')
        self.ax.set_title('Z-Axis Acceleration vs Time (RY Rotation)')
        self.ax.legend()

        self.canvas.draw()

        self.log(f"Plot updated: {len(samples)} samples over {t_rel[-1]:.0f} ms")

    def read_imu_samples(self, duration):
        samples = []
        start_time = time.time()

        while (time.time() - start_time) < duration:
            if self.serial_conn.in_waiting:
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()

                if line.startswith("IMU,"):
                    parts = line.split(',')
                    if len(parts) == 8:
                        try:
                            sample = {
                                'timestamp_pc': time.time(),
                                'accel': np.array([int(parts[2]), int(parts[3]), int(parts[4])]),
                                'gyro': np.array([int(parts[5]), int(parts[6]), int(parts[7])]),
                            }
                            samples.append(sample)
                        except ValueError:
                            continue

        return samples

    def cleanup(self):
        self.disconnect()


def main():
    root = tk.Tk()
    app = DelayMeasurementGUI(root)

    def on_closing():
        app.cleanup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()