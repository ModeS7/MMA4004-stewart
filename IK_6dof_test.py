#!/usr/bin/env python3
"""
Stewart Platform Manual Control GUI with IK Calculations

Requirements: pip install pyserial numpy
"""

import tkinter as tk
from tkinter import ttk
import serial
import serial.tools.list_ports
import numpy as np
import time


class StewartPlatformIK:
    """Inverse Kinematics calculator for Stewart Platform"""

    def __init__(self):
        # Geometry parameters (from original code)
        self.l0 = 73.025
        self.lf = 67.775
        self.d1 = 36.8893
        self.d2 = 38.1
        self.m = 12.7
        self.p1 = 31.75
        self.p2 = 145

        # Normal vectors for projection planes
        self.nab = np.array([np.sqrt(3) * 0.5, -0.5, 0])
        self.nac = np.array([np.sqrt(3) * 0.5, 0.5, 0])
        self.nbc = np.array([0, 1, 0])

        # Intermediate variables
        self.t = (self.lf ** 2 * np.sqrt(3)) / 2
        self.u = np.sqrt(self.l0 ** 2 + self.d1 ** 2) * np.sin((2 * np.pi / 3) - np.arctan(self.l0 / self.d1))

        # Base anchor positions
        self.a10 = np.array([(self.d2 - self.u * np.sqrt(3)) / 2, (-self.u - self.d2 * np.sqrt(3)) / 2, 0])
        self.a20 = np.array([-self.a10[0], self.a10[1], 0])
        self.b10 = np.array([(self.u * np.sqrt(3) + self.d2) / 2, (self.d2 * np.sqrt(3) - self.u) / 2, 0])
        self.b20 = np.array([self.d2, self.u, 0])
        self.c10 = np.array([-self.b20[0], self.b20[1], 0])
        self.c20 = np.array([-self.b10[0], self.b10[1], 0])

        # Vectors between base anchors
        self.ab = self.a20 - self.b10
        self.ac = self.a10 - self.c20
        self.bc = self.b20 - self.c10

        self.nz = 1.0
        self.to_deg = 180.0 / np.pi

    def calculate_angles(self, hx, hy, hz, nx, ny, ax):
        """
        Calculate servo angles for given platform position and orientation.

        Args:
            hx, hy, hz: Platform center position
            nx, ny: Platform tilt components (nz is fixed at 1)
            ax: Rotation parameter

        Returns:
            Array of 6 servo angles [theta0, theta1, theta2, theta3, theta4, theta5]
            or None if calculation fails
        """
        try:
            # Define platform frame vectors
            a = np.array([ax, 0, 0])

            # Normal vector (platform orientation)
            n = np.array([nx, ny, self.nz])
            mag_n = np.linalg.norm(n)
            n = n / mag_n

            # Platform center position
            h = np.array([hx, hy, hz])

            # STAGE 1: Calculate platform triangle vertices (a, b, c)
            e = np.zeros(3)
            g = np.zeros(3)
            k = np.zeros(3)

            # Point a calculation
            e[0] = a[0] - h[0]
            a[2] = ((n[1] * np.sqrt(self.lf ** 2 * (1 - n[0] ** 2) - e[0] ** 2) - n[2] * n[0] * e[0]) / (
                        1 - n[0] ** 2)) + h[2]
            g[0] = a[2] - h[2]
            a[1] = h[1] - np.sqrt(self.lf ** 2 - g[0] ** 2 - e[0] ** 2)
            k[0] = a[1] - h[1]

            w = np.sqrt(3) * (n[0] * g[0] - n[2] * e[0])

            # Point b calculation
            b = np.zeros(3)
            b[1] = h[1] + ((np.sqrt(w ** 2 - 3 * self.lf ** 2 * (1 - n[1] ** 2) + (2 * k[0]) ** 2) - w) / 2)
            k[1] = b[1] - h[1]
            b[0] = ((e[0] * k[1] - n[2] * self.t) / k[0]) + h[0]
            e[1] = b[0] - h[0]
            b[2] = ((n[0] * self.t + g[0] * k[1]) / k[0]) + h[2]
            g[1] = b[2] - h[2]

            # Point c calculation
            c = np.zeros(3)
            c[1] = h[1] + ((w + np.sqrt(w ** 2 - 3 * self.lf ** 2 * (1 - n[1] ** 2) + (2 * k[0]) ** 2)) / 2)
            k[2] = c[1] - h[1]
            c[0] = ((e[0] * k[2] + n[2] * self.t) / k[0]) + h[0]
            e[2] = c[0] - h[0]
            c[2] = ((g[0] * k[2] - n[0] * self.t) / k[0]) + h[2]
            g[2] = c[2] - h[2]

            # STAGE 2: Calculate platform anchor positions

            # a1
            a1f = np.zeros(3)
            a1f[0] = a[0] + (self.m / self.lf) * (n[2] * k[0] - n[1] * g[0])
            if e[0] == 0:
                a1f[1] = a[1]
                a1f[2] = a[2]
            else:
                a1f[1] = a[1] + ((a1f[0] - a[0]) * k[0] - n[2] * self.lf * self.m) / e[0]
                a1f[2] = a[2] + (n[1] * self.lf * self.m + (a1f[0] - a[0]) * g[0]) / e[0]
            a1 = a1f - self.a10

            # a2
            a2f = 2 * a - a1f
            a2 = a2f - self.a20

            # b1
            b1f = np.zeros(3)
            b1f[0] = b[0] + (self.m / self.lf) * (n[2] * k[1] - n[1] * g[1])
            b1f[1] = b[1] + ((b1f[0] - b[0]) * k[1] - n[2] * self.lf * self.m) / e[1]
            b1f[2] = b[2] + (n[1] * self.lf * self.m + (b1f[0] - b[0]) * g[1]) / e[1]
            b1 = b1f - self.b10

            # b2
            b2f = 2 * b - b1f
            b2 = b2f - self.b20

            # c1
            c1f = np.zeros(3)
            c1f[0] = c[0] + (self.m / self.lf) * (n[2] * k[2] - n[1] * g[2])
            c1f[1] = c[1] + ((c1f[0] - c[0]) * k[2] - n[2] * self.lf * self.m) / e[2]
            c1f[2] = c[2] + (n[1] * self.lf * self.m + (c1f[0] - c[0]) * g[2]) / e[2]
            c1 = c1f - self.c10

            # c2
            c2f = 2 * c - c1f
            c2 = c2f - self.c20

            # STAGE 3: Calculate servo angles using projection method
            theta = np.zeros(6)

            # theta_a1
            a1s = self.nac * np.dot(a1, self.nac)
            mag_a1s = np.linalg.norm(a1s)
            a1_proj = a1 - a1s
            mag_a1_proj = np.linalg.norm(a1_proj)
            mag_p2a1 = np.sqrt(self.p2 ** 2 - mag_a1s ** 2)
            theta[0] = np.arccos(-np.dot(a1_proj, self.ac) / (2 * self.d2 * mag_a1_proj))
            theta[0] = (theta[0] - np.arccos(
                (mag_a1_proj ** 2 + self.p1 ** 2 - mag_p2a1 ** 2) / (2 * mag_a1_proj * self.p1))) * self.to_deg

            # theta_a2
            a2s = self.nab * np.dot(a2, self.nab)
            mag_a2s = np.linalg.norm(a2s)
            a2_proj = a2 - a2s
            mag_a2_proj = np.linalg.norm(a2_proj)
            mag_p2a2 = np.sqrt(self.p2 ** 2 - mag_a2s ** 2)
            theta[1] = np.arccos(-np.dot(a2_proj, self.ab) / (2 * self.d2 * mag_a2_proj))
            theta[1] = (theta[1] - np.arccos(
                (mag_a2_proj ** 2 + self.p1 ** 2 - mag_p2a2 ** 2) / (2 * mag_a2_proj * self.p1))) * self.to_deg

            # theta_b1
            b1s = self.nab * np.dot(b1, self.nab)
            mag_b1s = np.linalg.norm(b1s)
            b1_proj = b1 - b1s
            mag_b1_proj = np.linalg.norm(b1_proj)
            mag_p2b1 = np.sqrt(self.p2 ** 2 - mag_b1s ** 2)
            theta[2] = np.arccos(np.dot(b1_proj, self.ab) / (2 * self.d2 * mag_b1_proj))
            theta[2] = (theta[2] - np.arccos(
                (mag_b1_proj ** 2 + self.p1 ** 2 - mag_p2b1 ** 2) / (2 * mag_b1_proj * self.p1))) * self.to_deg

            # theta_b2
            b2s = self.nbc * np.dot(b2, self.nbc)
            mag_b2s = np.linalg.norm(b2s)
            b2_proj = b2 - b2s
            mag_b2_proj = np.linalg.norm(b2_proj)
            mag_p2b2 = np.sqrt(self.p2 ** 2 - mag_b2s ** 2)
            theta[3] = np.arccos(-np.dot(b2_proj, self.bc) / (2 * self.d2 * mag_b2_proj))
            theta[3] = (theta[3] - np.arccos(
                (mag_b2_proj ** 2 + self.p1 ** 2 - mag_p2b2 ** 2) / (2 * mag_b2_proj * self.p1))) * self.to_deg

            # theta_c1
            c1s = self.nbc * np.dot(c1, self.nbc)
            mag_c1s = np.linalg.norm(c1s)
            c1_proj = c1 - c1s
            mag_c1_proj = np.linalg.norm(c1_proj)
            mag_p2c1 = np.sqrt(self.p2 ** 2 - mag_c1s ** 2)
            theta[4] = np.arccos(np.dot(c1_proj, self.bc) / (2 * self.d2 * mag_c1_proj))
            theta[4] = (theta[4] - np.arccos(
                (mag_c1_proj ** 2 + self.p1 ** 2 - mag_p2c1 ** 2) / (2 * mag_c1_proj * self.p1))) * self.to_deg

            # theta_c2
            c2s = self.nac * np.dot(c2, self.nac)
            mag_c2s = np.linalg.norm(c2s)
            c2_proj = c2 - c2s
            mag_c2_proj = np.linalg.norm(c2_proj)
            mag_p2c2 = np.sqrt(self.p2 ** 2 - mag_c2s ** 2)
            theta[5] = np.arccos(np.dot(c2_proj, self.ac) / (2 * self.d2 * mag_c2_proj))
            theta[5] = (theta[5] - np.arccos(
                (mag_c2_proj ** 2 + self.p1 ** 2 - mag_p2c2 ** 2) / (2 * mag_c2_proj * self.p1))) * self.to_deg

            # Validate angles
            for i in range(6):
                if abs(theta[i]) > 40:
                    print(f"ERROR: Angle {i} exceeds range: {theta[i]:.2f}°")
                    return None
                if np.isnan(theta[i]):
                    print(f"ERROR: Angle {i} is NaN")
                    return None

            return theta

        except Exception as e:
            print(f"IK Calculation Error: {e}")
            return None


class StewartControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stewart Platform Control")
        self.root.geometry("500x600")

        # IK Calculator
        self.ik = StewartPlatformIK()

        # Serial connection
        self.serial_conn = None
        self.is_connected = False

        # Current DOF values
        self.dof_values = {
            'hx': 0.0,
            'hy': 0.0,
            'hz': 118.19,
            'nx': 0.0,
            'ny': 0.0,
            'ax': 0.0
        }

        # DOF configuration: (min, max, resolution, default, label)
        self.dof_config = {
            'hx': (-50.0, 50.0, 0.1, 0.0, "X Position (mm)"),
            'hy': (-50.0, 50.0, 0.1, 0.0, "Y Position (mm)"),
            'hz': (100.0, 140.0, 0.1, 118.19, "Z Height (mm)"),
            'nx': (-0.25, 0.25, 0.01, 0.0, "X Tilt (nx)"),
            'ny': (-0.25, 0.25, 0.01, 0.0, "Y Tilt (ny)"),
            'ax': (-30.0, 30.0, 0.1, 0.0, "Rotation (ax)")
        }

        self.sliders = {}
        self.value_labels = {}

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

        # Sliders frame
        sliders_frame = ttk.LabelFrame(self.root, text="Manual Control (6 DOF)", padding=10)
        sliders_frame.pack(fill='both', expand=True, padx=10, pady=5)

        for idx, (dof, (min_val, max_val, res, default, label)) in enumerate(self.dof_config.items()):
            ttk.Label(sliders_frame, text=label).grid(row=idx, column=0, sticky='w', pady=5)

            slider = ttk.Scale(
                sliders_frame,
                from_=min_val,
                to=max_val,
                orient='horizontal',
                command=lambda val, d=dof: self.on_slider_change(d, val)
            )
            slider.set(default)
            slider.grid(row=idx, column=1, sticky='ew', padx=10, pady=5)
            self.sliders[dof] = slider

            value_label = ttk.Label(sliders_frame, text=f"{default:.2f}", width=8)
            value_label.grid(row=idx, column=2, pady=5)
            self.value_labels[dof] = value_label

        sliders_frame.columnconfigure(1, weight=1)

        # Calculated angles display
        angles_frame = ttk.LabelFrame(self.root, text="Calculated Servo Angles", padding=10)
        angles_frame.pack(fill='x', padx=10, pady=5)

        self.angle_labels = []
        for i in range(6):
            label = ttk.Label(angles_frame, text=f"θ{i}: 0.00°", font=('Courier', 9))
            label.grid(row=i // 3, column=i % 3, padx=10, pady=2, sticky='w')
            self.angle_labels.append(label)

        # Control buttons
        btn_frame = ttk.Frame(self.root, padding=10)
        btn_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(btn_frame, text="Home Position", command=self.home_position).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Reset All", command=self.reset_sliders).pack(side='left', padx=5)

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
            self.status_label.config(text="No port selected", foreground="red")
            return

        try:
            self.serial_conn = serial.Serial(port, 115200, timeout=1)
            time.sleep(2)
            self.is_connected = True
            self.connect_btn.config(text="Disconnect")
            self.status_label.config(text=f"Connected to {port}", foreground="green")

            # Send initial position
            self.calculate_and_send()

        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", foreground="red")

    def disconnect(self):
        if self.serial_conn:
            self.serial_conn.close()
            self.serial_conn = None
        self.is_connected = False
        self.connect_btn.config(text="Connect")
        self.status_label.config(text="Disconnected", foreground="red")

    def on_slider_change(self, dof, value):
        val = float(value)
        self.dof_values[dof] = val
        self.value_labels[dof].config(text=f"{val:.2f}")

        # Calculate and send
        self.calculate_and_send()

    def calculate_and_send(self):
        # Calculate IK
        angles = self.ik.calculate_angles(
            self.dof_values['hx'],
            self.dof_values['hy'],
            self.dof_values['hz'],
            self.dof_values['nx'],
            self.dof_values['ny'],
            self.dof_values['ax']
        )

        if angles is not None:
            # Update angle display
            for i in range(6):
                self.angle_labels[i].config(text=f"θ{i}: {angles[i]:6.2f}°")

            # Send to Teensy if connected
            if self.is_connected and self.serial_conn:
                self.send_angles(angles)
        else:
            # Clear angle display on error
            for i in range(6):
                self.angle_labels[i].config(text=f"θ{i}: ERROR")

    def send_angles(self, angles):
        try:
            command = ",".join([f"{angle:.3f}" for angle in angles]) + "\n"
            self.serial_conn.write(command.encode())
        except Exception as e:
            self.status_label.config(text=f"Send error: {str(e)}", foreground="red")

    def home_position(self):
        self.sliders['hx'].set(0.0)
        self.sliders['hy'].set(0.0)
        self.sliders['hz'].set(118.19)
        self.sliders['nx'].set(0.0)
        self.sliders['ny'].set(0.0)
        self.sliders['ax'].set(0.0)

    def reset_sliders(self):
        for dof, (_, _, _, default, _) in self.dof_config.items():
            self.sliders[dof].set(default)

    def cleanup(self):
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