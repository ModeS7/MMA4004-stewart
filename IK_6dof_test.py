#!/usr/bin/env python3
"""
Stewart Platform Control GUI with Clean IK Implementation
Based on Robert Eisele's proven method

Optimized for Teensy 4.1:
- USB operates at 480 Mbit/sec (baud rate parameter ignored)
- No auto-reset on serial connection
- Fast update rate (20ms) for responsive control

Requirements: pip install pyserial numpy
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import serial
import serial.tools.list_ports
import numpy as np
import time


class StewartPlatformIK:
    """Inverse kinematics using Robert Eisele's method."""

    def __init__(self):
        # Physical dimensions (mm)
        self.horn_length = 31.75
        self.rod_length = 145.0
        self.base = 73.025
        self.base_anchors = 36.8893
        self.platform = 67.775
        self.platform_anchors = 12.7

        # Distance from anchor center to top surface
        self.top_surface_offset = 26.0  # mm above anchor center

        base_angels = np.array([-np.pi / 2, np.pi / 6, np.pi * 5 / 6])
        platform_angels = np.array([-np.pi * 5 / 6, -np.pi / 6, np.pi / 2])

        self.base_anchors = self.claculate_home_coordinates(self.base, self.base_anchors, base_angels)
        platform_anchors_out_of_fase = self.claculate_home_coordinates(self.platform, self.platform_anchors,
                                                                       platform_angels)
        self.platform_anchors = np.roll(platform_anchors_out_of_fase, shift=-1, axis=0)

        print("Base anchors:\n", self.base_anchors)
        print("Platform anchors:\n", self.platform_anchors)

        # Calculate beta angles
        self.beta_angles = self._calculate_beta_angles()

        # Calculate home height from actual geometry (anchor center)
        base_pos = self.base_anchors[0]
        platform_pos = self.platform_anchors[0]

        horn_end_x = base_pos[0] + self.horn_length * np.cos(self.beta_angles[0])
        horn_end_y = base_pos[1] + self.horn_length * np.sin(self.beta_angles[0])

        dx = platform_pos[0] - horn_end_x
        dy = platform_pos[1] - horn_end_y
        horiz_dist_sq = dx ** 2 + dy ** 2

        self.home_height = np.sqrt(self.rod_length ** 2 - horiz_dist_sq)

        # Home height of the TOP SURFACE (what user interacts with when offset enabled)
        self.home_height_top_surface = self.home_height + self.top_surface_offset

        print(f"Calculated anchor center home height: {self.home_height:.2f}mm")
        print(f"Calculated top surface home height: {self.home_height_top_surface:.2f}mm")
        print(f"Horizontal distance (servo 0): {np.sqrt(horiz_dist_sq):.2f}mm")

    def claculate_home_coordinates(self, l, d, phi):
        """Calculate home coordinates for base or platform anchors."""
        angels = np.array([-np.pi / 2, np.pi / 2])
        xy = np.zeros((6, 3))
        for i in range(len(phi)):
            for j in range(len(angels)):
                x = l * np.cos(phi[i]) + d * np.cos(phi[i] + angels[j])
                y = l * np.sin(phi[i]) + d * np.sin(phi[i] + angels[j])
                xy[i * 2 + j] = np.array([x, y, 0])
        return xy

    def _calculate_beta_angles(self):
        """Calculate beta angles (servo horn orientations)."""
        beta_angles = np.zeros(6)

        # Pair 0,1 at back (camera side) - point toward each other along X
        beta_angles[0] = 0
        beta_angles[1] = np.pi

        # Pair 2,3 on right side
        dx_23 = self.base_anchors[3, 0] - self.base_anchors[2, 0]
        dy_23 = self.base_anchors[3, 1] - self.base_anchors[2, 1]
        angle_23 = np.arctan2(dy_23, dx_23)
        beta_angles[2] = angle_23
        beta_angles[3] = angle_23 + np.pi

        # Pair 4,5 on left side
        dx_54 = self.base_anchors[4, 0] - self.base_anchors[5, 0]
        dy_54 = self.base_anchors[4, 1] - self.base_anchors[5, 1]
        angle_54 = np.arctan2(dy_54, dx_54)
        beta_angles[5] = angle_54
        beta_angles[4] = angle_54 + np.pi

        return beta_angles

    def calculate_servo_angles(self, translation: np.ndarray, rotation: np.ndarray,
                               use_top_surface_offset: bool = True):
        """Calculate servo angles for desired pose.

        Args:
            translation: Desired position [x, y, z] in mm
            rotation: Desired orientation [rx, ry, rz] in degrees
            use_top_surface_offset: If True, translation is top surface position.
                                   If False, translation is anchor center position.

        Returns:
            Array of 6 servo angles in degrees, or None if unreachable
        """
        quat = self._euler_to_quaternion(np.radians(rotation))

        # Conditionally apply top surface offset
        if use_top_surface_offset:
            # The anchor center is offset below the top surface in platform's local frame
            offset_platform_frame = np.array([0, 0, -self.top_surface_offset])

            # Transform this offset to world coordinates using the current rotation
            offset_world_frame = self._rotate_vector(offset_platform_frame, quat)

            # Calculate where the anchor center needs to be
            anchor_center_translation = translation + offset_world_frame
        else:
            # Use translation directly as anchor center position
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
        """Convert Euler angles to quaternion [w, x, y, z]."""
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
        """Rotate vector by quaternion."""
        w, x, y, z = q
        vx, vy, vz = v

        return np.array([
            vx * (w * w + x * x - y * y - z * z) + vy * (2 * x * y - 2 * w * z) + vz * (2 * x * z + 2 * w * y),
            vx * (2 * x * y + 2 * w * z) + vy * (w * w - x * x + y * y - z * z) + vz * (2 * y * z - 2 * w * x),
            vx * (2 * x * z - 2 * w * y) + vy * (2 * y * z + 2 * w * x) + vz * (w * w - x * x - y * y + z * z)
        ])


class StewartControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stewart Platform Control")
        self.root.geometry("600x700")

        self.ik = StewartPlatformIK()
        self.serial_conn = None
        self.is_connected = False

        # Toggle for top surface offset
        self.use_top_surface_offset = tk.BooleanVar(value=True)

        # DOF values - Z represents TOP SURFACE position by default
        self.dof_values = {
            'x': 0.0, 'y': 0.0, 'z': self.ik.home_height_top_surface,
            'rx': 0.0, 'ry': 0.0, 'rz': 0.0
        }

        # DOF configuration (min, max, resolution, default, label)
        self.dof_config = {
            'x': (-30.0, 30.0, 0.1, 0.0, "X Position (mm) - Right+"),
            'y': (-30.0, 30.0, 0.1, 0.0, "Y Position (mm) - Away+"),
            'z': (self.ik.home_height_top_surface - 30,
                  self.ik.home_height_top_surface + 30,
                  0.1,
                  self.ik.home_height_top_surface,
                  f"Z Height (mm) - Top Surface [Home: {self.ik.home_height_top_surface:.1f}]"),
            'rx': (-15.0, 15.0, 0.1, 0.0, "Rotation X (°) - Roll"),
            'ry': (-15.0, 15.0, 0.1, 0.0, "Rotation Y (°) - Pitch"),
            'rz': (-15.0, 15.0, 0.1, 0.0, "Rotation Z (°) - Yaw")
        }

        self.sliders = {}
        self.value_labels = {}
        self.dof_labels = {}
        self.update_timer = None
        self.update_delay_ms = 30

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

        # Configuration frame
        config_frame = ttk.LabelFrame(self.root, text="Configuration", padding=10)
        config_frame.pack(fill='x', padx=10, pady=5)

        offset_checkbox = ttk.Checkbutton(
            config_frame,
            text=f"Use Top Surface Offset (+{self.ik.top_surface_offset}mm)",
            variable=self.use_top_surface_offset,
            command=self.on_offset_toggle
        )
        offset_checkbox.pack(anchor='w')

        # Sliders frame
        sliders_frame = ttk.LabelFrame(self.root, text="Manual Control (6 DOF)", padding=10)
        sliders_frame.pack(fill='both', expand=True, padx=10, pady=5)

        for idx, (dof, (min_val, max_val, res, default, label)) in enumerate(self.dof_config.items()):
            dof_label = ttk.Label(sliders_frame, text=label)
            dof_label.grid(row=idx, column=0, sticky='w', pady=5)
            self.dof_labels[dof] = dof_label

            slider = ttk.Scale(
                sliders_frame, from_=min_val, to=max_val, orient='horizontal',
                command=lambda val, d=dof: self.on_slider_change(d, val)
            )
            slider.grid(row=idx, column=1, sticky='ew', padx=10, pady=5)
            self.sliders[dof] = slider

            # Create value label BEFORE setting slider value
            value_label = ttk.Label(sliders_frame, text=f"{default:.2f}", width=10)
            value_label.grid(row=idx, column=2, pady=5)
            self.value_labels[dof] = value_label

            # Now set the slider value (this triggers the command callback)
            slider.set(default)

        sliders_frame.columnconfigure(1, weight=1)

        # Angles display
        angles_frame = ttk.LabelFrame(self.root, text="Calculated Servo Angles", padding=10)
        angles_frame.pack(fill='x', padx=10, pady=5)

        self.angle_labels = []
        servo_names = ['a1', 'a2', 'b1', 'b2', 'c1', 'c2']
        for i in range(6):
            label = ttk.Label(angles_frame, text=f"θ{servo_names[i]}: 0.00°",
                              font=('Courier', 9))
            label.grid(row=i // 3, column=i % 3, padx=10, pady=2, sticky='w')
            self.angle_labels.append(label)

        # Debug log
        log_frame = ttk.LabelFrame(self.root, text="Debug Log", padding=10)
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, font=('Courier', 8))
        self.log_text.pack(fill='both', expand=True)

        # Control buttons
        btn_frame = ttk.Frame(self.root, padding=10)
        btn_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(btn_frame, text="Home Position", command=self.home_position).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Clear Log", command=lambda: self.log_text.delete(1.0, tk.END)).pack(side='left',
                                                                                                        padx=5)

    def on_offset_toggle(self):
        """Handle toggle of top surface offset."""
        enabled = self.use_top_surface_offset.get()

        if enabled:
            # Switch to top surface mode
            home_z = self.ik.home_height_top_surface
            label_text = f"Z Height (mm) - Top Surface [Home: {home_z:.1f}]"
            self.log(f"Top surface offset ENABLED (+{self.ik.top_surface_offset}mm)")
        else:
            # Switch to anchor center mode
            home_z = self.ik.home_height
            label_text = f"Z Height (mm) - Anchor Center [Home: {home_z:.1f}]"
            self.log(f"Top surface offset DISABLED (anchor center)")

        # Update Z label
        self.dof_labels['z'].config(text=label_text)

        # Update Z slider range and default
        self.sliders['z'].config(from_=home_z - 30, to=home_z + 30)

        # Update home position value
        self.dof_values['z'] = home_z
        self.sliders['z'].set(home_z)
        self.value_labels['z'].config(text=f"{home_z:.2f}")

        # Recalculate with new mode
        self.calculate_and_send()

    def log(self, message):
        """Add message to debug log."""
        self.log_text.insert(tk.END, f"{message}\n")
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
            self.status_label.config(text="No port selected", foreground="red")
            return

        try:
            self.serial_conn = serial.Serial(port, 115200, timeout=1)
            time.sleep(0.1)
            self.is_connected = True
            self.connect_btn.config(text="Disconnect")
            self.status_label.config(text=f"Connected to {port}", foreground="green")
            self.log(f"Connected to {port}")
            self.calculate_and_send()
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", foreground="red")
            self.log(f"Connection error: {e}")

    def disconnect(self):
        if self.serial_conn:
            self.serial_conn.close()
            self.serial_conn = None
        self.is_connected = False
        self.connect_btn.config(text="Connect")
        self.status_label.config(text="Disconnected", foreground="red")
        self.log("Disconnected")

    def on_slider_change(self, dof, value):
        val = float(value)
        self.dof_values[dof] = val
        self.value_labels[dof].config(text=f"{val:.2f}")

        if self.update_timer is not None:
            self.root.after_cancel(self.update_timer)
        self.update_timer = self.root.after(self.update_delay_ms, self.calculate_and_send)

    def calculate_and_send(self):
        translation = np.array([
            self.dof_values['x'],
            self.dof_values['y'],
            self.dof_values['z']
        ])
        rotation = np.array([
            self.dof_values['rx'],
            self.dof_values['ry'],
            self.dof_values['rz']
        ])

        # Use toggle state to determine offset behavior
        angles = self.ik.calculate_servo_angles(
            translation,
            rotation,
            use_top_surface_offset=self.use_top_surface_offset.get()
        )

        servo_names = ['a1', 'a2', 'b1', 'b2', 'c1', 'c2']

        if angles is not None:
            for i in range(6):
                self.angle_labels[i].config(text=f"θ{servo_names[i]}: {angles[i]:6.2f}°")

            if self.is_connected and self.serial_conn:
                self.send_angles(angles)
        else:
            for i in range(6):
                self.angle_labels[i].config(text=f"θ{servo_names[i]}: ERROR")
            self.log("ERROR: IK calculation failed (unreachable position)")

    def send_angles(self, angles):
        try:
            command = ",".join([f"{angle:.3f}" for angle in angles]) + "\n"
            self.serial_conn.write(command.encode())
            self.status_label.config(foreground="blue")
            self.root.after(100, lambda: self.status_label.config(foreground="green") if self.is_connected else None)
        except Exception as e:
            self.status_label.config(text=f"Send error: {str(e)}", foreground="red")
            self.log(f"Send error: {e}")

    def home_position(self):
        for dof, (_, _, _, default, _) in self.dof_config.items():
            if dof == 'z':
                # Use correct home height based on offset toggle
                home_z = self.ik.home_height_top_surface if self.use_top_surface_offset.get() else self.ik.home_height
                self.sliders[dof].set(home_z)
            else:
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