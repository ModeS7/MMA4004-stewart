#!/usr/bin/env python3
"""
Stewart Platform Control GUI with Clean IK Implementation
Based on Robert Eisele's proven method
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import serial
import serial.tools.list_ports
import numpy as np
import time


class StewartPlatformIK:
    """Inverse kinematics using Robert Eisele's method."""

    def __init__(self, horn_length=31.75, rod_length=145.0, base=73.025,
                 base_anchors=36.8893, platform=67.775, platform_anchors=12.7,
                 top_surface_offset=26.0):
        # Physical dimensions (mm)
        self.horn_length = horn_length
        self.rod_length = rod_length
        self.base = base
        self.base_anchors = base_anchors
        self.platform = platform
        self.platform_anchors = platform_anchors

        # Distance from anchor center to top surface
        self.top_surface_offset = top_surface_offset

        base_angels = np.array([-np.pi / 2, np.pi / 6, np.pi * 5 / 6])
        platform_angels = np.array([-np.pi * 5 / 6, -np.pi / 6, np.pi / 2])

        self.base_anchors = self.claculate_home_coordinates(self.base, self.base_anchors, base_angels)
        platform_anchors_out_of_fase = self.claculate_home_coordinates(self.platform, self.platform_anchors,
                                                                       platform_angels)
        self.platform_anchors = np.roll(platform_anchors_out_of_fase, shift=-1, axis=0)

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

    def update_offset(self, new_offset):
        """Update top surface offset and recalculate home height."""
        self.top_surface_offset = new_offset
        self.home_height_top_surface = self.home_height + self.top_surface_offset

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


class PlatformParametersDialog:
    """Dialog for editing platform physical parameters."""

    def __init__(self, parent, current_params):
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Platform Parameters")
        self.dialog.geometry("400x350")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Parameter definitions (name, current_value, description)
        self.params = [
            ("horn_length", current_params.get("horn_length", 31.75), "Servo Horn Length (mm)"),
            ("rod_length", current_params.get("rod_length", 145.0), "Push Rod Length (mm)"),
            ("base", current_params.get("base", 73.025), "Base Radius (mm)"),
            ("base_anchors", current_params.get("base_anchors", 36.8893), "Base Anchor Offset (mm)"),
            ("platform", current_params.get("platform", 67.775), "Platform Radius (mm)"),
            ("platform_anchors", current_params.get("platform_anchors", 12.7), "Platform Anchor Offset (mm)"),
        ]

        self.entries = {}
        self.create_widgets()

        # Center dialog on parent
        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (self.dialog.winfo_width() // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill='both', expand=True)

        ttk.Label(main_frame, text="Platform Physical Parameters",
                  font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 15))

        # Create entry fields for each parameter
        for idx, (param_name, current_value, description) in enumerate(self.params, start=1):
            ttk.Label(main_frame, text=description).grid(row=idx, column=0, sticky='w', pady=5)

            entry = ttk.Entry(main_frame, width=15)
            entry.insert(0, str(current_value))
            entry.grid(row=idx, column=1, sticky='ew', pady=5, padx=(10, 0))

            self.entries[param_name] = entry

        main_frame.columnconfigure(1, weight=1)

        # Buttons frame
        btn_frame = ttk.Frame(self.dialog, padding=(20, 0, 20, 20))
        btn_frame.pack(fill='x')

        ttk.Button(btn_frame, text="Apply", command=self.on_apply).pack(side='right', padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.on_cancel).pack(side='right')
        ttk.Button(btn_frame, text="Reset to Defaults", command=self.on_reset).pack(side='left')

        # Bind Enter key to apply
        self.dialog.bind('<Return>', lambda e: self.on_apply())
        self.dialog.bind('<Escape>', lambda e: self.on_cancel())

    def on_reset(self):
        """Reset all values to defaults."""
        defaults = {
            "horn_length": 31.75,
            "rod_length": 145.0,
            "base": 73.025,
            "base_anchors": 36.8893,
            "platform": 67.775,
            "platform_anchors": 12.7,
        }

        for param_name, entry in self.entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, str(defaults[param_name]))

    def on_apply(self):
        """Validate and apply the parameters."""
        try:
            self.result = {}
            for param_name, entry in self.entries.items():
                value = float(entry.get())
                if value <= 0:
                    raise ValueError(f"{param_name} must be positive")
                self.result[param_name] = value

            self.dialog.destroy()
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Error: {str(e)}\n\nPlease enter valid positive numbers.",
                                 parent=self.dialog)

    def on_cancel(self):
        """Cancel without applying changes."""
        self.result = None
        self.dialog.destroy()

    def show(self):
        """Show dialog and wait for result."""
        self.dialog.wait_window()
        return self.result


class StewartControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stewart Platform Control")
        self.root.geometry("600x750")

        # Initialize with default parameters
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

        # Offset enable checkbox
        offset_checkbox = ttk.Checkbutton(
            config_frame,
            text=f"Use Top Surface Offset",
            variable=self.use_top_surface_offset,
            command=self.on_offset_toggle
        )
        offset_checkbox.pack(anchor='w', pady=2)

        # Offset slider
        offset_slider_frame = ttk.Frame(config_frame)
        offset_slider_frame.pack(fill='x', pady=5)

        self.offset_label = ttk.Label(offset_slider_frame, text="Top Surface Offset (mm):")
        self.offset_label.grid(row=0, column=0, sticky='w', padx=(0, 10))

        self.offset_slider = ttk.Scale(
            offset_slider_frame,
            from_=0.0,
            to=50.0,
            orient='horizontal',
            command=self.on_offset_slider_change
        )
        self.offset_slider.set(self.ik.top_surface_offset)
        self.offset_slider.grid(row=0, column=1, sticky='ew', padx=5)

        self.offset_value_label = ttk.Label(offset_slider_frame, text=f"{self.ik.top_surface_offset:.1f}", width=8)
        self.offset_value_label.grid(row=0, column=2, padx=5)

        offset_slider_frame.columnconfigure(1, weight=1)

        ttk.Button(config_frame, text="Edit Platform Parameters",
                   command=self.open_parameters_dialog).pack(anchor='w', pady=2)

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

    def on_offset_slider_change(self, value):
        """Handle offset slider changes."""
        new_offset = float(value)
        self.offset_value_label.config(text=f"{new_offset:.1f}")

        # Update IK offset
        self.ik.update_offset(new_offset)
        self.platform_params["top_surface_offset"] = new_offset

        # Update Z slider if in top surface mode
        if self.use_top_surface_offset.get():
            home_z = self.ik.home_height_top_surface
            self.dof_labels['z'].config(text=f"Z Height (mm) - Top Surface [Home: {home_z:.1f}]")
            self.sliders['z'].config(from_=home_z - 30, to=home_z + 30)

            # Maintain current Z position relative to new home
            current_relative_z = self.dof_values['z'] - (self.ik.home_height + new_offset - self.ik.top_surface_offset)
            new_z = home_z + current_relative_z
            self.dof_values['z'] = new_z
            self.sliders['z'].set(new_z)
            self.value_labels['z'].config(text=f"{new_z:.2f}")

        self.log(f"Top surface offset updated to {new_offset:.1f}mm")
        self.calculate_and_send()

    def open_parameters_dialog(self):
        """Open dialog to edit platform parameters."""
        dialog = PlatformParametersDialog(self.root, self.platform_params)
        new_params = dialog.show()

        if new_params:
            self.apply_new_parameters(new_params)

    def apply_new_parameters(self, new_params):
        """Apply new platform parameters and reinitialize IK."""
        try:
            # Preserve current offset
            new_params["top_surface_offset"] = self.platform_params["top_surface_offset"]

            # Store new parameters
            self.platform_params = new_params

            # Reinitialize IK with new parameters
            self.ik = StewartPlatformIK(**new_params)

            # Update UI elements that depend on home height
            self.update_z_slider_for_new_params()

            self.log("Platform parameters updated successfully")
            self.log(
                f"New home heights - Anchor: {self.ik.home_height:.2f}mm, Top: {self.ik.home_height_top_surface:.2f}mm")

            # Recalculate with new parameters
            self.calculate_and_send()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply parameters:\n{str(e)}")
            self.log(f"ERROR applying parameters: {e}")

    def update_z_slider_for_new_params(self):
        """Update Z slider range and position after parameter changes."""
        home_z = self.ik.home_height_top_surface if self.use_top_surface_offset.get() else self.ik.home_height

        # Update Z slider configuration
        self.sliders['z'].config(from_=home_z - 30, to=home_z + 30)

        # Update Z label
        if self.use_top_surface_offset.get():
            label_text = f"Z Height (mm) - Top Surface [Home: {home_z:.1f}]"
        else:
            label_text = f"Z Height (mm) - Anchor Center [Home: {home_z:.1f}]"

        self.dof_labels['z'].config(text=label_text)

        # Move to new home height
        self.dof_values['z'] = home_z
        self.sliders['z'].set(home_z)
        self.value_labels['z'].config(text=f"{home_z:.2f}")

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