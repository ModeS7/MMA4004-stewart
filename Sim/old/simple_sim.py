#!/usr/bin/env python3
"""
Basic Stewart Platform Simulator

Demonstrates servo dynamics and forward kinematics without ball physics.
Uses shared components from stewart_platform_core.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import time

from core import FirstOrderServo, StewartPlatformIK


class StewartSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stewart Platform Simulator with Servo Dynamics")
        self.root.geometry("700x900")

        # Initialize platform
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

        # Initialize 6 simulated servos
        self.servos = [FirstOrderServo(K=1.0, tau=0.1, delay=0.35) for _ in range(6)]

        # Simulation state
        self.simulation_running = False
        self.simulation_time = 0.0
        self.last_update_time = None
        self.update_rate_ms = 20  # 50 Hz update rate

        # UI state
        self.use_top_surface_offset = tk.BooleanVar(value=True)
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
        self.update_timer = None

        self.create_widgets()

    def create_widgets(self):
        # Simulation control
        sim_frame = ttk.LabelFrame(self.root, text="Simulation Control", padding=10)
        sim_frame.pack(fill='x', padx=10, pady=5)

        self.start_btn = ttk.Button(sim_frame, text="Start Simulation",
                                    command=self.start_simulation)
        self.start_btn.pack(side='left', padx=5)

        self.stop_btn = ttk.Button(sim_frame, text="Stop Simulation",
                                   command=self.stop_simulation, state='disabled')
        self.stop_btn.pack(side='left', padx=5)

        self.reset_btn = ttk.Button(sim_frame, text="Reset", command=self.reset_simulation)
        self.reset_btn.pack(side='left', padx=5)

        self.sim_time_label = ttk.Label(sim_frame, text="Time: 0.00s")
        self.sim_time_label.pack(side='right', padx=10)

        # Configuration
        config_frame = ttk.LabelFrame(self.root, text="Configuration", padding=10)
        config_frame.pack(fill='x', padx=10, pady=5)

        ttk.Checkbutton(config_frame, text="Use Top Surface Offset",
                        variable=self.use_top_surface_offset,
                        command=self.on_offset_toggle).pack(anchor='w')

        # Sliders for DOF control
        sliders_frame = ttk.LabelFrame(self.root, text="Commanded Pose (6 DOF)", padding=10)
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

        # Commanded angles display
        cmd_angles_frame = ttk.LabelFrame(self.root, text="Commanded Servo Angles (IK)", padding=10)
        cmd_angles_frame.pack(fill='x', padx=10, pady=5)

        self.cmd_angle_labels = []
        servo_names = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
        for i in range(6):
            label = ttk.Label(cmd_angles_frame, text=f"{servo_names[i]}: 0.00°",
                              font=('Courier', 9))
            label.grid(row=i // 3, column=i % 3, padx=10, pady=2, sticky='w')
            self.cmd_angle_labels.append(label)

        # Actual angles display
        actual_angles_frame = ttk.LabelFrame(self.root, text="Actual Servo Angles (Simulated)", padding=10)
        actual_angles_frame.pack(fill='x', padx=10, pady=5)

        self.actual_angle_labels = []
        for i in range(6):
            label = ttk.Label(actual_angles_frame, text=f"{servo_names[i]}: 0.00°",
                              font=('Courier', 9))
            label.grid(row=i // 3, column=i % 3, padx=10, pady=2, sticky='w')
            self.actual_angle_labels.append(label)

        # Actual pose display (FK result)
        fk_frame = ttk.LabelFrame(self.root, text="Actual Platform Pose (FK)", padding=10)
        fk_frame.pack(fill='x', padx=10, pady=5)

        fk_info_frame = ttk.Frame(fk_frame)
        fk_info_frame.pack(fill='x')

        ttk.Label(fk_info_frame, text="Position:", font=('TkDefaultFont', 9, 'bold')).grid(
            row=0, column=0, sticky='w', padx=5)
        self.fk_pos_label = ttk.Label(fk_info_frame, text="X: 0.00  Y: 0.00  Z: 0.00 mm",
                                      font=('Courier', 9))
        self.fk_pos_label.grid(row=0, column=1, sticky='w', padx=10)

        ttk.Label(fk_info_frame, text="Rotation:", font=('TkDefaultFont', 9, 'bold')).grid(
            row=1, column=0, sticky='w', padx=5)
        self.fk_rot_label = ttk.Label(fk_info_frame, text="Roll: 0.00  Pitch: 0.00  Yaw: 0.00°",
                                      font=('Courier', 9))
        self.fk_rot_label.grid(row=1, column=1, sticky='w', padx=10)

        ttk.Label(fk_info_frame, text="Iterations:", font=('TkDefaultFont', 9, 'bold')).grid(
            row=2, column=0, sticky='w', padx=5)
        self.fk_iter_label = ttk.Label(fk_info_frame, text="0", font=('Courier', 9))
        self.fk_iter_label.grid(row=2, column=1, sticky='w', padx=10)

        # Error display
        error_frame = ttk.LabelFrame(self.root, text="Position Error (Commanded vs Actual)", padding=10)
        error_frame.pack(fill='x', padx=10, pady=5)

        self.error_label = ttk.Label(error_frame, text="ΔPos: 0.00mm  ΔRot: 0.00°",
                                     font=('Courier', 9))
        self.error_label.pack()

        # Log
        log_frame = ttk.LabelFrame(self.root, text="Debug Log", padding=10)
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, font=('Courier', 8))
        self.log_text.pack(fill='both', expand=True)

    def log(self, message):
        """Add message to debug log."""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)

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

        if self.update_timer is not None:
            self.root.after_cancel(self.update_timer)
        self.update_timer = self.root.after(50, self.calculate_ik)

    def calculate_ik(self):
        """Calculate IK and send commands to simulated servos."""
        translation = np.array([self.dof_values['x'], self.dof_values['y'], self.dof_values['z']])
        rotation = np.array([self.dof_values['rx'], self.dof_values['ry'], self.dof_values['rz']])

        angles = self.ik.calculate_servo_angles(
            translation, rotation,
            use_top_surface_offset=self.use_top_surface_offset.get()
        )

        if angles is not None:
            # Update commanded angles display
            for i in range(6):
                self.cmd_angle_labels[i].config(text=f"S{i + 1}: {angles[i]:6.2f}°")

            # Send commands to simulated servos
            if self.simulation_running:
                for i, servo in enumerate(self.servos):
                    servo.send_command(angles[i], self.simulation_time)
        else:
            for i in range(6):
                self.cmd_angle_labels[i].config(text=f"S{i + 1}: ERROR")
            self.log("ERROR: IK failed (unreachable)")

    def start_simulation(self):
        """Start the simulation loop."""
        self.simulation_running = True
        self.last_update_time = time.time()
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.log("Simulation started")
        self.simulation_loop()

    def stop_simulation(self):
        """Stop the simulation loop."""
        self.simulation_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.log("Simulation stopped")

    def reset_simulation(self):
        """Reset simulation to initial state."""
        was_running = self.simulation_running
        if was_running:
            self.stop_simulation()

        # Reset servos
        for servo in self.servos:
            servo.reset()

        # Reset time
        self.simulation_time = 0.0
        self.last_update_time = None
        self.sim_time_label.config(text="Time: 0.00s")

        # Reset sliders to home
        for dof, (_, _, _, default, _) in self.dof_config.items():
            if dof == 'z':
                home_z = (self.ik.home_height_top_surface if self.use_top_surface_offset.get()
                          else self.ik.home_height)
                self.sliders[dof].set(home_z)
            else:
                self.sliders[dof].set(default)

        # Update displays
        self.update_displays()
        self.log("Simulation reset")

        if was_running:
            self.start_simulation()

    def simulation_loop(self):
        """Main simulation loop."""
        if not self.simulation_running:
            return

        current_time = time.time()
        if self.last_update_time is not None:
            dt = current_time - self.last_update_time
            self.simulation_time += dt

            # Update all servos
            for servo in self.servos:
                servo.update(dt, self.simulation_time)

            # Update displays
            self.update_displays()

        self.last_update_time = current_time
        self.sim_time_label.config(text=f"Time: {self.simulation_time:.2f}s")

        # Schedule next update
        self.root.after(self.update_rate_ms, self.simulation_loop)

    def update_displays(self):
        """Update all display elements with current simulation state."""
        # Get current servo angles
        actual_angles = np.array([servo.get_angle() for servo in self.servos])

        # Update actual angles display
        for i in range(6):
            self.actual_angle_labels[i].config(text=f"S{i + 1}: {actual_angles[i]:6.2f}°")

        # Calculate FK from actual angles
        translation, rotation, success, iterations = self.ik.calculate_forward_kinematics(
            actual_angles,
            use_top_surface_offset=self.use_top_surface_offset.get()
        )

        if success:
            # Update FK display
            self.fk_pos_label.config(
                text=f"X: {translation[0]:6.2f}  Y: {translation[1]:6.2f}  Z: {translation[2]:6.2f} mm"
            )
            self.fk_rot_label.config(
                text=f"Roll: {rotation[0]:6.2f}  Pitch: {rotation[1]:6.2f}  Yaw: {rotation[2]:6.2f}°"
            )
            self.fk_iter_label.config(text=f"{iterations}")

            # Calculate error between commanded and actual pose
            cmd_translation = np.array([self.dof_values['x'], self.dof_values['y'],
                                        self.dof_values['z']])
            cmd_rotation = np.array([self.dof_values['rx'], self.dof_values['ry'],
                                     self.dof_values['rz']])

            pos_error = np.linalg.norm(translation - cmd_translation)
            rot_error = np.linalg.norm(rotation - cmd_rotation)

            self.error_label.config(text=f"ΔPos: {pos_error:.2f}mm  ΔRot: {rot_error:.2f}°")
        else:
            self.fk_pos_label.config(text="FK FAILED")
            self.fk_rot_label.config(text="FK FAILED")
            self.fk_iter_label.config(text="N/A")
            self.error_label.config(text="Cannot calculate error")


def main():
    root = tk.Tk()
    app = StewartSimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()