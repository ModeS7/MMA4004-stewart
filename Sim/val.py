#!/usr/bin/env python3
"""
Real-Time Hardware Validation Simulator

Replays hardware test data with real-time physics simulation.
Allows tuning servo and ball parameters to match hardware behavior.

Usage:
    python realtime_validation.py <csv_file>
"""

import sys
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch

from core.core import StewartPlatformIK, SimpleBallPhysics2D, FirstOrderServo
from core.utils import SimulationConfig


class ValidationSimulator:
    """Real-time hardware validation simulator with parameter tuning."""

    def __init__(self, root, csv_file):
        self.root = root
        self.csv_file = csv_file

        self.root.title(f"Hardware Validation - {csv_file}")
        self.root.geometry("1400x900")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Colors (matching your simulator)
        self.colors = {
            'bg': '#1e1e1e',
            'panel_bg': '#2d2d2d',
            'widget_bg': '#3d3d3d',
            'fg': '#e0e0e0',
            'highlight': '#007acc',
            'button_bg': '#0e639c',
            'button_fg': '#ffffff',
            'entry_bg': '#3d3d3d',
            'border': '#555555',
            'success': '#4ec9b0',
            'warning': '#ce9178'
        }

        self.root.configure(bg=self.colors['bg'])
        self.setup_dark_theme()

        # Load CSV data
        self.df = pd.read_csv(csv_file)
        self.test_runs = self._split_test_runs()

        if len(self.test_runs) == 0:
            messagebox.showerror("Error", "No test runs found in CSV")
            sys.exit(1)

        self.current_run_idx = 0
        self.current_run = None

        # Initialize IK
        self.ik = StewartPlatformIK(
            horn_length=31.75,
            rod_length=145.0,
            base=73.025,
            base_anchors=36.8893,
            platform=67.775,
            platform_anchors=12.7,
            top_surface_offset=26.0
        )

        # Initialize servos
        self.servos = [FirstOrderServo(K=1.0, tau=0.1, delay=0.0) for _ in range(6)]

        # Initialize ball physics
        self.ball_physics = SimpleBallPhysics2D(
            ball_radius=0.02,
            ball_mass=0.0027,
            gravity=9.81,
            rolling_friction=0.0225,
            sphere_type='hollow'
        )

        # Simulation state
        self.simulation_time = 0.0
        self.data_index = 0

        # Ball states
        self.sim_ball_pos = None
        self.sim_ball_vel = None
        self.sim_ball_omega = None
        self.hw_ball_pos = (0.0, 0.0)
        self.ball_active = True  # Track if ball is still on platform

        # Trajectory history
        self.sim_trajectory = {'time': [], 'x': [], 'y': []}
        self.hw_trajectory = {'time': [], 'x': [], 'y': []}

        # Error tracking
        self.error_history = []

        self._build_gui()
        self.load_test_run(0)

    def setup_dark_theme(self):
        """Configure ttk widgets for dark mode."""
        style = ttk.Style()
        style.theme_use('default')

        style.configure('TFrame', background=self.colors['bg'])
        style.configure('TLabelframe',
                        background=self.colors['panel_bg'],
                        foreground=self.colors['fg'],
                        borderwidth=1,
                        relief='solid')
        style.configure('TLabelframe.Label',
                        background=self.colors['panel_bg'],
                        foreground=self.colors['highlight'],
                        font=('Segoe UI', 9, 'bold'))
        style.configure('TLabel',
                        background=self.colors['panel_bg'],
                        foreground=self.colors['fg'],
                        font=('Segoe UI', 9))
        style.configure('TButton',
                        background=self.colors['button_bg'],
                        foreground=self.colors['button_fg'],
                        borderwidth=0,
                        focuscolor='none',
                        font=('Segoe UI', 9))
        style.map('TButton',
                  background=[('active', self.colors['highlight']),
                              ('pressed', '#005a9e')])
        style.configure('TScale',
                        background=self.colors['panel_bg'],
                        troughcolor=self.colors['widget_bg'],
                        borderwidth=0)

    def _split_test_runs(self):
        """Split CSV data into individual test runs."""
        runs = []
        current_run = []
        current_angles = None

        for idx in range(len(self.df)):
            rx = self.df.loc[idx, 'rx_deg']
            ry = self.df.loc[idx, 'ry_deg']
            angles = (rx, ry)

            if current_angles is None:
                current_angles = angles

            if angles != current_angles and len(current_run) > 5:
                runs.append({
                    'data': self.df.iloc[current_run].copy().reset_index(drop=True),
                    'rx': current_angles[0],
                    'ry': current_angles[1]
                })
                current_run = []
                current_angles = angles

            current_run.append(idx)

        if len(current_run) > 5:
            runs.append({
                'data': self.df.iloc[current_run].copy().reset_index(drop=True),
                'rx': current_angles[0],
                'ry': current_angles[1]
            })

        return runs

    def _build_gui(self):
        """Build GUI."""
        main_frame = ttk.Frame(self.root, style='TFrame')
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Left panel - controls
        left_panel = ttk.Frame(main_frame, style='TFrame', width=400)
        left_panel.pack(side='left', fill='both', padx=(0, 5))
        left_panel.pack_propagate(False)

        # Right panel - plot
        right_panel = ttk.Frame(main_frame, style='TFrame')
        right_panel.pack(side='left', fill='both', expand=True)

        self._build_control_panel(left_panel)
        self._build_plot_panel(right_panel)

    def _build_control_panel(self, parent):
        """Build control panel with parameter sliders."""

        # Run selection
        run_frame = ttk.LabelFrame(parent, text="Test Run Selection", padding=10)
        run_frame.pack(fill='x', pady=(0, 10))

        self.run_label = ttk.Label(run_frame, text="", font=('Consolas', 10))
        self.run_label.pack()

        btn_frame = ttk.Frame(run_frame)
        btn_frame.pack(fill='x', pady=(5, 0))

        ttk.Button(btn_frame, text="◀ Prev", command=self.prev_run, width=10).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Next ▶", command=self.next_run, width=10).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="↻ Re-run", command=self.run_simulation, width=10).pack(side='left', padx=5)

        # Servo parameters
        servo_frame = ttk.LabelFrame(parent, text="Servo Dynamics", padding=10)
        servo_frame.pack(fill='x', pady=(0, 10))

        self.servo_sliders = {}
        self.servo_labels = {}

        servo_params = [
            ('tau', 'Time Constant τ (s)', 0.01, 0.5, 0.1),
            ('delay', 'Delay (s)', 0.0, 2.0, 0.0),
        ]

        for param_name, label, min_val, max_val, default in servo_params:
            self._create_slider(servo_frame, param_name, label, min_val, max_val, default,
                                self.servo_sliders, self.servo_labels, self.on_servo_change)

        # Ball physics parameters
        physics_frame = ttk.LabelFrame(parent, text="Ball Physics", padding=10)
        physics_frame.pack(fill='x', pady=(0, 10))

        self.physics_sliders = {}
        self.physics_labels = {}

        physics_params = [
            ('friction', 'Rolling Friction', 0.0, 0.05, 0.0225),
            ('mass', 'Mass (kg)', 0.001, 0.01, 0.0027),
            ('radius', 'Radius (m)', 0.01, 0.08, 0.04),
            ('air_density', 'Air Density (kg/m³)', 0.0, 2.0, 1.225),
        ]

        for param_name, label, min_val, max_val, default in physics_params:
            self._create_slider(physics_frame, param_name, label, min_val, max_val, default,
                                self.physics_sliders, self.physics_labels, self.on_physics_change)

        # Metrics
        metrics_frame = ttk.LabelFrame(parent, text="Error Metrics", padding=10)
        metrics_frame.pack(fill='x', pady=(0, 10))

        self.error_label = ttk.Label(metrics_frame, text="Current Error: 0.0 mm", font=('Consolas', 9))
        self.error_label.pack(anchor='w', pady=2)

        self.mean_error_label = ttk.Label(metrics_frame, text="Mean Error: 0.0 mm", font=('Consolas', 9))
        self.mean_error_label.pack(anchor='w', pady=2)

        self.max_error_label = ttk.Label(metrics_frame, text="Max Error: 0.0 mm", font=('Consolas', 9))
        self.max_error_label.pack(anchor='w', pady=2)

    def _create_slider(self, parent, name, label, min_val, max_val, default,
                       slider_dict, label_dict, callback):
        """Create a parameter slider."""
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=3)

        ttk.Label(frame, text=label, font=('Segoe UI', 9)).grid(row=0, column=0, sticky='w')

        slider = ttk.Scale(frame, from_=min_val, to=max_val, orient='horizontal')
        slider.grid(row=0, column=1, sticky='ew', padx=5)
        slider.set(default)
        slider_dict[name] = slider

        value_label = ttk.Label(frame, text=f"{default:.4f}", width=8, font=('Consolas', 9))
        value_label.grid(row=0, column=2)
        label_dict[name] = value_label

        slider.config(command=lambda val, n=name, l=value_label, cb=callback:
        self._on_slider_change(n, val, l, cb))

        frame.columnconfigure(1, weight=1)

    def _on_slider_change(self, name, value, label, callback):
        """Handle slider change."""
        val = float(value)
        label.config(text=f"{val:.4f}")
        callback(name, val)

    def on_servo_change(self, param, value):
        """Update servo parameters."""
        for servo in self.servos:
            if param == 'tau':
                servo.tau = value
            elif param == 'delay':
                servo.delay = value

    def on_physics_change(self, param, value):
        """Update ball physics parameters."""
        if param == 'friction':
            self.ball_physics.mu_roll = value
        elif param == 'mass':
            self.ball_physics.mass = value
            self.ball_physics.update_sphere_type(self.ball_physics.sphere_type)
        elif param == 'radius':
            self.ball_physics.radius = value
            self.ball_physics.update_sphere_type(self.ball_physics.sphere_type)
        elif param == 'air_density':
            self.ball_physics.set_air_resistance(air_density=value)

    def _build_plot_panel(self, parent):
        """Build plot panel."""
        plot_frame = ttk.LabelFrame(parent, text="Position vs Time", padding=10)
        plot_frame.pack(fill='both', expand=True)

        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(12, 8), facecolor=self.colors['panel_bg'])

        # Create 2 subplots: X vs time, Y vs time
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
        self.ax_x = self.fig.add_subplot(gs[0])
        self.ax_y = self.fig.add_subplot(gs[1])

        for ax in [self.ax_x, self.ax_y]:
            ax.set_facecolor(self.colors['widget_bg'])

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        self.setup_plot()

    def setup_plot(self):
        """Setup matplotlib plots."""
        # === X vs Time Plot ===
        self.ax_x.clear()
        self.ax_x.set_xlabel('Time (s)', color=self.colors['fg'], fontsize=10)
        self.ax_x.set_ylabel('X Position (mm)', color=self.colors['fg'], fontsize=10)
        self.ax_x.set_title('X Coordinate vs Time', color=self.colors['fg'],
                            fontsize=11, fontweight='bold')
        self.ax_x.grid(True, alpha=0.2, linestyle='--', color=self.colors['fg'])
        self.ax_x.tick_params(colors=self.colors['fg'], labelsize=8)

        for spine in self.ax_x.spines.values():
            spine.set_color(self.colors['border'])

        self.sim_x_line, = self.ax_x.plot([], [], color=self.colors['highlight'],
                                          linewidth=2, label='Sim X')
        self.hw_x_line, = self.ax_x.plot([], [], 'o', color='#ff4444',
                                         markersize=3, alpha=0.6, label='HW X')

        legend_x = self.ax_x.legend(loc='upper right', fontsize=8,
                                    facecolor=self.colors['panel_bg'],
                                    edgecolor=self.colors['border'],
                                    labelcolor=self.colors['fg'])
        legend_x.get_frame().set_alpha(0.9)

        # === Y vs Time Plot ===
        self.ax_y.clear()
        self.ax_y.set_xlabel('Time (s)', color=self.colors['fg'], fontsize=10)
        self.ax_y.set_ylabel('Y Position (mm)', color=self.colors['fg'], fontsize=10)
        self.ax_y.set_title('Y Coordinate vs Time', color=self.colors['fg'],
                            fontsize=11, fontweight='bold')
        self.ax_y.grid(True, alpha=0.2, linestyle='--', color=self.colors['fg'])
        self.ax_y.tick_params(colors=self.colors['fg'], labelsize=8)

        for spine in self.ax_y.spines.values():
            spine.set_color(self.colors['border'])

        self.sim_y_line, = self.ax_y.plot([], [], color=self.colors['highlight'],
                                          linewidth=2, label='Sim Y')
        self.hw_y_line, = self.ax_y.plot([], [], 'o', color='#ff4444',
                                         markersize=3, alpha=0.6, label='HW Y')

        legend_y = self.ax_y.legend(loc='upper right', fontsize=8,
                                    facecolor=self.colors['panel_bg'],
                                    edgecolor=self.colors['border'],
                                    labelcolor=self.colors['fg'])
        legend_y.get_frame().set_alpha(0.9)

        self.canvas.draw()

    def load_test_run(self, run_idx):
        """Load a test run."""
        if run_idx < 0 or run_idx >= len(self.test_runs):
            return

        self.current_run_idx = run_idx
        self.current_run = self.test_runs[run_idx]

        self.run_label.config(
            text=f"Run {run_idx + 1}/{len(self.test_runs)}: "
                 f"rx={self.current_run['rx']:+.1f}°, ry={self.current_run['ry']:+.1f}°"
        )

        self.run_simulation()

    def prev_run(self):
        """Load previous test run."""
        if self.current_run_idx > 0:
            self.load_test_run(self.current_run_idx - 1)

    def next_run(self):
        """Load next test run."""
        if self.current_run_idx < len(self.test_runs) - 1:
            self.load_test_run(self.current_run_idx + 1)

    def reset_simulation(self):
        """Reset simulation state."""
        self.simulation_time = 0.0
        self.data_index = 0
        self.error_history = []
        self.ball_active = True

        # Clear trajectory history
        self.sim_trajectory = {'time': [], 'x': [], 'y': []}
        self.hw_trajectory = {'time': [], 'x': [], 'y': []}

        # Reset servos
        for servo in self.servos:
            servo.reset()

        # Initialize ball at detected starting position
        df = self.current_run['data']

        # Find first valid detection
        for idx in range(len(df)):
            if df.loc[idx, 'ball_detected'] and not pd.isna(df.loc[idx, 'ball_x_mm']):
                ball_x_mm = df.loc[idx, 'ball_x_mm']
                ball_y_mm = df.loc[idx, 'ball_y_mm']
                ball_x_m = ball_x_mm / 1000.0
                ball_y_m = ball_y_mm / 1000.0
                ball_z_m = (self.ik.home_height_top_surface / 1000.0) + self.ball_physics.radius

                self.sim_ball_pos = torch.tensor([[ball_x_m, ball_y_m, ball_z_m]], dtype=torch.float32)
                break
        else:
            # Default to center
            ball_z_m = (self.ik.home_height_top_surface / 1000.0) + self.ball_physics.radius
            self.sim_ball_pos = torch.tensor([[0.0, 0.0, ball_z_m]], dtype=torch.float32)

        self.sim_ball_vel = torch.zeros((1, 3), dtype=torch.float32)
        self.sim_ball_omega = torch.zeros((1, 3), dtype=torch.float32)

        self.sim_x_line.set_data([], [])
        self.hw_x_line.set_data([], [])
        self.sim_y_line.set_data([], [])
        self.hw_y_line.set_data([], [])

    def run_simulation(self):
        """Run simulation as fast as possible."""
        self.reset_simulation()

        df = self.current_run['data']

        # Run through all data points as fast as possible
        for idx in range(len(df)):
            self.data_index = idx

            # Get time step
            if idx > 0:
                sim_dt = df.loc[idx, 'time'] - df.loc[idx - 1, 'time']
            else:
                sim_dt = df.loc[idx, 'time']

            self.simulation_time = df.loc[idx, 'time']

            # Get servo angles from CSV
            servo_angles = np.array([
                df.loc[idx, 's0'],
                df.loc[idx, 's1'],
                df.loc[idx, 's2'],
                df.loc[idx, 's3'],
                df.loc[idx, 's4'],
                df.loc[idx, 's5']
            ])

            # Send to servos
            for i, servo in enumerate(self.servos):
                servo.send_command(servo_angles[i], self.simulation_time)

            # Update servos
            for servo in self.servos:
                servo.update(sim_dt, self.simulation_time)

            # Get actual servo angles
            actual_angles = np.array([servo.get_angle() for servo in self.servos])

            # Forward kinematics
            translation, rotation, success, _ = self.ik.calculate_forward_kinematics(
                actual_angles, use_top_surface_offset=True
            )

            if success:
                # Create platform pose
                platform_pose = torch.tensor([[
                    translation[0] / 1000,
                    translation[1] / 1000,
                    translation[2] / 1000,
                    rotation[0],
                    rotation[1],
                    rotation[2]
                ]], dtype=torch.float32)

                # Step physics only if ball is still active
                if self.ball_active:
                    self.sim_ball_pos, self.sim_ball_vel, self.sim_ball_omega, _ = \
                        self.ball_physics.step(
                            self.sim_ball_pos,
                            self.sim_ball_vel,
                            self.sim_ball_omega,
                            platform_pose,
                            sim_dt
                        )

                    # Check if ball has fallen off platform
                    sim_x = self.sim_ball_pos[0, 0].item() * 1000
                    sim_y = self.sim_ball_pos[0, 1].item() * 1000
                    if abs(sim_x) > 100 or abs(sim_y) > 100:
                        self.ball_active = False

                # Store simulation trajectory only if ball is active
                if self.ball_active:
                    sim_x = self.sim_ball_pos[0, 0].item() * 1000
                    sim_y = self.sim_ball_pos[0, 1].item() * 1000
                    self.sim_trajectory['time'].append(self.simulation_time)
                    self.sim_trajectory['x'].append(sim_x)
                    self.sim_trajectory['y'].append(sim_y)

            # Get hardware ball position
            if df.loc[idx, 'ball_detected'] and not pd.isna(df.loc[idx, 'ball_x_mm']):
                hw_x = df.loc[idx, 'ball_x_mm']
                hw_y = df.loc[idx, 'ball_y_mm']
                self.hw_ball_pos = (hw_x, hw_y)

                # Store hardware trajectory
                self.hw_trajectory['time'].append(self.simulation_time)
                self.hw_trajectory['x'].append(hw_x)
                self.hw_trajectory['y'].append(hw_y)

                # Calculate error only if ball is still active
                if self.ball_active:
                    sim_x = self.sim_ball_pos[0, 0].item() * 1000
                    sim_y = self.sim_ball_pos[0, 1].item() * 1000
                    error = np.sqrt((sim_x - hw_x) ** 2 + (sim_y - hw_y) ** 2)
                    self.error_history.append(error)

        # Update GUI and plots once at the end
        self.update_gui()
        self.update_plot()

    def update_gui(self):
        """Update GUI labels."""
        if self.error_history:
            current_error = self.error_history[-1]
            mean_error = np.mean(self.error_history)
            max_error = np.max(self.error_history)

            self.error_label.config(text=f"Current Error: {current_error:.1f} mm")
            self.mean_error_label.config(text=f"Mean Error: {mean_error:.1f} mm")
            self.max_error_label.config(text=f"Max Error: {max_error:.1f} mm")

    def update_plot(self):
        """Update all plots."""
        # === X vs Time ===
        if len(self.sim_trajectory['time']) > 0:
            self.sim_x_line.set_data(self.sim_trajectory['time'], self.sim_trajectory['x'])

        if len(self.hw_trajectory['time']) > 0:
            self.hw_x_line.set_data(self.hw_trajectory['time'], self.hw_trajectory['x'])

        # Auto-scale X plot
        if len(self.sim_trajectory['time']) > 0 or len(self.hw_trajectory['time']) > 0:
            all_times = self.sim_trajectory['time'] + self.hw_trajectory['time']
            all_x = self.sim_trajectory['x'] + self.hw_trajectory['x']
            if all_times and all_x:
                self.ax_x.set_xlim(min(all_times), max(all_times) + 0.1)
                x_margin = (max(all_x) - min(all_x)) * 0.1 + 1
                self.ax_x.set_ylim(min(all_x) - x_margin, max(all_x) + x_margin)

        # === Y vs Time ===
        if len(self.sim_trajectory['time']) > 0:
            self.sim_y_line.set_data(self.sim_trajectory['time'], self.sim_trajectory['y'])

        if len(self.hw_trajectory['time']) > 0:
            self.hw_y_line.set_data(self.hw_trajectory['time'], self.hw_trajectory['y'])

        # Auto-scale Y plot
        if len(self.sim_trajectory['time']) > 0 or len(self.hw_trajectory['time']) > 0:
            all_times = self.sim_trajectory['time'] + self.hw_trajectory['time']
            all_y = self.sim_trajectory['y'] + self.hw_trajectory['y']
            if all_times and all_y:
                self.ax_y.set_xlim(min(all_times), max(all_times) + 0.1)
                y_margin = (max(all_y) - min(all_y)) * 0.1 + 1
                self.ax_y.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)

        self.canvas.draw()

    def on_closing(self):
        """Clean shutdown."""
        self.root.quit()
        self.root.destroy()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python realtime_validation.py <csv_file>")
        print("\nExample:")
        print("  python realtime_validation.py step_response_20250117_143022.csv")
        sys.exit(1)

    csv_file = sys.argv[1]

    try:
        root = tk.Tk()
        app = ValidationSimulator(root, csv_file)
        root.mainloop()

    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found")
        sys.exit(1)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()