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
from matplotlib.patches import Circle, Rectangle
import torch
import time

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
            ball_radius=0.04,
            ball_mass=0.0027,
            gravity=9.81,
            rolling_friction=0.001,
            sphere_type='hollow'
        )

        # Simulation state
        self.simulation_running = False
        self.simulation_time = 0.0
        self.playback_speed = 1.0
        self.data_index = 0
        self.last_update_time = None

        # Ball states
        self.sim_ball_pos = None
        self.sim_ball_vel = None
        self.sim_ball_omega = None
        self.hw_ball_pos = (0.0, 0.0)

        # Error tracking
        self.error_history = []
        self.max_error_history = 100

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

        # Playback controls
        playback_frame = ttk.LabelFrame(parent, text="Playback", padding=10)
        playback_frame.pack(fill='x', pady=(0, 10))

        self.time_label = ttk.Label(playback_frame, text="Time: 0.00s", font=('Consolas', 10, 'bold'))
        self.time_label.pack()

        pb_btn_frame = ttk.Frame(playback_frame)
        pb_btn_frame.pack(fill='x', pady=(5, 0))

        self.play_btn = ttk.Button(pb_btn_frame, text="▶ Play", command=self.start_simulation, width=10)
        self.play_btn.pack(side='left', padx=5)

        self.stop_btn = ttk.Button(pb_btn_frame, text="⏸ Stop", command=self.stop_simulation,
                                   state='disabled', width=10)
        self.stop_btn.pack(side='left', padx=5)

        ttk.Button(pb_btn_frame, text="↻ Reset", command=self.reset_simulation, width=10).pack(side='left', padx=5)

        # Speed control
        speed_frame = ttk.Frame(playback_frame)
        speed_frame.pack(fill='x', pady=(10, 0))

        ttk.Label(speed_frame, text="Speed:", font=('Segoe UI', 9)).pack(side='left')

        self.speed_slider = ttk.Scale(speed_frame, from_=0.1, to=2.0, orient='horizontal',
                                      command=self.on_speed_change)
        self.speed_slider.set(1.0)
        self.speed_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.speed_label = ttk.Label(speed_frame, text="1.0x", width=5, font=('Consolas', 9))
        self.speed_label.pack(side='left')

        # Servo parameters
        servo_frame = ttk.LabelFrame(parent, text="Servo Dynamics", padding=10)
        servo_frame.pack(fill='x', pady=(0, 10))

        self.servo_sliders = {}
        self.servo_labels = {}

        servo_params = [
            ('tau', 'Time Constant τ (s)', 0.01, 0.5, 0.1),
            ('delay', 'Delay (s)', 0.0, 1.0, 0.0),
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
            ('friction', 'Rolling Friction', 0.0, 0.01, 0.001),
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

    def on_speed_change(self, value):
        """Update playback speed."""
        self.playback_speed = float(value)
        self.speed_label.config(text=f"{self.playback_speed:.1f}x")

    def _build_plot_panel(self, parent):
        """Build plot panel."""
        plot_frame = ttk.LabelFrame(parent, text="Ball Trajectory (Top View)", padding=10)
        plot_frame.pack(fill='both', expand=True)

        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(8, 8), facecolor=self.colors['panel_bg'])
        self.ax.set_facecolor(self.colors['widget_bg'])

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        self.setup_plot()

    def setup_plot(self):
        """Setup matplotlib plot."""
        self.ax.clear()
        self.ax.set_xlim(-120, 120)
        self.ax.set_ylim(-120, 120)
        self.ax.set_xlabel('X (mm)', color=self.colors['fg'], fontsize=11)
        self.ax.set_ylabel('Y (mm)', color=self.colors['fg'], fontsize=11)
        self.ax.set_title('Ball Position (Top View)', color=self.colors['fg'],
                          fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.2, linestyle='--', color=self.colors['fg'])
        self.ax.set_aspect('equal')
        self.ax.tick_params(colors=self.colors['fg'])

        for spine in self.ax.spines.values():
            spine.set_color(self.colors['border'])

        # Platform boundary
        platform_square = Rectangle((-100, -100), 200, 200,
                                    fill=False,
                                    edgecolor=self.colors['fg'],
                                    linewidth=2,
                                    linestyle='--',
                                    alpha=0.5)
        self.ax.add_patch(platform_square)

        # Trajectory lines
        self.sim_trail, = self.ax.plot([], [], color=self.colors['highlight'],
                                       linewidth=2, alpha=0.7, label='Simulation')
        self.hw_trail, = self.ax.plot([], [], 'o', color='#ff4444',
                                      markersize=4, alpha=0.5, label='Hardware')

        # Ball markers
        self.sim_ball = Circle((0, 0), 3.0, color=self.colors['highlight'],
                               alpha=0.8, zorder=10)
        self.ax.add_patch(self.sim_ball)

        self.hw_ball = Circle((0, 0), 3.0, color='#ff4444',
                              alpha=0.8, zorder=10)
        self.ax.add_patch(self.hw_ball)

        legend = self.ax.legend(loc='upper right', fontsize=9,
                                facecolor=self.colors['panel_bg'],
                                edgecolor=self.colors['border'],
                                labelcolor=self.colors['fg'])
        legend.get_frame().set_alpha(0.9)

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

        self.reset_simulation()

    def prev_run(self):
        """Load previous test run."""
        if self.current_run_idx > 0:
            self.load_test_run(self.current_run_idx - 1)

    def next_run(self):
        """Load next test run."""
        if self.current_run_idx < len(self.test_runs) - 1:
            self.load_test_run(self.current_run_idx + 1)

    def reset_simulation(self):
        """Reset simulation to start."""
        self.stop_simulation()

        self.simulation_time = 0.0
        self.data_index = 0
        self.error_history = []

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

        self.sim_trail.set_data([], [])
        self.hw_trail.set_data([], [])

        self.update_plot()

    def start_simulation(self):
        """Start playback."""
        self.simulation_running = True
        self.last_update_time = time.time()

        self.play_btn.config(state='disabled')
        self.stop_btn.config(state='normal')

        self.simulation_loop()

    def stop_simulation(self):
        """Stop playback."""
        self.simulation_running = False

        self.play_btn.config(state='normal')
        self.stop_btn.config(state='disabled')

    def simulation_loop(self):
        """Main simulation update loop."""
        if not self.simulation_running:
            return

        current_time = time.time()
        if self.last_update_time is not None:
            real_dt = current_time - self.last_update_time
            sim_dt = real_dt * self.playback_speed

            self.simulation_time += sim_dt

            # Update based on CSV data
            df = self.current_run['data']

            # Find current data point based on simulation time
            while self.data_index < len(df) - 1:
                if df.loc[self.data_index + 1, 'time'] <= self.simulation_time:
                    self.data_index += 1
                else:
                    break

            if self.data_index < len(df):
                # Get servo angles from CSV
                servo_angles = np.array([
                    df.loc[self.data_index, 's0'],
                    df.loc[self.data_index, 's1'],
                    df.loc[self.data_index, 's2'],
                    df.loc[self.data_index, 's3'],
                    df.loc[self.data_index, 's4'],
                    df.loc[self.data_index, 's5']
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

                    # Step physics
                    self.sim_ball_pos, self.sim_ball_vel, self.sim_ball_omega, _ = \
                        self.ball_physics.step(
                            self.sim_ball_pos,
                            self.sim_ball_vel,
                            self.sim_ball_omega,
                            platform_pose,
                            sim_dt
                        )

                # Get hardware ball position
                if df.loc[self.data_index, 'ball_detected'] and \
                        not pd.isna(df.loc[self.data_index, 'ball_x_mm']):
                    hw_x = df.loc[self.data_index, 'ball_x_mm']
                    hw_y = df.loc[self.data_index, 'ball_y_mm']
                    self.hw_ball_pos = (hw_x, hw_y)

                    # Calculate error
                    sim_x = self.sim_ball_pos[0, 0].item() * 1000
                    sim_y = self.sim_ball_pos[0, 1].item() * 1000
                    error = np.sqrt((sim_x - hw_x) ** 2 + (sim_y - hw_y) ** 2)
                    self.error_history.append(error)
                    if len(self.error_history) > self.max_error_history:
                        self.error_history.pop(0)

                self.update_gui()
                self.update_plot()

            # Check if finished
            if self.data_index >= len(df) - 1:
                self.stop_simulation()

        self.last_update_time = current_time

        if self.simulation_running:
            self.root.after(20, self.simulation_loop)

    def update_gui(self):
        """Update GUI labels."""
        self.time_label.config(text=f"Time: {self.simulation_time:.2f}s")

        if self.error_history:
            current_error = self.error_history[-1]
            mean_error = np.mean(self.error_history)
            max_error = np.max(self.error_history)

            self.error_label.config(text=f"Current Error: {current_error:.1f} mm")
            self.mean_error_label.config(text=f"Mean Error: {mean_error:.1f} mm")
            self.max_error_label.config(text=f"Max Error: {max_error:.1f} mm")

    def update_plot(self):
        """Update plot."""
        # Simulation ball
        sim_x = self.sim_ball_pos[0, 0].item() * 1000
        sim_y = self.sim_ball_pos[0, 1].item() * 1000
        self.sim_ball.center = (sim_x, sim_y)

        # Hardware ball
        self.hw_ball.center = self.hw_ball_pos

        # Update trails
        df = self.current_run['data']

        # Sim trail
        sim_trail_x = []
        sim_trail_y = []
        for i in range(max(0, self.data_index - 50), self.data_index + 1):
            if i < len(df):
                # Would need to store sim positions, for now just show current
                pass

        # HW trail
        hw_trail_x = []
        hw_trail_y = []
        for i in range(max(0, self.data_index - 50), self.data_index + 1):
            if i < len(df) and df.loc[i, 'ball_detected']:
                hw_trail_x.append(df.loc[i, 'ball_x_mm'])
                hw_trail_y.append(df.loc[i, 'ball_y_mm'])

        self.hw_trail.set_data(hw_trail_x, hw_trail_y)

        self.canvas.draw_idle()

    def on_closing(self):
        """Clean shutdown."""
        self.simulation_running = False
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