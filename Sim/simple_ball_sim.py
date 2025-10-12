#!/usr/bin/env python3
"""
Simple Stewart Platform Simulator with 2D Ball Physics (RK4)
With Dark Mode
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle, Circle

from core import FirstOrderServo, StewartPlatformIK, SimpleBallPhysics2D


class StewartSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stewart Platform - Simple 2D Ball Physics (RK4)")
        self.root.geometry("1200x900")

        # Dark mode colors
        self.colors = {
            'bg': '#1e1e1e',  # Main background
            'panel_bg': '#2d2d2d',  # Panel background
            'widget_bg': '#3d3d3d',  # Widget background
            'fg': '#e0e0e0',  # Foreground text
            'highlight': '#007acc',  # Highlight blue
            'button_bg': '#0e639c',  # Button background
            'button_fg': '#ffffff',  # Button text
            'entry_bg': '#3d3d3d',  # Entry background
            'border': '#555555'  # Border color
        }

        # Configure root window
        self.root.configure(bg=self.colors['bg'])

        # Configure ttk style for dark mode
        self.setup_dark_theme()

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
        self.servos = [FirstOrderServo(K=1.0, tau=0.1, delay=0.35) for _ in range(6)]

        self.ball_physics = SimpleBallPhysics2D(
            ball_radius=0.01,
            gravity=9.81,
            friction_coef=0.1
        )

        ball_start_height = (self.ik.home_height_top_surface / 1000) + self.ball_physics.radius
        self.ball_pos = torch.tensor([[0.0, 0.0, ball_start_height]], dtype=torch.float32)
        self.ball_vel = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        self.ball_omega = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

        self.simulation_running = False
        self.simulation_time = 0.0
        self.last_update_time = None
        self.update_rate_ms = 20

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

    def setup_dark_theme(self):
        """Configure ttk widgets for dark mode."""
        style = ttk.Style()

        # Configure colors
        style.theme_use('default')

        # Frame
        style.configure('TFrame', background=self.colors['bg'])
        style.configure('Card.TFrame', background=self.colors['panel_bg'], relief='flat')

        # LabelFrame
        style.configure('TLabelframe',
                        background=self.colors['panel_bg'],
                        foreground=self.colors['fg'],
                        borderwidth=1,
                        relief='solid')
        style.configure('TLabelframe.Label',
                        background=self.colors['panel_bg'],
                        foreground=self.colors['highlight'],
                        font=('Segoe UI', 9, 'bold'))

        # Label
        style.configure('TLabel',
                        background=self.colors['panel_bg'],
                        foreground=self.colors['fg'],
                        font=('Segoe UI', 9))

        # Button
        style.configure('TButton',
                        background=self.colors['button_bg'],
                        foreground=self.colors['button_fg'],
                        borderwidth=0,
                        focuscolor='none',
                        font=('Segoe UI', 9))
        style.map('TButton',
                  background=[('active', self.colors['highlight']),
                              ('pressed', '#005a9e')])

        # Scale
        style.configure('TScale',
                        background=self.colors['panel_bg'],
                        troughcolor=self.colors['widget_bg'],
                        borderwidth=0)

        # Checkbutton
        style.configure('TCheckbutton',
                        background=self.colors['panel_bg'],
                        foreground=self.colors['fg'],
                        font=('Segoe UI', 9))

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, style='TFrame')
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)

        left_panel = ttk.Frame(main_frame, style='TFrame', width=400)
        left_panel.pack(side='left', fill='both', expand=False, padx=(0, 5))
        left_panel.pack_propagate(False)

        middle_panel = ttk.Frame(main_frame, style='TFrame', width=450)
        middle_panel.pack(side='left', fill='both', expand=False, padx=5)
        middle_panel.pack_propagate(False)

        right_panel = ttk.Frame(main_frame, style='TFrame')
        right_panel.pack(side='right', fill='both', expand=True, padx=(5, 0))

        # === LEFT PANEL ===

        # Simulation control
        sim_frame = ttk.LabelFrame(left_panel, text="Simulation Control", padding=10)
        sim_frame.pack(fill='x', pady=(0, 10))

        btn_frame = ttk.Frame(sim_frame)
        btn_frame.pack(fill='x')

        self.start_btn = ttk.Button(btn_frame, text="▶ Start", command=self.start_simulation, width=10)
        self.start_btn.pack(side='left', padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="⏸ Stop", command=self.stop_simulation, state='disabled', width=10)
        self.stop_btn.pack(side='left', padx=5)

        self.reset_btn = ttk.Button(btn_frame, text="↻ Reset", command=self.reset_simulation, width=10)
        self.reset_btn.pack(side='left', padx=5)

        self.sim_time_label = ttk.Label(sim_frame, text="Time: 0.00s", font=('Consolas', 10, 'bold'))
        self.sim_time_label.pack(pady=(10, 0))

        # Ball control
        ball_frame = ttk.LabelFrame(left_panel, text="Ball Control", padding=10)
        ball_frame.pack(fill='x', pady=(0, 10))

        ball_btn_frame = ttk.Frame(ball_frame)
        ball_btn_frame.pack()

        ttk.Button(ball_btn_frame, text="⟲ Reset Ball", command=self.reset_ball, width=15).pack(side='left', padx=5)
        ttk.Button(ball_btn_frame, text="⇝ Push Ball", command=self.push_ball, width=15).pack(side='left', padx=5)

        # Physics info
        physics_info_frame = ttk.LabelFrame(left_panel, text="Physics Info", padding=10)
        physics_info_frame.pack(fill='x', pady=(0, 10))

        info_items = [
            ("Type:", "2D (XY only)"),
            ("Integration:", "RK4 (4th order)"),
            ("Bouncing:", "Disabled"),
            ("Friction:", f"{self.ball_physics.friction}")
        ]

        for label, value in info_items:
            frame = ttk.Frame(physics_info_frame)
            frame.pack(fill='x', pady=2)
            ttk.Label(frame, text=label, font=('Segoe UI', 9, 'bold')).pack(side='left')
            ttk.Label(frame, text=value, font=('Consolas', 9)).pack(side='left', padx=(5, 0))

        # Ball state
        ball_info_frame = ttk.LabelFrame(left_panel, text="Ball State", padding=10)
        ball_info_frame.pack(fill='x', pady=(0, 10))

        self.ball_pos_label = ttk.Label(ball_info_frame, text="Position: (0.0, 0.0) mm", font=('Consolas', 9))
        self.ball_pos_label.pack(anchor='w', pady=2)

        self.ball_vel_label = ttk.Label(ball_info_frame, text="Velocity: (0.0, 0.0) mm/s", font=('Consolas', 9))
        self.ball_vel_label.pack(anchor='w', pady=2)

        # Config
        config_frame = ttk.LabelFrame(left_panel, text="Configuration", padding=10)
        config_frame.pack(fill='x', pady=(0, 10))

        ttk.Checkbutton(config_frame, text="Use Top Surface Offset",
                        variable=self.use_top_surface_offset,
                        command=self.on_offset_toggle).pack(anchor='w')

        # Sliders
        sliders_frame = ttk.LabelFrame(left_panel, text="Commanded Pose (6 DOF)", padding=10)
        sliders_frame.pack(fill='both', expand=True)

        for idx, (dof, (min_val, max_val, res, default, label)) in enumerate(self.dof_config.items()):
            ttk.Label(sliders_frame, text=label, font=('Segoe UI', 9)).grid(row=idx, column=0, sticky='w', pady=8)

            slider = ttk.Scale(sliders_frame, from_=min_val, to=max_val, orient='horizontal',
                               command=lambda val, d=dof: self.on_slider_change(d, val))
            slider.grid(row=idx, column=1, sticky='ew', padx=10, pady=8)
            self.sliders[dof] = slider

            value_label = ttk.Label(sliders_frame, text=f"{default:.2f}", width=8, font=('Consolas', 9))
            value_label.grid(row=idx, column=2, pady=8)
            self.value_labels[dof] = value_label
            slider.set(default)

        sliders_frame.columnconfigure(1, weight=1)

        # === MIDDLE PANEL ===

        # Commanded angles
        cmd_angles_frame = ttk.LabelFrame(middle_panel, text="Commanded Servo Angles (IK)", padding=10)
        cmd_angles_frame.pack(fill='x', pady=(0, 10))

        self.cmd_angle_labels = []
        for i in range(6):
            label = ttk.Label(cmd_angles_frame, text=f"S{i + 1}: 0.00°", font=('Consolas', 9))
            label.grid(row=i // 3, column=i % 3, padx=15, pady=5, sticky='w')
            self.cmd_angle_labels.append(label)

        # Actual angles
        actual_angles_frame = ttk.LabelFrame(middle_panel, text="Actual Servo Angles", padding=10)
        actual_angles_frame.pack(fill='x', pady=(0, 10))

        self.actual_angle_labels = []
        for i in range(6):
            label = ttk.Label(actual_angles_frame, text=f"S{i + 1}: 0.00°", font=('Consolas', 9))
            label.grid(row=i // 3, column=i % 3, padx=15, pady=5, sticky='w')
            self.actual_angle_labels.append(label)

        # FK
        fk_frame = ttk.LabelFrame(middle_panel, text="Platform Pose (FK)", padding=10)
        fk_frame.pack(fill='x', pady=(0, 10))

        self.fk_pos_label = ttk.Label(fk_frame, text="X: 0.00  Y: 0.00  Z: 0.00 mm", font=('Consolas', 9))
        self.fk_pos_label.pack(anchor='w', pady=2)

        self.fk_rot_label = ttk.Label(fk_frame, text="Roll: 0.00  Pitch: 0.00  Yaw: 0.00°", font=('Consolas', 9))
        self.fk_rot_label.pack(anchor='w', pady=2)

        # Log
        log_frame = ttk.LabelFrame(middle_panel, text="Debug Log", padding=10)
        log_frame.pack(fill='both', expand=True)

        # Create scrolled text with dark theme
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=10,
            font=('Consolas', 8),
            bg=self.colors['widget_bg'],
            fg=self.colors['fg'],
            insertbackground=self.colors['fg'],
            selectbackground=self.colors['highlight'],
            selectforeground=self.colors['button_fg'],
            relief='flat',
            borderwidth=0
        )
        self.log_text.pack(fill='both', expand=True)

        # === RIGHT PANEL ===
        plot_frame = ttk.LabelFrame(right_panel, text="Ball Position (Top View)", padding=10)
        plot_frame.pack(fill='both', expand=True)

        # Dark mode matplotlib
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(6, 6), facecolor=self.colors['panel_bg'])
        self.ax.set_facecolor(self.colors['widget_bg'])

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        self.setup_plot()
        self.log("Simple 2D physics with RK4 initialized")
        self.log("Dark mode enabled")

    def setup_plot(self):
        """Setup the matplotlib plot with dark theme."""
        from matplotlib.patches import Rectangle

        self.ax.clear()
        self.ax.set_xlim(-120, 120)
        self.ax.set_ylim(-120, 120)
        self.ax.set_xlabel('X (mm)', color=self.colors['fg'], fontsize=10)
        self.ax.set_ylabel('Y (mm)', color=self.colors['fg'], fontsize=10)
        self.ax.set_title('Ball Position (Top View)', color=self.colors['fg'], fontsize=11, fontweight='bold')
        self.ax.grid(True, alpha=0.2, linestyle='--', color=self.colors['fg'])
        self.ax.set_aspect('equal')

        # Change tick colors
        self.ax.tick_params(colors=self.colors['fg'])

        # Change spine colors
        for spine in self.ax.spines.values():
            spine.set_color(self.colors['border'])

        # Platform square (±100mm)
        platform_square = Rectangle((-100, -100), 200, 200,
                                    fill=False,
                                    edgecolor=self.colors['fg'],
                                    linewidth=2,
                                    linestyle='--',
                                    label='Platform Edge',
                                    alpha=0.5)
        self.ax.add_patch(platform_square)

        # Center marker
        self.ax.plot(0, 0, '+', color=self.colors['highlight'],
                     markersize=12, markeredgewidth=2, label='Center')

        # Ball
        self.ball_circle = Circle((0, 0), 3.0, color='#ff4444', alpha=0.8,
                                  zorder=10, label='Ball')
        self.ax.add_patch(self.ball_circle)

        self.tilt_arrow = None

        legend = self.ax.legend(loc='upper right', fontsize=8, facecolor=self.colors['panel_bg'],
                                edgecolor=self.colors['border'], labelcolor=self.colors['fg'])
        legend.get_frame().set_alpha(0.9)

        self.canvas.draw()

    def update_plot(self):
        """Update the plot with current ball position."""
        ball_x = self.ball_pos[0, 0].item() * 1000
        ball_y = self.ball_pos[0, 1].item() * 1000
        self.ball_circle.center = (ball_x, ball_y)

        if self.tilt_arrow is not None:
            self.tilt_arrow.remove()
            self.tilt_arrow = None

        rx = self.dof_values['rx']
        ry = self.dof_values['ry']

        if abs(rx) > 0.5 or abs(ry) > 0.5:
            dx = -np.sin(np.radians(ry))
            dy = -np.sin(np.radians(rx))
            magnitude = np.sqrt(dx ** 2 + dy ** 2)

            if magnitude > 0:
                dx = (dx / magnitude) * 30
                dy = (dy / magnitude) * 30
                self.tilt_arrow = self.ax.arrow(0, 0, dx, dy, head_width=8, head_length=10,
                                                fc=self.colors['highlight'],
                                                ec=self.colors['highlight'],
                                                alpha=0.6, linewidth=2, zorder=5)

        self.canvas.draw_idle()

    def reset_ball(self):
        home_z = self.ik.home_height_top_surface if self.use_top_surface_offset.get() else self.ik.home_height
        ball_start_height = (home_z / 1000) + self.ball_physics.radius

        self.ball_pos = torch.tensor([[0.0, 0.0, ball_start_height]], dtype=torch.float32)
        self.ball_vel = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        self.ball_omega = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        self.update_plot()
        self.log("Ball reset to center")

    def push_ball(self):
        vx = np.random.uniform(-0.05, 0.05)
        vy = np.random.uniform(-0.05, 0.05)
        self.ball_vel = torch.tensor([[vx, vy, 0.0]], dtype=torch.float32)
        self.log(f"Ball pushed: vx={vx:.3f}, vy={vy:.3f} m/s")

    def log(self, message):
        self.log_text.insert(tk.END, f"[{self.simulation_time:.2f}s] {message}\n")
        self.log_text.see(tk.END)

    def on_offset_toggle(self):
        enabled = self.use_top_surface_offset.get()
        home_z = self.ik.home_height_top_surface if enabled else self.ik.home_height

        self.sliders['z'].config(from_=home_z - 30, to=home_z + 30)
        self.dof_values['z'] = home_z
        self.sliders['z'].set(home_z)
        self.value_labels['z'].config(text=f"{home_z:.2f}")

        ball_start_height = (home_z / 1000) + self.ball_physics.radius
        self.ball_pos[0, 2] = ball_start_height
        self.log(f"Offset: {'Top Surface' if enabled else 'Anchor Center'}")

    def on_slider_change(self, dof, value):
        val = float(value)
        self.dof_values[dof] = val
        self.value_labels[dof].config(text=f"{val:.2f}")

        if self.update_timer is not None:
            self.root.after_cancel(self.update_timer)
        self.update_timer = self.root.after(50, self.calculate_ik)

    def calculate_ik(self):
        translation = np.array([self.dof_values['x'], self.dof_values['y'], self.dof_values['z']])
        rotation = np.array([self.dof_values['rx'], self.dof_values['ry'], self.dof_values['rz']])

        angles = self.ik.calculate_servo_angles(translation, rotation, self.use_top_surface_offset.get())

        if angles is not None:
            for i in range(6):
                self.cmd_angle_labels[i].config(text=f"S{i + 1}: {angles[i]:6.2f}°")

            if self.simulation_running:
                for i, servo in enumerate(self.servos):
                    servo.send_command(angles[i], self.simulation_time)
        else:
            for i in range(6):
                self.cmd_angle_labels[i].config(text=f"S{i + 1}: ERROR")

    def start_simulation(self):
        self.simulation_running = True
        self.last_update_time = time.time()
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.log("Simulation started")
        self.simulation_loop()

    def stop_simulation(self):
        self.simulation_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.log("Simulation stopped")

    def reset_simulation(self):
        was_running = self.simulation_running
        if was_running:
            self.stop_simulation()

        for servo in self.servos:
            servo.reset()

        self.simulation_time = 0.0
        self.last_update_time = None
        self.sim_time_label.config(text="Time: 0.00s")

        for dof, (_, _, _, default, _) in self.dof_config.items():
            if dof == 'z':
                home_z = (self.ik.home_height_top_surface if self.use_top_surface_offset.get()
                          else self.ik.home_height)
                self.sliders[dof].set(home_z)
            else:
                self.sliders[dof].set(default)

        self.reset_ball()
        self.log("Simulation reset")

        if was_running:
            self.start_simulation()

    def simulation_loop(self):
        if not self.simulation_running:
            return

        current_time = time.time()
        if self.last_update_time is not None:
            dt = current_time - self.last_update_time
            self.simulation_time += dt

            for servo in self.servos:
                servo.update(dt, self.simulation_time)

            actual_angles = np.array([servo.get_angle() for servo in self.servos])

            for i in range(6):
                self.actual_angle_labels[i].config(text=f"S{i + 1}: {actual_angles[i]:6.2f}°")

            translation, rotation, success, _ = self.ik.calculate_forward_kinematics(
                actual_angles, use_top_surface_offset=self.use_top_surface_offset.get()
            )

            if success:
                self.fk_pos_label.config(
                    text=f"X: {translation[0]:6.2f}  Y: {translation[1]:6.2f}  Z: {translation[2]:6.2f} mm"
                )
                self.fk_rot_label.config(
                    text=f"Roll: {rotation[0]:6.2f}  Pitch: {rotation[1]:6.2f}  Yaw: {rotation[2]:6.2f}°"
                )

                try:
                    platform_pose = torch.tensor([[
                        translation[0] / 1000, translation[1] / 1000, translation[2] / 1000,
                        rotation[0], rotation[1], rotation[2]
                    ]], dtype=torch.float32)

                    self.ball_pos, self.ball_vel, self.ball_omega, contact_info = \
                        self.ball_physics.step(self.ball_pos, self.ball_vel, platform_pose, dt)

                    if contact_info.get('fell_off', False):
                        self.log("Ball fell off platform")

                    ball_x_mm = self.ball_pos[0, 0].item() * 1000
                    ball_y_mm = self.ball_pos[0, 1].item() * 1000
                    vel_x_mm = self.ball_vel[0, 0].item() * 1000
                    vel_y_mm = self.ball_vel[0, 1].item() * 1000

                    self.ball_pos_label.config(text=f"Position: ({ball_x_mm:.1f}, {ball_y_mm:.1f}) mm")
                    self.ball_vel_label.config(text=f"Velocity: ({vel_x_mm:.1f}, {vel_y_mm:.1f}) mm/s")

                except Exception as e:
                    self.log(f"Physics error: {str(e)}")
                    self.reset_ball()

            self.update_plot()

        self.last_update_time = current_time
        self.sim_time_label.config(text=f"Time: {self.simulation_time:.2f}s")

        self.root.after(self.update_rate_ms, self.simulation_loop)


def main():
    root = tk.Tk()
    app = StewartSimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()