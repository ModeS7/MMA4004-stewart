#!/usr/bin/env python3
"""
Stewart Platform Simulator - Base Class

Reusable simulator with pluggable controller support.
Eliminates duplication between PID and LQR implementations.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle, Circle
from abc import ABC, abstractmethod

from core import FirstOrderServo, StewartPlatformIK, SimpleBallPhysics2D, PatternFactory


class ControllerConfig(ABC):
    """
    Abstract base for controller-specific configuration.

    Defines interface for controller parameter UI and behavior.
    """

    @abstractmethod
    def get_controller_name(self) -> str:
        """Return display name for controller (e.g., 'PID', 'LQR')."""
        pass

    @abstractmethod
    def create_controller(self, **kwargs):
        """Create and return controller instance."""
        pass

    @abstractmethod
    def create_parameter_widgets(self, parent_frame, colors, on_param_change_callback):
        """
        Create controller-specific parameter widgets.

        Args:
            parent_frame: ttk.Frame to add widgets to
            colors: dict of color scheme
            on_param_change_callback: function to call when parameters change

        Returns:
            dict: {
                'sliders': {name: slider_widget},
                'value_labels': {name: label_widget},
                'scalar_vars': {name: IntVar} (optional),
                'update_fn': callable to update controller parameters
            }
        """
        pass

    @abstractmethod
    def get_scalar_values(self) -> list:
        """Return list of scalar multipliers for parameters."""
        pass

    @abstractmethod
    def create_info_widgets(self, parent_frame, colors, controller_instance):
        """
        Create controller-specific info widgets (optional).

        Args:
            parent_frame: ttk.Frame to add widgets to
            colors: dict of color scheme
            controller_instance: the controller object
        """
        pass


class BaseStewartSimulator:
    """
    Base Stewart Platform Simulator with pluggable controller support.

    Handles all common functionality:
    - GUI layout and dark theme
    - Simulation loop and physics
    - Ball control and visualization
    - Servo dynamics and kinematics
    - Pattern/trajectory system
    """

    def __init__(self, root, controller_config: ControllerConfig):
        self.root = root
        self.controller_config = controller_config

        controller_name = controller_config.get_controller_name()
        self.root.title(f"Stewart Platform - {controller_name} Ball Balancing Control")
        self.root.geometry("1400x900")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Dark mode colors
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

        # Platform setup
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
        self.servos = [FirstOrderServo(K=1.0, tau=0.1, delay=0.0) for _ in range(6)]

        # Ball physics
        self.ball_physics = SimpleBallPhysics2D(
            ball_radius=0.04,
            ball_mass=0.0027,
            gravity=9.81,
            rolling_friction=0.001,
            sphere_type='hollow'
        )

        # Controller (created by subclass config)
        self.controller = None
        self.controller_enabled = tk.BooleanVar(value=False)

        # Trajectory pattern
        self.current_pattern = PatternFactory.create('static', x=0.0, y=0.0)
        self.pattern_type = tk.StringVar(value='static')
        self.pattern_start_time = 0.0

        # Ball state
        ball_start_height = (self.ik.home_height_top_surface / 1000) + self.ball_physics.radius
        self.ball_pos = torch.tensor([[0.0, 0.0, ball_start_height]], dtype=torch.float32)
        self.ball_vel = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        self.ball_omega = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

        # Simulation state
        self.simulation_running = False
        self.simulation_time = 0.0
        self.last_update_time = None
        self.update_rate_ms = 20
        self.simulation_loop_id = None

        # DOF configuration
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

        # Controller-specific widgets (populated by config)
        self.controller_widgets = {}

        # Platform motion tracking
        self.prev_platform_angles = {'rx': 0.0, 'ry': 0.0}
        self.platform_angular_vel = {'rx': 0.0, 'ry': 0.0}
        self.platform_angular_accel = {'rx': 0.0, 'ry': 0.0}

        # Build GUI
        self.create_widgets()

    def setup_dark_theme(self):
        """Configure ttk widgets for dark mode."""
        style = ttk.Style()
        style.theme_use('default')

        style.configure('TFrame', background=self.colors['bg'])
        style.configure('Card.TFrame', background=self.colors['panel_bg'], relief='flat')

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

        style.configure('TCheckbutton',
                        background=self.colors['panel_bg'],
                        foreground=self.colors['fg'],
                        font=('Segoe UI', 9))

        style.configure('TCombobox',
                        fieldbackground=self.colors['widget_bg'],
                        background=self.colors['button_bg'],
                        foreground=self.colors['fg'],
                        arrowcolor=self.colors['fg'],
                        selectbackground=self.colors['highlight'],
                        selectforeground=self.colors['button_fg'])
        style.map('TCombobox',
                  fieldbackground=[('readonly', self.colors['widget_bg'])],
                  selectbackground=[('readonly', self.colors['widget_bg'])],
                  foreground=[('readonly', self.colors['fg'])])

        self.root.option_add('*TCombobox*Listbox.background', self.colors['widget_bg'])
        self.root.option_add('*TCombobox*Listbox.foreground', self.colors['fg'])
        self.root.option_add('*TCombobox*Listbox.selectBackground', self.colors['highlight'])
        self.root.option_add('*TCombobox*Listbox.selectForeground', self.colors['button_fg'])
        self.root.option_add('*TCombobox*Listbox.font', ('Segoe UI', 9))

    def create_widgets(self):
        """Create all GUI widgets."""
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

        # Build panels
        self._create_left_panel(left_panel)
        self._create_middle_panel(middle_panel)
        self._create_right_panel(right_panel)

        # Initialize controller
        self._initialize_controller()

        controller_name = self.controller_config.get_controller_name()
        self.log(f"{controller_name} ball balancing control initialized")
        self.log(f"Configure {controller_name} parameters and enable to start automatic balancing")

    def _create_left_panel(self, parent):
        """Create left panel widgets (control, parameters, ball, config, DOF)."""
        # Simulation control
        sim_frame = ttk.LabelFrame(parent, text="Simulation Control", padding=10)
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

        # Controller frame
        controller_name = self.controller_config.get_controller_name()
        ctrl_frame = ttk.LabelFrame(parent, text=f"{controller_name} Ball Balancing", padding=10)
        ctrl_frame.pack(fill='x', pady=(0, 10))

        enable_frame = ttk.Frame(ctrl_frame)
        enable_frame.pack(fill='x', pady=(0, 10))

        ttk.Checkbutton(enable_frame, text=f"Enable {controller_name} Control",
                        variable=self.controller_enabled,
                        command=self.on_controller_toggle).pack(side='left')

        self.controller_status_label = ttk.Label(enable_frame, text="●",
                                                 foreground=self.colors['border'],
                                                 font=('Segoe UI', 14))
        self.controller_status_label.pack(side='left', padx=(10, 0))

        # Controller-specific parameter widgets
        self.controller_widgets = self.controller_config.create_parameter_widgets(
            ctrl_frame, self.colors, self.on_controller_param_change
        )

        # Controller-specific info widgets
        self.controller_config.create_info_widgets(ctrl_frame, self.colors, self.controller)

        # Trajectory pattern
        pattern_frame = ttk.LabelFrame(ctrl_frame, text="Trajectory Pattern", padding=10)
        pattern_frame.pack(fill='x', pady=(10, 0))

        selector_frame = ttk.Frame(pattern_frame)
        selector_frame.pack(fill='x', pady=(0, 5))

        ttk.Label(selector_frame, text="Pattern:", font=('Segoe UI', 9)).pack(side='left', padx=(0, 10))

        pattern_combo = ttk.Combobox(selector_frame, textvariable=self.pattern_type,
                                     width=15, state='readonly',
                                     values=['static', 'circle', 'figure8', 'star'])
        pattern_combo.pack(side='left', padx=5)
        pattern_combo.bind('<<ComboboxSelected>>', self.on_pattern_change)

        ttk.Button(selector_frame, text="Reset", command=self.reset_pattern, width=8).pack(side='left', padx=5)

        self.pattern_info_label = ttk.Label(pattern_frame,
                                            text="Tracking: Center (0, 0)",
                                            font=('Consolas', 8),
                                            foreground=self.colors['success'])
        self.pattern_info_label.pack(anchor='w', pady=(5, 0))

        # Ball control
        ball_frame = ttk.LabelFrame(parent, text="Ball Control", padding=10)
        ball_frame.pack(fill='x', pady=(0, 10))

        ball_btn_frame = ttk.Frame(ball_frame)
        ball_btn_frame.pack()

        ttk.Button(ball_btn_frame, text="⟲ Reset Ball", command=self.reset_ball, width=15).pack(side='left', padx=5)
        ttk.Button(ball_btn_frame, text="⇝ Push Ball", command=self.push_ball, width=15).pack(side='left', padx=5)

        # Ball state
        ball_info_frame = ttk.LabelFrame(parent, text="Ball State", padding=10)
        ball_info_frame.pack(fill='x', pady=(0, 10))

        self.ball_pos_label = ttk.Label(ball_info_frame, text="Position: (0.0, 0.0) mm", font=('Consolas', 9))
        self.ball_pos_label.pack(anchor='w', pady=2)

        self.ball_vel_label = ttk.Label(ball_info_frame, text="Velocity: (0.0, 0.0) mm/s", font=('Consolas', 9))
        self.ball_vel_label.pack(anchor='w', pady=2)

        # Config
        config_frame = ttk.LabelFrame(parent, text="Configuration", padding=10)
        config_frame.pack(fill='x', pady=(0, 10))

        ttk.Checkbutton(config_frame, text="Use Top Surface Offset",
                        variable=self.use_top_surface_offset,
                        command=self.on_offset_toggle).pack(anchor='w')

        # Manual DOF sliders
        sliders_frame = ttk.LabelFrame(parent, text="Manual Pose Control (6 DOF)", padding=10)
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

    def _create_middle_panel(self, parent):
        """Create middle panel widgets (angles, FK, output, log)."""
        # Commanded angles
        cmd_angles_frame = ttk.LabelFrame(parent, text="Commanded Servo Angles (IK)", padding=10)
        cmd_angles_frame.pack(fill='x', pady=(0, 10))

        self.cmd_angle_labels = []
        for i in range(6):
            label = ttk.Label(cmd_angles_frame, text=f"S{i + 1}: 0.00°", font=('Consolas', 9))
            label.grid(row=i // 3, column=i % 3, padx=15, pady=5, sticky='w')
            self.cmd_angle_labels.append(label)

        # Actual angles
        actual_angles_frame = ttk.LabelFrame(parent, text="Actual Servo Angles", padding=10)
        actual_angles_frame.pack(fill='x', pady=(0, 10))

        self.actual_angle_labels = []
        for i in range(6):
            label = ttk.Label(actual_angles_frame, text=f"S{i + 1}: 0.00°", font=('Consolas', 9))
            label.grid(row=i // 3, column=i % 3, padx=15, pady=5, sticky='w')
            self.actual_angle_labels.append(label)

        # FK
        fk_frame = ttk.LabelFrame(parent, text="Platform Pose (FK)", padding=10)
        fk_frame.pack(fill='x', pady=(0, 10))

        self.fk_pos_label = ttk.Label(fk_frame, text="X: 0.00  Y: 0.00  Z: 0.00 mm", font=('Consolas', 9))
        self.fk_pos_label.pack(anchor='w', pady=2)

        self.fk_rot_label = ttk.Label(fk_frame, text="Roll: 0.00  Pitch: 0.00  Yaw: 0.00°", font=('Consolas', 9))
        self.fk_rot_label.pack(anchor='w', pady=2)

        # Controller output
        controller_name = self.controller_config.get_controller_name()
        output_frame = ttk.LabelFrame(parent, text=f"{controller_name} Output", padding=10)
        output_frame.pack(fill='x', pady=(0, 10))

        self.controller_output_label = ttk.Label(output_frame, text="Tilt: rx=0.00°  ry=0.00°", font=('Consolas', 9))
        self.controller_output_label.pack(anchor='w', pady=2)

        self.controller_error_label = ttk.Label(output_frame, text="Error: (0.0, 0.0) mm", font=('Consolas', 9))
        self.controller_error_label.pack(anchor='w', pady=2)

        # Log
        log_frame = ttk.LabelFrame(parent, text="Debug Log", padding=10)
        log_frame.pack(fill='both', expand=True)

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

    def _create_right_panel(self, parent):
        """Create right panel widgets (visualization)."""
        plot_frame = ttk.LabelFrame(parent, text="Ball Position (Top View)", padding=10)
        plot_frame.pack(fill='both', expand=True)

        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(6, 6), facecolor=self.colors['panel_bg'])
        self.ax.set_facecolor(self.colors['widget_bg'])

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        self.setup_plot()

    def setup_plot(self):
        """Setup matplotlib plot."""
        self.ax.clear()
        self.ax.set_xlim(-120, 120)
        self.ax.set_ylim(-120, 120)
        self.ax.set_xlabel('X (mm)', color=self.colors['fg'], fontsize=10)
        self.ax.set_ylabel('Y (mm)', color=self.colors['fg'], fontsize=10)
        self.ax.set_title('Ball Position (Top View)', color=self.colors['fg'], fontsize=11, fontweight='bold')
        self.ax.grid(True, alpha=0.2, linestyle='--', color=self.colors['fg'])
        self.ax.set_aspect('equal')
        self.ax.tick_params(colors=self.colors['fg'])

        for spine in self.ax.spines.values():
            spine.set_color(self.colors['border'])

        platform_square = Rectangle((-100, -100), 200, 200,
                                    fill=False,
                                    edgecolor=self.colors['fg'],
                                    linewidth=2,
                                    linestyle='--',
                                    label='Platform Edge',
                                    alpha=0.5)
        self.ax.add_patch(platform_square)

        self.trajectory_line, = self.ax.plot([], [], '--', color=self.colors['highlight'],
                                             alpha=0.3, linewidth=1, label='Trajectory')
        self.target_marker, = self.ax.plot([0], [0], 'x', color=self.colors['success'],
                                           markersize=10, markeredgewidth=2, label='Target')

        self.ball_circle = Circle((0, 0), 3.0, color='#ff4444', alpha=0.8,
                                  zorder=10, label='Ball')
        self.ax.add_patch(self.ball_circle)

        self.tilt_arrow = None

        legend = self.ax.legend(loc='upper right', fontsize=8, facecolor=self.colors['panel_bg'],
                                edgecolor=self.colors['border'], labelcolor=self.colors['fg'])
        legend.get_frame().set_alpha(0.9)

        self.canvas.draw()

    def update_plot(self):
        """Update plot with current state."""
        ball_x = self.ball_pos[0, 0].item() * 1000
        ball_y = self.ball_pos[0, 1].item() * 1000
        self.ball_circle.center = (ball_x, ball_y)

        if self.pattern_type.get() != 'static':
            pattern_periods = {'circle': 10.0, 'figure8': 12.0, 'star': 15.0}
            period = pattern_periods.get(self.pattern_type.get(), 10.0)

            t_samples = np.linspace(0, period, 100)
            path_x, path_y = [], []
            for t in t_samples:
                x, y = self.current_pattern.get_position(t)
                path_x.append(x)
                path_y.append(y)

            self.trajectory_line.set_data(path_x, path_y)

            pattern_time = self.simulation_time - self.pattern_start_time
            target_x, target_y = self.current_pattern.get_position(pattern_time)
            self.target_marker.set_data([target_x], [target_y])
        else:
            self.trajectory_line.set_data([], [])
            self.target_marker.set_data([0], [0])

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
                color = self.colors['success'] if self.controller_enabled.get() else self.colors['highlight']
                self.tilt_arrow = self.ax.arrow(0, 0, dx, dy, head_width=8, head_length=10,
                                                fc=color, ec=color,
                                                alpha=0.6, linewidth=2, zorder=5)

        self.canvas.draw_idle()

    def _initialize_controller(self):
        """Initialize controller using config (called after widgets created)."""
        # Controller config should handle creating the controller with proper params
        pass

    def on_controller_toggle(self):
        """Handle controller enable/disable."""
        enabled = self.controller_enabled.get()

        if enabled:
            self.controller.reset()
            self.reset_pattern()
            self.controller_status_label.config(foreground=self.colors['success'])

            controller_name = self.controller_config.get_controller_name()
            self.log(f"{controller_name} control ENABLED - automatic balancing active")

            # Disable manual tilt controls
            self.sliders['rx'].config(state='disabled')
            self.sliders['ry'].config(state='disabled')

            # Initialize with zero tilt
            self.dof_values['rx'] = 0.0
            self.dof_values['ry'] = 0.0
            self.value_labels['rx'].config(text="0.00")
            self.value_labels['ry'].config(text="0.00")

            self.calculate_ik()
        else:
            self.controller_status_label.config(foreground=self.colors['border'])
            controller_name = self.controller_config.get_controller_name()
            self.log(f"{controller_name} control DISABLED - manual control active")

            # Re-enable manual tilt controls
            self.sliders['rx'].config(state='normal')
            self.sliders['ry'].config(state='normal')

            self.sliders['rx'].set(self.dof_values['rx'])
            self.sliders['ry'].set(self.dof_values['ry'])

    def on_controller_param_change(self):
        """Callback when controller parameters change."""
        if 'update_fn' in self.controller_widgets:
            self.controller_widgets['update_fn']()

    def on_pattern_change(self, event=None):
        """Handle pattern selection change."""
        pattern_type = self.pattern_type.get()

        pattern_configs = {
            'static': ('static', {'x': 0.0, 'y': 0.0}, "Tracking: Center (0, 0)"),
            'circle': ('circle', {'radius': 50.0, 'period': 10.0, 'clockwise': True},
                       "Tracking: Circle (r=50mm, T=10s)"),
            'figure8': ('figure8', {'width': 60.0, 'height': 40.0, 'period': 12.0},
                        "Tracking: Figure-8 (60×40mm, T=12s)"),
            'star': ('star', {'radius': 60.0, 'period': 15.0},
                     "Tracking: 5-Point Star (r=60mm, T=15s)")
        }

        if pattern_type in pattern_configs:
            pattern_name, params, info = pattern_configs[pattern_type]
            self.current_pattern = PatternFactory.create(pattern_name, **params)
            self.pattern_info_label.config(text=info)
            self.reset_pattern()
            self.update_plot()
            self.log(f"Pattern changed to: {pattern_type}")

    def reset_pattern(self):
        """Reset pattern timing."""
        self.pattern_start_time = self.simulation_time
        self.current_pattern.reset()
        self.log(f"Pattern reset at t={self.simulation_time:.2f}s")

        if self.controller_enabled.get():
            self.controller.reset()

    def apply_random_tilt(self, max_degrees=1.0):
        """Apply small random tilt (simulates disturbance)."""
        if not self.controller_enabled.get():
            return

        random_rx = np.random.uniform(-max_degrees, max_degrees)
        random_ry = np.random.uniform(-max_degrees, max_degrees)

        self.dof_values['rx'] = random_rx
        self.dof_values['ry'] = random_ry

        self.value_labels['rx'].config(text=f"{random_rx:.2f}")
        self.value_labels['ry'].config(text=f"{random_ry:.2f}")

        self.calculate_ik()
        self.log(f"Platform disturbance: rx={random_rx:.2f}°, ry={random_ry:.2f}°")

    def reset_ball(self):
        """Reset ball to center."""
        home_z = self.ik.home_height_top_surface if self.use_top_surface_offset.get() else self.ik.home_height
        ball_start_height = (home_z / 1000) + self.ball_physics.radius

        self.ball_pos = torch.tensor([[0.0, 0.0, ball_start_height]], dtype=torch.float32)
        self.ball_vel = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        self.ball_omega = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

        if self.controller_enabled.get():
            self.controller.reset()
            self.apply_random_tilt()

        self.update_plot()
        self.log("Ball reset to center")

    def push_ball(self):
        """Apply random velocity to ball."""
        vx = np.random.uniform(-0.05, 0.05)
        vy = np.random.uniform(-0.05, 0.05)
        self.ball_vel = torch.tensor([[vx, vy, 0.0]], dtype=torch.float32)
        self.log(f"Ball pushed: vx={vx:.3f}, vy={vy:.3f} m/s")

    def log(self, message):
        """Add message to debug log."""
        self.log_text.insert(tk.END, f"[{self.simulation_time:.2f}s] {message}\n")
        self.log_text.see(tk.END)

    def on_offset_toggle(self):
        """Handle top surface offset toggle."""
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
        """Handle manual DOF slider change."""
        val = float(value)
        self.dof_values[dof] = val
        self.value_labels[dof].config(text=f"{val:.2f}")

        if self.update_timer is not None:
            self.root.after_cancel(self.update_timer)
        self.update_timer = self.root.after(50, self.calculate_ik)

    def calculate_ik(self):
        """Calculate inverse kinematics for current pose."""
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
        """Start simulation loop."""
        self.simulation_running = True
        self.last_update_time = time.time()
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.log("Simulation started")
        self.simulation_loop()

    def stop_simulation(self):
        """Stop simulation loop."""
        self.simulation_running = False

        if self.simulation_loop_id is not None:
            self.root.after_cancel(self.simulation_loop_id)
            self.simulation_loop_id = None

        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.log("Simulation stopped")

    def reset_simulation(self):
        """Reset simulation to initial state."""
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

        if self.controller_enabled.get():
            self.controller.reset()

        self.log("Simulation reset")

        if was_running:
            self.start_simulation()

    def simulation_loop(self):
        """Main simulation update loop."""
        if not self.simulation_running:
            self.simulation_loop_id = None
            return

        current_time = time.time()
        if self.last_update_time is not None:
            dt = current_time - self.last_update_time
            self.simulation_time += dt

            # Controller update
            if self.controller_enabled.get():
                try:
                    ball_x_mm = self.ball_pos[0, 0].item() * 1000
                    ball_y_mm = self.ball_pos[0, 1].item() * 1000
                    ball_vx_mm_s = self.ball_vel[0, 0].item() * 1000
                    ball_vy_mm_s = self.ball_vel[0, 1].item() * 1000

                    pattern_time = self.simulation_time - self.pattern_start_time
                    target_x, target_y = self.current_pattern.get_position(pattern_time)
                    target_pos_mm = (target_x, target_y)

                    # Call controller update (signature depends on controller type)
                    # This is handled by subclass implementation
                    rx, ry = self._update_controller(
                        (ball_x_mm, ball_y_mm),
                        (ball_vx_mm_s, ball_vy_mm_s),
                        target_pos_mm,
                        dt
                    )

                    self.dof_values['rx'] = rx
                    self.dof_values['ry'] = ry

                    self.value_labels['rx'].config(text=f"{rx:.2f}")
                    self.value_labels['ry'].config(text=f"{ry:.2f}")

                    error_x = ball_x_mm - target_pos_mm[0]
                    error_y = ball_y_mm - target_pos_mm[1]
                    self.controller_output_label.config(text=f"Tilt: rx={rx:.2f}°  ry={ry:.2f}°")
                    self.controller_error_label.config(text=f"Error: ({error_x:.1f}, {error_y:.1f}) mm")

                    if int(self.simulation_time * 0.5) % 2 == 0 and int((self.simulation_time - dt) * 0.5) % 2 == 1:
                        self.log(f"Target: ({target_x:.1f},{target_y:.1f})mm "
                                 f"Ball: ({ball_x_mm:.1f},{ball_y_mm:.1f})mm "
                                 f"Error: ({error_x:.1f},{error_y:.1f})mm")

                    translation = np.array([self.dof_values['x'], self.dof_values['y'], self.dof_values['z']])
                    rotation = np.array([rx, ry, self.dof_values['rz']])

                    angles = self.ik.calculate_servo_angles(translation, rotation, self.use_top_surface_offset.get())

                    if angles is not None:
                        for i in range(6):
                            self.cmd_angle_labels[i].config(text=f"S{i + 1}: {angles[i]:6.2f}°")
                            self.servos[i].send_command(angles[i], self.simulation_time)
                    else:
                        controller_name = self.controller_config.get_controller_name()
                        self.log(f"{controller_name}: IK solution out of range")

                except Exception as e:
                    controller_name = self.controller_config.get_controller_name()
                    self.log(f"{controller_name} control error: {str(e)}")
                    self.controller_enabled.set(False)
                    self.on_controller_toggle()

            # Update servos
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
                        self.ball_physics.step(
                            self.ball_pos, self.ball_vel, self.ball_omega, platform_pose, dt,
                            platform_angular_accel=self.platform_angular_accel
                        )

                    if contact_info.get('fell_off', False):
                        self.log("Ball fell off platform")
                        if self.controller_enabled.get():
                            self.apply_random_tilt()

                    ball_x_mm = self.ball_pos[0, 0].item() * 1000
                    ball_y_mm = self.ball_pos[0, 1].item() * 1000
                    vel_x_mm = self.ball_vel[0, 0].item() * 1000
                    vel_y_mm = self.ball_vel[0, 1].item() * 1000

                    self.ball_pos_label.config(text=f"Position: ({ball_x_mm:.1f}, {ball_y_mm:.1f}) mm")
                    self.ball_vel_label.config(text=f"Velocity: ({vel_x_mm:.1f}, {vel_y_mm:.1f}) mm/s")

                except Exception as e:
                    self.log(f"Physics error: {str(e)}")
                    self.reset_ball()

                # Update platform angular acceleration
                rx_now = rotation[0]
                ry_now = rotation[1]

                omega_rx = (rx_now - self.prev_platform_angles['rx']) / dt
                omega_ry = (ry_now - self.prev_platform_angles['ry']) / dt

                alpha_rx = (omega_rx - self.platform_angular_vel['rx']) / dt
                alpha_ry = (omega_ry - self.platform_angular_vel['ry']) / dt

                self.platform_angular_vel['rx'] = omega_rx
                self.platform_angular_vel['ry'] = omega_ry
                self.platform_angular_accel['rx'] = alpha_rx
                self.platform_angular_accel['ry'] = alpha_ry

                self.prev_platform_angles['rx'] = rx_now
                self.prev_platform_angles['ry'] = ry_now

            self.update_plot()

        self.last_update_time = current_time
        self.sim_time_label.config(text=f"Time: {self.simulation_time:.2f}s")

        self.simulation_loop_id = self.root.after(self.update_rate_ms, self.simulation_loop)

    def _update_controller(self, ball_pos_mm, ball_vel_mm_s, target_pos_mm, dt):
        """
        Update controller and return control output.

        Override in subclass if needed to handle different controller signatures.

        Args:
            ball_pos_mm: (x, y) ball position in mm
            ball_vel_mm_s: (vx, vy) ball velocity in mm/s
            target_pos_mm: (x, y) target position in mm
            dt: timestep in seconds

        Returns:
            (rx, ry): platform tilt angles in degrees
        """
        # Default implementation assumes controller.update takes position and target
        # Override for controllers with different signatures
        raise NotImplementedError("Subclass must implement _update_controller")

    def on_closing(self):
        """Clean shutdown when window is closed."""
        self.simulation_running = False

        if self.simulation_loop_id is not None:
            try:
                self.root.after_cancel(self.simulation_loop_id)
                self.simulation_loop_id = None
            except:
                pass

        if self.update_timer is not None:
            try:
                self.root.after_cancel(self.update_timer)
            except:
                pass

        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass