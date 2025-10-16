#!/usr/bin/env python3
"""
Stewart Platform Simulator - Modular Base Class

Reusable simulator with pluggable controller support and modular GUI.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle, Circle
from abc import ABC, abstractmethod

from core import FirstOrderServo, StewartPlatformIK, SimpleBallPhysics2D, PatternFactory
from control_core import clip_tilt_vector
from utils import (
    MAX_TILT_ANGLE_DEG, MAX_SERVO_ANGLE_DEG, PLATFORM_HALF_SIZE_MM,
    SimulationConfig, format_vector_2d, format_time, format_error_context
)
from gui_builder import GUIBuilder, create_standard_layout
import gui_modules as gm


class ControllerConfig(ABC):
    """Abstract base for controller-specific configuration."""

    @abstractmethod
    def get_controller_name(self) -> str:
        """Return display name for controller."""
        pass

    @abstractmethod
    def create_controller(self, **kwargs):
        """Create and return controller instance."""
        pass

    @abstractmethod
    def get_scalar_values(self) -> list:
        """Return list of scalar multipliers for parameters."""
        pass

    def get_scaled_param(self, param_name, sliders, scalar_vars):
        """Extract and scale a parameter value from widgets."""
        raw = float(sliders[param_name].get())
        scalar = self.get_scalar_values()[scalar_vars[param_name].get()]
        return raw * scalar

    def create_parameter_slider(self, parent, param_name, label, default,
                                sliders, value_labels, scalar_vars,
                                on_change_callback):
        """Create standard parameter slider with scalar multiplier."""
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=5)

        ttk.Label(frame, text=label, font=('Segoe UI', 9)).grid(
            row=0, column=0, sticky='w', pady=2
        )

        slider = ttk.Scale(frame, from_=0.0, to=10.0, orient='horizontal')
        slider.grid(row=0, column=1, sticky='ew', padx=10)
        slider.set(default)
        sliders[param_name] = slider

        value_label = ttk.Label(frame, text=f"{default:.2f}",
                                width=6, font=('Consolas', 9))
        value_label.grid(row=0, column=2)
        value_labels[param_name] = value_label

        scalar_var = tk.IntVar(value=getattr(self, 'default_scalar_idx', 4))
        scalar_vars[param_name] = scalar_var

        scalar_combo = ttk.Combobox(
            frame, width=12, state='readonly',
            values=[f'×{s:.7g}' for s in self.get_scalar_values()]
        )
        scalar_combo.grid(row=0, column=3, padx=(5, 0))
        scalar_combo.current(scalar_var.get())

        slider.config(command=lambda val, p=param_name:
        self._on_slider_change(p, val, value_labels, on_change_callback))
        scalar_combo.bind('<<ComboboxSelected>>',
                          lambda e, c=scalar_combo, v=scalar_var, p=param_name:
                          self._on_scalar_change(c, v, p, on_change_callback))

        frame.columnconfigure(1, weight=1)

    def _on_slider_change(self, param_name, value, value_labels, callback):
        """Handle slider value change."""
        val = float(value)
        value_labels[param_name].config(text=f"{val:.2f}")
        callback()

    def _on_scalar_change(self, combo, var, param_name, callback):
        """Handle scalar selection change."""
        var.set(combo.current())
        callback()


class BaseStewartSimulator:
    """
    Base Stewart Platform Simulator with modular GUI.

    Subclasses define layout via get_layout_config().
    """

    def __init__(self, root, controller_config: ControllerConfig):
        self.root = root
        self.controller_config = controller_config

        controller_name = controller_config.get_controller_name()
        self.root.title(f"Stewart Platform - {controller_name} Ball Balancing Control")
        self.root.geometry("1400x900")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

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

        # Initialize simulation components
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
        self.servos = [FirstOrderServo(K=1.0, tau=SimulationConfig.DEFAULT_SERVO_TAU,
                                       delay=SimulationConfig.DEFAULT_SERVO_DELAY)
                       for _ in range(6)]

        self.ball_physics = SimpleBallPhysics2D(
            ball_radius=0.04,
            ball_mass=0.0027,
            gravity=9.81,
            rolling_friction=0.001,
            sphere_type='hollow'
        )

        # Controller state
        self.controller = None
        self.controller_enabled = tk.BooleanVar(value=False)

        # Pattern state with parameter tracking
        self.current_pattern = PatternFactory.create('static', x=0.0, y=0.0)
        self.pattern_type = tk.StringVar(value='static')
        self.pattern_start_time = 0.0
        self.pattern_params = {}  # Store current pattern parameters

        # Ball state
        ball_start_height = (self.ik.home_height_top_surface / 1000) + self.ball_physics.radius
        self.ball_pos = torch.tensor([[0.0, 0.0, ball_start_height]], dtype=torch.float32)
        self.ball_vel = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        self.ball_omega = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

        # Simulation state
        self.simulation_running = False
        self.simulation_time = 0.0
        self.last_update_time = None
        self.update_rate_ms = SimulationConfig.UPDATE_RATE_MS
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
            'rx': (-MAX_TILT_ANGLE_DEG, MAX_TILT_ANGLE_DEG, 0.1, 0.0, "Roll (°)"),
            'ry': (-MAX_TILT_ANGLE_DEG, MAX_TILT_ANGLE_DEG, 0.1, 0.0, "Pitch (°)"),
            'rz': (-MAX_TILT_ANGLE_DEG, MAX_TILT_ANGLE_DEG, 0.1, 0.0, "Yaw (°)")
        }

        # Platform angular state (for physics)
        self.prev_platform_angles = {'rx': 0.0, 'ry': 0.0}
        self.platform_angular_vel = {'rx': 0.0, 'ry': 0.0}
        self.platform_angular_accel = {'rx': 0.0, 'ry': 0.0}

        # Tracking variables for GUI updates
        self.last_cmd_angles = np.zeros(6)
        self.last_fk_translation = np.zeros(3)
        self.last_fk_rotation = np.zeros(3)

        self.update_timer = None

        # Create controller parameter widgets first (needed for GUI builder)
        self._create_controller_param_widgets()

        # Build modular GUI
        self._build_modular_gui()

        # Initialize controller
        self._initialize_controller()

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

    def _create_controller_param_widgets(self):
        """Create controller parameter widgets (for controller module)."""
        controller_name = self.controller_config.get_controller_name()

        if controller_name == "PID":
            self.param_definitions = [
                ('kp', 'P (Proportional)', 3.0, 4),
                ('ki', 'I (Integral)', 1.0, 4),
                ('kd', 'D (Derivative)', 3.0, 4)
            ]
        elif controller_name == "LQR":
            self.param_definitions = [
                ('Q_pos', 'Q Position Weight', 1.0, 7),
                ('Q_vel', 'Q Velocity Weight', 1.0, 6),
                ('R', 'R Control Weight', 1.0, 5)
            ]
        else:
            self.param_definitions = []

        self.controller_widgets = {
            'sliders': {},
            'value_labels': {},
            'scalar_vars': {},
            'update_fn': lambda: None,
            'param_definitions': self.param_definitions
        }

    def _build_modular_gui(self):
        """Build GUI using modular system."""
        module_registry = {
            'simulation_control': gm.SimulationControlModule,
            'controller': gm.ControllerModule,
            'trajectory_pattern': gm.TrajectoryPatternModule,
            'ball_control': gm.BallControlModule,
            'ball_state': gm.BallStateModule,
            'configuration': gm.ConfigurationModule,
            'manual_pose': gm.ManualPoseControlModule,
            'servo_angles': gm.ServoAnglesModule,
            'platform_pose': gm.PlatformPoseModule,
            'controller_output': gm.ControllerOutputModule,
            'debug_log': gm.DebugLogModule,
            'serial_connection': gm.SerialConnectionModule,
            'performance_stats': gm.PerformanceStatsModule,
        }

        layout_config = self.get_layout_config()
        callbacks = self._create_callbacks()

        self.gui_builder = GUIBuilder(self.root, module_registry)
        self.gui_modules = self.gui_builder.build(layout_config, self.colors, callbacks)

        if 'plot_panel' in self.gui_modules:
            self._create_plot(self.gui_modules['plot_panel'])

    def _create_callbacks(self):
        """Create callback dictionary for modules."""
        return {
            'start': self.start_simulation,
            'stop': self.stop_simulation,
            'reset': self.reset_simulation,
            'controller_enabled_var': self.controller_enabled,
            'toggle_controller': self.on_controller_toggle,
            'param_change': self.on_controller_param_change,
            'pattern_change': self.on_pattern_change,
            'pattern_reset': self.reset_pattern,
            'pattern_param_change': self.on_pattern_param_change,  # Centralized here
            'reset_ball': self.reset_ball,
            'push_ball': self.push_ball,
            'toggle_offset': self.on_offset_toggle,
            'slider_change': self.on_slider_change,
        }

    def on_pattern_param_change(self, param_name, value):
        """
        Centralized pattern parameter update handler.
        Called when pattern sliders change - updates pattern with new parameters.
        """
        pattern_type = self.pattern_type.get()

        # Update stored parameters
        self.pattern_params[param_name] = value

        # Create new pattern with updated parameters
        if pattern_type == 'circle':
            radius = self.pattern_params.get('radius', 50.0)
            period = self.pattern_params.get('period', 10.0)
            self.current_pattern = PatternFactory.create('circle',
                                                         radius=radius,
                                                         period=period,
                                                         clockwise=True)

        elif pattern_type == 'figure8':
            width = self.pattern_params.get('width', 60.0)
            height = self.pattern_params.get('height', 40.0)
            period = self.pattern_params.get('period', 12.0)
            self.current_pattern = PatternFactory.create('figure8',
                                                         width=width,
                                                         height=height,
                                                         period=period)

        elif pattern_type == 'star':
            radius = self.pattern_params.get('radius', 60.0)
            period = self.pattern_params.get('period', 15.0)
            self.current_pattern = PatternFactory.create('star',
                                                         radius=radius,
                                                         period=period)

        # Reset pattern timing and update visualization
        self.reset_pattern()
        self.update_plot()

    @abstractmethod
    def get_layout_config(self):
        """Return layout configuration for this simulator."""
        raise NotImplementedError

    def _create_plot(self, parent):
        """Create matplotlib plot."""
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
        self.ax.set_title('Ball Position (Top View)', color=self.colors['fg'],
                          fontsize=11, fontweight='bold')
        self.ax.grid(True, alpha=0.2, linestyle='--', color=self.colors['fg'])
        self.ax.set_aspect('equal')
        self.ax.tick_params(colors=self.colors['fg'])

        for spine in self.ax.spines.values():
            spine.set_color(self.colors['border'])

        platform_square = Rectangle((-PLATFORM_HALF_SIZE_MM, -PLATFORM_HALF_SIZE_MM),
                                    PLATFORM_HALF_SIZE_MM * 2, PLATFORM_HALF_SIZE_MM * 2,
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

        legend = self.ax.legend(loc='upper right', fontsize=8,
                                facecolor=self.colors['panel_bg'],
                                edgecolor=self.colors['border'],
                                labelcolor=self.colors['fg'])
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

    def log(self, message):
        """Add message to debug log."""
        if 'debug_log' in self.gui_modules:
            self.gui_modules['debug_log'].log(message, self.simulation_time)

    def update_gui_modules(self):
        """Update all GUI modules with current state."""
        ball_x_mm = self.ball_pos[0, 0].item() * 1000
        ball_y_mm = self.ball_pos[0, 1].item() * 1000
        vel_x_mm = self.ball_vel[0, 0].item() * 1000
        vel_y_mm = self.ball_vel[0, 1].item() * 1000

        state = {
            'simulation_time': self.simulation_time,
            'controller_enabled': self.controller_enabled.get(),
            'ball_pos': (ball_x_mm, ball_y_mm),
            'ball_vel': (vel_x_mm, vel_y_mm),
            'dof_values': self.dof_values,
            'cmd_angles': self.last_cmd_angles,
            'actual_angles': [s.get_angle() for s in self.servos],
            'fk_translation': self.last_fk_translation,
            'fk_rotation': self.last_fk_rotation,
        }

        if self.controller_enabled.get():
            rx = self.dof_values['rx']
            ry = self.dof_values['ry']
            magnitude = np.sqrt(rx ** 2 + ry ** 2)
            magnitude_percent = (magnitude / MAX_TILT_ANGLE_DEG) * 100

            state['controller_output'] = (rx, ry)
            state['controller_magnitude'] = (magnitude, magnitude_percent)

            pattern_time = self.simulation_time - self.pattern_start_time
            target_x, target_y = self.current_pattern.get_position(pattern_time)
            error_x = ball_x_mm - target_x
            error_y = ball_y_mm - target_y
            state['controller_error'] = (error_x, error_y)

        rx = self.dof_values['rx']
        ry = self.dof_values['ry']
        _, _, magnitude = clip_tilt_vector(rx, ry, MAX_TILT_ANGLE_DEG)
        state['tilt_magnitude'] = magnitude

        pattern_configs = {
            'static': "Tracking: Center (0, 0)",
            'circle': "Tracking: Circle (r=50mm, T=10s)",
            'figure8': "Tracking: Figure-8 (60×40mm, T=12s)",
            'star': "Tracking: 5-Point Star (r=60mm, T=15s)"
        }
        state['pattern_info'] = pattern_configs.get(self.pattern_type.get(), "")

        self.gui_builder.update_modules(state)

    def on_controller_toggle(self):
        """Handle controller enable/disable."""
        enabled = self.controller_enabled.get()

        if enabled:
            self.controller.reset()
            self.reset_pattern()

            controller_name = self.controller_config.get_controller_name()
            self.log(f"{controller_name} control ENABLED")

            if 'manual_pose' in self.gui_modules:
                manual_pose = self.gui_modules['manual_pose']
                manual_pose.sliders['rx'].config(state='disabled')
                manual_pose.sliders['ry'].config(state='disabled')

            self.dof_values['rx'] = 0.0
            self.dof_values['ry'] = 0.0

            self.calculate_ik()
        else:
            controller_name = self.controller_config.get_controller_name()
            self.log(f"{controller_name} control DISABLED")

            if 'manual_pose' in self.gui_modules:
                manual_pose = self.gui_modules['manual_pose']
                manual_pose.sliders['rx'].config(state='normal')
                manual_pose.sliders['ry'].config(state='normal')

    def on_pattern_change(self, event=None):
        """Handle pattern selection change."""
        pattern_type = self.pattern_type.get()

        # Reset parameter storage for new pattern
        self.pattern_params.clear()

        pattern_configs = {
            'static': ('static', {'x': 0.0, 'y': 0.0}),
            'circle': ('circle', {'radius': 50.0, 'period': 10.0, 'clockwise': True}),
            'figure8': ('figure8', {'width': 60.0, 'height': 40.0, 'period': 12.0}),
            'star': ('star', {'radius': 60.0, 'period': 15.0})
        }

        if pattern_type in pattern_configs:
            pattern_name, params = pattern_configs[pattern_type]

            # Store initial parameters
            for key, value in params.items():
                if key != 'clockwise':  # Don't store non-adjustable params
                    self.pattern_params[key] = value

            self.current_pattern = PatternFactory.create(pattern_name, **params)
            self.reset_pattern()
            self.update_plot()
            self.log(f"Pattern changed to: {pattern_type}")

    def reset_pattern(self):
        """Reset pattern timing."""
        self.pattern_start_time = self.simulation_time
        self.current_pattern.reset()
        self.log(f"Pattern reset at t={format_time(self.simulation_time)}")

        if self.controller_enabled.get():
            self.controller.reset()

    def reset_ball(self):
        """Reset ball to center."""
        home_z = self.ik.home_height_top_surface if self.use_top_surface_offset.get() else self.ik.home_height
        ball_start_height = (home_z / 1000) + self.ball_physics.radius

        self.ball_pos = torch.tensor([[0.0, 0.0, ball_start_height]], dtype=torch.float32)
        self.ball_vel = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        self.ball_omega = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

        if self.controller_enabled.get():
            self.controller.reset()

        self.update_plot()
        self.log("Ball reset to center")

    def push_ball(self):
        """Apply random velocity to ball."""
        vx = np.random.uniform(-0.05, 0.05)
        vy = np.random.uniform(-0.05, 0.05)
        self.ball_vel = torch.tensor([[vx, vy, 0.0]], dtype=torch.float32)
        self.log(f"Ball pushed: vx={vx:.3f}, vy={vy:.3f} m/s")

    def on_offset_toggle(self):
        """Handle top surface offset toggle."""
        enabled = self.use_top_surface_offset.get()
        home_z = self.ik.home_height_top_surface if enabled else self.ik.home_height

        if 'manual_pose' in self.gui_modules:
            manual_pose = self.gui_modules['manual_pose']
            z_config = self.dof_config['z']
            new_config = (home_z - 30, home_z + 30, z_config[2], home_z, z_config[4])
            self.dof_config['z'] = new_config

            manual_pose.sliders['z'].config(from_=home_z - 30, to=home_z + 30)

        self.dof_values['z'] = home_z

        ball_start_height = (home_z / 1000) + self.ball_physics.radius
        self.ball_pos[0, 2] = ball_start_height
        self.log(f"Offset: {'Top Surface' if enabled else 'Anchor Center'}")

    def on_slider_change(self, dof, value):
        """Handle manual DOF slider change."""
        val = float(value)
        self.dof_values[dof] = val

        if self.update_timer is not None:
            self.root.after_cancel(self.update_timer)
        self.update_timer = self.root.after(50, self.calculate_ik)

    def calculate_ik(self):
        """Calculate inverse kinematics for current pose."""
        translation = np.array([self.dof_values['x'],
                                self.dof_values['y'],
                                self.dof_values['z']])

        rx_limited, ry_limited, tilt_mag = clip_tilt_vector(
            self.dof_values['rx'],
            self.dof_values['ry'],
            MAX_TILT_ANGLE_DEG
        )

        if tilt_mag > MAX_TILT_ANGLE_DEG and not self.controller_enabled.get():
            self.dof_values['rx'] = rx_limited
            self.dof_values['ry'] = ry_limited

        rotation = np.array([rx_limited, ry_limited, self.dof_values['rz']])

        angles = self.ik.calculate_servo_angles(translation, rotation,
                                                self.use_top_surface_offset.get())

        if angles is not None:
            self.last_cmd_angles = angles
            if self.simulation_running:
                for i, servo in enumerate(self.servos):
                    servo.send_command(angles[i], self.simulation_time)

    def start_simulation(self):
        """Start simulation loop."""
        self.simulation_running = True
        self.last_update_time = time.time()
        self.log("Simulation started")

        if 'simulation_control' in self.gui_modules:
            sim_ctrl = self.gui_modules['simulation_control']
            sim_ctrl.start_btn.config(state='disabled')
            sim_ctrl.stop_btn.config(state='normal')

        self.simulation_loop()

    def stop_simulation(self):
        """Stop simulation loop."""
        self.simulation_running = False

        if self.simulation_loop_id is not None:
            self.root.after_cancel(self.simulation_loop_id)
            self.simulation_loop_id = None

        if 'simulation_control' in self.gui_modules:
            sim_ctrl = self.gui_modules['simulation_control']
            sim_ctrl.start_btn.config(state='normal')
            sim_ctrl.stop_btn.config(state='disabled')

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

        for dof, (_, _, _, default, _) in self.dof_config.items():
            if dof == 'z':
                home_z = (self.ik.home_height_top_surface if self.use_top_surface_offset.get()
                          else self.ik.home_height)
                self.dof_values[dof] = home_z
            else:
                self.dof_values[dof] = default

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

            if self.controller_enabled.get():
                try:
                    ball_x_mm = self.ball_pos[0, 0].item() * 1000
                    ball_y_mm = self.ball_pos[0, 1].item() * 1000
                    ball_vx_mm_s = self.ball_vel[0, 0].item() * 1000
                    ball_vy_mm_s = self.ball_vel[0, 1].item() * 1000

                    pattern_time = self.simulation_time - self.pattern_start_time
                    target_x, target_y = self.current_pattern.get_position(pattern_time)
                    target_pos_mm = (target_x, target_y)

                    rx_raw, ry_raw = self._update_controller(
                        (ball_x_mm, ball_y_mm),
                        (ball_vx_mm_s, ball_vy_mm_s),
                        target_pos_mm,
                        dt
                    )

                    rx, ry, tilt_mag = clip_tilt_vector(rx_raw, ry_raw, MAX_TILT_ANGLE_DEG)

                    if tilt_mag > MAX_TILT_ANGLE_DEG:
                        controller_name = self.controller_config.get_controller_name()
                        self.log(f"{controller_name} output clipped: "
                                 f"({rx_raw:.2f}, {ry_raw:.2f}) → ({rx:.2f}, {ry:.2f})")

                    self.dof_values['rx'] = rx
                    self.dof_values['ry'] = ry

                    translation = np.array([self.dof_values['x'],
                                            self.dof_values['y'],
                                            self.dof_values['z']])
                    rotation = np.array([rx, ry, self.dof_values['rz']])

                    angles = self.ik.calculate_servo_angles(translation, rotation,
                                                            self.use_top_surface_offset.get())

                    if angles is not None:
                        self.last_cmd_angles = angles
                        for i in range(6):
                            self.servos[i].send_command(angles[i], self.simulation_time)
                    else:
                        controller_name = self.controller_config.get_controller_name()
                        self.log(f"{controller_name}: IK solution out of range")

                except Exception as e:
                    controller_name = self.controller_config.get_controller_name()
                    error_msg = format_error_context(
                        self.simulation_time,
                        (ball_x_mm, ball_y_mm),
                        (ball_vx_mm_s, ball_vy_mm_s),
                        str(e)
                    )
                    self.log(f"{controller_name} error:\n{error_msg}")
                    self.controller_enabled.set(False)
                    self.on_controller_toggle()

            for servo in self.servos:
                servo.update(dt, self.simulation_time)

            actual_angles = np.array([servo.get_angle() for servo in self.servos])

            translation, rotation, success, _ = self.ik.calculate_forward_kinematics(
                actual_angles, use_top_surface_offset=self.use_top_surface_offset.get()
            )

            if success:
                self.last_fk_translation = translation
                self.last_fk_rotation = rotation

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

                except Exception as e:
                    self.log(format_error_context(
                        self.simulation_time,
                        self.ball_pos[0, :2],
                        self.ball_vel[0, :2] * 1000,
                        f"Physics error: {str(e)}"
                    ))
                    self.reset_ball()

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

            self.update_gui_modules()
            self.update_plot()

        self.last_update_time = current_time
        self.simulation_loop_id = self.root.after(self.update_rate_ms, self.simulation_loop)

    @abstractmethod
    def _initialize_controller(self):
        """Initialize controller (implemented by subclass)."""
        pass

    @abstractmethod
    def _update_controller(self, ball_pos_mm, ball_vel_mm_s, target_pos_mm, dt):
        """Update controller and return control output (implemented by subclass)."""
        pass

    def on_controller_param_change(self):
        """Callback when controller parameters change."""
        pass

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