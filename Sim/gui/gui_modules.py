#!/usr/bin/env python3
"""
Modular GUI Components for Stewart Platform Simulators

Each module is a self-contained, reusable widget panel.
Modules communicate with the main simulator through callbacks.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import serial.tools.list_ports

from core.utils import format_time, format_vector_2d, MAX_TILT_ANGLE_DEG


class GUIModule:
    """Base class for all GUI modules."""

    def __init__(self, parent, colors, callbacks=None):
        """
        Args:
            parent: Parent tkinter widget
            colors: Color scheme dict
            callbacks: Dict of callback functions to simulator
        """
        self.parent = parent
        self.colors = colors
        self.callbacks = callbacks or {}
        self.frame = None

    def create(self):
        """Create and return the module's frame."""
        raise NotImplementedError

    def update(self, state):
        """Update module with new state."""
        pass


class SimulationControlModule(GUIModule):
    """Start/Stop/Reset simulation controls."""

    def create(self):
        self.frame = ttk.LabelFrame(self.parent, text="Simulation Control", padding=10)

        btn_frame = ttk.Frame(self.frame)
        btn_frame.pack(fill='x')

        self.start_btn = ttk.Button(btn_frame, text="▶ Start",
                                    command=self.callbacks.get('start'),
                                    width=10)
        self.start_btn.pack(side='left', padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="⏸ Stop",
                                   command=self.callbacks.get('stop'),
                                   state='disabled', width=10)
        self.stop_btn.pack(side='left', padx=5)

        self.reset_btn = ttk.Button(btn_frame, text="↻ Reset",
                                    command=self.callbacks.get('reset'),
                                    width=10)
        self.reset_btn.pack(side='left', padx=5)

        self.time_label = ttk.Label(self.frame, text="Time: 0.00s",
                                    font=('Consolas', 10, 'bold'))
        self.time_label.pack(pady=(10, 0))

        return self.frame

    def update(self, state):
        if 'simulation_time' in state:
            self.time_label.config(text=f"Time: {format_time(state['simulation_time'])}")


class ControllerModule(GUIModule):
    """Controller enable/disable and parameter tuning."""

    def __init__(self, parent, colors, callbacks, controller_config, controller_widgets):
        super().__init__(parent, colors, callbacks)
        self.controller_config = controller_config
        self.controller_widgets = controller_widgets
        self.controller_enabled = callbacks.get('controller_enabled_var')

    def create(self):
        controller_name = self.controller_config.get_controller_name()
        self.frame = ttk.LabelFrame(self.parent, text=f"{controller_name} Ball Balancing", padding=10)

        enable_frame = ttk.Frame(self.frame)
        enable_frame.pack(fill='x', pady=(0, 10))

        ttk.Checkbutton(enable_frame, text=f"Enable {controller_name} Control",
                        variable=self.controller_enabled,
                        command=self.callbacks.get('toggle_controller')).pack(side='left')

        self.status_label = ttk.Label(enable_frame, text="●",
                                      foreground=self.colors['border'],
                                      font=('Segoe UI', 14))
        self.status_label.pack(side='left', padx=(10, 0))

        if 'param_definitions' in self.controller_widgets:
            param_definitions = self.controller_widgets['param_definitions']
            sliders = self.controller_widgets['sliders']
            value_labels = self.controller_widgets['value_labels']
            scalar_vars = self.controller_widgets['scalar_vars']

            for param_def in param_definitions:
                if len(param_def) == 4:
                    param_name, label, default, default_scalar_idx = param_def
                else:
                    param_name, label, default = param_def
                    default_scalar_idx = 4

                old_idx = getattr(self.controller_config, 'default_scalar_idx', 4)
                self.controller_config.default_scalar_idx = default_scalar_idx

                self.controller_config.create_parameter_slider(
                    self.frame, param_name, label, default,
                    sliders, value_labels, scalar_vars,
                    self.callbacks.get('param_change')
                )

                self.controller_config.default_scalar_idx = old_idx

        return self.frame

    def update(self, state):
        if 'controller_enabled' in state and state['controller_enabled']:
            self.status_label.config(foreground=self.colors['success'])
        else:
            self.status_label.config(foreground=self.colors['border'])


class TrajectoryPatternModule(GUIModule):
    """Trajectory pattern selection with dynamic parameter controls."""

    def __init__(self, parent, colors, callbacks, pattern_var):
        super().__init__(parent, colors, callbacks)
        self.pattern_var = pattern_var
        self.param_widgets = {}

    def create(self):
        self.frame = ttk.LabelFrame(self.parent, text="Trajectory Pattern", padding=10)

        selector_frame = ttk.Frame(self.frame)
        selector_frame.pack(fill='x', pady=(0, 5))

        ttk.Label(selector_frame, text="Pattern:",
                  font=('Segoe UI', 9)).pack(side='left', padx=(0, 10))

        pattern_combo = ttk.Combobox(selector_frame, textvariable=self.pattern_var,
                                     width=15, state='readonly',
                                     values=['static', 'circle', 'figure8', 'star'])
        pattern_combo.pack(side='left', padx=5)
        pattern_combo.bind('<<ComboboxSelected>>', self._on_pattern_change)

        ttk.Button(selector_frame, text="Reset",
                   command=self.callbacks.get('pattern_reset'),
                   width=8).pack(side='left', padx=5)

        self.info_label = ttk.Label(self.frame,
                                    text="Tracking: Center (0, 0)",
                                    font=('Consolas', 8),
                                    foreground=self.colors['success'])
        self.info_label.pack(anchor='w', pady=(5, 0))

        self.params_container = ttk.Frame(self.frame)
        self.params_container.pack(fill='x', pady=(10, 0))

        self._update_pattern_params()

        return self.frame

    def _on_pattern_change(self, event=None):
        self._update_pattern_params()
        if self.callbacks.get('pattern_change'):
            self.callbacks['pattern_change']()

    def _update_pattern_params(self):
        """Update parameter sliders based on selected pattern."""
        for widget in self.params_container.winfo_children():
            widget.destroy()
        self.param_widgets.clear()

        pattern_type = self.pattern_var.get()

        pattern_params = {
            'static': [],
            'circle': [
                ('radius', 'Radius (mm)', 10.0, 100.0, 50.0, 1.0),
                ('period', 'Period (s)', 3.0, 30.0, 10.0, 0.5)
            ],
            'figure8': [
                ('width', 'Width (mm)', 10.0, 150.0, 60.0, 1.0),
                ('height', 'Height (mm)', 10.0, 100.0, 40.0, 1.0),
                ('period', 'Period (s)', 3.0, 30.0, 12.0, 0.5)
            ],
            'star': [
                ('radius', 'Radius (mm)', 10.0, 100.0, 60.0, 1.0),
                ('period', 'Period (s)', 3.0, 30.0, 15.0, 0.5)
            ]
        }

        params = pattern_params.get(pattern_type, [])

        if not params:
            ttk.Label(self.params_container,
                     text="No adjustable parameters",
                     font=('Segoe UI', 8, 'italic'),
                     foreground=self.colors['border']).pack(pady=5)
            return

        for param_name, label, min_val, max_val, default, resolution in params:
            self._create_param_slider(param_name, label, min_val, max_val, default, resolution)

    def _create_param_slider(self, param_name, label, min_val, max_val, default, resolution):
        frame = ttk.Frame(self.params_container)
        frame.pack(fill='x', pady=3)

        ttk.Label(frame, text=label,
                  font=('Segoe UI', 9)).grid(row=0, column=0, sticky='w', padx=(0, 5))

        slider = ttk.Scale(frame, from_=min_val, to=max_val, orient='horizontal')
        slider.grid(row=0, column=1, sticky='ew', padx=5)
        slider.set(default)

        value_label = ttk.Label(frame, text=f"{default:.1f}",
                               width=6, font=('Consolas', 9))
        value_label.grid(row=0, column=2)

        def on_change(val):
            value = float(val)
            value_label.config(text=f"{value:.1f}")
            if self.callbacks.get('pattern_param_change'):
                self.callbacks['pattern_param_change'](param_name, value)

        slider.config(command=on_change)
        frame.columnconfigure(1, weight=1)

        self.param_widgets[param_name] = {
            'slider': slider,
            'label': value_label,
            'frame': frame
        }

    def update(self, state):
        if 'pattern_info' in state:
            self.info_label.config(text=state['pattern_info'])


class BallControlModule(GUIModule):
    """Ball reset and push buttons."""

    def create(self):
        self.frame = ttk.LabelFrame(self.parent, text="Ball Control", padding=10)

        btn_frame = ttk.Frame(self.frame)
        btn_frame.pack()

        ttk.Button(btn_frame, text="Reset Ball",
                   command=self.callbacks.get('reset_ball'),
                   width=15).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Push Ball",
                   command=self.callbacks.get('push_ball'),
                   width=15).pack(side='left', padx=5)

        return self.frame


class BallStateModule(GUIModule):
    """Ball position and velocity display."""

    def create(self):
        self.frame = ttk.LabelFrame(self.parent, text="Ball State", padding=10)

        self.pos_label = ttk.Label(self.frame,
                                   text="Position: (0.0, 0.0) mm",
                                   font=('Consolas', 9))
        self.pos_label.pack(anchor='w', pady=2)

        self.vel_label = ttk.Label(self.frame,
                                   text="Velocity: (0.0, 0.0) mm/s",
                                   font=('Consolas', 9))
        self.vel_label.pack(anchor='w', pady=2)

        return self.frame

    def update(self, state):
        if 'ball_pos' in state:
            self.pos_label.config(text=f"Position: {format_vector_2d(state['ball_pos'])}")
        if 'ball_vel' in state:
            if isinstance(state['ball_vel'], tuple) and len(state['ball_vel']) == 2:
                if isinstance(state['ball_vel'][0], str):
                    self.vel_label.config(text=f"Status: {state['ball_vel'][0]}")
                else:
                    self.vel_label.config(text=f"Velocity: {format_vector_2d(state['ball_vel'], 'mm/s')}")
            elif isinstance(state['ball_vel'], str):
                self.vel_label.config(text=f"Status: {state['ball_vel']}")


class ConfigurationModule(GUIModule):
    """Configuration options."""

    def __init__(self, parent, colors, callbacks, use_offset_var):
        super().__init__(parent, colors, callbacks)
        self.use_offset_var = use_offset_var

    def create(self):
        self.frame = ttk.LabelFrame(self.parent, text="Configuration", padding=10)

        ttk.Checkbutton(self.frame, text="Use Top Surface Offset",
                        variable=self.use_offset_var,
                        command=self.callbacks.get('toggle_offset')).pack(anchor='w')

        return self.frame


class ManualPoseControlModule(GUIModule):
    """6 DOF manual control sliders."""

    def __init__(self, parent, colors, callbacks, dof_config):
        super().__init__(parent, colors, callbacks)
        self.dof_config = dof_config
        self.sliders = {}
        self.value_labels = {}

    def create(self):
        self.frame = ttk.LabelFrame(self.parent, text="Manual Pose Control (6 DOF)", padding=10)

        for idx, (dof, (min_val, max_val, res, default, label)) in enumerate(self.dof_config.items()):
            ttk.Label(self.frame, text=label,
                      font=('Segoe UI', 9)).grid(row=idx, column=0, sticky='w', pady=8)

            slider = ttk.Scale(self.frame, from_=min_val, to=max_val,
                               orient='horizontal',
                               command=lambda val, d=dof: self._on_slider_change(d, val))
            slider.grid(row=idx, column=1, sticky='ew', padx=10, pady=8)
            self.sliders[dof] = slider

            value_label = ttk.Label(self.frame, text=f"{default:.2f}",
                                    width=8, font=('Consolas', 9))
            value_label.grid(row=idx, column=2, pady=8)
            self.value_labels[dof] = value_label
            slider.set(default)

        self.frame.columnconfigure(1, weight=1)

        tilt_info_frame = ttk.Frame(self.frame)
        tilt_info_frame.grid(row=len(self.dof_config), column=0,
                             columnspan=3, sticky='ew', pady=(10, 5))

        ttk.Label(tilt_info_frame, text="Tilt Vector:",
                  font=('Segoe UI', 9, 'bold')).pack(side='left', padx=(0, 10))
        self.tilt_magnitude_label = ttk.Label(tilt_info_frame,
                                              text="0.00° (0.0%)",
                                              font=('Consolas', 9),
                                              foreground=self.colors['success'])
        self.tilt_magnitude_label.pack(side='left')

        return self.frame

    def _on_slider_change(self, dof, value):
        val = float(value)
        self.value_labels[dof].config(text=f"{val:.2f}")

        if self.callbacks.get('slider_change'):
            self.callbacks['slider_change'](dof, val)

    def update(self, state):
        if 'dof_values' in state:
            for dof, val in state['dof_values'].items():
                if dof in self.value_labels:
                    self.value_labels[dof].config(text=f"{val:.2f}")

        if 'tilt_magnitude' in state:
            mag = state['tilt_magnitude']
            percent = (mag / MAX_TILT_ANGLE_DEG) * 100

            if percent > 80:
                color = self.colors['warning']
            elif percent > 60:
                color = '#ffa500'
            else:
                color = self.colors['success']

            self.tilt_magnitude_label.config(
                text=f"{mag:.2f}° ({percent:.1f}%)",
                foreground=color
            )


class ServoAnglesModule(GUIModule):
    """Display commanded and actual servo angles."""

    def __init__(self, parent, colors, callbacks, show_actual=True):
        super().__init__(parent, colors, callbacks)
        self.show_actual = show_actual

    def create(self):
        container = ttk.Frame(self.parent)

        cmd_frame = ttk.LabelFrame(container, text="Commanded Servo Angles (IK)", padding=10)
        cmd_frame.pack(fill='x', pady=(0, 10))

        self.cmd_labels = []
        for i in range(6):
            label = ttk.Label(cmd_frame, text=f"S{i + 1}: 0.00°",
                              font=('Consolas', 9))
            label.grid(row=i // 3, column=i % 3, padx=15, pady=5, sticky='w')
            self.cmd_labels.append(label)

        if self.show_actual:
            actual_frame = ttk.LabelFrame(container, text="Actual Servo Angles", padding=10)
            actual_frame.pack(fill='x')

            self.actual_labels = []
            for i in range(6):
                label = ttk.Label(actual_frame, text=f"S{i + 1}: 0.00°",
                                  font=('Consolas', 9))
                label.grid(row=i // 3, column=i % 3, padx=15, pady=5, sticky='w')
                self.actual_labels.append(label)

        self.frame = container
        return self.frame

    def update(self, state):
        if 'cmd_angles' in state:
            for i, angle in enumerate(state['cmd_angles']):
                self.cmd_labels[i].config(text=f"S{i + 1}: {angle:6.2f}°")

        if self.show_actual and 'actual_angles' in state:
            for i, angle in enumerate(state['actual_angles']):
                self.actual_labels[i].config(text=f"S{i + 1}: {angle:6.2f}°")


class PlatformPoseModule(GUIModule):
    """Display platform pose from forward kinematics."""

    def create(self):
        self.frame = ttk.LabelFrame(self.parent, text="Platform Pose (FK)", padding=10)

        self.pos_label = ttk.Label(self.frame, text="X: 0.00  Y: 0.00  Z: 0.00 mm",
                                   font=('Consolas', 9))
        self.pos_label.pack(anchor='w', pady=2)

        self.rot_label = ttk.Label(self.frame, text="Roll: 0.00  Pitch: 0.00  Yaw: 0.00°",
                                   font=('Consolas', 9))
        self.rot_label.pack(anchor='w', pady=2)

        return self.frame

    def update(self, state):
        if 'fk_translation' in state:
            t = state['fk_translation']
            self.pos_label.config(
                text=f"X: {t[0]:6.2f}  Y: {t[1]:6.2f}  Z: {t[2]:6.2f} mm"
            )

        if 'fk_rotation' in state:
            r = state['fk_rotation']
            self.rot_label.config(
                text=f"Roll: {r[0]:6.2f}  Pitch: {r[1]:6.2f}  Yaw: {r[2]:6.2f}°"
            )


class ControllerOutputModule(GUIModule):
    """Display controller output and error."""

    def __init__(self, parent, colors, callbacks, controller_name):
        super().__init__(parent, colors, callbacks)
        self.controller_name = controller_name

    def create(self):
        self.frame = ttk.LabelFrame(self.parent, text=f"{self.controller_name} Output", padding=10)

        self.output_label = ttk.Label(self.frame,
                                      text="Tilt: rx=0.00°  ry=0.00°",
                                      font=('Consolas', 9))
        self.output_label.pack(anchor='w', pady=2)

        self.magnitude_label = ttk.Label(self.frame,
                                         text="Magnitude: 0.00° (0%)",
                                         font=('Consolas', 9))
        self.magnitude_label.pack(anchor='w', pady=2)

        self.error_label = ttk.Label(self.frame,
                                     text="Error: (0.0, 0.0) mm",
                                     font=('Consolas', 9))
        self.error_label.pack(anchor='w', pady=2)

        return self.frame

    def update(self, state):
        if 'controller_output' in state:
            rx, ry = state['controller_output']
            self.output_label.config(text=f"Tilt: rx={rx:.2f}°  ry={ry:.2f}°")

        if 'controller_magnitude' in state:
            mag, percent = state['controller_magnitude']
            self.magnitude_label.config(text=f"Magnitude: {mag:.2f}° ({percent:.1f}%)")

        if 'controller_error' in state:
            error = state['controller_error']
            self.error_label.config(text=f"Error: {format_vector_2d(error)}")


class DebugLogModule(GUIModule):
    """Debug log display."""

    def __init__(self, parent, colors, callbacks, height=10):
        super().__init__(parent, colors, callbacks)
        self.height = height

    def create(self):
        self.frame = ttk.LabelFrame(self.parent, text="Debug Log", padding=10)

        self.log_text = scrolledtext.ScrolledText(
            self.frame,
            height=self.height,
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

        return self.frame

    def log(self, message, timestamp=None):
        if timestamp is not None:
            msg = f"[{format_time(timestamp)}] {message}\n"
        else:
            msg = f"{message}\n"
        self.log_text.insert(tk.END, msg)
        self.log_text.see(tk.END)


class SerialConnectionModule(GUIModule):
    """Serial port connection for hardware."""

    def __init__(self, parent, colors, callbacks, port_var):
        super().__init__(parent, colors, callbacks)
        self.port_var = port_var
        self.port_combo = None
        self.port_status_label = None

    def create(self):
        self.frame = ttk.LabelFrame(self.parent, text="Serial Connection", padding=10)

        port_frame = ttk.Frame(self.frame)
        port_frame.pack(fill='x', pady=(0, 5))

        ttk.Label(port_frame, text="Port:").pack(side='left', padx=(0, 5))

        self.port_combo = ttk.Combobox(port_frame, textvariable=self.port_var,
                                       width=15, state='readonly')
        self.port_combo.pack(side='left', padx=5)

        ttk.Button(port_frame, text="Refresh",
                   command=self._refresh_ports,
                   width=8).pack(side='left')

        self.port_status_label = ttk.Label(self.frame, text="",
                                           font=('Consolas', 8),
                                           foreground=self.colors['fg'])
        self.port_status_label.pack(fill='x', pady=(0, 5))

        btn_frame = ttk.Frame(self.frame)
        btn_frame.pack(fill='x', pady=(5, 0))

        self.connect_btn = ttk.Button(btn_frame, text="Connect",
                                      command=self.callbacks.get('connect'),
                                      width=12)
        self.connect_btn.pack(side='left', padx=5)

        self.disconnect_btn = ttk.Button(btn_frame, text="Disconnect",
                                         command=self.callbacks.get('disconnect'),
                                         state='disabled', width=12)
        self.disconnect_btn.pack(side='left', padx=5)

        self.status_label = ttk.Label(self.frame, text="Not connected",
                                      foreground=self.colors['border'])
        self.status_label.pack(pady=(5, 0))

        self._refresh_ports()

        return self.frame

    def _refresh_ports(self):
        try:
            ports = list(serial.tools.list_ports.comports())
            port_names = [port.device for port in ports]

            self.port_combo['values'] = port_names

            if port_names:
                self.port_combo.current(0)
                self.port_status_label.config(
                    text=f"Found {len(port_names)} port(s)",
                    foreground=self.colors['success']
                )
                self.connect_btn.config(state='normal')
            else:
                self.port_status_label.config(
                    text="No serial ports found",
                    foreground=self.colors['warning']
                )
                self.connect_btn.config(state='disabled')

        except Exception as e:
            self.port_status_label.config(
                text=f"Error: {str(e)}",
                foreground=self.colors['warning']
            )
            self.connect_btn.config(state='disabled')

    def update(self, state):
        if 'connected' in state:
            if state['connected']:
                self.status_label.config(text="Connected",
                                         foreground=self.colors['success'])
                self.connect_btn.config(state='disabled')
                self.disconnect_btn.config(state='normal')
            else:
                self.status_label.config(text="Not connected",
                                         foreground=self.colors['border'])
                self.connect_btn.config(state='normal')
                self.disconnect_btn.config(state='disabled')


class PerformanceStatsModule(GUIModule):
    """Performance statistics for 100Hz hardware mode."""

    def create(self):
        self.frame = ttk.LabelFrame(self.parent, text="100Hz MODE", padding=10)

        self.fps_label = ttk.Label(self.frame, text="Control Loop: 0 Hz",
                                   font=('Consolas', 10, 'bold'),
                                   foreground=self.colors['success'])
        self.fps_label.pack()

        self.cache_label = ttk.Label(self.frame, text="IK Cache: 0.0%",
                                     font=('Consolas', 9))
        self.cache_label.pack()

        self.timeout_label = ttk.Label(self.frame, text="IK Timeouts: 0",
                                       font=('Consolas', 9))
        self.timeout_label.pack()

        ttk.Button(self.frame, text="Show Statistics",
                   command=self.callbacks.get('show_stats')).pack(pady=(5, 0))

        return self.frame

    def update(self, state):
        if 'fps' in state:
            self.fps_label.config(text=f"Control: {state['fps']:.1f} Hz")

        if 'cache_hit_rate' in state:
            self.cache_label.config(text=f"IK Cache: {state['cache_hit_rate'] * 100:.1f}%")

        if 'ik_timeouts' in state:
            self.timeout_label.config(text=f"IK Timeouts: {state['ik_timeouts']}")


class BallFilterModule(GUIModule):
    """Ball position EMA filter control."""

    def __init__(self, parent, colors, callbacks, ball_filter):
        super().__init__(parent, colors, callbacks)
        self.ball_filter = ball_filter

    def create(self):
        self.frame = ttk.LabelFrame(self.parent, text="Ball Position Filter (EMA)", padding=10)

        slider_frame = ttk.Frame(self.frame)
        slider_frame.pack(fill='x')

        ttk.Label(slider_frame, text="α:",
                  font=('Segoe UI', 9, 'bold')).pack(side='left', padx=(0, 5))

        self.alpha_slider = ttk.Scale(
            slider_frame, from_=0.0, to=1.0, orient='horizontal',
            command=self._on_alpha_change
        )
        self.alpha_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.alpha_value_label = ttk.Label(
            slider_frame, text=f"{self.ball_filter.get_alpha():.2f}",
            width=4, font=('Consolas', 10, 'bold'),
            foreground=self.colors['highlight']
        )
        self.alpha_value_label.pack(side='left', padx=(5, 0))

        info_label = ttk.Label(
            self.frame,
            text="0=Smooth/Lag  →  1=Raw/Responsive",
            font=('Segoe UI', 7, 'italic'),
            foreground=self.colors['border']
        )
        info_label.pack(anchor='w', pady=(3, 0))

        self.alpha_slider.set(self.ball_filter.get_alpha())

        return self.frame

    def _on_alpha_change(self, value):
        alpha = float(value)
        self.ball_filter.set_alpha(alpha)
        self.alpha_value_label.config(text=f"{alpha:.2f}")


class Pixy2CameraModule(GUIModule):
    """Pixy2 camera noise model configuration and control."""

    def __init__(self, parent, colors, callbacks, camera=None):
        super().__init__(parent, colors, callbacks)
        self.camera = camera

    def create(self):
        self.frame = ttk.LabelFrame(self.parent, text="Pixy2 Camera Model", padding=10)

        # Enable/Disable camera noise
        enable_frame = ttk.Frame(self.frame)
        enable_frame.pack(fill='x', pady=(0, 10))

        self.camera_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(enable_frame, text="Enable Camera Noise",
                        variable=self.camera_enabled,
                        command=self._on_enable_toggle).pack(side='left')

        self.status_indicator = ttk.Label(enable_frame, text="●",
                                          foreground=self.colors['success'],
                                          font=('Segoe UI', 14))
        self.status_indicator.pack(side='left', padx=(10, 0))

        # Separator
        ttk.Separator(self.frame, orient='horizontal').pack(fill='x', pady=(5, 10))

        # Pixel size parameter
        self._create_parameter_row(
            "Pixel Size:",
            min_val=0.5,
            max_val=3.0,
            default=1.4,
            resolution=0.1,
            param_name='pixel_size',
            unit='mm'
        )

        # Sub-pixel noise parameter
        self._create_parameter_row(
            "Sub-pixel Noise:",
            min_val=0.0,
            max_val=1.0,
            default=0.4,
            resolution=0.01,
            param_name='noise_std',
            unit='mm'
        )

        # Detection rate parameter
        self._create_parameter_row(
            "Detection Rate:",
            min_val=0.90,
            max_val=1.0,
            default=0.999,
            resolution=0.001,
            param_name='detection_rate',
            unit='',
            format_str="{:.3f}"
        )

        # Sample rate parameter
        self._create_parameter_row(
            "Sample Rate:",
            min_val=0.0,
            max_val=60.0,
            default=19.3,
            resolution=0.1,
            param_name='sample_rate',
            unit='Hz',
            format_str="{:.1f}"
        )

        # Info label for sample rate
        info_label = ttk.Label(
            self.frame,
            text="(0 Hz = sample every frame)",
            font=('Segoe UI', 7, 'italic'),
            foreground=self.colors['border']
        )
        info_label.pack(anchor='w', pady=(0, 10))

        # Separator
        ttk.Separator(self.frame, orient='horizontal').pack(fill='x', pady=(5, 10))

        # Camera statistics display
        stats_frame = ttk.Frame(self.frame)
        stats_frame.pack(fill='x', pady=(0, 5))

        ttk.Label(stats_frame, text="Camera Stats:",
                  font=('Segoe UI', 9, 'bold')).pack(anchor='w', pady=(0, 5))

        self.measurement_label = ttk.Label(stats_frame,
                                           text="Raw: (0.0, 0.0) mm",
                                           font=('Consolas', 8))
        self.measurement_label.pack(anchor='w', pady=2)

        self.quantized_label = ttk.Label(stats_frame,
                                         text="Quantized: (0.0, 0.0) mm",
                                         font=('Consolas', 8))
        self.quantized_label.pack(anchor='w', pady=2)

        self.sample_status_label = ttk.Label(stats_frame,
                                             text="Last sample: never",
                                             font=('Consolas', 8))
        self.sample_status_label.pack(anchor='w', pady=2)

        # Control buttons
        btn_frame = ttk.Frame(self.frame)
        btn_frame.pack(fill='x', pady=(10, 0))

        ttk.Button(btn_frame, text="Reset Camera",
                   command=self._on_reset_camera,
                   width=15).pack(side='left', padx=5)

        ttk.Button(btn_frame, text="Preset: Real",
                   command=self._load_real_preset,
                   width=15).pack(side='left', padx=5)

        # Preset info
        preset_info = ttk.Label(
            self.frame,
            text="Real preset: measured values from actual Pixy2",
            font=('Segoe UI', 7, 'italic'),
            foreground=self.colors['border']
        )
        preset_info.pack(anchor='w', pady=(5, 0))

        return self.frame

    def _create_parameter_row(self, label, min_val, max_val, default,
                              resolution, param_name, unit='', format_str="{:.2f}"):
        """Create a parameter control row with slider and value display."""
        frame = ttk.Frame(self.frame)
        frame.pack(fill='x', pady=5)

        ttk.Label(frame, text=label,
                  font=('Segoe UI', 9)).grid(row=0, column=0, sticky='w', padx=(0, 10))

        slider = ttk.Scale(frame, from_=min_val, to=max_val, orient='horizontal')
        slider.grid(row=0, column=1, sticky='ew', padx=5)
        slider.set(default)

        value_text = format_str.format(default) + (f" {unit}" if unit else "")
        value_label = ttk.Label(frame, text=value_text,
                                width=10, font=('Consolas', 9),
                                foreground=self.colors['highlight'])
        value_label.grid(row=0, column=2)

        # Store references
        if not hasattr(self, 'sliders'):
            self.sliders = {}
            self.value_labels = {}

        self.sliders[param_name] = slider
        self.value_labels[param_name] = value_label

        # Bind slider change
        def on_change(val):
            value = float(val)
            value_text = format_str.format(value) + (f" {unit}" if unit else "")
            value_label.config(text=value_text)
            self._on_param_change(param_name, value)

        slider.config(command=on_change)
        frame.columnconfigure(1, weight=1)

    def _on_enable_toggle(self):
        """Handle camera enable/disable."""
        enabled = self.camera_enabled.get()

        if enabled:
            self.status_indicator.config(foreground=self.colors['success'])
            # Enable all sliders
            for slider in self.sliders.values():
                slider.config(state='normal')
        else:
            self.status_indicator.config(foreground=self.colors['border'])
            # Disable all sliders
            for slider in self.sliders.values():
                slider.config(state='disabled')

        if self.callbacks.get('camera_enable_change'):
            self.callbacks['camera_enable_change'](enabled)

    def _on_param_change(self, param_name, value):
        """Handle parameter change."""
        if not self.camera or not self.camera_enabled.get():
            return

        # Update camera parameters
        if param_name == 'pixel_size':
            self.camera.pixel_size = value
        elif param_name == 'noise_std':
            self.camera.set_noise_level(value)
        elif param_name == 'detection_rate':
            self.camera.detection_rate = value
        elif param_name == 'sample_rate':
            self.camera.set_sample_rate(value)

        if self.callbacks.get('camera_param_change'):
            self.callbacks['camera_param_change'](param_name, value)

    def _on_reset_camera(self):
        """Reset camera state."""
        if self.camera:
            self.camera.reset()

        if self.callbacks.get('camera_reset'):
            self.callbacks['camera_reset']()

    def _load_real_preset(self):
        """Load real-world measured parameters."""
        # Measured values from your noise analysis
        presets = {
            'pixel_size': 1.4,
            'noise_std': 0.4,  # Tuned to match your X-axis std dev
            'detection_rate': 0.999,
            'sample_rate': 19.3
        }

        for param_name, value in presets.items():
            if param_name in self.sliders:
                self.sliders[param_name].set(value)
                # Trigger update
                self._on_param_change(param_name, value)

        if self.callbacks.get('log'):
            self.callbacks['log']("Loaded real Pixy2 camera preset")

    def update(self, state):
        """Update camera statistics display."""
        if 'camera_raw_measurement' in state:
            x, y = state['camera_raw_measurement']
            if x is not None and y is not None:
                self.measurement_label.config(
                    text=f"Raw: ({x:.2f}, {y:.2f}) mm"
                )

        if 'camera_quantized_measurement' in state:
            x, y = state['camera_quantized_measurement']
            if x is not None and y is not None:
                self.quantized_label.config(
                    text=f"Quantized: ({x:.2f}, {y:.2f}) mm"
                )

        if 'camera_last_sample_time' in state:
            time_val = state['camera_last_sample_time']
            if time_val >= 0:
                self.sample_status_label.config(
                    text=f"Last sample: {time_val:.3f}s"
                )

        if 'camera_is_new_sample' in state and state['camera_is_new_sample']:
            # Flash the status indicator briefly
            original_color = self.status_indicator.cget('foreground')
            self.status_indicator.config(foreground='#ffffff')
            self.frame.after(50, lambda: self.status_indicator.config(
                foreground=original_color
            ))


class KalmanFilterModule(GUIModule):
    """Kalman filter configuration and monitoring."""

    def __init__(self, parent, colors, callbacks, kalman_filter=None):
        super().__init__(parent, colors, callbacks)
        self.kalman_filter = kalman_filter

    def create(self):
        self.frame = ttk.LabelFrame(self.parent, text="Kalman Filter", padding=10)

        # Enable/Disable control
        enable_frame = ttk.Frame(self.frame)
        enable_frame.pack(fill='x', pady=(0, 10))

        self.filter_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(enable_frame, text="Enable Kalman Filter",
                        variable=self.filter_enabled,
                        command=self._on_enable_toggle).pack(side='left')

        self.status_indicator = ttk.Label(enable_frame, text="●",
                                          foreground=self.colors['border'],
                                          font=('Segoe UI', 14))
        self.status_indicator.pack(side='left', padx=(10, 0))

        # Separator
        ttk.Separator(self.frame, orient='horizontal').pack(fill='x', pady=(5, 10))

        # Process Noise slider
        self._create_parameter_slider(
            label="Process Noise (Q):",
            min_val=0.01,
            max_val=10.0,
            default=1.0,
            param_name='process_noise',
            tooltip="Model trust: Lower=trust model, Higher=trust measurements"
        )

        # Measurement Noise slider
        self._create_parameter_slider(
            label="Measurement Noise (R):",
            min_val=0.01,
            max_val=10.0,
            default=1.0,
            param_name='measurement_noise',
            tooltip="Camera trust: Lower=trust camera, Higher=smooth more"
        )

        # Separator
        ttk.Separator(self.frame, orient='horizontal').pack(fill='x', pady=(10, 10))

        # Filter state display
        state_frame = ttk.Frame(self.frame)
        state_frame.pack(fill='x', pady=(0, 5))

        ttk.Label(state_frame, text="Filter State:",
                  font=('Segoe UI', 9, 'bold')).pack(anchor='w', pady=(0, 5))

        self.position_label = ttk.Label(state_frame,
                                        text="Position: (0.0, 0.0) mm",
                                        font=('Consolas', 8))
        self.position_label.pack(anchor='w', pady=2)

        self.velocity_label = ttk.Label(state_frame,
                                        text="Velocity: (0.0, 0.0) mm/s",
                                        font=('Consolas', 8))
        self.velocity_label.pack(anchor='w', pady=2)

        self.uncertainty_label = ttk.Label(state_frame,
                                           text="Uncertainty: ±0.0 mm",
                                           font=('Consolas', 8))
        self.uncertainty_label.pack(anchor='w', pady=2)

        # Statistics display
        stats_frame = ttk.Frame(self.frame)
        stats_frame.pack(fill='x', pady=(5, 5))

        self.stats_label = ttk.Label(stats_frame,
                                     text="Updates: 0/0 (0.0%)",
                                     font=('Consolas', 8),
                                     foreground=self.colors['border'])
        self.stats_label.pack(anchor='w', pady=2)

        # Control buttons
        btn_frame = ttk.Frame(self.frame)
        btn_frame.pack(fill='x', pady=(10, 0))

        ttk.Button(btn_frame, text="Reset Filter",
                   command=self._on_reset_filter,
                   width=15).pack(side='left', padx=5)

        ttk.Button(btn_frame, text="Show Details",
                   command=self._on_show_details,
                   width=15).pack(side='left', padx=5)

        return self.frame

    def _create_parameter_slider(self, label, min_val, max_val, default,
                                 param_name, tooltip=""):
        """Create a parameter control row with slider and value display."""
        frame = ttk.Frame(self.frame)
        frame.pack(fill='x', pady=5)

        label_widget = ttk.Label(frame, text=label,
                                 font=('Segoe UI', 9))
        label_widget.grid(row=0, column=0, sticky='w', padx=(0, 10))

        slider = ttk.Scale(frame, from_=min_val, to=max_val, orient='horizontal')
        slider.grid(row=0, column=1, sticky='ew', padx=5)
        slider.set(default)

        value_label = ttk.Label(frame, text=f"{default:.2f}",
                                width=6, font=('Consolas', 9),
                                foreground=self.colors['highlight'])
        value_label.grid(row=0, column=2)

        # Store references
        if not hasattr(self, 'sliders'):
            self.sliders = {}
            self.value_labels = {}

        self.sliders[param_name] = slider
        self.value_labels[param_name] = value_label

        # Bind slider change
        def on_change(val):
            value = float(val)
            value_label.config(text=f"{value:.2f}")
            self._on_param_change(param_name, value)

        slider.config(command=on_change)
        frame.columnconfigure(1, weight=1)

        # Tooltip (optional)
        if tooltip:
            info_label = ttk.Label(frame, text="ⓘ",
                                   font=('Segoe UI', 8),
                                   foreground=self.colors['border'])
            info_label.grid(row=0, column=3, padx=(2, 0))
            # In a real implementation, you'd add a tooltip on hover

    def _on_enable_toggle(self):
        """Handle filter enable/disable."""
        enabled = self.filter_enabled.get()

        if enabled:
            self.status_indicator.config(foreground=self.colors['success'])
            # Enable sliders
            for slider in self.sliders.values():
                slider.config(state='normal')
        else:
            self.status_indicator.config(foreground=self.colors['border'])
            # Disable sliders
            for slider in self.sliders.values():
                slider.config(state='disabled')

        if self.callbacks.get('kalman_enable_change'):
            self.callbacks['kalman_enable_change'](enabled)

    def _on_param_change(self, param_name, value):
        """Handle parameter change."""
        if not self.kalman_filter or not self.filter_enabled.get():
            return

        # Update filter parameters
        if param_name == 'process_noise':
            self.kalman_filter.set_process_noise(value)
        elif param_name == 'measurement_noise':
            self.kalman_filter.set_measurement_noise(value)

        if self.callbacks.get('kalman_param_change'):
            self.callbacks['kalman_param_change'](param_name, value)

    def _on_reset_filter(self):
        """Reset filter state."""
        if self.kalman_filter:
            self.kalman_filter.reset()

        if self.callbacks.get('kalman_reset'):
            self.callbacks['kalman_reset']()

    def _on_show_details(self):
        """Show detailed filter information."""
        if not self.kalman_filter:
            return

        from tkinter import messagebox

        stats = self.kalman_filter.get_statistics()
        pos, vel, _ = self.kalman_filter.get_state()
        std_pos = self.kalman_filter.get_position_uncertainty()
        std_vel = self.kalman_filter.get_velocity_uncertainty()

        details = "Kalman Filter Details\n"
        details += "=" * 50 + "\n\n"

        details += "Current State:\n"
        details += f"  Position: ({pos[0]:.2f}, {pos[1]:.2f}) mm\n"
        details += f"  Velocity: ({vel[0]:.2f}, {vel[1]:.2f}) mm/s\n\n"

        details += "Uncertainty (1σ):\n"
        details += f"  Position: (±{std_pos[0]:.3f}, ±{std_pos[1]:.3f}) mm\n"
        details += f"  Velocity: (±{std_vel[0]:.3f}, ±{std_vel[1]:.3f}) mm/s\n\n"

        details += "Statistics:\n"
        details += f"  Predictions: {stats['predictions']}\n"
        details += f"  Updates: {stats['updates']}\n"
        details += f"  Update ratio: {stats['update_ratio'] * 100:.1f}%\n"
        details += f"  Last measurement: {stats['last_measurement_time']:.3f}s\n\n"

        details += "Parameters:\n"
        details += f"  Process noise scale: {stats['process_noise_scale']:.3f}\n"
        details += f"  Measurement noise scale: {stats['measurement_noise_scale']:.3f}\n"

        messagebox.showinfo("Kalman Filter Details", details)

    def update(self, state):
        """Update filter state display."""
        if 'kalman_position' in state:
            pos = state['kalman_position']
            self.position_label.config(
                text=f"Position: ({pos[0]:.2f}, {pos[1]:.2f}) mm"
            )

        if 'kalman_velocity' in state:
            vel = state['kalman_velocity']
            self.velocity_label.config(
                text=f"Velocity: ({vel[0]:.2f}, {vel[1]:.2f}) mm/s"
            )

        if 'kalman_uncertainty' in state:
            std = state['kalman_uncertainty']
            avg_std = (std[0] + std[1]) / 2.0
            self.uncertainty_label.config(
                text=f"Uncertainty: ±{avg_std:.3f} mm"
            )

        if 'kalman_stats' in state:
            stats = state['kalman_stats']
            self.stats_label.config(
                text=f"Updates: {stats['updates']}/{stats['predictions']} "
                     f"({stats['update_ratio'] * 100:.1f}%)"
            )