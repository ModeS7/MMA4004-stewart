def reset_pattern(self):
    """Reset pattern timing."""
    self.pattern_start_time = self.simulation_time
    self.current_pattern.reset()
    self.log(f"Pattern reset at t={format_time(self.simulation_time)}")

    if self.controller_enabled.get():
        self.controller.reset()

    # Reset velocity estimator (no ball_filter in hardware)
    self.velocity_estimator.reset()  # !/usr/bin/env python3


"""
Stewart Platform Real Hardware Controller - LQR

Features:
- 100Hz dedicated control thread
- Pixy2 camera integration
- Minimal position smoothing (α=0.95) for clean velocity estimates
- Full LQR control with position + velocity feedback
- Modular GUI with scrollable columns
- Velocity saturation to reject extreme spikes
- Garbage collection optimization
- Optimized baud rates (USB 200k, Maestro 250k)
- Windows thread priority + timer resolution

Filtering approach:
- Position: 95% new data, 5% smoothed (minimal lag, removes jitter)
- Velocity: Computed from smoothed position (clean, usable for control)
"""

import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import time
import threading
import serial.tools.list_ports
import gc
import sys
import ctypes

from setup.base_simulator import BaseStewartSimulator
from setup.hardware_controller_config import SerialController, IKCache, WindowsTimerManager, ThreadPriorityManager
from core.control_core import clip_tilt_vector, LQRController
from core.utils import ControlLoopConfig, GUIConfig, MAX_TILT_ANGLE_DEG, MAX_SERVO_ANGLE_DEG, format_time, \
    format_vector_2d
from gui.gui_builder import create_standard_layout

THREAD_PRIORITY_TIME_CRITICAL = 15


class LQRHardwareControllerConfig:
    """LQR controller configuration for hardware."""

    def __init__(self, ball_physics_params):
        self.scalar_values = [0.0000001, 0.000001, 0.00001, 0.0001,
                              0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        self.default_weights = {'Q_pos': 1.0, 'Q_vel': 1.0, 'R': 1.0}
        self.default_scalar_indices = {'Q_pos': 7, 'Q_vel': 6, 'R': 5}
        self.ball_physics_params = ball_physics_params
        self.controller_ref = None

    def get_controller_name(self) -> str:
        return "LQR (Hardware)"

    def create_controller(self, **kwargs):
        return LQRController(
            Q_pos=kwargs.get('Q_pos', 1.0),
            Q_vel=kwargs.get('Q_vel', 0.1),
            R=kwargs.get('R', 0.01),
            output_limit=kwargs.get('output_limit', 15.0),
            ball_physics_params=self.ball_physics_params
        )

    def get_scalar_values(self) -> list:
        return self.scalar_values

    def get_scaled_param(self, param_name, sliders, scalar_vars):
        """Extract and scale a parameter value from widgets."""
        raw = float(sliders[param_name].get())
        scalar = self.scalar_values[scalar_vars[param_name].get()]
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

        scalar_var = tk.IntVar(value=self.default_scalar_indices.get(param_name, 4))
        scalar_vars[param_name] = scalar_var

        scalar_combo = ttk.Combobox(
            frame, width=12, state='readonly',
            values=[f'×{s:.7g}' for s in self.scalar_values]
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
        val = float(value)
        value_labels[param_name].config(text=f"{val:.2f}")
        callback()

    def _on_scalar_change(self, combo, var, param_name, callback):
        var.set(combo.current())
        callback()


class VelocityEstimator:
    """Estimate velocity from position with minimal smoothing for noise rejection."""

    def __init__(self, max_velocity_mm_s=500.0, position_alpha=0.95):
        """
        Args:
            max_velocity_mm_s: Maximum velocity saturation limit
            position_alpha: Position filter coefficient (0.95 = 95% new, 5% old)
        """
        self.prev_pos_x = 0.0
        self.prev_pos_y = 0.0
        self.filtered_pos_x = 0.0
        self.filtered_pos_y = 0.0
        self.initialized = False
        self.max_velocity = max_velocity_mm_s
        self.position_alpha = position_alpha

    def update(self, pos_x_raw, pos_y_raw, dt):
        """
        Estimate velocity from position with minimal smoothing.

        Args:
            pos_x_raw: Raw X position from camera (mm)
            pos_y_raw: Raw Y position from camera (mm)
            dt: Time step (seconds)

        Returns:
            (pos_x_filtered, pos_y_filtered, vel_x, vel_y)
        """
        if not self.initialized:
            self.filtered_pos_x = pos_x_raw
            self.filtered_pos_y = pos_y_raw
            self.prev_pos_x = pos_x_raw
            self.prev_pos_y = pos_y_raw
            self.initialized = True
            return pos_x_raw, pos_y_raw, 0.0, 0.0

        # Minimal position smoothing (α=0.95: 95% new, 5% old)
        self.filtered_pos_x = (self.position_alpha * pos_x_raw +
                               (1.0 - self.position_alpha) * self.filtered_pos_x)
        self.filtered_pos_y = (self.position_alpha * pos_y_raw +
                               (1.0 - self.position_alpha) * self.filtered_pos_y)

        if dt <= 0:
            return self.filtered_pos_x, self.filtered_pos_y, 0.0, 0.0

        # Velocity from filtered position (much cleaner)
        vel_x_raw = (self.filtered_pos_x - self.prev_pos_x) / dt
        vel_y_raw = (self.filtered_pos_y - self.prev_pos_y) / dt

        # Saturate to reject any remaining extreme spikes
        vel_x = np.clip(vel_x_raw, -self.max_velocity, self.max_velocity)
        vel_y = np.clip(vel_y_raw, -self.max_velocity, self.max_velocity)

        # Update previous filtered position
        self.prev_pos_x = self.filtered_pos_x
        self.prev_pos_y = self.filtered_pos_y

        return self.filtered_pos_x, self.filtered_pos_y, vel_x, vel_y

    def reset(self):
        """Reset estimator state."""
        self.initialized = False


class HardwareStewartSimulator(BaseStewartSimulator):
    """Hardware-specific Stewart Platform Simulator with LQR control."""

    def __init__(self, root):
        self.port_var = tk.StringVar()

        ball_physics_params = {
            'radius': 0.02,
            'mass': 0.0027,
            'gravity': 9.81,
            'mass_factor': 1.667
        }

        config = LQRHardwareControllerConfig(ball_physics_params)

        # Velocity estimator with minimal position smoothing
        # α=0.95: 95% new data, 5% old (barely noticeable lag, removes camera jitter)
        # Max velocity: 500 mm/s is reasonable for ping pong ball
        self.velocity_estimator = VelocityEstimator(
            max_velocity_mm_s=500.0,
            position_alpha=0.95
        )

        super().__init__(root, config)

        self.root.title("Stewart Platform - Real Hardware Control (LQR, 100Hz)")

        self.serial_controller = None
        self.connected = False

        # Camera calibration
        self.pixy_width_mm = 350.0
        self.pixy_height_mm = 266.0
        self.pixels_to_mm_x = self.pixy_width_mm / 316.0
        self.pixels_to_mm_y = self.pixy_height_mm / 208.0

        # Ball state
        self.ball_pos_mm = (0.0, 0.0)
        self.ball_vel_mm_s = (0.0, 0.0)
        self.ball_detected = False
        self.last_ball_update = 0
        self.ball_history_x = []
        self.ball_history_y = []
        self.max_history = 100

        # IK cache for performance
        self.ik_cache = IKCache(max_size=5000)

        # Pre-allocated buffers
        self._translation_buffer = np.zeros(3, dtype=np.float64)
        self._rotation_buffer = np.zeros(3, dtype=np.float64)

        # Control thread
        self.control_thread = None
        self.last_sent_angles = None
        self.angle_change_threshold = 0.2

        # Windows optimization
        self.priority_manager = ThreadPriorityManager()
        self.control_thread_id = None
        self.timer_manager = WindowsTimerManager()

        # Performance monitoring
        self.actual_fps = 0.0
        self.timing_stats = {
            'ik_time': [],
            'send_time': [],
            'total_time': []
        }
        self.timing_breakpoints = {}
        self.ik_timeout_count = 0

        # Debug logging
        self.debug_counter = 0
        self.debug_interval = 50  # Log every 50 loops (0.5s at 100Hz)

        # GUI update timing
        self.last_gui_update = time.time()
        self.gui_update_count = 0

        # Disable Start button until connected
        if 'simulation_control' in self.gui_modules:
            self.gui_modules['simulation_control'].start_btn.config(state='disabled')

        self.log("LQR Hardware controller initialized (100Hz mode)")
        self.log("Position: Minimal smoothing (α=0.95, barely noticeable lag)")
        self.log("Velocity: Clean estimates from filtered position")
        self.log("Debug: Control values logged to console every 0.5s")
        self.log("Optimizations: GC optimization, optimized baud rates")

    def _create_controller_param_widgets(self):
        """Override to use LQR-specific defaults."""
        self.param_definitions = [
            ('Q_pos', 'Q Position Weight', 1.0, 7),
            ('Q_vel', 'Q Velocity Weight', 1.0, 5),  # Moderate default with filtered velocity
            ('R', 'R Control Weight', 1.0, 5)
        ]

        self.controller_widgets = {
            'sliders': {},
            'value_labels': {},
            'scalar_vars': {},
            'update_fn': lambda: None,
            'param_definitions': self.param_definitions
        }

    def get_layout_config(self):
        """Define hardware-specific GUI layout with scrollable columns."""
        layout = create_standard_layout(scrollable_columns=False, include_plot=True)

        layout['columns'][0]['modules'] = [
            {'type': 'performance_stats'},
            {'type': 'serial_connection', 'args': {'port_var': self.port_var}},
            {'type': 'simulation_control'},
            {'type': 'controller',
             'args': {'controller_config': self.controller_config,
                      'controller_widgets': self.controller_widgets}},
            {'type': 'trajectory_pattern', 'args': {'pattern_var': self.pattern_type}},
            {'type': 'ball_state'},
            {'type': 'configuration', 'args': {'use_offset_var': self.use_top_surface_offset}},
        ]

        layout['columns'][1]['modules'] = [
            {'type': 'servo_angles', 'args': {'show_actual': False}},
            {'type': 'platform_pose'},
            {'type': 'controller_output', 'args': {'controller_name': 'LQR (Hardware)'}},
            {'type': 'manual_pose', 'args': {'dof_config': self.dof_config}},
            {'type': 'debug_log', 'args': {'height': 8}},
        ]

        return layout

    def _create_callbacks(self):
        """Create callback dictionary including hardware-specific callbacks."""
        callbacks = super()._create_callbacks()

        callbacks.update({
            'connect': self.connect_serial,
            'disconnect': self.disconnect_serial,
            'show_stats': self.show_timing_stats,
        })

        return callbacks

    def _build_modular_gui(self):
        """Override to add gain matrix button after GUI is built."""
        super()._build_modular_gui()

        if 'controller' in self.gui_modules:
            controller_frame = self.gui_modules['controller'].frame

            info_frame = ttk.Frame(controller_frame)
            info_frame.pack(fill='x', pady=(10, 0))

            ttk.Button(info_frame, text="Show Gain Matrix",
                       command=self.show_gain_matrix,
                       width=20).pack(side='left', padx=5)

    def show_gain_matrix(self):
        """Display LQR gain matrix in popup."""
        if self.controller is None or not hasattr(self.controller, 'get_gain_matrix'):
            messagebox.showerror("Error", "Controller not initialized")
            return

        K = self.controller.get_gain_matrix()
        if K is None:
            messagebox.showerror("Error", "LQR gain matrix not computed")
            return

        popup = tk.Toplevel(self.root)
        popup.title("LQR Gain Matrix")
        popup.configure(bg=self.colors['bg'])
        popup.geometry("500x300")

        text = tk.Text(popup,
                       bg=self.colors['widget_bg'],
                       fg=self.colors['fg'],
                       font=('Consolas', 9),
                       wrap='none')
        text.pack(fill='both', expand=True, padx=10, pady=10)

        text.insert('1.0', "LQR Gain Matrix K (2x4):\n")
        text.insert('end', "State: [x(m), y(m), vx(m/s), vy(m/s)]\n")
        text.insert('end', "Control: [ry(deg), rx(deg)]\n\n")
        text.insert('end', "K = [ry/state]\n")
        text.insert('end', f"    {K[0, :]}\n\n")
        text.insert('end', "K = [rx/state]\n")
        text.insert('end', f"    {K[1, :]}\n\n")
        text.insert('end', "Interpretation:\n")
        text.insert('end', f"- Position gain: {K[0, 0]:.4f} deg/(m error)\n")
        text.insert('end', f"- Velocity gain: {K[0, 2]:.4f} deg/(m/s)\n")

        text.config(state='disabled')

    def refresh_ports(self):
        """Refresh available serial ports."""
        if 'serial_connection' in self.gui_modules:
            self.gui_modules['serial_connection']._refresh_ports()

    def prewarm_ik_cache(self):
        """Pre-calculate common IK solutions."""
        self.log("Pre-warming IK cache...")

        tilts = np.arange(-15, 16, 2)
        count = 0
        start_time = time.time()

        for rx in tilts:
            for ry in tilts:
                translation = np.array([0.0, 0.0, self.ik.home_height_top_surface])
                rotation = np.array([float(rx), float(ry), 0.0])

                angles = self.ik.calculate_servo_angles(
                    translation, rotation,
                    self.use_top_surface_offset.get()
                )

                if angles is not None:
                    self.ik_cache.put(translation, rotation, angles)
                    count += 1

        elapsed = time.time() - start_time
        self.log(f"Pre-warmed {count} poses in {elapsed:.2f}s")

    def connect_serial(self):
        """Connect to hardware."""
        port = self.port_var.get()
        if not port:
            messagebox.showerror("Error", "No port selected")
            return

        self.serial_controller = SerialController(port)
        success, message = self.serial_controller.connect()

        if success:
            self.connected = True
            self.log(f"Connected to {port}")

            time.sleep(0.5)
            self.serial_controller.set_servo_speed(0)
            time.sleep(0.1)
            self.serial_controller.set_servo_acceleration(0)
            time.sleep(0.2)
            self.log("Servos: Speed=0 (unlimited), Accel=0")

            success_timer, msg_timer = self.timer_manager.set_high_resolution()
            self.log(msg_timer)

            self.prewarm_ik_cache()

            if 'simulation_control' in self.gui_modules:
                self.gui_modules['simulation_control'].start_btn.config(state='normal')
        else:
            messagebox.showerror("Error", message)
            self.log(f"Error: {message}")

    def disconnect_serial(self):
        """Disconnect from hardware."""
        if self.simulation_running:
            self.stop_simulation()

        if self.serial_controller:
            self.serial_controller.disconnect()

        self.connected = False

        if 'simulation_control' in self.gui_modules:
            self.gui_modules['simulation_control'].start_btn.config(state='disabled')

        self.velocity_estimator.reset()

        self.log("Disconnected")

    def _initialize_controller(self):
        """Initialize LQR controller with parameters from widgets."""
        sliders = self.controller_widgets['sliders']
        scalar_vars = self.controller_widgets['scalar_vars']

        Q_pos = self.controller_config.get_scaled_param('Q_pos', sliders, scalar_vars)
        Q_vel = self.controller_config.get_scaled_param('Q_vel', sliders, scalar_vars)
        R = self.controller_config.get_scaled_param('R', sliders, scalar_vars)

        self.controller = self.controller_config.create_controller(
            Q_pos=Q_pos, Q_vel=Q_vel, R=R, output_limit=15.0
        )

        self.controller_config.controller_ref = self.controller
        self.log(f"LQR initialized: Q_pos={Q_pos:.6f}, Q_vel={Q_vel:.6f}, R={R:.6f}")

    def on_controller_param_change(self):
        """Update controller when parameters change."""
        if self.controller is None:
            return

        sliders = self.controller_widgets['sliders']
        scalar_vars = self.controller_widgets['scalar_vars']

        Q_pos = self.controller_config.get_scaled_param('Q_pos', sliders, scalar_vars)
        Q_vel = self.controller_config.get_scaled_param('Q_vel', sliders, scalar_vars)
        R = self.controller_config.get_scaled_param('R', sliders, scalar_vars)

        self.controller.set_weights(Q_pos=Q_pos, Q_vel=Q_vel, R=R)

        if self.controller_enabled.get():
            self.log(f"LQR weights updated: Q_pos={Q_pos:.6f}, Q_vel={Q_vel:.6f}, R={R:.6f}")

    def start_simulation(self):
        """Start 100Hz hardware control thread."""
        if not self.connected:
            return

        self.simulation_running = True
        self.simulation_time = 0.0
        self.ik_timeout_count = 0

        gc.disable()
        self.log("Control started (100Hz, GC disabled)")

        self.control_thread = threading.Thread(target=self._control_thread_func, daemon=True)
        self.control_thread.start()

        self.control_thread_id = self.control_thread.ident
        if self.priority_manager.set_thread_priority(self.control_thread_id, THREAD_PRIORITY_TIME_CRITICAL):
            self.log("Thread priority: TIME_CRITICAL")
        else:
            if sys.platform.startswith('win'):
                self.log("Note: Could not set thread priority")

        self.last_gui_update = time.time()
        self.gui_update_count = 0
        self._gui_update_loop()

    def _control_thread_func(self):
        """Dedicated 100Hz control thread with LQR controller."""
        loop_interval = ControlLoopConfig.INTERVAL_S
        max_ik_time = ControlLoopConfig.IK_TIMEOUT_S

        timing_breakpoints = {
            'ball_read': [],
            'ball_process': [],
            'vel_estimate': [],
            'pattern_calc': [],
            'lqr_update': [],
            'ik_total': [],
            'serial_send': [],
            'sleep': []
        }
        max_breakpoint_samples = 1000

        self.timing_breakpoints = timing_breakpoints

        while self.simulation_running:
            loop_start = time.perf_counter()

            # Read ball data
            t0 = time.perf_counter()
            ball_data = self.serial_controller.get_latest_ball_data()
            ball_read_time = (time.perf_counter() - t0) * 1000
            timing_breakpoints['ball_read'].append(ball_read_time)

            if ball_data is not None:
                t1 = time.perf_counter()
                self.last_ball_update = self.simulation_time

                pixy_x = ball_data['x']
                pixy_y = ball_data['y']

                # Camera coordinate transformation
                CAMERA_HEIGHT_PIXELS = 208.0
                CAMERA_CENTER_X = 158.0
                CAMERA_CENTER_Y = 104.0

                ball_x_mm = (pixy_x - CAMERA_CENTER_X) * self.pixels_to_mm_x
                ball_y_mm = ((CAMERA_HEIGHT_PIXELS - pixy_y) - CAMERA_CENTER_Y) * self.pixels_to_mm_y

                # Store raw position (no filtering)
                self.ball_pos_mm = (ball_x_mm, ball_y_mm)
                self.ball_detected = ball_data['detected']

                if self.ball_detected:
                    self.ball_history_x.append(ball_x_mm)
                    self.ball_history_y.append(ball_y_mm)
                    if len(self.ball_history_x) > self.max_history:
                        self.ball_history_x.pop(0)
                        self.ball_history_y.pop(0)

                ball_process_time = (time.perf_counter() - t1) * 1000
                timing_breakpoints['ball_process'].append(ball_process_time)

                # Estimate velocity from raw position
                t2 = time.perf_counter()
                pos_x_filtered, pos_y_filtered, vel_x_mm_s, vel_y_mm_s = self.velocity_estimator.update(
                    ball_x_mm, ball_y_mm, loop_interval
                )
                self.ball_vel_mm_s = (vel_x_mm_s, vel_y_mm_s)
                vel_estimate_time = (time.perf_counter() - t2) * 1000
                timing_breakpoints['vel_estimate'].append(vel_estimate_time)
            else:
                timing_breakpoints['ball_process'].append(0.0)
                timing_breakpoints['vel_estimate'].append(0.0)

            if self.controller_enabled.get() and self.ball_detected:
                # Calculate target from pattern
                t3 = time.perf_counter()
                pattern_time = self.simulation_time - self.pattern_start_time
                target_x, target_y = self.current_pattern.get_position(pattern_time)
                target_pos_mm = (target_x, target_y)
                pattern_calc_time = (time.perf_counter() - t3) * 1000
                timing_breakpoints['pattern_calc'].append(pattern_calc_time)

                # LQR controller update
                t4 = time.perf_counter()
                rx, ry = self.controller.update(
                    self.ball_pos_mm,
                    self.ball_vel_mm_s,
                    target_pos_mm
                )
                lqr_update_time = (time.perf_counter() - t4) * 1000
                timing_breakpoints['lqr_update'].append(lqr_update_time)

                # Debug logging (every 0.5s)
                self.debug_counter += 1
                if self.debug_counter >= self.debug_interval:
                    self.debug_counter = 0
                    vel_mag = np.sqrt(self.ball_vel_mm_s[0] ** 2 + self.ball_vel_mm_s[1] ** 2)
                    vel_sat = " [SAT]" if vel_mag >= 490 else ""
                    print(f"[LQR] Pos:({self.ball_pos_mm[0]:.1f},{self.ball_pos_mm[1]:.1f})mm "
                          f"Vel:({self.ball_vel_mm_s[0]:.1f},{self.ball_vel_mm_s[1]:.1f})mm/s{vel_sat} "
                          f"Target:({target_pos_mm[0]:.1f},{target_pos_mm[1]:.1f})mm "
                          f"Control:({rx:.2f},{ry:.2f})°")

                self.dof_values['rx'] = rx
                self.dof_values['ry'] = ry

                # Inverse kinematics
                start_ik = time.perf_counter()

                self._translation_buffer[0] = self.dof_values['x']
                self._translation_buffer[1] = self.dof_values['y']
                self._translation_buffer[2] = self.dof_values['z']

                self._rotation_buffer[0] = self.dof_values['rx']
                self._rotation_buffer[1] = self.dof_values['ry']
                self._rotation_buffer[2] = self.dof_values['rz']

                angles = self.ik_cache.get(self._translation_buffer, self._rotation_buffer)

                if angles is None:
                    angles = self.ik.calculate_servo_angles(
                        self._translation_buffer,
                        self._rotation_buffer,
                        self.use_top_surface_offset.get()
                    )

                    ik_time = time.perf_counter() - start_ik

                    if ik_time > max_ik_time:
                        if self.last_sent_angles is not None:
                            angles = self.last_sent_angles
                            self.ik_timeout_count += 1
                    elif angles is not None:
                        self.ik_cache.put(
                            self._translation_buffer,
                            self._rotation_buffer,
                            angles
                        )
                        self.timing_stats['ik_time'].append(ik_time * 1000)
                else:
                    ik_time = time.perf_counter() - start_ik
                    self.timing_stats['ik_time'].append(ik_time * 1000)

                ik_total_time = (time.perf_counter() - start_ik) * 1000
                timing_breakpoints['ik_total'].append(ik_total_time)

                # Send to servos
                if angles is not None:
                    if (self.last_sent_angles is None or
                            not np.allclose(angles, self.last_sent_angles,
                                            atol=self.angle_change_threshold)):

                        send_start = time.perf_counter()
                        success = self.serial_controller.send_servo_angles(angles)
                        send_time = (time.perf_counter() - send_start) * 1000
                        timing_breakpoints['serial_send'].append(send_time)

                        if success:
                            self.last_sent_angles = angles.copy()

                            total_time = (time.perf_counter() - loop_start) * 1000
                            self.timing_stats['send_time'].append(send_time)
                            self.timing_stats['total_time'].append(total_time)

                            for key in self.timing_stats:
                                if len(self.timing_stats[key]) > 1000:
                                    self.timing_stats[key].pop(0)
                    else:
                        timing_breakpoints['serial_send'].append(0.0)
                else:
                    timing_breakpoints['serial_send'].append(0.0)
            else:
                timing_breakpoints['pattern_calc'].append(0.0)
                timing_breakpoints['lqr_update'].append(0.0)
                timing_breakpoints['ik_total'].append(0.0)
                timing_breakpoints['serial_send'].append(0.0)

            self.simulation_time += loop_interval

            # Sleep to maintain 100Hz
            t_sleep = time.perf_counter()
            elapsed = time.perf_counter() - loop_start

            if elapsed > 0.050:
                self.log(f"WARNING: Loop took {elapsed * 1000:.1f}ms - Windows preemption detected")
                timing_breakpoints['sleep'].append(0.0)
            else:
                sleep_time = loop_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                sleep_actual_time = (time.perf_counter() - t_sleep) * 1000
                timing_breakpoints['sleep'].append(sleep_actual_time)

            # Limit breakpoint history
            for key in timing_breakpoints:
                if len(timing_breakpoints[key]) > max_breakpoint_samples:
                    timing_breakpoints[key].pop(0)

    def _gui_update_loop(self):
        """Separate GUI update loop at lower frequency."""
        if not self.simulation_running:
            return

        self.update_gui_modules()

        if self.gui_update_count % 2 == 0:
            self._update_hardware_plot()

        self.gui_update_count += 1

        self.root.after(GUIConfig.UPDATE_INTERVAL_MS, self._gui_update_loop)

    def update_gui_modules(self):
        """Override to add hardware-specific state."""
        status = f"Detected | Vel: {format_vector_2d(self.ball_vel_mm_s, 'mm/s')}" if self.ball_detected else "Not detected"

        state = {
            'simulation_time': self.simulation_time,
            'controller_enabled': self.controller_enabled.get(),
            'ball_pos': self.ball_pos_mm,
            'ball_vel': status,
            'dof_values': self.dof_values,
            'connected': self.connected,
            'fps': ControlLoopConfig.FREQUENCY_HZ,
            'cache_hit_rate': self.ik_cache.get_hit_rate(),
            'ik_timeouts': self.ik_timeout_count,
        }

        if self.controller_enabled.get():
            rx = self.dof_values['rx']
            ry = self.dof_values['ry']
            magnitude = np.sqrt(rx ** 2 + ry ** 2)
            magnitude_percent = (magnitude / 15.0) * 100

            pattern_time = self.simulation_time - self.pattern_start_time
            target_x, target_y = self.current_pattern.get_position(pattern_time)
            error_x = target_x - self.ball_pos_mm[0]
            error_y = target_y - self.ball_pos_mm[1]

            state['controller_output'] = (rx, ry)
            state['controller_magnitude'] = (magnitude, magnitude_percent)
            state['controller_error'] = (error_x, error_y)

        if self.last_sent_angles is not None:
            state['cmd_angles'] = self.last_sent_angles

        pattern_configs = {
            'static': "Tracking: Center (0, 0)",
            'circle': "Tracking: Circle (r=50mm, T=10s)",
            'figure8': "Tracking: Figure-8 (60×40mm, T=12s)",
            'star': "Tracking: 5-Point Star (r=60mm, T=15s)"
        }
        state['pattern_info'] = pattern_configs.get(self.pattern_type.get(), "")

        self.gui_builder.update_modules(state)

    def setup_plot(self):
        """Setup plot for hardware."""
        super().setup_plot()

        self.ball_trail, = self.ax.plot([], [], 'r-', alpha=0.3, linewidth=1,
                                        label='Ball Trail')

        legend = self.ax.legend(loc='upper right', fontsize=8,
                                facecolor=self.colors['panel_bg'],
                                edgecolor=self.colors['border'],
                                labelcolor=self.colors['fg'])
        legend.get_frame().set_alpha(0.9)

        self.canvas.draw()

    def _update_hardware_plot(self):
        """Update plot with hardware data."""
        try:
            if self.ball_detected:
                self.ball_circle.center = self.ball_pos_mm
                self.ball_circle.set_alpha(0.8)
            else:
                self.ball_circle.set_alpha(0.2)

            if len(self.ball_history_x) > 1:
                self.ball_trail.set_data(self.ball_history_x, self.ball_history_y)

            if self.pattern_type.get() != 'static':
                pattern_time = self.simulation_time - self.pattern_start_time
                target_x, target_y = self.current_pattern.get_position(pattern_time)
                self.target_marker.set_data([target_x], [target_y])

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
                    color = self.colors['success']
                    self.tilt_arrow = self.ax.arrow(0, 0, dx, dy,
                                                    head_width=8, head_length=10,
                                                    fc=color, ec=color,
                                                    alpha=0.6, linewidth=2, zorder=5)

            self.canvas.draw_idle()
        except:
            pass

    def show_timing_stats(self):
        """Show performance statistics with detailed breakpoint analysis."""
        print("\n" + "=" * 70)
        print("LQR HARDWARE CONTROL - DETAILED TIMING BREAKDOWN")
        print("=" * 70 + "\n")

        if hasattr(self, 'timing_breakpoints') and self.timing_breakpoints:
            breakpoint_names = {
                'ball_read': 'Ball Data Read (Queue)',
                'ball_process': 'Ball Processing (Transform/History)',
                'vel_estimate': 'Velocity Estimation',
                'pattern_calc': 'Pattern Calculation',
                'lqr_update': 'LQR Controller Update',
                'ik_total': 'IK Total (Cache+Calc)',
                'serial_send': 'Serial Send',
                'sleep': 'Sleep/Timing'
            }

            for key, name in breakpoint_names.items():
                if key in self.timing_breakpoints and self.timing_breakpoints[key]:
                    data = [x for x in self.timing_breakpoints[key] if x > 0]
                    if data:
                        avg = np.mean(data)
                        max_val = np.max(data)
                        min_val = np.min(data)

                        marker = "SPIKE SOURCE!" if max_val > 50 else ""

                        print(f"{name}:{marker}")
                        print(f"  Average: {avg:.3f} ms")
                        print(f"  Min: {min_val:.3f} ms")
                        print(f"  Max: {max_val:.3f} ms")

                        if max_val > 10:
                            p95 = np.percentile(data, 95)
                            p99 = np.percentile(data, 99)
                            print(f"  95th percentile: {p95:.3f} ms")
                            print(f"  99th percentile: {p99:.3f} ms")
                        print()
        else:
            print("No timing breakpoint data collected yet!")

        print("=" * 70 + "\n")

        stats_msg = "Performance Statistics (LQR Hardware, 100Hz Mode)\n"
        stats_msg += "=" * 60 + "\n\n"

        if self.timing_stats['ik_time']:
            stats_msg += "IK Calculation Time (ms):\n"
            stats_msg += f"  Average: {np.mean(self.timing_stats['ik_time']):.3f}\n"
            stats_msg += f"  Min: {np.min(self.timing_stats['ik_time']):.3f}\n"
            stats_msg += f"  Max: {np.max(self.timing_stats['ik_time']):.3f}\n\n"

            stats_msg += "Serial Send Time (ms):\n"
            stats_msg += f"  Average: {np.mean(self.timing_stats['send_time']):.3f}\n"
            stats_msg += f"  Min: {np.min(self.timing_stats['send_time']):.3f}\n"
            stats_msg += f"  Max: {np.max(self.timing_stats['send_time']):.3f}\n\n"

            stats_msg += "Total Loop Time (ms):\n"
            stats_msg += f"  Average: {np.mean(self.timing_stats['total_time']):.3f}\n"
            stats_msg += f"  Min: {np.min(self.timing_stats['total_time']):.3f}\n"
            stats_msg += f"  Max: {np.max(self.timing_stats['total_time']):.3f}\n\n"

        hit_rate = self.ik_cache.get_hit_rate()
        stats_msg += "IK Cache Statistics:\n"
        stats_msg += f"  Hit Rate: {hit_rate * 100:.1f}%\n"
        stats_msg += f"  Hits: {self.ik_cache.hits}\n"
        stats_msg += f"  Misses: {self.ik_cache.misses}\n"
        stats_msg += f"  Cache Size: {len(self.ik_cache.cache)}/{self.ik_cache.max_size}\n\n"

        stats_msg += "Optimizations Active:\n"
        stats_msg += f"  Position smoothing (α=0.95) for clean velocity\n"
        stats_msg += f"  Velocity saturation (500 mm/s max)\n"
        stats_msg += f"  GC Disabled during control\n"
        stats_msg += f"  USB 200k, Maestro 250k baud\n"
        stats_msg += f"  Thread Priority TIME_CRITICAL\n"
        stats_msg += f"  Windows Timer 1ms + Pre-allocated buffers\n"
        stats_msg += f"  IK Timeouts: {self.ik_timeout_count}\n\n"

        stats_msg += "DETAILED BREAKDOWN PRINTED TO CONSOLE"

        messagebox.showinfo("Performance Statistics", stats_msg)

    def calculate_ik(self):
        """Calculate inverse kinematics and send to hardware."""
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

            if self.connected and not self.simulation_running:
                self.serial_controller.send_servo_angles(angles)

    def on_controller_toggle(self):
        """Override to handle manual control disabling for hardware."""
        enabled = self.controller_enabled.get()

        if enabled:
            self.controller.reset()
            self.velocity_estimator.reset()
            self.reset_pattern()
            self.log("LQR control ENABLED")

            if 'manual_pose' in self.gui_modules:
                manual_pose = self.gui_modules['manual_pose']
                manual_pose.sliders['rx'].config(state='disabled')
                manual_pose.sliders['ry'].config(state='disabled')
                manual_pose.sliders['x'].config(state='disabled')
                manual_pose.sliders['y'].config(state='disabled')
                manual_pose.sliders['z'].config(state='disabled')
        else:
            self.log("LQR control DISABLED")

            if 'manual_pose' in self.gui_modules:
                manual_pose = self.gui_modules['manual_pose']
                manual_pose.sliders['rx'].config(state='normal')
                manual_pose.sliders['ry'].config(state='normal')
                manual_pose.sliders['x'].config(state='normal')
                manual_pose.sliders['y'].config(state='normal')
                manual_pose.sliders['z'].config(state='normal')

    def reset_pattern(self):
        """Override to avoid calling non-existent ball_filter."""
        self.pattern_start_time = self.simulation_time
        self.current_pattern.reset()
        self.log(f"Pattern reset at t={format_time(self.simulation_time)}")

        if self.controller_enabled.get():
            self.controller.reset()

        # Reset velocity estimator (no ball_filter in this hardware version)
        self.velocity_estimator.reset()

    def _update_controller(self, ball_pos_mm, ball_vel_mm_s, target_pos_mm, dt):
        """Hardware controller update (not used - control thread handles it)."""
        return self.controller.update(ball_pos_mm, ball_vel_mm_s, target_pos_mm)

    def stop_simulation(self):
        """Stop the control thread."""
        self.simulation_running = False

        if self.control_thread:
            self.control_thread.join(timeout=1.0)

        if self.serial_controller:
            while not self.serial_controller.command_queue.empty():
                try:
                    self.serial_controller.command_queue.get_nowait()
                except:
                    break

        gc.enable()
        gc.collect()

        self.log("Control stopped (GC re-enabled)")

    def on_closing(self):
        """Clean shutdown."""
        if self.simulation_running:
            self.stop_simulation()

        if self.connected:
            self.disconnect_serial()

        self.timer_manager.restore_default()

        gc.enable()
        gc.collect()

        super().on_closing()


def main():
    """Launch LQR hardware controller."""
    root = tk.Tk()
    app = HardwareStewartSimulator(root)

    app.log("=" * 50)
    app.log("LQR Hardware Controller - Ready")
    app.log("=" * 50)
    app.log("")
    app.log("Optimizations Active:")
    app.log("   Minimal Position Smoothing (α=0.95)")
    app.log("   Velocity Saturation (500 mm/s)")
    app.log("   Debug Logging (console every 0.5s)")
    app.log("   GC Optimization")
    app.log("   Optimized Baud Rates")
    app.log("   Windows Thread Priority")
    app.log("   Windows Timer + Pre-allocated Arrays")
    app.log("")
    app.log("Quick Start:")
    app.log("1. Select serial port and click 'Connect'")
    app.log("2. Enable LQR Control for optimal balancing")
    app.log("3. Click 'Start' to begin 100Hz control loop")
    app.log("4. Select trajectory patterns to track")
    app.log("")
    app.log("LQR Tuning Tips:")
    app.log("- Position smoothing: α=0.95 (95% new, 5% old)")
    app.log("- This gives clean velocity for LQR feedback")
    app.log("- Increase Q_pos for tighter position control")
    app.log("- Increase Q_vel for more damping")
    app.log("- Decrease R for more aggressive control")
    app.log("- Watch console for debug output every 0.5s")
    app.log("- Click 'Show Gain Matrix' to see computed gains")
    app.log("")

    root.mainloop()


if __name__ == "__main__":
    main()