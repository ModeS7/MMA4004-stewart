#!/usr/bin/env python3
"""
Stewart Platform Real Hardware Controller

Features:
- 100Hz dedicated control thread
- Pixy2 camera integration
- Modular GUI with scrollable columns
- Garbage collection optimization
- Optimized baud rates (USB 200k, Maestro 250k)
- Windows thread priority
- Windows timer resolution + Pre-allocated NumPy arrays
"""

import sys
from PyQt6.QtWidgets import QApplication, QMessageBox
import numpy as np
import time
import threading
import serial.tools.list_ports
import gc
import sys
import ctypes

from setup.base_simulator import BaseStewartSimulator
from setup.hardware_controller_config import HardwareControllerConfig, SerialController, IKCache
from core.control_core import clip_tilt_vector, PIDController, KalmanFilter
from core.utils import ControlLoopConfig, GUIConfig, MAX_TILT_ANGLE_DEG, MAX_SERVO_ANGLE_DEG, format_time, format_vector_2d
from gui.gui_builder import create_standard_layout

THREAD_PRIORITY_IDLE = -15
THREAD_PRIORITY_LOWEST = -2
THREAD_PRIORITY_BELOW_NORMAL = -1
THREAD_PRIORITY_NORMAL = 0
THREAD_PRIORITY_ABOVE_NORMAL = 1
THREAD_PRIORITY_HIGHEST = 2
THREAD_PRIORITY_TIME_CRITICAL = 15


class WindowsTimerManager:
    """Windows multimedia timer resolution manager. Reduces timer granularity from 15.6ms to 1ms."""

    def __init__(self):
        self.timer_set = False
        self.is_windows = sys.platform.startswith('win')

    def set_high_resolution(self):
        """Set Windows timer to 1ms resolution."""
        if not self.is_windows:
            return False, "Not Windows - timer not set"

        try:
            timeBeginPeriod = ctypes.windll.winmm.timeBeginPeriod
            result = timeBeginPeriod(1)
            if result == 0:
                self.timer_set = True
                return True, "Windows timer set to 1ms"
            else:
                return False, f"Timer set failed: {result}"
        except Exception as e:
            return False, f"Timer error: {str(e)}"

    def restore_default(self):
        """Restore default timer resolution."""
        if self.timer_set:
            try:
                timeEndPeriod = ctypes.windll.winmm.timeEndPeriod
                timeEndPeriod(1)
                self.timer_set = False
            except:
                pass


class ThreadPriorityManager:
    """Windows thread priority manager. Elevates control thread priority to reduce jitter."""

    def __init__(self):
        self.is_windows = sys.platform.startswith('win')
        self.kernel32 = None

        if self.is_windows:
            try:
                self.kernel32 = ctypes.windll.kernel32
            except (AttributeError, OSError):
                self.is_windows = False

    def set_thread_priority(self, thread_id, priority=THREAD_PRIORITY_ABOVE_NORMAL):
        """
        Set thread priority on Windows.

        Args:
            thread_id: Thread ID from thread.ident
            priority: Priority level (1=ABOVE_NORMAL, 2=HIGHEST)

        Returns:
            True if successful, False otherwise
        """
        if not self.is_windows or self.kernel32 is None:
            return False

        try:
            handle = self.kernel32.OpenThread(0x0020, False, thread_id)
            if not handle:
                return False

            result = self.kernel32.SetThreadPriority(handle, priority)
            self.kernel32.CloseHandle(handle)

            return bool(result)
        except Exception:
            return False


class HardwareStewartSimulator(BaseStewartSimulator):
    """Hardware-specific Stewart Platform Simulator with modular GUI."""

    def __init__(self, app):
        self.port_var = ''
        config = HardwareControllerConfig()

        ball_physics_params = {
            'radius': 0.02,
            'mass': 0.0027,
            'gravity': 9.81,
            'mass_factor': 1.667
        }

        self.kalman_filter = KalmanFilter(
            process_noise_scale=1.0,
            measurement_noise_scale=1.0,
            ball_physics_params=ball_physics_params,
            dt=ControlLoopConfig.INTERVAL_S
        )
        self.kalman_enabled = False

        # PID-specific: Option to use Kalman velocity for derivative
        self.use_kalman_derivative = False

        super().__init__(app, config)

        self.setWindowTitle("Stewart Platform - Real Hardware Control (100Hz)")

        self.serial_controller = None
        self.connected = False

        self.pixy_width_mm = 350.0
        self.pixy_height_mm = 266.0
        self.pixels_to_mm_x = self.pixy_width_mm / 316.0
        self.pixels_to_mm_y = self.pixy_height_mm / 208.0

        self.ball_pos_mm = (0.0, 0.0)
        self.ball_detected = False
        self.last_ball_update = 0
        self.ball_history_x = []
        self.ball_history_y = []
        self.max_history = 100

        self.ik_cache = IKCache(max_size=5000)

        self._translation_buffer = np.zeros(3, dtype=np.float64)
        self._rotation_buffer = np.zeros(3, dtype=np.float64)

        self.control_thread = None
        self.last_sent_angles = None
        self.angle_change_threshold = 0.2

        self.priority_manager = ThreadPriorityManager()
        self.control_thread_id = None

        self.timer_manager = WindowsTimerManager()

        self.actual_fps = 0.0
        self.timing_stats = {
            'ik_time': [],
            'send_time': [],
            'total_time': []
        }
        self.timing_breakpoints = {}
        self.ik_timeout_count = 0

        self.last_gui_update = time.time()
        self.gui_update_count = 0

        if 'simulation_control' in self.gui_modules:
            self.gui_modules['simulation_control'].start_btn.setEnabled(False)

        self.log("Hardware controller initialized (100Hz mode)")

    def _create_controller_param_widgets(self):
        """Override to use hardware-specific defaults."""
        self.param_definitions = [
            ('kp', 'P (Proportional)', 1.0, 6),
            ('ki', 'I (Integral)', 1.0, 6),
            ('kd', 'D (Derivative)', 4.0, 5)
        ]

        self.controller_widgets = {
            'sliders': {},
            'value_labels': {},
            'scalar_vars': {},
            'update_fn': lambda: None,
            'param_definitions': self.param_definitions
        }

    def get_layout_config(self):
        """Define hardware-specific GUI layout (matches simulation layout)."""
        layout = create_standard_layout(scrollable_columns=False, include_plot=True)

        layout['columns'][0]['modules'] = [
            {'type': 'performance_stats'},
            {'type': 'serial_connection', 'args': {'port_var': self.port_var}},
            {'type': 'simulation_control'},
            {'type': 'trajectory_pattern', 'args': {'pattern_var': self.pattern_type}},
            {'type': 'ball_state'},
            {'type': 'configuration', 'args': {'use_offset_var': self.use_top_surface_offset}},
            {'type': 'kalman_filter',
             'args': {'kalman_filter': self.kalman_filter}},
        ]

        layout['columns'][1]['modules'] = [
            {'type': 'controller',
             'args': {'controller_config': self.controller_config,
                      'controller_widgets': self.controller_widgets}},
            {'type': 'servo_angles', 'args': {'show_actual': False}},
            {'type': 'platform_pose'},
            {'type': 'controller_output', 'args': {'controller_name': 'PID (Hardware)'}},
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
            'kalman_enable_change': self.on_kalman_enable_change,
            'kalman_param_change': self.on_kalman_param_change,
            'kalman_reset': self.on_kalman_reset,
        })

        return callbacks

    def _build_modular_gui(self):
        """Override to add PID Kalman derivative option."""
        super()._build_modular_gui()

        if 'controller' in self.gui_modules:
            from PyQt6.QtWidgets import QWidget, QHBoxLayout, QCheckBox, QLabel
            from PyQt6.QtCore import Qt

            controller_frame = self.gui_modules['controller'].widget
            controller_layout = controller_frame.layout()

            derivative_widget = QWidget()
            derivative_layout = QHBoxLayout(derivative_widget)
            derivative_layout.setContentsMargins(0, 10, 0, 0)

            derivative_checkbox = QCheckBox("Use Kalman Velocity for Derivative")
            derivative_checkbox.setChecked(self.use_kalman_derivative)
            derivative_checkbox.stateChanged.connect(self._on_kalman_derivative_toggle)
            derivative_layout.addWidget(derivative_checkbox)

            self.derivative_checkbox_ref = derivative_checkbox

            self.derivative_status = QLabel("ⓘ")
            self.derivative_status.setStyleSheet(f"color: {self.colors['border']}; font-size: 10pt;")
            derivative_layout.addWidget(self.derivative_status)
            derivative_layout.addStretch()

            controller_layout.addWidget(derivative_widget)

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
                    self.use_top_surface_offset
                )

                if angles is not None:
                    self.ik_cache.put(translation, rotation, angles)
                    count += 1

        elapsed = time.time() - start_time
        self.log(f"Pre-warmed {count} poses in {elapsed:.2f}s")

    def connect_serial(self):
        """Connect to hardware."""
        port = self.port_var
        if not port:
            QMessageBox.critical(self, "Error", "No port selected")
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
                self.gui_modules['simulation_control'].start_btn.setEnabled(True)
        else:
            QMessageBox.critical(self, "Error", message)
            self.log(f"Error: {message}")

    def disconnect_serial(self):
        """Disconnect from hardware."""
        if self.simulation_running:
            self.stop_simulation()

        if self.serial_controller:
            self.serial_controller.disconnect()

        self.connected = False

        if 'simulation_control' in self.gui_modules:
            self.gui_modules['simulation_control'].start_btn.setEnabled(False)

        self.log("Disconnected")

    def _initialize_controller(self):
        """Initialize hardware PID controller."""
        sliders = self.controller_widgets['sliders']
        scalar_vars = self.controller_widgets['scalar_vars']

        kp = self.controller_config.get_scaled_param('kp', sliders, scalar_vars)
        ki = self.controller_config.get_scaled_param('ki', sliders, scalar_vars)
        kd = self.controller_config.get_scaled_param('kd', sliders, scalar_vars)

        self.controller = self.controller_config.create_controller(
            kp=kp, ki=ki, kd=kd, output_limit=15.0
        )

        self.log(f"PID initialized: Kp={kp:.6f}, Ki={ki:.6f}, Kd={kd:.6f}")

    def on_controller_param_change(self):
        """Update controller when parameters change."""
        if self.controller is None:
            return

        sliders = self.controller_widgets['sliders']
        scalar_vars = self.controller_widgets['scalar_vars']

        kp = self.controller_config.get_scaled_param('kp', sliders, scalar_vars)
        ki = self.controller_config.get_scaled_param('ki', sliders, scalar_vars)
        kd = self.controller_config.get_scaled_param('kd', sliders, scalar_vars)

        self.controller.set_gains(kp, ki, kd)

        if self.controller_enabled:
            self.log(f"PID gains updated: Kp={kp:.6f}, Ki={ki:.6f}, Kd={kd:.6f}")

    def on_kalman_enable_change(self, enabled):
        """Handle Kalman filter enable/disable."""
        self.kalman_enabled = enabled
        if enabled:
            self.kalman_filter.reset(self.ball_pos_mm)
        else:
            # Disable Kalman derivative if Kalman is disabled
            if self.use_kalman_derivative:
                self.use_kalman_derivative = False
                if hasattr(self, 'derivative_checkbox_ref'):
                    self.derivative_checkbox_ref.setChecked(False)
                self._on_kalman_derivative_toggle()
        self.log(f"Kalman filter: {'ENABLED' if enabled else 'DISABLED'}")

    def on_kalman_param_change(self, param_name, value):
        """Handle Kalman parameter change."""
        param_labels = {
            'process_noise': 'Process noise',
            'measurement_noise': 'Measurement noise'
        }
        label = param_labels.get(param_name, param_name)
        self.log(f"Kalman {label}: {value:.2f}")

    def on_kalman_reset(self):
        """Handle Kalman filter reset."""
        self.kalman_filter.reset(self.ball_pos_mm)
        self.log("Kalman filter reset")

    def _on_kalman_derivative_toggle(self):
        """Handle PID derivative mode toggle."""
        enabled = self.derivative_checkbox_ref.isChecked()
        if enabled and not self.kalman_enabled:
            # Can't use Kalman derivative without Kalman enabled
            self.use_kalman_derivative = False
            self.derivative_checkbox_ref.setChecked(False)
            self.log("Enable Kalman filter first to use Kalman derivative")
            return

        self.use_kalman_derivative = enabled
        mode = "Kalman velocity" if enabled else "finite difference"
        self.log(f"PID derivative: {mode}")

        if hasattr(self, 'derivative_status'):
            self.derivative_status.setText("✓" if enabled else "ⓘ")
            color = self.colors['success'] if enabled else self.colors['border']
            self.derivative_status.setStyleSheet(f"color: {color}; font-size: 10pt;")

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
        """Dedicated 100Hz control thread with detailed timing instrumentation."""
        loop_interval = ControlLoopConfig.INTERVAL_S
        max_ik_time = ControlLoopConfig.IK_TIMEOUT_S

        timing_breakpoints = {
            'ball_read': [],
            'ball_process': [],
            'pattern_calc': [],
            'kalman_predict': [],
            'kalman_update': [],
            'pid_update': [],
            'ik_total': [],
            'serial_send': [],
            'sleep': []
        }
        max_breakpoint_samples = 1000

        self.timing_breakpoints = timing_breakpoints

        while self.simulation_running:
            loop_start = time.perf_counter()

            t0 = time.perf_counter()
            ball_data = self.serial_controller.get_latest_ball_data()
            ball_read_time = (time.perf_counter() - t0) * 1000
            timing_breakpoints['ball_read'].append(ball_read_time)

            if ball_data is not None:
                t1 = time.perf_counter()
                self.last_ball_update = self.simulation_time

                pixy_x = ball_data['x']
                pixy_y = ball_data['y']

                # Camera dimensions: 316×208 pixels, origin at top-left
                # Invert Y so (0,0) moves to bottom-left, then center it
                CAMERA_HEIGHT_PIXELS = 208.0
                CAMERA_CENTER_X = 158.0
                CAMERA_CENTER_Y = 104.0

                ball_x_mm = (pixy_x - CAMERA_CENTER_X) * self.pixels_to_mm_x
                ball_y_mm = ((CAMERA_HEIGHT_PIXELS - pixy_y) - CAMERA_CENTER_Y) * self.pixels_to_mm_y

                # Use raw position
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
            else:
                timing_breakpoints['ball_process'].append(0.0)

            if self.kalman_enabled:
                t_kalman_pred = time.perf_counter()

                # Get platform angles from FK
                if hasattr(self, 'last_fk_rotation'):
                    rx_deg = self.last_fk_rotation[0]
                    ry_deg = self.last_fk_rotation[1]
                else:
                    rx_deg = self.dof_values['rx']
                    ry_deg = self.dof_values['ry']

                self.kalman_filter.predict([rx_deg, ry_deg])

                kalman_pred_time = (time.perf_counter() - t_kalman_pred) * 1000
                timing_breakpoints['kalman_predict'].append(kalman_pred_time)
            else:
                timing_breakpoints['kalman_predict'].append(0.0)

            if self.controller_enabled and self.ball_detected:

                if self.kalman_enabled:
                    t_kalman_upd = time.perf_counter()
                    self.kalman_filter.update(self.ball_pos_mm, self.simulation_time)
                    kalman_upd_time = (time.perf_counter() - t_kalman_upd) * 1000
                    timing_breakpoints['kalman_update'].append(kalman_upd_time)

                    # Get filtered estimates
                    filtered_x, filtered_y = self.kalman_filter.get_position_mm()
                    filtered_vx, filtered_vy = self.kalman_filter.get_velocity_mm_s()
                    ball_pos_mm = (filtered_x, filtered_y)
                    ball_vel_mm_s = (filtered_vx, filtered_vy)
                else:
                    timing_breakpoints['kalman_update'].append(0.0)
                    ball_pos_mm = self.ball_pos_mm
                    ball_vel_mm_s = (0.0, 0.0)

                t2 = time.perf_counter()
                pattern_time = self.simulation_time - self.pattern_start_time
                target_x, target_y = self.current_pattern.get_position(pattern_time)
                target_pos_mm = (target_x, target_y)
                pattern_calc_time = (time.perf_counter() - t2) * 1000
                timing_breakpoints['pattern_calc'].append(pattern_calc_time)

                t3 = time.perf_counter()

                if self.use_kalman_derivative and self.kalman_enabled:
                    # Use Kalman velocity for derivative term
                    error_x = ball_pos_mm[0] - target_pos_mm[0]
                    error_y = ball_pos_mm[1] - target_pos_mm[1]

                    error_dot_x = ball_vel_mm_s[0]
                    error_dot_y = ball_vel_mm_s[1]

                    # Manual PID computation
                    output_x = (self.controller.kp * error_x +
                                self.controller.ki * self.controller.integral_x +
                                self.controller.kd * error_dot_x)
                    output_y = (self.controller.kp * error_y +
                                self.controller.ki * self.controller.integral_y +
                                self.controller.kd * error_dot_y)

                    # Update integrals
                    self.controller.integral_x += error_x * loop_interval
                    self.controller.integral_y += error_y * loop_interval

                    # Apply limits
                    output_x = np.clip(output_x, -MAX_TILT_ANGLE_DEG, MAX_TILT_ANGLE_DEG)
                    output_y = np.clip(output_y, -MAX_TILT_ANGLE_DEG, MAX_TILT_ANGLE_DEG)

                    rx = output_y
                    ry = -output_x
                else:
                    # Standard PID (finite difference derivative)
                    rx, ry = self.controller.update(ball_pos_mm, target_pos_mm, loop_interval)

                pid_update_time = (time.perf_counter() - t3) * 1000
                timing_breakpoints['pid_update'].append(pid_update_time)

                self.dof_values['rx'] = rx
                self.dof_values['ry'] = ry

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
                        self.use_top_surface_offset
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
                timing_breakpoints['kalman_update'].append(0.0)
                timing_breakpoints['pid_update'].append(0.0)
                timing_breakpoints['ik_total'].append(0.0)
                timing_breakpoints['serial_send'].append(0.0)

            self.simulation_time += loop_interval

            t_sleep = time.perf_counter()
            elapsed = time.perf_counter() - loop_start

            if elapsed > 0.050:
                print(f"WARNING: Loop took {elapsed * 1000:.1f}ms - Windows preemption detected")
                timing_breakpoints['sleep'].append(0.0)
            else:
                sleep_time = loop_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                sleep_actual_time = (time.perf_counter() - t_sleep) * 1000
                timing_breakpoints['sleep'].append(sleep_actual_time)

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

        # Schedule next update using QTimer
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(GUIConfig.UPDATE_INTERVAL_MS, self._gui_update_loop)

    def update_gui_modules(self):
        """Override to add hardware-specific state."""
        status = "Detected" if self.ball_detected else "Not detected"

        state = {
            'simulation_time': self.simulation_time,
            'controller_enabled': self.controller_enabled,
            'ball_pos': self.ball_pos_mm,
            'ball_vel': status,
            'dof_values': self.dof_values,
            'connected': self.connected,
            'fps': ControlLoopConfig.FREQUENCY_HZ,
            'cache_hit_rate': self.ik_cache.get_hit_rate(),
            'ik_timeouts': self.ik_timeout_count,
        }

        if self.controller_enabled:
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
        state['pattern_info'] = pattern_configs.get(self.pattern_type, "")

        self.gui_builder.update_modules(state)

        if self.kalman_enabled and hasattr(self, 'kalman_filter'):
            pos, vel, _ = self.kalman_filter.get_state()
            std_pos = self.kalman_filter.get_position_uncertainty()
            stats = self.kalman_filter.get_statistics()

            kalman_state = {
                'kalman_position': pos,
                'kalman_velocity': vel,
                'kalman_uncertainty': std_pos,
                'kalman_stats': stats
            }

            if 'kalman_filter' in self.gui_modules:
                self.gui_modules['kalman_filter'].update(kalman_state)

    def setup_plot(self):
        """Setup plot for hardware (using PyQtGraph)."""
        super().setup_plot()
        # PyQtGraph plot is set up in base class, no matplotlib needed

    def _update_hardware_plot(self):
        """Update plot with hardware data (using PyQtGraph)."""
        # Plot updates are handled by base class update_plot() method
        # Just call it with current state
        self.update_plot()

    def show_timing_stats(self):
        """Show performance statistics with detailed breakpoint analysis."""
        print("\n" + "=" * 70)
        print("DETAILED TIMING BREAKDOWN")
        print("=" * 70 + "\n")

        if hasattr(self, 'timing_breakpoints') and self.timing_breakpoints:
            breakpoint_names = {
                'ball_read': 'Ball Data Read (Queue)',
                'ball_process': 'Ball Processing (Transform/History)',
                'pattern_calc': 'Pattern Calculation',
                'pid_update': 'PID Controller Update',
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

        stats_msg = "Performance Statistics (100Hz Hardware Mode)\n"
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
        stats_msg += f"  GC Disabled during control\n"
        stats_msg += f"  USB 200k, Maestro 250k baud\n"
        stats_msg += f"  Thread Priority TIME_CRITICAL\n"
        stats_msg += f"  Windows Timer 1ms + Pre-allocated buffers\n"
        stats_msg += f"  IK Timeouts: {self.ik_timeout_count}\n\n"

        stats_msg += "DETAILED BREAKDOWN PRINTED TO CONSOLE"

        QMessageBox.information(self, "Performance Statistics", stats_msg)

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
                                                self.use_top_surface_offset)

        if angles is not None:
            self.last_cmd_angles = angles

            if self.connected and not self.simulation_running:
                self.serial_controller.send_servo_angles(angles)

    def on_controller_toggle(self):
        """Override to handle manual control disabling for hardware."""
        enabled = self.controller_enabled

        if enabled:
            self.controller.reset()
            self.reset_pattern()
            self.log("PID control ENABLED")

            if 'manual_pose' in self.gui_modules:
                manual_pose = self.gui_modules['manual_pose']
                manual_pose.sliders['rx'].setEnabled(False)
                manual_pose.sliders['ry'].setEnabled(False)
                manual_pose.sliders['x'].setEnabled(False)
                manual_pose.sliders['y'].setEnabled(False)
                manual_pose.sliders['z'].setEnabled(False)
        else:
            self.log("PID control DISABLED")

            if 'manual_pose' in self.gui_modules:
                manual_pose = self.gui_modules['manual_pose']
                manual_pose.sliders['rx'].setEnabled(True)
                manual_pose.sliders['ry'].setEnabled(True)
                manual_pose.sliders['x'].setEnabled(True)
                manual_pose.sliders['y'].setEnabled(True)
                manual_pose.sliders['z'].setEnabled(True)

    def _update_controller(self, ball_pos_mm, ball_vel_mm_s, target_pos_mm, dt):
        """Hardware controller update (not used - control thread handles it)."""
        return self.controller.update(ball_pos_mm, target_pos_mm, dt)

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
    """Launch hardware controller."""
    app = QApplication(sys.argv)
    simulator = HardwareStewartSimulator(app)

    simulator.log("=" * 50)
    simulator.log("Hardware Controller - Ready")
    simulator.log("=" * 50)
    simulator.log("")
    simulator.log("")
    simulator.log("Quick Start:")
    simulator.log("1. Select serial port and click 'Connect'")
    simulator.log("2. Enable PID Control for automatic balancing")
    simulator.log("3. Click 'Start' to begin 100Hz control loop")
    simulator.log("4. Select trajectory patterns to track")
    simulator.log("")

    simulator.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()