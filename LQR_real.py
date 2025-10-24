"""
Stewart Platform Real Hardware Controller - LQR

Features:
- Variable frequency control thread (50-500Hz)
- Pixy2 camera integration
- Full LQR control with position
- Modular GUI with scrollable columns
- Garbage collection optimization
- Optimized baud rates (USB 200k, Maestro 250k)
- Windows thread priority + timer resolution
"""

from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget, QHBoxLayout, QPushButton, QDialog, QVBoxLayout, QTextEdit
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg
import numpy as np
import time
import threading
from queue import Queue, Empty
import gc
import sys
import torch

from setup.base_simulator import BaseStewartSimulator, ControllerConfig
from setup.hardware_controller_config import SerialController, IKCache, WindowsTimerManager, ThreadPriorityManager
from core.control_core import clip_tilt_vector, LQRController, KalmanFilter
from core.utils import ControlLoopConfig, GUIConfig, MAX_TILT_ANGLE_DEG, format_time
from gui.gui_builder import create_standard_layout

THREAD_PRIORITY_TIME_CRITICAL = 15


class LQRHardwareControllerConfig(ControllerConfig):
    """LQR controller configuration for hardware."""

    def __init__(self, ball_physics_params):
        self.scalar_values = [0.0000001, 0.000001, 0.00001, 0.0001,
                              0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        self.default_weights = {'Q_pos': 1.0, 'Q_vel': 1.0, 'R': 1.0}
        self.default_scalar_indices = {'Q_pos': 9, 'Q_vel': 5, 'R': 5}
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


class HardwareStewartSimulator(BaseStewartSimulator):
    """Hardware-specific Stewart Platform Simulator with LQR control."""

    def __init__(self, app):
        self.port_var = ''

        # Plot control
        self.plot_enabled = True
        self.plot_rate = 10  # 10 Hz default
        self.plot_divisor = 10  # Update every Nth loop
        self.plot_drops = 0

        ball_physics_params = {
            'radius': 0.02,
            'mass': 0.0027,
            'gravity': 9.81,
            'mass_factor': 1.667
        }

        config = LQRHardwareControllerConfig(ball_physics_params)

        # Control frequency (variable 50-500Hz) - MUST be set before super().__init__()
        self.control_frequency = ControlLoopConfig.DEFAULT_FREQUENCY_HZ
        self.control_interval = 1.0 / self.control_frequency
        self.actual_frequency = self.control_frequency

        # Kalman filter for ball state estimation
        self.kalman_filter = KalmanFilter(
            process_noise_scale=1.0,
            measurement_noise_scale=1.0,
            ball_physics_params=ball_physics_params,
            dt=self.control_interval
        )
        self.kalman_enabled = False

        super().__init__(app, config)

        self.setWindowTitle(f"Stewart Platform - Real Hardware Control (LQR, {self.control_frequency}Hz)")

        self.serial_controller = None
        self.connected = False

        # Camera calibration
        self.pixy_width_mm = 350.0
        self.pixy_height_mm = 266.0
        self.pixels_to_mm_x = self.pixy_width_mm / 316.0
        self.pixels_to_mm_y = self.pixy_height_mm / 208.0

        # Ball state
        self.ball_pos_mm = (0.0, 0.0)
        self.ball_detected = False
        self.last_ball_update = 0
        self.ball_history_x = []
        self.ball_history_y = []
        self.max_history = 100

        # Ball position for plotting (compatible with base class)
        self.ball_pos = torch.tensor([[0.0, 0.0, 0.02]], dtype=torch.float32)

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
        self.first_ball_data_received = False

        # GUI update timing
        self.last_gui_update = time.time()
        self.gui_update_count = 0

        # Thread-safe queue for GUI updates (non-blocking control thread)
        self.gui_state_queue = Queue(maxsize=1)
        self.plot_state_queue = Queue(maxsize=1)

        # Disable Start button until connected
        if 'simulation_control' in self.gui_modules:
            self.gui_modules['simulation_control'].start_btn.setEnabled(False)

        self.log("LQR Hardware controller initialized")
        self.log("Optimizations: GC optimization, optimized baud rates")

    def _create_controller_param_widgets(self):
        """Override to use LQR-specific defaults."""
        self.param_definitions = [
            ('Q_pos', 'Q Position Weight', 1.0, 9),
            ('Q_vel', 'Q Velocity Weight', 1.0, 5),
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
        """Define hardware-specific GUI layout (matches simulation layout)."""
        layout = create_standard_layout(scrollable_columns=True, include_plot=True)

        layout['columns'][0]['modules'] = [
            {'type': 'performance_stats'},
            {'type': 'serial_connection', 'args': {'port_var': self.port_var}},
            {'type': 'simulation_control'},
            {'type': 'control_frequency',
             'args': {'frequency_var': self.control_frequency,
                      'min_freq': ControlLoopConfig.MIN_FREQUENCY_HZ,
                      'max_freq': ControlLoopConfig.MAX_FREQUENCY_HZ}},
            {'type': 'trajectory_pattern', 'args': {'pattern_var': self.pattern_type}},
            {'type': 'ball_state'},
            {'type': 'configuration', 'args': {'use_offset_var': self.use_top_surface_offset}},
            {'type': 'kalman_filter',
             'args': {'kalman_filter': self.kalman_filter}},
            {'type': 'plot_control',
             'args': {'plot_enabled_var': self.plot_enabled,
                      'plot_rate_var': self.plot_rate}},
        ]

        layout['columns'][1]['modules'] = [
            {'type': 'controller',
             'args': {'controller_config': self.controller_config,
                      'controller_widgets': self.controller_widgets}},
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
            'kalman_enable_change': self.on_kalman_enable_change,
            'kalman_param_change': self.on_kalman_param_change,
            'kalman_reset': self.on_kalman_reset,
            'plot_enable_change': self.on_plot_enable_change,
            'plot_rate_change': self.on_plot_rate_change,
            'frequency_change': self.on_frequency_change,
        })

        return callbacks

    def _build_modular_gui(self):
        """Override to add gain matrix button after GUI is built."""
        super()._build_modular_gui()

        if 'controller' in self.gui_modules:
            controller_frame = self.gui_modules['controller'].widget
            controller_layout = controller_frame.layout()

            info_widget = QWidget()
            info_layout = QHBoxLayout(info_widget)
            info_layout.setContentsMargins(0, 10, 0, 0)

            gain_matrix_btn = QPushButton("Show Gain Matrix")
            gain_matrix_btn.clicked.connect(self.show_gain_matrix)
            gain_matrix_btn.setFixedWidth(150)
            info_layout.addWidget(gain_matrix_btn)
            info_layout.addStretch()

            controller_layout.addWidget(info_widget)

    def show_gain_matrix(self):
        """Display LQR gain matrix in popup."""
        if self.controller is None or not hasattr(self.controller, 'get_gain_matrix'):
            QMessageBox.critical(self, "Error", "Controller not initialized")
            return

        K = self.controller.get_gain_matrix()
        if K is None:
            QMessageBox.critical(self, "Error", "LQR gain matrix not computed")
            return

        popup = QDialog(self)
        popup.setWindowTitle("LQR Gain Matrix")
        popup.setGeometry(100, 100, 500, 300)

        layout = QVBoxLayout(popup)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: {self.colors['widget_bg']};
                color: {self.colors['fg']};
                font-family: 'Consolas', monospace;
                font-size: 9pt;
            }}
        """)

        content = "LQR Gain Matrix K (2x4):\n"
        content += "State: [x(m), y(m), vx(m/s), vy(m/s)]\n"
        content += "Control: [ry(deg), rx(deg)]\n\n"
        content += "K = [ry/state]\n"
        content += f"    {K[0, :]}\n\n"
        content += "K = [rx/state]\n"
        content += f"    {K[1, :]}\n\n"
        content += "Interpretation:\n"
        content += f"- Position gain: {K[0, 0]:.4f} deg/(m error)\n"
        content += f"- Velocity gain: {K[0, 2]:.4f} deg/(m/s)\n"

        text_edit.setText(content)
        layout.addWidget(text_edit)

        popup.exec()

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
        if 'serial_connection' in self.gui_modules:
            port = self.gui_modules['serial_connection'].port_combo.currentText()
        else:
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

        if self.controller_enabled:
            self.log(f"LQR weights updated: Q_pos={Q_pos:.6f}, Q_vel={Q_vel:.6f}, R={R:.6f}")

    def on_kalman_enable_change(self, enabled):
        """Handle Kalman filter enable/disable."""
        self.kalman_enabled = enabled
        if enabled:
            self.kalman_filter.reset(self.ball_pos_mm)
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

    def on_plot_enable_change(self, enabled):
        self.log(f"Plot updates: {'ENABLED' if enabled else 'DISABLED'}")

    def on_plot_rate_change(self, rate):
        self.plot_divisor = max(1, self.control_frequency // rate)  # Convert Hz to divisor

    def on_frequency_change(self, frequency):
        """Handle control frequency change."""
        self.control_frequency = frequency
        self.control_interval = 1.0 / frequency
        self.log(f"Control frequency: {frequency} Hz ({self.control_interval*1000:.2f} ms period)")
        self.setWindowTitle(f"Stewart Platform - Real Hardware Control (LQR, {frequency}Hz)")

        # Update Kalman filter dt and rebuild matrices
        self.kalman_filter.dt = self.control_interval
        self.kalman_filter._build_system_matrices()

        # Recalculate plot divisor
        if hasattr(self, 'plot_rate'):
            self.plot_divisor = max(1, frequency // self.plot_rate)

    def start_simulation(self):
        """Start hardware control thread."""
        if not self.connected:
            return

        self.simulation_running = True
        self.simulation_time = 0.0
        self.ik_timeout_count = 0
        self.first_ball_data_received = False

        gc.disable()
        self.log(f"Control started ({self.control_frequency}Hz, GC disabled)")
        self.log("Waiting for ball data from Pixy2 camera...")

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
        """Dedicated high-frequency control thread with LQR controller (50-500Hz variable)."""
        max_ik_time = ControlLoopConfig.IK_TIMEOUT_S

        # Frequency tracking
        loop_count = 0
        freq_measure_start = time.perf_counter()

        timing_breakpoints = {
            'ball_read': [],
            'ball_process': [],
            'pattern_calc': [],
            'kalman_predict': [],
            'kalman_update': [],
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

                if not self.first_ball_data_received:
                    self.first_ball_data_received = True
                    print(f"[BALL DATA] First data received at t={self.simulation_time:.2f}s")

                pixy_x = ball_data['x']
                pixy_y = ball_data['y']

                # Camera coordinate transformation
                CAMERA_HEIGHT_PIXELS = 208.0
                CAMERA_CENTER_X = 158.0
                CAMERA_CENTER_Y = 104.0

                ball_x_mm = (pixy_x - CAMERA_CENTER_X) * self.pixels_to_mm_x
                ball_y_mm = ((CAMERA_HEIGHT_PIXELS - pixy_y) - CAMERA_CENTER_Y) * self.pixels_to_mm_y

                # Use raw position
                self.ball_pos_mm = (ball_x_mm, ball_y_mm)
                self.ball_detected = ball_data['detected']

                # Update ball_pos for plotting (convert mm to m)
                self.ball_pos[0, 0] = ball_x_mm / 1000.0
                self.ball_pos[0, 1] = ball_y_mm / 1000.0

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

                # Get platform angles from FK (actual angles, not commanded)
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
                    ball_vel_mm_s = (0.0, 0.0)  # No velocity without Kalman

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
                    ball_vel_mm_s,
                    target_pos_mm
                )
                lqr_update_time = (time.perf_counter() - t4) * 1000
                timing_breakpoints['lqr_update'].append(lqr_update_time)

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
                timing_breakpoints['kalman_update'].append(0.0)
                timing_breakpoints['lqr_update'].append(0.0)
                timing_breakpoints['ik_total'].append(0.0)
                timing_breakpoints['serial_send'].append(0.0)

            self.simulation_time += self.control_interval

            # Queue GUI state update (non-blocking)
            if self.gui_update_count % 2 == 0:  # Update every other loop (50Hz)
                gui_state = {
                    'simulation_time': self.simulation_time,
                    'controller_enabled': self.controller_enabled,
                    'ball_pos': self.ball_pos_mm,
                    'ball_vel': "Detected" if self.ball_detected else "Not detected",
                    'dof_values': self.dof_values.copy(),
                    'connected': self.connected,
                    'fps': self.control_frequency,
                    'actual_frequency': self.actual_frequency,
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

                    gui_state['controller_output'] = (rx, ry)
                    gui_state['controller_magnitude'] = (magnitude, magnitude_percent)
                    gui_state['controller_error'] = (error_x, error_y)

                if self.last_sent_angles is not None:
                    gui_state['cmd_angles'] = self.last_sent_angles

                pattern_configs = {
                    'static': "Tracking: Center (0, 0)",
                    'circle': "Tracking: Circle (r=50mm, T=10s)",
                    'figure8': "Tracking: Figure-8 (60×40mm, T=12s)",
                    'star': "Tracking: 5-Point Star (r=60mm, T=15s)"
                }
                gui_state['pattern_info'] = pattern_configs.get(self.pattern_type, "")

                # Non-blocking queue put
                try:
                    self.gui_state_queue.put_nowait(gui_state)
                except:
                    pass  # Drop frame if GUI can't keep up

            self.gui_update_count += 1

            # Plot updates (controlled by toggle and rate)
            if self.gui_update_count % self.plot_divisor == 0 and self.plot_enabled:
                plot_state = {
                    'ball_pos': self.ball_pos_mm,
                    'ball_detected': self.ball_detected,
                    'ball_history_x': list(self.ball_history_x) if self.ball_history_x else [],
                    'ball_history_y': list(self.ball_history_y) if self.ball_history_y else [],
                    'pattern_type': self.pattern_type,
                    'pattern_time': self.simulation_time - self.pattern_start_time,
                    'dof_values': (self.dof_values['rx'], self.dof_values['ry']),
                }
                try:
                    self.plot_state_queue.put_nowait(plot_state)
                except:
                    self.plot_drops += 1

            # Sleep to maintain target frequency
            t_sleep = time.perf_counter()
            elapsed = time.perf_counter() - loop_start

            # Warn if loop took more than 2× period (severe overrun)
            if elapsed > self.control_interval * 2:
                print(f"WARNING: Loop took {elapsed * 1000:.1f}ms (target: {self.control_interval*1000:.1f}ms)")
                timing_breakpoints['sleep'].append(0.0)
            else:
                sleep_time = self.control_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                sleep_actual_time = (time.perf_counter() - t_sleep) * 1000
                timing_breakpoints['sleep'].append(sleep_actual_time)

            # Measure actual frequency (every 100 loops)
            loop_count += 1
            if loop_count >= 100:
                freq_measure_end = time.perf_counter()
                elapsed_time = freq_measure_end - freq_measure_start
                self.actual_frequency = loop_count / elapsed_time
                loop_count = 0
                freq_measure_start = freq_measure_end

            # Limit breakpoint history
            for key in timing_breakpoints:
                if len(timing_breakpoints[key]) > max_breakpoint_samples:
                    timing_breakpoints[key].pop(0)

    def _gui_update_loop(self):
        """GUI update loop - pulls state from queue (main thread only)."""
        if not self.simulation_running:
            return

        # Get GUI state from queue
        try:
            state = self.gui_state_queue.get_nowait()
            self.gui_builder.update_modules(state)

            # Update Kalman filter GUI if enabled
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

        except Empty:
            pass  # No new state, skip update

        # Get plot state from queue
        try:
            plot_state = self.plot_state_queue.get_nowait()
            self._update_hardware_plot_from_state(plot_state)
        except Empty:
            pass  # No new plot data

        # Schedule next update using QTimer
        QTimer.singleShot(GUIConfig.UPDATE_INTERVAL_MS, self._gui_update_loop)

    def setup_plot(self):
        """Setup plot for hardware (using PyQtGraph)."""
        super().setup_plot()

        # Add ball trail plot item
        plot_item = self.plot_widget.getPlotItem()
        self.ball_trail = plot_item.plot([], [], pen=pg.mkPen(color='#ff8888', width=2, style=Qt.PenStyle.DashLine))

    def _update_hardware_plot_from_state(self, state):
        """Update plot from queued state with ball trail."""
        # Update base plot (ball position, trajectory, target)
        self.update_plot()

        # Update ball trail from history
        if 'ball_history_x' in state and 'ball_history_y' in state:
            history_x = state['ball_history_x']
            history_y = state['ball_history_y']
            if history_x and history_y and len(history_x) > 1:
                self.ball_trail.setData(history_x, history_y)
    def show_timing_stats(self):
        """Show performance statistics with detailed breakpoint analysis."""
        print("\n" + "=" * 70)
        print("LQR HARDWARE CONTROL - DETAILED TIMING BREAKDOWN")
        print("=" * 70 + "\n")

        if hasattr(self, 'timing_breakpoints') and self.timing_breakpoints:
            breakpoint_names = {
                'ball_read': 'Ball Data Read (Queue)',
                'ball_process': 'Ball Processing (Transform/History)',
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

        stats_msg = f"Performance Statistics (LQR Hardware, {self.control_frequency}Hz)\n"
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

        if tilt_mag > MAX_TILT_ANGLE_DEG and not self.controller_enabled:
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
        # Toggle the state (call parent implementation)
        self.controller_enabled = not self.controller_enabled
        enabled = self.controller_enabled

        if enabled:
            self.controller.reset()
            self.reset_pattern()
            self.log("LQR control ENABLED")

            if 'manual_pose' in self.gui_modules:
                manual_pose = self.gui_modules['manual_pose']
                manual_pose.sliders['rx'].setEnabled(False)
                manual_pose.sliders['ry'].setEnabled(False)
                manual_pose.sliders['x'].setEnabled(False)
                manual_pose.sliders['y'].setEnabled(False)
                manual_pose.sliders['z'].setEnabled(False)
        else:
            self.log("LQR control DISABLED")

            if 'manual_pose' in self.gui_modules:
                manual_pose = self.gui_modules['manual_pose']
                manual_pose.sliders['rx'].setEnabled(True)
                manual_pose.sliders['ry'].setEnabled(True)
                manual_pose.sliders['x'].setEnabled(True)
                manual_pose.sliders['y'].setEnabled(True)
                manual_pose.sliders['z'].setEnabled(True)

    def reset_pattern(self):
        """Reset pattern timing."""
        self.pattern_start_time = self.simulation_time
        self.current_pattern.reset()
        self.log(f"Pattern reset at t={format_time(self.simulation_time)}")

        if self.controller_enabled:
            self.controller.reset()

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
    app = QApplication(sys.argv)
    simulator = HardwareStewartSimulator(app)

    simulator.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()