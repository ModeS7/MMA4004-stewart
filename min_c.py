#!/usr/bin/env python3
"""
Minimal Stewart Platform Controller

Lightweight controller with essential features only.
Supports PID/LQR control in simulation and hardware modes.
"""

import sys
import gc
import time
import threading
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QWidget, QMessageBox
from PyQt6.QtCore import QTimer, Qt

from setup.base_simulator import BaseStewartSimulator
from setup.hardware_controller_config import (WindowsTimerManager,
                                              ThreadPriorityManager, HardwareControllerConfig,
                                              LQRControllerConfig,
                                              SerialController, IKCache)
from gui import gui_modules as gm
from gui.gui_builder import GUIBuilder
from core.control_core import PIDController, LQRController, KalmanFilter, clip_tilt_vector
from core.utils import (MAX_TILT_ANGLE_DEG, ControlLoopConfig, Pixy2CameraConfig,
                         BallPhysicsConfig, VisualizationConfig, HardwareConnectionConfig,
                         PIDConfig, PerformanceConfig, GUIConfig)


class MinimalController(BaseStewartSimulator):
    """Lightweight controller with minimal GUI for rapid development."""

    def __init__(self, app):
        # Mode selection
        self.operation_mode = 'sim'  # 'sim' or 'real'
        self.controller_type_selection = 'Manual'  # 'PID', 'LQR', or 'Manual'

        # Hardware components
        self.serial_controller = None
        self.connected = False
        self.port_var = ''
        self.ik_cache = None
        self.timer_manager = WindowsTimerManager()
        self.priority_manager = ThreadPriorityManager()
        self.control_frequency = ControlLoopConfig.DEFAULT_FREQUENCY_HZ
        self.control_interval = 1.0 / ControlLoopConfig.DEFAULT_FREQUENCY_HZ
        self.use_kalman_derivative = False

        # Camera calibration parameters (pixels to mm conversion)
        self.pixy_width_mm = Pixy2CameraConfig.FOV_WIDTH_MM
        self.pixy_height_mm = Pixy2CameraConfig.FOV_HEIGHT_MM
        self.pixels_to_mm_x = Pixy2CameraConfig.PIXELS_TO_MM_X
        self.pixels_to_mm_y = Pixy2CameraConfig.PIXELS_TO_MM_Y
        self.last_ball_update = 0.0
        self.ball_pos_mm = np.array([0.0, 0.0])
        self.ball_detected = False

        # Performance tracking
        self.performance_data = {
            'loop_times': [],
            'ik_times': [],
            'serial_times': []
        }

        # Initialize Kalman filter (required before super().__init__)
        self.kalman_filter = KalmanFilter(
            process_noise_scale=1.0,
            measurement_noise_scale=1.0,
            ball_physics_params=BallPhysicsConfig.as_dict(),
            dt=self.control_interval
        )
        self.kalman_enabled = True

        # Ball position trail visualization
        self.ball_history_x = []
        self.ball_history_y = []
        self.max_history = VisualizationConfig.BALL_TRAIL_MAX_HISTORY

        # Create controller config
        controller_config = self._create_controller_config()

        # Call parent constructor
        super().__init__(app, controller_config)

        # Window title
        self.setWindowTitle("Stewart Platform - Minimal Controller")

    def _get_controller_type(self):
        """Override to return selected controller type."""
        return self.controller_type_selection

    def _create_controller_config(self):
        """Create controller config based on mode and controller type."""
        controller_name = self.controller_type_selection
        if "PID" in controller_name:
            return HardwareControllerConfig()
        elif "LQR" in controller_name:
            if self.operation_mode == 'real':
                return LQRControllerConfig(mode='hardware')
            else:
                return LQRControllerConfig(mode='simulation')
        else:
            return HardwareControllerConfig()

    def _initialize_controller(self):
        """Initialize controller based on current type."""
        if self.controller_type_selection == 'Manual':
            self.controller = None
            return

        sliders = self.controller_widgets['sliders']
        scalar_vars = self.controller_widgets['scalar_vars']

        controller_name = self.controller_config.get_controller_name()
        if "PID" in controller_name and 'kp' not in sliders:
            self.controller = None
            return
        if "LQR" in controller_name and 'Q_pos' not in sliders:
            self.controller = None
            return

        if "PID" in controller_name:
            kp = self.controller_config.get_scaled_param('kp', sliders, scalar_vars)
            ki = self.controller_config.get_scaled_param('ki', sliders, scalar_vars)
            kd = self.controller_config.get_scaled_param('kd', sliders, scalar_vars)

            self.controller = PIDController(
                kp=kp, ki=ki, kd=kd,
                output_limit=15.0,
                derivative_filter_alpha=PIDConfig.HW_DERIVATIVE_FILTER_ALPHA
            )
            self.log(f"PID initialized: Kp={kp:.6f}, Ki={ki:.6f}, Kd={kd:.6f}")

        elif "LQR" in controller_name:
            Q_pos = self.controller_config.get_scaled_param('Q_pos', sliders, scalar_vars)
            Q_vel = self.controller_config.get_scaled_param('Q_vel', sliders, scalar_vars)
            R = self.controller_config.get_scaled_param('R', sliders, scalar_vars)

            try:
                self.controller = LQRController(
                    Q_pos=Q_pos,
                    Q_vel=Q_vel,
                    R=R,
                    output_limit=15.0
                )
                self.log(f"LQR initialized: Q_pos={Q_pos:.2e}, Q_vel={Q_vel:.2e}, R={R:.2e}")
            except Exception as e:
                self.log(f"LQR initialization failed: {str(e)}")
                self.log("Adjust Q/R parameters and retry")
                self.controller = None

    def _update_controller(self, ball_pos_mm, ball_vel_mm_s, target_pos_mm, dt):
        """Update controller and return control output.

        In simulation mode, applies Kalman filtering when enabled for improved
        state estimation. Returns platform tilt angles (rx, ry) in degrees.
        """
        if self.controller is None:
            return None

        # In simulation mode, handle Kalman filter
        if self.operation_mode == 'sim':
            if self.kalman_enabled:
                # Update Kalman dt to match actual simulation timestep
                self.kalman_filter.set_dt(dt)

                # Kalman predict step (using actual platform angles from previous timestep)
                if hasattr(self, 'last_fk_rotation'):
                    rx_deg = self.last_fk_rotation[0]
                    ry_deg = self.last_fk_rotation[1]
                else:
                    rx_deg = self.dof_values['rx']
                    ry_deg = self.dof_values['ry']

                self.kalman_filter.predict([rx_deg, ry_deg])

                # Kalman update step (with camera measurement)
                self.kalman_filter.update(ball_pos_mm, self.simulation_time)

                # Get filtered estimates
                filtered_x, filtered_y = self.kalman_filter.get_position_mm()
                filtered_vx, filtered_vy = self.kalman_filter.get_velocity_mm_s()

                ball_pos_filtered = (filtered_x, filtered_y)
                ball_vel_filtered = (filtered_vx, filtered_vy)
            else:
                # Kalman filter disabled
                ball_pos_filtered = ball_pos_mm
                ball_vel_filtered = (0.0, 0.0)
        else:
            # Hardware mode uses pre-processed values from control thread
            ball_pos_filtered = ball_pos_mm
            ball_vel_filtered = ball_vel_mm_s

        controller_name = self.controller_config.get_controller_name()
        if "PID" in controller_name:
            rx, ry = self.controller.update(ball_pos_filtered, target_pos_mm, dt)
        elif "LQR" in controller_name:
            rx, ry = self.controller.update(ball_pos_filtered, ball_vel_filtered, target_pos_mm)
        else:
            return None

        return rx, ry

    def get_layout_config(self):
        """Return minimal layout configuration."""
        # Custom layout with narrow left column for controls, plot takes remaining space
        layout = {
            'columns': [
                {
                    'width': 350,  # Narrower control column
                    'scrollable': True,
                    'modules': []
                }
            ],
            'plot': {
                'enabled': True,
                'title': 'Ball Position (Top View)'
            }
        }

        # LEFT COLUMN: Minimal essential controls only
        left_modules = [
            {'type': 'mode_selector', 'args': {'current_mode': self.operation_mode}},
            {'type': 'controller_selector', 'args': {'current_controller': self.controller_type_selection}},
        ]

        # Hardware: Serial connection
        if self.operation_mode == 'real':
            left_modules.append({'type': 'serial_connection',
                                'args': {'port_var': self.port_var, 'connected_var': self.connected}})

        # Simulation control
        left_modules.append({'type': 'simulation_control'})

        # Trajectory pattern
        left_modules.append({'type': 'trajectory_pattern', 'args': {'pattern_var': self.pattern_type}})

        # Controller parameters (if not Manual)
        if self.controller_type_selection != 'Manual':
            left_modules.append({'type': 'controller',
                                'args': {'controller_config': self.controller_config,
                                        'controller_widgets': self.controller_widgets}})

        # Manual pose control
        left_modules.append({'type': 'manual_pose', 'args': {'dof_config': self.dof_config}})

        # Ball control (sim only)
        if self.operation_mode == 'sim':
            left_modules.append({'type': 'ball_control'})

        layout['columns'][0]['modules'] = left_modules

        return layout

    def _create_callbacks(self):
        """Create callback dictionary."""
        callbacks = super()._create_callbacks()

        callbacks.update({
            'mode_change': self.on_mode_change,
            'controller_type_change': self.on_controller_type_change,
        })

        if self.operation_mode == 'real':
            callbacks.update({
                'connect': self.connect_serial,
                'disconnect': self.disconnect_serial,
            })

        return callbacks

    def _build_modular_gui(self):
        """Build GUI using modular system."""
        module_registry = {
            'mode_selector': gm.ModeSelectionModule,
            'controller_selector': gm.ControllerSelectionModule,
            'simulation_control': gm.SimulationControlModule,
            'controller': gm.ControllerModule,
            'trajectory_pattern': gm.TrajectoryPatternModule,
            'ball_control': gm.BallControlModule,
            'manual_pose': gm.ManualPoseControlModule,
            'serial_connection': gm.SerialConnectionModule,
        }

        layout_config = self.get_layout_config()
        callbacks = self._create_callbacks()

        self.gui_builder = GUIBuilder(self.central_widget, module_registry)
        self.gui_modules = self.gui_builder.build(layout_config, self.colors, callbacks)

        if 'plot_panel' in self.gui_modules:
            self._create_plot(self.gui_modules['plot_panel'])

        self.update_gui_modules()

    def on_mode_change(self, mode):
        """Handle mode change between sim and real."""
        if mode == self.operation_mode:
            return

        # Stop simulation if running
        if self.simulation_running:
            self.stop_simulation()

        # Disable controller if enabled
        if self.controller_enabled:
            self.controller_enabled = False

        # Clean up hardware resources
        if self.operation_mode == 'real' and self.connected:
            self.disconnect_serial()

        self.operation_mode = mode
        self.log(f"Operation mode changed to: {mode.upper()}")

        # Update configuration and rebuild GUI
        self.controller_config = self._create_controller_config()
        self._create_controller_param_widgets()
        self._rebuild_gui()

        QApplication.processEvents()

        self._initialize_controller()

        # Update window title
        self.setWindowTitle(f"Stewart Platform - Minimal Controller [{mode.upper()}]")

    def on_controller_type_change(self, controller_type):
        """Handle controller type change between PID/LQR/Manual."""
        if controller_type == self.controller_type_selection:
            return

        # Stop simulation if running
        if self.simulation_running:
            self.stop_simulation()

        # Disable controller before switching if currently enabled
        if self.controller_enabled:
            self.controller_enabled = False
            self.log("Controller disabled for mode transition")

        self.controller_type_selection = controller_type
        self.log(f"Controller type changed to: {controller_type}")

        self.controller_config = self._create_controller_config()
        self._create_controller_param_widgets()
        self._rebuild_gui()

        QApplication.processEvents()

        self._initialize_controller()

    def on_controller_param_change(self):
        """Update controller when parameters change."""
        if self.controller is None:
            return

        sliders = self.controller_widgets['sliders']
        scalar_vars = self.controller_widgets['scalar_vars']

        if self.controller_type_selection == 'PID':
            kp = self.controller_config.get_scaled_param('kp', sliders, scalar_vars)
            ki = self.controller_config.get_scaled_param('ki', sliders, scalar_vars)
            kd = self.controller_config.get_scaled_param('kd', sliders, scalar_vars)

            self.controller.set_gains(kp, ki, kd)

            if self.controller_enabled:
                self.log(f"PID gains updated: Kp={kp:.6f}, Ki={ki:.6f}, Kd={kd:.6f}")

        elif self.controller_type_selection == 'LQR':
            Q_pos = self.controller_config.get_scaled_param('Q_pos', sliders, scalar_vars)
            Q_vel = self.controller_config.get_scaled_param('Q_vel', sliders, scalar_vars)
            R = self.controller_config.get_scaled_param('R', sliders, scalar_vars)

            self.controller.set_weights(Q_pos=Q_pos, Q_vel=Q_vel, R=R)

            if self.controller_enabled:
                self.log(f"LQR weights updated: Q_pos={Q_pos:.6f}, Q_vel={Q_vel:.6f}, R={R:.6f}")

    def _rebuild_gui(self):
        """Rebuild GUI after mode/controller change."""
        # Clear old GUI
        if hasattr(self, 'gui_modules'):
            for module in self.gui_modules.values():
                if module and hasattr(module, 'widget'):
                    try:
                        module.widget.deleteLater()
                    except:
                        pass

        old_central = self.central_widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        old_central.deleteLater()

        gc.collect()

        # Rebuild GUI
        self._build_modular_gui()

        QApplication.processEvents()

    def start_simulation(self):
        """Start simulation or hardware control based on mode."""
        if self.operation_mode == 'real':
            # Hardware mode requires active connection
            if not self.connected:
                self.log("Hardware connection required before starting")
                return

            if self.simulation_running:
                return

            self.simulation_running = True
            self.simulation_time = 0.0

            gc.disable()
            self.log(f"Control loop started: {self.control_frequency} Hz (garbage collection disabled)")

            self.control_thread = threading.Thread(target=self._control_thread_func, daemon=True)
            self.control_thread.start()

            self.last_gui_update = time.time()
            self._gui_update_loop()
        else:
            # Simulation mode: use parent's timer-based loop
            super().start_simulation()

    def _control_thread_func(self):
        """Hardware control thread."""
        while self.simulation_running:
            loop_start = time.perf_counter()

            # Get ball data from Pixy2 camera via serial
            ball_data = self.serial_controller.get_latest_ball_data()

            if ball_data is not None:
                self.last_ball_update = self.simulation_time

                pixy_x = ball_data['x']
                pixy_y = ball_data['y']

                # Camera dimensions: 316×208 pixels, origin at top-left
                CAMERA_HEIGHT_PIXELS = 208.0

                ball_x_mm = (pixy_x - Pixy2CameraConfig.CENTER_X) * self.pixels_to_mm_x
                ball_y_mm = (CAMERA_HEIGHT_PIXELS - pixy_y - Pixy2CameraConfig.CENTER_Y) * self.pixels_to_mm_y

                self.ball_pos_mm = np.array([ball_x_mm, ball_y_mm])
                self.ball_detected = ball_data.get('detected', False)

                # Update ball_pos tensor for plotting (convert mm to m)
                self.ball_pos[0, 0] = ball_x_mm / 1000.0
                self.ball_pos[0, 1] = ball_y_mm / 1000.0

                # Track ball history for trail (only if detected)
                if self.ball_detected:
                    self.ball_history_x.append(ball_x_mm)
                    self.ball_history_y.append(ball_y_mm)
                    if len(self.ball_history_x) > self.max_history:
                        self.ball_history_x.pop(0)
                        self.ball_history_y.pop(0)

            # Kalman filter (always enabled in unified controller)
            if self.kalman_enabled:
                # Predict step (always predict, even without new measurement)
                rx_deg = self.prev_platform_angles.get('rx', 0.0)
                ry_deg = self.prev_platform_angles.get('ry', 0.0)
                self.kalman_filter.predict([rx_deg, ry_deg])

                # Update step (only when ball is detected)
                if ball_data is not None and self.ball_detected:
                    self.kalman_filter.update(self.ball_pos_mm, self.simulation_time)

                # Get filtered estimates
                filtered_x, filtered_y = self.kalman_filter.get_position_mm()
                filtered_vx, filtered_vy = self.kalman_filter.get_velocity_mm_s()
                ball_pos_mm = np.array([filtered_x, filtered_y])
                ball_vel_mm_s = np.array([filtered_vx, filtered_vy])
            else:
                # Use raw camera data (shouldn't happen, but handle it)
                ball_pos_mm = self.ball_pos_mm
                ball_vel_mm_s = np.array([0.0, 0.0])

            # Get target position
            pattern_time = self.simulation_time - self.pattern_start_time
            target_x, target_y = self.current_pattern.get_position(pattern_time)
            target_pos_mm = (target_x, target_y)

            # Update controller if enabled and ball is detected
            if self.controller_enabled and self.controller is not None and self.ball_detected:
                control_output = self._update_controller(
                    ball_pos_mm, ball_vel_mm_s, target_pos_mm, self.control_interval
                )

                if control_output is not None:
                    rx, ry = control_output
                    rx, ry, _ = clip_tilt_vector(rx, ry, 15.0)

                    # Update dof_values so GUI reflects controller state
                    self.dof_values['rx'] = rx
                    self.dof_values['ry'] = ry

                    # Calculate servo angles
                    translation = np.array([0.0, 0.0, self.ik.home_height_top_surface])
                    rotation = np.array([rx, ry, 0.0])

                    # Check cache first
                    if self.ik_cache:
                        angles = self.ik_cache.get(translation, rotation)
                        if angles is None:
                            angles = self.ik.calculate_servo_angles(translation, rotation, self.use_top_surface_offset)
                            if angles is not None:
                                self.ik_cache.put(translation, rotation, angles)
                    else:
                        angles = self.ik.calculate_servo_angles(translation, rotation, self.use_top_surface_offset)

                    if angles is not None:
                        self.serial_controller.send_servo_angles(angles)
                        self.last_cmd_angles = angles

                        self.prev_platform_angles['rx'] = rx
                        self.prev_platform_angles['ry'] = ry

            # Manual control when controller disabled
            elif not self.controller_enabled:
                translation = np.array([self.dof_values['x'], self.dof_values['y'], self.dof_values['z']])
                rotation = np.array([self.dof_values['rx'], self.dof_values['ry'], self.dof_values['rz']])

                angles = self.ik.calculate_servo_angles(translation, rotation, self.use_top_surface_offset)
                if angles is not None:
                    self.serial_controller.send_servo_angles(angles)
                    self.last_cmd_angles = angles

            # Timing control
            loop_time = (time.perf_counter() - loop_start)
            sleep_time = self.control_interval - loop_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            self.simulation_time += self.control_interval

    def _gui_update_loop(self):
        """Update GUI at 10 Hz while control thread is active."""
        if not self.simulation_running:
            return

        # Update GUI modules
        self.update_gui_modules()

        # Update plot
        self.update_plot()

        # Update ball trail from history
        if hasattr(self, 'ball_trail') and self.ball_history_x and len(self.ball_history_x) > 1:
            self.ball_trail.setData(self.ball_history_x, self.ball_history_y)

        # Schedule next update
        QTimer.singleShot(GUIConfig.PLOT_DISABLED_INTERVAL_MS, self._gui_update_loop)

    def _create_plot(self, plot_panel):
        """Override to add ball trail plot item."""
        # Call parent to create standard plot
        super()._create_plot(plot_panel)

        # Add ball trail plot item (dashed red line)
        plot_item = self.plot_widget.getPlotItem()
        self.ball_trail = plot_item.plot([], [], pen=pg.mkPen(color='#ff8888', width=2, style=Qt.PenStyle.DashLine))

    def connect_serial(self):
        """Establish serial connection to hardware."""
        if self.connected and self.serial_controller is not None:
            self.log("Connection already established")
            return

        if 'serial_connection' in self.gui_modules:
            port = self.gui_modules['serial_connection'].port_combo.currentText()
        else:
            return

        if not port:
            QMessageBox.critical(self, "Error", "No port selected")
            return

        self.serial_controller = SerialController(port)
        success, message = self.serial_controller.connect()

        if success:
            self.connected = True
            self.log(f"Connected to {port}")

            time.sleep(HardwareConnectionConfig.POST_CONNECTION_DELAY_S)
            self.serial_controller.set_servo_speed(0)
            time.sleep(HardwareConnectionConfig.POST_SERVO_SPEED_DELAY_S)
            self.serial_controller.set_servo_acceleration(0)
            time.sleep(HardwareConnectionConfig.POST_SERVO_ACCEL_DELAY_S)
            self.log("Servo parameters configured: Speed=0, Acceleration=0")

            success_timer, msg_timer = self.timer_manager.set_high_resolution()
            self.log(msg_timer)

            self.initialize_ik_cache()

            if 'simulation_control' in self.gui_modules:
                self.gui_modules['simulation_control'].start_btn.setEnabled(True)

            if 'serial_connection' in self.gui_modules:
                self.gui_modules['serial_connection'].update({'connected': True})
        else:
            QMessageBox.critical(self, "Error", message)
            self.log(f"Error: {message}")

    def disconnect_serial(self):
        """Disconnect from serial hardware."""
        if self.simulation_running:
            self.stop_simulation()

        if self.serial_controller:
            self.serial_controller.disconnect()

        self.connected = False

        if 'simulation_control' in self.gui_modules:
            self.gui_modules['simulation_control'].start_btn.setEnabled(False)

        if 'serial_connection' in self.gui_modules:
            self.gui_modules['serial_connection'].update({'connected': False})

        self.log("Disconnected")

    def initialize_ik_cache(self):
        """Pre-compute common inverse kinematics solutions."""
        if not hasattr(self, 'ik_cache') or self.ik_cache is None:
            self.ik_cache = IKCache(max_size=PerformanceConfig.IK_CACHE_SIZE)

        self.log("Initializing IK cache...")
        tilts = np.arange(PerformanceConfig.IK_PREWARM_TILT_RANGE[0],
                          PerformanceConfig.IK_PREWARM_TILT_RANGE[1] + 1,
                          PerformanceConfig.IK_PREWARM_TILT_STEP)
        count = 0
        for rx in tilts:
            for ry in tilts:
                translation = np.array([0.0, 0.0, self.ik.home_height_top_surface])
                rotation = np.array([float(rx), float(ry), 0.0])
                angles = self.ik.calculate_servo_angles(translation, rotation, self.use_top_surface_offset)
                if angles is not None:
                    self.ik_cache.put(translation, rotation, angles)
                    count += 1

        self.log(f"IK cache initialized: {count} poses cached")

    def update_gui_modules(self):
        """Override to provide hardware-specific state in real mode."""
        if self.operation_mode == 'real':
            # Hardware mode: use actual hardware state
            ball_x_mm = self.ball_pos[0, 0].item() * 1000
            ball_y_mm = self.ball_pos[0, 1].item() * 1000

            # Get velocity from Kalman filter (always enabled in unified)
            vel_x_mm, vel_y_mm = self.kalman_filter.get_velocity_mm_s()

            # Calculate FK from sent servo angles
            if hasattr(self, 'last_cmd_angles') and self.last_cmd_angles is not None:
                fk_translation, fk_rotation, fk_success, _ = self.ik.calculate_forward_kinematics(
                    self.last_cmd_angles,
                    use_top_surface_offset=self.use_top_surface_offset
                )
                if fk_success:
                    self.last_fk_translation = fk_translation
                    self.last_fk_rotation = fk_rotation
                else:
                    fk_translation = np.zeros(3)
                    fk_rotation = np.zeros(3)
            else:
                fk_translation = np.zeros(3)
                fk_rotation = np.zeros(3)

            state = {
                'simulation_time': self.simulation_time,
                'controller_enabled': self.controller_enabled,
                'ball_pos': (ball_x_mm, ball_y_mm),
                'ball_vel': (vel_x_mm, vel_y_mm),
                'dof_values': self.dof_values,
                'cmd_angles': self.last_cmd_angles,
                'actual_angles': self.last_cmd_angles,  # No servo simulation in hardware
                'fk_translation': fk_translation,
                'fk_rotation': fk_rotation,
                'frequency': self.control_frequency,
            }

            # Add Kalman filter state (always enabled in unified, flat keys for GUI module)
            kalman_pos = self.kalman_filter.get_position_mm()
            kalman_vel = self.kalman_filter.get_velocity_mm_s()
            kalman_std_pos = self.kalman_filter.get_position_uncertainty()
            kalman_stats = self.kalman_filter.get_statistics()

            state['kalman_position'] = kalman_pos
            state['kalman_velocity'] = kalman_vel
            state['kalman_uncertainty'] = kalman_std_pos
            state['kalman_stats'] = kalman_stats

            if self.controller_enabled:
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

            # Update all modules (skip plot_panel which is a Qt widget, not a GUIModule)
            for key, module in self.gui_modules.items():
                if key == 'plot_panel':
                    continue
                if module and hasattr(module, 'update') and hasattr(module, 'widget'):
                    module.update(state)
        else:
            # Simulation mode: use parent implementation first
            super().update_gui_modules()

            # Then update Kalman filter GUI (always enabled in unified)
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


def main():
    """Launch minimal controller application."""
    app = QApplication(sys.argv)
    controller = MinimalController(app)
    controller.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
