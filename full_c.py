#!/usr/bin/env python3
"""
Comprehensive Stewart Platform Controller
Unified interface supporting all features from PID_sim, LQR_sim, PID_real, LQR_real

Inherits from BaseStewartSimulator and adds:
- Mode switching: Simulation / Hardware
- Controller switching: PID / LQR / Manual
- Conditional GUI module loading based on mode and controller
"""

import sys
import gc
import time
import threading
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QMessageBox, QDialog, QVBoxLayout, QLabel, QTextEdit,
                              QPushButton, QApplication, QWidget, QHBoxLayout, QCheckBox)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import QTimer, Qt

from setup.base_simulator import BaseStewartSimulator
from setup.hardware_controller_config import (SerialController, IKCache, WindowsTimerManager,
                                               ThreadPriorityManager, HardwareControllerConfig,
                                               LQRControllerConfig)
from gui import gui_modules as gm
from gui.gui_builder import create_standard_layout, GUIBuilder
from core.control_core import (PIDController, LQRController, KalmanFilter, clip_tilt_vector,
                                OrientationKalmanFilter, apply_imu_transforms)
from core.utils import IKZOptimizationConfig, MAX_TILT_ANGLE_DEG

# Control configurations
DEFAULT_HW_FREQUENCY_HZ = 250


class ComprehensiveStewartController(BaseStewartSimulator):
    """Comprehensive unified Stewart Platform controller with all features."""

    def __init__(self, app):
        # Mode selection
        self.operation_mode = 'real'  # 'sim' or 'real'
        self.controller_type_selection = 'LQR'  # 'PID', 'LQR', or 'Manual'

        # Hardware-specific components (initialize before super().__init__)
        self.serial_controller = None
        self.connected = False
        self.port_var = ''
        self.ik_cache = None
        self.timer_manager = WindowsTimerManager()
        self.priority_manager = ThreadPriorityManager()
        self.control_frequency = DEFAULT_HW_FREQUENCY_HZ
        self.control_interval = 1.0 / DEFAULT_HW_FREQUENCY_HZ
        self.use_kalman_derivative = False  # PID-specific

        # Pixy2 camera calibration (pixels to mm)
        # Field of view at platform surface (calibrated for current camera distance)
        self.pixy_width_mm = 558.0   # Was 350.0 before camera repositioning
        self.pixy_height_mm = 424.0  # Was 266.0 before camera repositioning
        self.pixels_to_mm_x = self.pixy_width_mm / 316.0  # = 1.766 mm/pixel
        self.pixels_to_mm_y = self.pixy_height_mm / 208.0  # = 2.038 mm/pixel
        self.last_ball_update = 0.0
        self.ball_pos_mm = np.array([0.0, 0.0])  # Initialize to center
        self.ball_detected = False  # Track whether ball is currently detected

        # Performance tracking for hardware mode
        self.performance_data = {
            'loop_times': [],
            'ik_times': [],
            'serial_times': []
        }

        # Plot control settings
        self.plot_enabled = True
        self.plot_rate_hz = 10

        # Initialize Kalman filter (required before super().__init__)
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
            dt=self.control_interval
        )
        self.kalman_enabled = False  # User enables via GUI toggle

        # Ball history for trail visualization
        self.ball_history_x = []
        self.ball_history_y = []
        self.max_history = 100  # ~5 seconds at camera rate

        # IMU orientation tracking (defaults from rot_core.py)
        ACCEL_NOISE = 1.0
        GYRO_NOISE = 0.0224
        PROCESS_NOISE_ANGLE = 0.001
        PROCESS_NOISE_BIAS = 0.00001
        GYRO_BIAS_X = 0.112679
        GYRO_BIAS_Y = 0.031500
        ACCEL_AXIS_FLIP = np.array([1, 1, 1])
        GYRO_AXIS_FLIP = np.array([1, 1, 1])
        ACCEL_ROTATION = np.eye(3)
        GYRO_ROTATION = np.eye(3)
        ACCEL_MAGNITUDE_THRESHOLD = 1.0
        GYRO_MAGNITUDE_THRESHOLD = 0.5

        # Initialize IMU Kalman filter
        self.orientation_kalman = OrientationKalmanFilter(
            accel_noise=ACCEL_NOISE,
            gyro_noise=GYRO_NOISE,
            process_noise_angle=PROCESS_NOISE_ANGLE,
            process_noise_bias=PROCESS_NOISE_BIAS,
            accel_axis_flip=ACCEL_AXIS_FLIP,
            gyro_axis_flip=GYRO_AXIS_FLIP,
            accel_rotation=ACCEL_ROTATION,
            gyro_rotation=GYRO_ROTATION,
            initial_bias_x=GYRO_BIAS_X,
            initial_bias_y=GYRO_BIAS_Y,
            gyro_scale_multiplier=6.6,
            accel_magnitude_threshold=ACCEL_MAGNITUDE_THRESHOLD,
            gyro_magnitude_threshold=GYRO_MAGNITUDE_THRESHOLD
        )

        # IMU state
        self.current_rx_imu = 0.0
        self.current_ry_imu = 0.0
        self.imu_tilt_correction_enabled = False
        self.imu_compensation_gain = 1.0  # Full compensation

        # IMU initialization and calibration
        self.imu_initializing = False
        self.initialization_duration = 3.0  # 3 seconds stabilization
        self.initialization_start_time = None
        self.initialization_time_remaining = 0.0
        self.imu_calibrating = False
        self.calibration_duration = 10.0  # 10 seconds calibration
        self.calibration_start_time = None
        self.calibration_time_remaining = 0.0
        self.calibration_raw_data = {'gyro': [], 'accel': [], 'mag': []}
        self.calibrated_gravity = None

        # Create controller config based on mode and controller type
        controller_config = self._create_controller_config()

        # Call parent constructor
        super().__init__(app, controller_config)

        # Override window title
        self.setWindowTitle(f"Stewart Platform - {self.controller_type_selection} [{self.operation_mode.upper()}]")

    def _get_controller_type(self):
        """Override to return selected controller type."""
        return self.controller_type_selection

    def _create_controller_config(self):
        """Create appropriate controller config based on mode and controller type."""
        if self.controller_type_selection == 'PID':
            return HardwareControllerConfig()
        elif self.controller_type_selection == 'LQR':
            if self.operation_mode == 'real':
                return LQRControllerConfig(mode='hardware')
            else:
                return LQRControllerConfig(mode='simulation')
        else:  # Manual
            # For Manual mode, use a dummy config (won't be used)
            return HardwareControllerConfig()

    def _initialize_controller(self):
        """Initialize controller based on current type."""
        if self.controller_type_selection == 'Manual':
            self.controller = None
            return

        sliders = self.controller_widgets['sliders']
        scalar_vars = self.controller_widgets['scalar_vars']

        # Check if sliders exist (GUI might not be built yet)
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
                derivative_filter_alpha=0.1
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
                self.log("Try adjusting Q/R parameters")
                self.controller = None

    def _update_controller(self, ball_pos_mm, ball_vel_mm_s, target_pos_mm, dt):
        """Update controller and return control output.

        Handles Kalman filter in simulation mode:
        - When enabled: predict, update, use filtered position/velocity
        - When disabled: use raw position, velocity = (0, 0)
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
                # No Kalman filter - no velocity available
                ball_pos_filtered = ball_pos_mm
                ball_vel_filtered = (0.0, 0.0)  # Zero velocity when filter disabled
        else:
            # Hardware mode: position and velocity already handled in control thread
            ball_pos_filtered = ball_pos_mm
            ball_vel_filtered = ball_vel_mm_s

        controller_name = self.controller_config.get_controller_name()
        if "PID" in controller_name:
            # PID uses position error and dt
            rx, ry = self.controller.update(ball_pos_filtered, target_pos_mm, dt)
        elif "LQR" in controller_name:
            # LQR uses position, velocity, and target
            rx, ry = self.controller.update(ball_pos_filtered, ball_vel_filtered, target_pos_mm)
        else:
            return None

        return rx, ry

    def _create_callbacks(self):
        """Create callback dictionary with all features."""
        callbacks = super()._create_callbacks()

        # Add mode/controller switching
        callbacks.update({
            'mode_change': self.on_mode_change,
            'controller_type_change': self.on_controller_type_change,
        })

        # Add plot control callbacks
        callbacks.update({
            'plot_enable_change': self.on_plot_enable_change,
            'plot_rate_change': self.on_plot_rate_change,
        })

        # Add hardware callbacks
        if self.operation_mode == 'real':
            callbacks.update({
                'connect': self.connect_serial,
                'disconnect': self.disconnect_serial,
                'show_stats': self.show_timing_stats,
                'frequency_change': self.on_frequency_change,
            })

        # Add Kalman filter callbacks
        callbacks.update({
            'kalman_enable_change': self.on_kalman_enable_change,
            'kalman_param_change': self.on_kalman_param_change,
            'kalman_reset': self.on_kalman_reset,
        })

        # Add IMU callbacks (hardware only)
        if self.operation_mode == 'real':
            callbacks.update({
                'imu_tilt_correction_toggle': self.on_imu_tilt_correction_toggle,
                'imu_kalman_param_change': self.on_imu_kalman_param_change,
                'imu_motion_param_change': self.on_imu_motion_param_change,
                'imu_detection_toggle': self.on_imu_detection_toggle,
                'imu_mag_toggle': self.on_imu_mag_toggle,
            })

        # Add LQR-specific callbacks
        if self.controller_type_selection == 'LQR':
            callbacks['show_gain_matrix'] = self.show_gain_matrix

        # Add PID-specific callbacks
        if self.controller_type_selection == 'PID':
            callbacks['kalman_derivative_toggle'] = self.on_kalman_derivative_toggle

        return callbacks

    def get_layout_config(self):
        """Return layout configuration based on mode and controller."""
        # Always scrollable columns
        layout = create_standard_layout(scrollable_columns=True, include_plot=True)

        # Build left column modules
        left_modules = []

        # Mode selector (always show)
        left_modules.append({'type': 'mode_selector', 'args': {'current_mode': self.operation_mode}})

        # Controller selector (always show)
        left_modules.append({'type': 'controller_selector',
                            'args': {'current_controller': self.controller_type_selection}})

        # Hardware-only: Serial connection
        if self.operation_mode == 'real':
            left_modules.append({'type': 'serial_connection',
                                'args': {'port_var': self.port_var, 'connected_var': self.connected}})
            left_modules.append({'type': 'control_frequency',
                               'args': {'frequency_var': self.control_frequency,
                                       'min_freq': 50, 'max_freq': 500}})

        # Simulation control
        left_modules.append({'type': 'simulation_control'})

        # Trajectory pattern
        left_modules.append({'type': 'trajectory_pattern', 'args': {'pattern_var': self.pattern_type}})

        # Simulation-only: Ball control
        if self.operation_mode == 'sim':
            left_modules.append({'type': 'ball_control'})

        # Simulation-only: Camera noise
        if self.operation_mode == 'sim':
            left_modules.append({'type': 'pixy2_camera',
                               'args': {'camera': getattr(self, 'pixy_camera', None)}})

        # Kalman filter
        left_modules.append({'type': 'kalman_filter',
                           'args': {'kalman_filter': getattr(self, 'kalman_filter', None)}})

        # Hardware-only: IMU modules
        if self.operation_mode == 'real':
            left_modules.append({'type': 'imu_kalman_parameters',
                               'args': {'orientation_kalman': getattr(self, 'orientation_kalman', None)}})
            left_modules.append({'type': 'imu_motion_detection'})

        # Configuration
        left_modules.append({'type': 'configuration',
                           'args': {'use_offset_var': self.use_top_surface_offset}})

        # Plot control
        left_modules.append({'type': 'plot_control',
                           'args': {'plot_enabled_var': self.plot_enabled,
                                   'plot_rate_var': self.plot_rate_hz}})

        # Ball state
        left_modules.append({'type': 'ball_state'})

        layout['columns'][0]['modules'] = left_modules

        # Build right column modules
        right_modules = []

        # Controller parameters (if not Manual mode)
        if self.controller_type_selection != 'Manual':
            right_modules.append({'type': 'controller',
                                 'args': {'controller_config': self.controller_config,
                                         'controller_widgets': self.controller_widgets}})

        # Servo angles
        right_modules.append({'type': 'servo_angles', 'args': {'show_actual': self.operation_mode == 'sim'}})

        # Platform pose
        right_modules.append({'type': 'platform_pose'})

        # Controller output
        right_modules.append({'type': 'controller_output',
                            'args': {'controller_name': self.controller_type_selection}})

        # Manual pose
        right_modules.append({'type': 'manual_pose', 'args': {'dof_config': self.dof_config}})

        # IK Z Optimization
        right_modules.append({'type': 'ik_z_optimization'})

        # Hardware-only: Performance stats
        if self.operation_mode == 'real':
            right_modules.append({'type': 'performance_stats'})

        # Debug log
        right_modules.append({'type': 'debug_log', 'args': {'height': 8}})

        layout['columns'][1]['modules'] = right_modules

        return layout

    def _build_modular_gui(self):
        """Override to add mode/controller selector modules and build GUI."""
        # Create extended module registry with our new modules
        module_registry = {
            'mode_selector': gm.ModeSelectionModule,
            'controller_selector': gm.ControllerSelectionModule,
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
            'pixy2_camera': gm.Pixy2CameraModule,
            'kalman_filter': gm.KalmanFilterModule,
            'control_frequency': gm.ControlFrequencyModule,
            'plot_control': gm.PlotControlModule,
            'imu_kalman_parameters': gm.IMUKalmanParametersModule,
            'imu_motion_detection': gm.IMUMotionDetectionModule,
            'ik_z_optimization': gm.IKZOptimizationModule,
        }

        layout_config = self.get_layout_config()
        callbacks = self._create_callbacks()

        self.gui_builder = GUIBuilder(self.central_widget, module_registry)
        self.gui_modules = self.gui_builder.build(layout_config, self.colors, callbacks)

        if 'plot_panel' in self.gui_modules:
            self._create_plot(self.gui_modules['plot_panel'])

        # Add controller-specific widgets
        self._add_controller_specific_widgets()

    def _add_controller_specific_widgets(self):
        """Add controller-specific widgets (PID Kalman derivative, LQR gain matrix)."""
        if self.controller_type_selection == 'PID' and 'controller' in self.gui_modules:
            # Add "Use Kalman Velocity for Derivative" checkbox
            controller_frame = self.gui_modules['controller'].widget
            controller_layout = controller_frame.layout()

            derivative_widget = QWidget()
            derivative_layout = QHBoxLayout(derivative_widget)
            derivative_layout.setContentsMargins(0, 10, 0, 0)

            derivative_checkbox = QCheckBox("Use Kalman Velocity for Derivative")
            derivative_checkbox.setChecked(self.use_kalman_derivative)
            derivative_checkbox.stateChanged.connect(self.on_kalman_derivative_toggle)
            derivative_layout.addWidget(derivative_checkbox)

            self.derivative_status = QLabel("[OFF]")
            self.derivative_status.setStyleSheet(f"color: {self.colors['border']}; font-size: 10pt;")
            derivative_layout.addWidget(self.derivative_status)
            derivative_layout.addStretch()

            controller_layout.addWidget(derivative_widget)

        elif self.controller_type_selection == 'LQR' and 'controller' in self.gui_modules:
            # Add "Show Gain Matrix" button
            controller_frame = self.gui_modules['controller'].widget
            controller_layout = controller_frame.layout()

            gain_widget = QWidget()
            gain_layout = QHBoxLayout(gain_widget)
            gain_layout.setContentsMargins(0, 10, 0, 0)

            gain_btn = QPushButton("Show Gain Matrix")
            gain_btn.clicked.connect(self.show_gain_matrix)
            gain_btn.setMinimumWidth(150)
            gain_layout.addWidget(gain_btn)
            gain_layout.addStretch()

            controller_layout.addWidget(gain_widget)

    # ============================================================================
    # Hardware-specific methods
    # ============================================================================

    def connect_serial(self):
        """Connect to hardware."""
        # Prevent double connection
        if self.connected and self.serial_controller is not None:
            self.log("Already connected")
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

            time.sleep(0.5)
            self.serial_controller.set_servo_speed(0)
            time.sleep(0.1)
            self.serial_controller.set_servo_acceleration(0)
            time.sleep(0.2)
            self.log("Servos: Speed=0, Accel=0")

            success_timer, msg_timer = self.timer_manager.set_high_resolution()
            self.log(msg_timer)

            self.prewarm_ik_cache()

            # Start IMU initialization and calibration sequence
            self.start_imu_initialization()

            if 'simulation_control' in self.gui_modules:
                self.gui_modules['simulation_control'].start_btn.setEnabled(True)

            if 'serial_connection' in self.gui_modules:
                self.gui_modules['serial_connection'].update({'connected': True})
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

        if 'serial_connection' in self.gui_modules:
            self.gui_modules['serial_connection'].update({'connected': False})

        self.log("Disconnected")

    def start_imu_initialization(self):
        """Start 3-second IMU initialization phase."""
        self.imu_initializing = True
        self.initialization_start_time = time.time()
        self.initialization_time_remaining = self.initialization_duration
        self.log("IMU initialization: 3s stabilization...")

    def start_imu_calibration(self):
        """Start 10-second IMU calibration phase."""
        self.imu_calibrating = True
        self.calibration_start_time = time.time()
        self.calibration_time_remaining = self.calibration_duration
        self.calibration_raw_data = {'gyro': [], 'accel': [], 'mag': []}
        self.log("IMU calibration: Keep platform stationary for 10s...")

    def finish_imu_calibration(self):
        """Process calibration data and initialize Kalman filter."""
        gyro_data = self.calibration_raw_data['gyro']
        accel_data = self.calibration_raw_data['accel']

        if not gyro_data or not accel_data:
            self.log("WARNING: No calibration data collected!")
            self.imu_calibrating = False
            return

        # Convert to arrays
        accel_samples = np.array([sample[1] for sample in accel_data])
        gyro_samples = np.array([sample[1] for sample in gyro_data])

        # Calculate mean raw values
        accel_mean_raw = np.mean(accel_samples, axis=0)
        gyro_mean_raw = np.mean(gyro_samples, axis=0)

        # Apply transformations (same as in OrientationKalmanFilter)
        accel_scale = 0.001 * 9.81  # LSM303: 1mg/LSB -> m/s²
        gyro_scale = 0.00875 * np.pi / 180  # L3GD20: 8.75 mdps/LSB -> rad/s

        accel_mean = apply_imu_transforms(accel_mean_raw,
                                         self.orientation_kalman.accel_axis_flip,
                                         self.orientation_kalman.accel_rotation,
                                         accel_scale)
        gyro_mean = apply_imu_transforms(gyro_mean_raw,
                                        self.orientation_kalman.gyro_axis_flip,
                                        self.orientation_kalman.gyro_rotation,
                                        gyro_scale)

        # Store calibrated gravity for initialization
        self.calibrated_gravity = accel_mean.copy()

        # Initialize Kalman filter
        self.orientation_kalman.initialize(accel_mean_raw, calibrated_gravity=self.calibrated_gravity)

        # Check tilt
        ax, ay, az = accel_mean
        tilt_x = np.arctan2(ay, az)
        tilt_y = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        tilt_x_deg = np.degrees(tilt_x)
        tilt_y_deg = np.degrees(tilt_y)

        if abs(tilt_x_deg) > 5 or abs(tilt_y_deg) > 5:
            self.log(f"WARNING: IMU tilted during calibration! RX={tilt_x_deg:.1f}°, RY={tilt_y_deg:.1f}°")
        else:
            self.log(f"IMU level check: RX={tilt_x_deg:.1f}°, RY={tilt_y_deg:.1f}° (good)")

        self.log(f"Gyro bias [°/s]: X={np.degrees(gyro_mean[0]):.4f}, Y={np.degrees(gyro_mean[1]):.4f}")
        self.log("IMU calibration complete!")
        self.imu_calibrating = False

    def prewarm_ik_cache(self):
        """Pre-calculate common IK solutions."""
        if not hasattr(self, 'ik_cache') or self.ik_cache is None:
            self.ik_cache = IKCache(max_size=5000)

        self.log("Pre-warming IK cache...")
        tilts = np.arange(-15, 16, 2)
        count = 0
        for rx in tilts:
            for ry in tilts:
                translation = np.array([0.0, 0.0, self.ik.home_height_top_surface])
                rotation = np.array([float(rx), float(ry), 0.0])
                angles = self.ik.calculate_servo_angles(translation, rotation, self.use_top_surface_offset)
                if angles is not None:
                    self.ik_cache.put(translation, rotation, angles)
                    count += 1

        self.log(f"IK cache pre-warmed: {count} poses")

    def on_frequency_change(self, frequency):
        """Handle control frequency change."""
        self.control_frequency = frequency
        self.control_interval = 1.0 / frequency

        # Update Kalman filter dt
        if hasattr(self, 'kalman_filter') and self.kalman_filter:
            self.kalman_filter.set_dt(self.control_interval)

        # Update window title
        self.setWindowTitle(f"Stewart Platform - {self.controller_type_selection} [{self.operation_mode.upper()}] @ {frequency}Hz")
        self.log(f"Control frequency: {frequency}Hz")

    def on_plot_enable_change(self, enabled):
        """Handle plot enable/disable."""
        self.plot_enabled = enabled
        status = "ENABLED" if enabled else "DISABLED"
        self.log(f"Plot updates: {status}")

    def on_plot_rate_change(self, rate):
        """Handle plot refresh rate change."""
        self.plot_rate_hz = rate
        self.log(f"Plot refresh rate: {rate} Hz")

    def show_timing_stats(self):
        """Show performance timing statistics."""
        if not hasattr(self, 'performance_data') or not self.performance_data:
            QMessageBox.information(self, "Performance Stats", "No data available. Start the controller first.")
            return

        # Calculate statistics
        loop_times = self.performance_data.get('loop_times', [])
        ik_times = self.performance_data.get('ik_times', [])
        serial_times = self.performance_data.get('serial_times', [])

        if not loop_times:
            QMessageBox.information(self, "Performance Stats", "No timing data collected yet.")
            return

        stats_text = "=== PERFORMANCE STATISTICS ===\n\n"

        stats_text += f"Loop Time (ms):\n"
        stats_text += f"  Avg: {np.mean(loop_times):.2f}\n"
        stats_text += f"  Min: {np.min(loop_times):.2f}\n"
        stats_text += f"  Max: {np.max(loop_times):.2f}\n"
        stats_text += f"  95th: {np.percentile(loop_times, 95):.2f}\n\n"

        if ik_times:
            stats_text += f"IK Time (ms):\n"
            stats_text += f"  Avg: {np.mean(ik_times):.2f}\n"
            stats_text += f"  Max: {np.max(ik_times):.2f}\n\n"

        if serial_times:
            stats_text += f"Serial Send Time (ms):\n"
            stats_text += f"  Avg: {np.mean(serial_times):.2f}\n"
            stats_text += f"  Max: {np.max(serial_times):.2f}\n\n"

        if hasattr(self, 'ik_cache') and self.ik_cache:
            hit_rate = self.ik_cache.get_hit_rate() * 100
            stats_text += f"IK Cache:\n"
            stats_text += f"  Hit rate: {hit_rate:.1f}%\n"
            stats_text += f"  Hits: {self.ik_cache.hits}\n"
            stats_text += f"  Misses: {self.ik_cache.misses}\n"

        # Show dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Performance Statistics")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout()

        text_edit = QTextEdit()
        text_edit.setPlainText(stats_text)
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Consolas", 10))
        layout.addWidget(text_edit)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.setLayout(layout)
        dialog.exec()

    # ============================================================================
    # PID-specific methods
    # ============================================================================

    def on_kalman_derivative_toggle(self):
        """Toggle Kalman derivative option for PID."""
        self.use_kalman_derivative = not self.use_kalman_derivative

        if hasattr(self, 'derivative_status'):
            if self.use_kalman_derivative:
                self.derivative_status.setText("[ON]")
                self.derivative_status.setStyleSheet(f"color: {self.colors['success']}; font-size: 10pt;")
            else:
                self.derivative_status.setText("[OFF]")
                self.derivative_status.setStyleSheet(f"color: {self.colors['border']}; font-size: 10pt;")

        status = "ON" if self.use_kalman_derivative else "OFF"
        self.log(f"Kalman derivative: {status}")

    # ============================================================================
    # LQR-specific methods
    # ============================================================================

    def show_gain_matrix(self):
        """Show LQR gain matrix dialog."""
        if not hasattr(self, 'controller') or self.controller is None:
            QMessageBox.warning(self, "Gain Matrix", "Controller not initialized")
            return

        if not isinstance(self.controller, LQRController):
            QMessageBox.warning(self, "Gain Matrix", "Not an LQR controller")
            return

        K = self.controller.K

        gain_text = "=== LQR GAIN MATRIX ===\n\n"
        gain_text += "State vector: [x, y, vx, vy]\n"
        gain_text += "Control output: [ry, rx]\n\n"
        gain_text += "K matrix (2x4):\n"
        gain_text += f"  ry: [{K[0,0]:8.4f}, {K[0,1]:8.4f}, {K[0,2]:8.4f}, {K[0,3]:8.4f}]\n"
        gain_text += f"  rx: [{K[1,0]:8.4f}, {K[1,1]:8.4f}, {K[1,2]:8.4f}, {K[1,3]:8.4f}]\n\n"
        gain_text += "Position gains:\n"
        gain_text += f"  K_x:  {abs(K[0,0]):.4f}\n"
        gain_text += f"  K_y:  {abs(K[1,1]):.4f}\n\n"
        gain_text += "Velocity gains:\n"
        gain_text += f"  K_vx: {abs(K[0,2]):.4f}\n"
        gain_text += f"  K_vy: {abs(K[1,3]):.4f}\n"

        # Show dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("LQR Gain Matrix")
        dialog.setMinimumWidth(450)

        layout = QVBoxLayout()

        text_edit = QTextEdit()
        text_edit.setPlainText(gain_text)
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Consolas", 10))
        layout.addWidget(text_edit)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.setLayout(layout)
        dialog.exec()

    # ============================================================================
    # Controller parameter methods
    # ============================================================================

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

    # ============================================================================
    # Kalman filter methods
    # ============================================================================

    def on_kalman_enable_change(self, enabled):
        """Handle Kalman filter enable/disable."""
        self.kalman_enabled = enabled
        if enabled:
            self.kalman_filter.reset(self.ball_pos_mm)
        else:
            # Disable Kalman derivative if Kalman is disabled
            if self.use_kalman_derivative:
                self.use_kalman_derivative = False
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

    # ============================================================================
    # IMU callbacks
    # ============================================================================

    def on_imu_tilt_correction_toggle(self, enabled):
        """Handle IMU tilt correction enable/disable."""
        self.imu_tilt_correction_enabled = enabled
        self.log(f"IMU tilt correction: {'ENABLED' if enabled else 'DISABLED'}")

    def on_imu_kalman_param_change(self, param_name, value):
        """Handle IMU Kalman filter parameter change."""
        param_map = {
            'accel_noise': 'accel_noise',
            'gyro_noise': 'gyro_noise',
            'process_noise_angle': 'process_noise_angle',
            'process_noise_bias': 'process_noise_bias'
        }

        if param_name in param_map:
            attr_name = param_map[param_name]
            setattr(self.orientation_kalman, attr_name, value)
            self.log(f"IMU Kalman {param_name}: {value:.4f}")

    def on_imu_motion_param_change(self, param_name, value):
        """Handle IMU motion detection parameter change."""
        if param_name == 'accel_threshold':
            self.orientation_kalman.accel_magnitude_threshold = value
            self.log(f"IMU accel threshold: {value:.2f} m/s²")
        elif param_name == 'gyro_threshold':
            self.orientation_kalman.gyro_magnitude_threshold = value
            self.log(f"IMU gyro threshold: {value:.2f} rad/s")

    def on_imu_detection_toggle(self, enabled):
        """Handle IMU motion detection enable/disable."""
        self.orientation_kalman.enable_rejection = enabled
        self.log(f"IMU motion detection: {'ENABLED' if enabled else 'DISABLED'}")

    def on_imu_mag_toggle(self, enabled):
        """Handle magnetometer enable/disable."""
        self.orientation_kalman.use_magnetometer = enabled
        self.log(f"Magnetometer backup: {'ENABLED' if enabled else 'DISABLED'}")

    # ============================================================================
    # Mode/controller switching
    # ============================================================================

    def on_mode_change(self, mode):
        """Handle mode change (sim/real)."""
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
        self.log(f"Mode changed to: {mode.upper()}")

        # Rebuild GUI dynamically
        self._rebuild_gui()

        # Update window title
        self.setWindowTitle(f"Stewart Platform - {self.controller_type_selection} [{mode.upper()}]")

    def on_controller_type_change(self, controller_type):
        """Handle controller type change (PID/LQR/Manual)."""
        if controller_type == self.controller_type_selection:
            return

        # Stop simulation if running
        if self.simulation_running:
            self.stop_simulation()

        self.controller_type_selection = controller_type
        self.log(f"Controller changed to: {controller_type}")

        # Update controller config
        self.controller_config = self._create_controller_config()

        # Recreate controller parameter widgets for new controller type
        self._create_controller_param_widgets()

        # Rebuild GUI dynamically
        self._rebuild_gui()

        # Process events to ensure GUI is fully built
        QApplication.processEvents()

        # Reinitialize controller (after GUI is built)
        self._initialize_controller()

        # Update window title
        self.setWindowTitle(f"Stewart Platform - {controller_type} [{self.operation_mode.upper()}]")

    def _rebuild_gui(self):
        """Rebuild entire GUI with new mode/controller configuration."""
        # Clear old GUI modules
        if hasattr(self, 'gui_modules'):
            for module in self.gui_modules.values():
                if module and hasattr(module, 'widget'):
                    try:
                        module.widget.deleteLater()
                    except:
                        pass

        # Clear central widget
        old_central = self.central_widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        old_central.deleteLater()

        # Force garbage collection
        gc.collect()

        # Rebuild GUI
        self._build_modular_gui()

        # Process events to ensure GUI is updated
        QApplication.processEvents()

    def start_simulation(self):
        """Start simulation or hardware control based on mode."""
        if self.operation_mode == 'real':
            # Hardware mode: use control thread
            if not self.connected:
                self.log("Connect to hardware first")
                return

            if self.simulation_running:
                return

            self.simulation_running = True
            self.simulation_time = 0.0

            gc.disable()
            self.log(f"Control started ({self.control_frequency}Hz, GC disabled)")

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

            # Handle IMU initialization and calibration phases
            if self.imu_initializing:
                elapsed = time.time() - self.initialization_start_time
                self.initialization_time_remaining = max(0.0, self.initialization_duration - elapsed)

                if elapsed >= self.initialization_duration:
                    self.imu_initializing = False
                    self.initialization_time_remaining = 0.0
                    self.start_imu_calibration()
                else:
                    # Just drain queues during initialization
                    self.serial_controller.get_imu_data_batch()
                    time.sleep(self.control_interval)
                    continue

            if self.imu_calibrating:
                elapsed = time.time() - self.calibration_start_time
                self.calibration_time_remaining = max(0.0, self.calibration_duration - elapsed)

                # Collect calibration data
                gyro_batch, accel_batch, mag_batch = self.serial_controller.get_imu_data_batch()
                self.calibration_raw_data['gyro'].extend(gyro_batch)
                self.calibration_raw_data['accel'].extend(accel_batch)
                self.calibration_raw_data['mag'].extend(mag_batch)

                if elapsed >= self.calibration_duration:
                    self.finish_imu_calibration()
                    self.calibration_time_remaining = 0.0

                time.sleep(self.control_interval)
                continue

            # Normal operation: Process IMU data
            gyro_data, accel_data, mag_data = self.serial_controller.get_single_imu_sample()

            # Debug: Log IMU data reception (first 5 samples)
            if not hasattr(self, '_imu_data_debug_count'):
                self._imu_data_debug_count = 0
            if self._imu_data_debug_count < 5:
                if gyro_data or accel_data:
                    self.log(f"IMU data: Gyro={'Yes' if gyro_data else 'No'} Accel={'Yes' if accel_data else 'No'} Mag={'Yes' if mag_data else 'No'}")
                    self._imu_data_debug_count += 1

            if gyro_data is not None:
                timestamp_us, gyro_raw = gyro_data
                self.orientation_kalman.predict(gyro_raw, self.control_interval)

            if accel_data is not None:
                timestamp_us, accel_raw = accel_data
                self.orientation_kalman.update(accel_raw, mag_raw=mag_data[1] if mag_data else None)

            # Get IMU orientation
            self.current_rx_imu = np.degrees(self.orientation_kalman.state[0])
            self.current_ry_imu = np.degrees(self.orientation_kalman.state[1])

            # Get ball data from Pixy2 camera via serial
            ball_data = self.serial_controller.get_latest_ball_data()

            if ball_data is not None:
                self.last_ball_update = self.simulation_time

                pixy_x = ball_data['x']
                pixy_y = ball_data['y']

                # Camera dimensions: 316×208 pixels, origin at top-left
                CAMERA_HEIGHT_PIXELS = 208.0
                CAMERA_CENTER_X = 145.0
                CAMERA_CENTER_Y = 102.0

                ball_x_mm = (pixy_x - CAMERA_CENTER_X) * self.pixels_to_mm_x
                ball_y_mm = (CAMERA_HEIGHT_PIXELS - pixy_y - CAMERA_CENTER_Y) * self.pixels_to_mm_y

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

            # Kalman filter (only if enabled)
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
                # Use raw camera data
                ball_pos_mm = self.ball_pos_mm
                ball_vel_mm_s = np.array([0.0, 0.0])  # No velocity without Kalman

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
                    rx_ctrl, ry_ctrl = control_output
                    rx_ctrl, ry_ctrl, _ = clip_tilt_vector(rx_ctrl, ry_ctrl, 15.0)

                    # IMU tilt correction: add compensation (if enabled)
                    if self.imu_tilt_correction_enabled:
                        # IMU compensation: oppose platform tilt (up to ±15°)
                        rx_imu_comp = -self.current_rx_imu * self.imu_compensation_gain
                        ry_imu_comp = -self.current_ry_imu * self.imu_compensation_gain
                        rx_imu_comp, ry_imu_comp, _ = clip_tilt_vector(rx_imu_comp, ry_imu_comp, 15.0)

                        # Combine: controller output + IMU compensation (no additional clipping)
                        rx = rx_ctrl + rx_imu_comp
                        ry = ry_ctrl + ry_imu_comp

                        # Debug logging (first 10 samples only)
                        if not hasattr(self, '_imu_debug_count'):
                            self._imu_debug_count = 0
                        if self._imu_debug_count < 10:
                            self.log(f"IMU correction: RX_IMU={self.current_rx_imu:.2f}° RY_IMU={self.current_ry_imu:.2f}° | "
                                   f"Comp=({rx_imu_comp:.2f}°, {ry_imu_comp:.2f}°) | "
                                   f"Ctrl=({rx_ctrl:.2f}°, {ry_ctrl:.2f}°) → Final=({rx:.2f}°, {ry:.2f}°)")
                            self._imu_debug_count += 1
                    else:
                        rx = rx_ctrl
                        ry = ry_ctrl

                    # Update dof_values so GUI reflects final combined state
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
                # Start with manual dof values
                rx = self.dof_values['rx']
                ry = self.dof_values['ry']

                # Apply IMU tilt correction to manual values (if enabled)
                if self.imu_tilt_correction_enabled:
                    # IMU compensation: oppose platform tilt (clipped to ±15°)
                    rx_imu_comp = -self.current_rx_imu * self.imu_compensation_gain
                    ry_imu_comp = -self.current_ry_imu * self.imu_compensation_gain
                    rx_imu_comp, ry_imu_comp, _ = clip_tilt_vector(rx_imu_comp, ry_imu_comp, 15.0)

                    # Add compensation to manual values (no additional clipping)
                    rx = rx + rx_imu_comp
                    ry = ry + ry_imu_comp

                    # Note: Don't update dof_values here to avoid feedback loop accumulation

                translation = np.array([self.dof_values['x'], self.dof_values['y'], self.dof_values['z']])
                rotation = np.array([rx, ry, self.dof_values['rz']])

                # Apply Z optimization if enabled
                if self.z_optimization_enabled:
                    # Use current Z as search center (not home Z)
                    search_translation = translation.copy()

                    optimized_translation, angles, success = self.ik.optimize_z_offset(
                        search_translation, rotation,
                        use_top_surface_offset=self.use_top_surface_offset,
                        z_search_range=IKZOptimizationConfig.Z_SEARCH_RANGE_MM,
                        max_iterations=IKZOptimizationConfig.MAX_ITERATIONS,
                        tolerance=IKZOptimizationConfig.TOLERANCE_DEG,
                        ik_cache=self.ik_cache if hasattr(self, 'ik_cache') else None
                    )

                    if success and angles is not None:
                        self.z_offset = optimized_translation[2] - translation[2]
                        max_angle = np.max(angles)
                        min_angle = np.min(angles)
                        self.servo_balance = (max_angle, min_angle)
                    else:
                        angles = self.ik.calculate_servo_angles(translation, rotation, self.use_top_surface_offset)
                        if angles is not None:
                            max_angle = np.max(angles)
                            min_angle = np.min(angles)
                            self.servo_balance = (max_angle, min_angle)
                            self.z_offset = 0.0
                else:
                    angles = self.ik.calculate_servo_angles(translation, rotation, self.use_top_surface_offset)
                    if angles is not None:
                        max_angle = np.max(angles)
                        min_angle = np.min(angles)
                        self.servo_balance = (max_angle, min_angle)
                        self.z_offset = 0.0

                if angles is not None:
                    self.serial_controller.send_servo_angles(angles)
                    self.last_cmd_angles = angles

            # Track performance
            loop_time = (time.perf_counter() - loop_start) * 1000
            if len(self.performance_data['loop_times']) < 1000:
                self.performance_data['loop_times'].append(loop_time)

            # Timing control
            sleep_time = self.control_interval - (loop_time / 1000)
            if sleep_time > 0:
                time.sleep(sleep_time)

            self.simulation_time += self.control_interval

    def _gui_update_loop(self):
        """Update GUI at lower rate while control thread runs."""
        if not self.simulation_running:
            return

        # Update GUI modules
        self.update_gui_modules()

        # Update plot
        if self.plot_enabled:
            self.update_plot()

            # Update ball trail from history
            if hasattr(self, 'ball_trail') and self.ball_history_x and len(self.ball_history_x) > 1:
                self.ball_trail.setData(self.ball_history_x, self.ball_history_y)

        # Calculate plot update rate
        plot_interval_ms = int(1000 / self.plot_rate_hz) if self.plot_enabled else 100

        # Schedule next update
        QTimer.singleShot(plot_interval_ms, self._gui_update_loop)

    def _create_plot(self, plot_panel):
        """Override to add ball trail plot item."""
        # Call parent to create standard plot
        super()._create_plot(plot_panel)

        # Add ball trail plot item (dashed red line)
        plot_item = self.plot_widget.getPlotItem()
        self.ball_trail = plot_item.plot([], [], pen=pg.mkPen(color='#ff8888', width=2, style=Qt.PenStyle.DashLine))

    def update_gui_modules(self):
        """Override to provide hardware-specific state."""
        if self.operation_mode == 'real':
            # Hardware mode: use actual hardware state
            ball_x_mm = self.ball_pos[0, 0].item() * 1000
            ball_y_mm = self.ball_pos[0, 1].item() * 1000

            # Get velocity from Kalman filter if enabled
            if self.kalman_enabled:
                vel_x_mm, vel_y_mm = self.kalman_filter.get_velocity_mm_s()
            else:
                vel_x_mm, vel_y_mm = 0.0, 0.0

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
                'z_optimization_enabled': self.z_optimization_enabled,
                'z_offset': self.z_offset,
                'servo_balance': self.servo_balance,
            }

            # Add Kalman filter state (flat keys for GUI module)
            if self.kalman_enabled:
                kalman_pos = self.kalman_filter.get_position_mm()
                kalman_vel = self.kalman_filter.get_velocity_mm_s()
                kalman_std_pos = self.kalman_filter.get_position_uncertainty()
                kalman_stats = self.kalman_filter.get_statistics()

                state['kalman_position'] = kalman_pos
                state['kalman_velocity'] = kalman_vel
                state['kalman_uncertainty'] = kalman_std_pos
                state['kalman_stats'] = kalman_stats

            # Hardware-specific stats
            if hasattr(self, 'ik_cache') and self.ik_cache:
                state['cache_hit_rate'] = self.ik_cache.get_hit_rate()

            state['frequency'] = self.control_frequency

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

            # Add IMU orientation state
            state['imu_orientation'] = (self.current_rx_imu, self.current_ry_imu)
            state['imu_bias'] = (self.orientation_kalman.state[2], self.orientation_kalman.state[3])

            # Add IMU calibration status
            state['imu_initializing'] = self.imu_initializing
            state['imu_calibrating'] = self.imu_calibrating
            state['initialization_time_remaining'] = self.initialization_time_remaining
            state['calibration_time_remaining'] = self.calibration_time_remaining

            # Add IMU motion detection statistics
            state['imu_rejection_stats'] = (
                self.orientation_kalman.rejected_accel_count,
                self.orientation_kalman.total_accel_count
            )
            state['imu_mag_updates'] = self.orientation_kalman.mag_update_count

            # Update all modules (skip plot_panel which is a Qt widget, not a GUIModule)
            for key, module in self.gui_modules.items():
                if key == 'plot_panel':
                    continue
                if module and hasattr(module, 'update') and hasattr(module, 'widget'):
                    module.update(state)
        else:
            # Simulation mode: use parent implementation first
            super().update_gui_modules()

            # Then update Kalman filter GUI if enabled
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
    """Launch comprehensive controller."""

    app = QApplication(sys.argv)
    controller = ComprehensiveStewartController(app)
    controller.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
