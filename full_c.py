#!/usr/bin/env python3
"""
Stewart Platform Controller

Full-featured controller supporting multiple operation modes and control strategies.

Features:
- Mode switching: Simulation / Hardware
- Controller types: PID / LQR / Manual
- IMU tilt correction and calibration
- Kalman filtering for state estimation
- Dynamic GUI configuration
"""

import sys
import time
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QMessageBox, QDialog, QVBoxLayout, QLabel, QTextEdit,
                              QPushButton, QApplication, QWidget, QHBoxLayout, QCheckBox,
                              QTabWidget, QGroupBox)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import QTimer, Qt

from setup.base_hardware import HardwareControllerBase, SerialController
from gui import gui_modules as gm
from gui.gui_builder import create_standard_layout, GUIBuilder
from core.control_core import IMUControllerMixin, LQRController, clip_tilt_vector
from core.utils import (IKZOptimizationConfig, MAX_TILT_ANGLE_DEG, MAX_CONTROLLER_OUTPUT_DEG,
                         MAX_YAW_ANGLE_DEG, Pixy2CameraConfig, BallPhysicsConfig,
                         HardwareConnectionConfig, GUIConfig, VisualizationConfig, ControlLoopConfig,
                         GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL, PerformanceConfig)


class StewartController(IMUControllerMixin, HardwareControllerBase):
    """Stewart Platform controller with full feature set."""

    def __init__(self, app: QApplication) -> None:
        """Initialize the Stewart Platform controller.

        Args:
            app: The QApplication instance for the GUI.
        """
        # Mode selection
        self.operation_mode = 'real'  # 'sim' or 'real'
        self.controller_type_selection = 'LQR'  # 'PID', 'LQR', or 'Manual'

        # Plot control settings (unique to full controller, from GUIConfig)
        self.plot_enabled = GUIConfig.DEFAULT_PLOT_ENABLED
        self.plot_rate_hz = GUIConfig.DEFAULT_PLOT_RATE_HZ

        # Initialize IMU system (MUST be called before super().__init__)
        self._init_imu_system()

        # Create controller config based on mode and controller type
        controller_config = self._create_controller_config()

        # Call parent constructor (HardwareControllerBase handles hardware init)
        super().__init__(app, controller_config)

        # Effective platform angles (gravity frame) for Kalman filter prediction
        # Excludes IMU compensation to maintain consistent physics model
        self.prev_effective_angles = {'rx': 0.0, 'ry': 0.0}

        # Ball trail history (for hardware mode)
        self.max_history = VisualizationConfig.BALL_TRAIL_MAX_HISTORY
        self.ball_history_x = []
        self.ball_history_y = []

        # Performance data recording
        self.csv_writer = None
        self.csv_file = None
        self.recording = False
        self.recording_start_time = 0.0
        self.sample_count = 0
        self.recording_filename = None

        # Z optimization (controlled by GUI toggle in IKZOptimizationModule)
        self.z_optimization_enabled = IKZOptimizationConfig.ENABLED  # Default from config
        self.z_offset = 0.0  # Current Z offset from optimization (mm)
        self.servo_balance = (0.0, 0.0)  # (max_angle, min_angle) in degrees

        # Override window title
        self.setWindowTitle(f"Stewart Platform - {self.controller_type_selection} [{self.operation_mode.upper()}]")

    def _update_controller(self, ball_pos_mm: Tuple[float, float], ball_vel_mm_s: Tuple[float, float],
                           target_pos_mm: Tuple[float, float], dt: float) -> Optional[Tuple[float, float]]:
        """Update controller and return control output.

        In simulation mode, applies Kalman filtering when enabled for improved
        state estimation. Returns platform tilt angles (rx, ry) in degrees.

        Args:
            ball_pos_mm: Ball position in millimeters (x, y).
            ball_vel_mm_s: Ball velocity in mm/s (vx, vy).
            target_pos_mm: Target position in millimeters (x, y).
            dt: Time step in seconds.

        Returns:
            Tuple of (rx, ry) tilt angles in degrees, or None if controller unavailable.
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

                # Kalman update step (with camera measurement) - convert mm to meters
                ball_pos_m = [ball_pos_mm[0] / 1000.0, ball_pos_mm[1] / 1000.0]
                self.kalman_filter.update(ball_pos_m, self.simulation_time)

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
        elif "MPC" in controller_name:
            # MPC uses same interface as LQR (position + velocity in mm)
            # Pass actual dt for adaptive QP rebuild on timing variations
            rx, ry = self.controller.update(
                ball_pos_mm=ball_pos_filtered,
                ball_vel_mm_s=ball_vel_filtered,
                target_pos_mm=target_pos_mm,
                actual_dt=dt
            )
        elif self.controller_type_selection == 'RL':
            platform_tilt = (self.prev_effective_angles['rx'], self.prev_effective_angles['ry'])
            rx, ry = self.controller.update(
                ball_pos_filtered, ball_vel_filtered, target_pos_mm,
                platform_tilt_deg=platform_tilt
            )
        else:
            return None

        return rx, ry

    def _create_callbacks(self) -> Dict[str, Any]:
        """Create callback dictionary with all features.

        Returns:
            Dictionary mapping callback names to callable functions.
        """
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

        # Add Z optimization callback
        callbacks['z_optimization_toggle'] = self.on_z_optimization_toggle

        # Add performance data recording callbacks (both sim and hardware)
        callbacks.update({
            'start_recording': self.start_recording,
            'stop_recording': self.stop_recording,
        })

        # Add video recording callbacks (hardware only)
        if self.operation_mode == 'real':
            callbacks.update({
                'start_video_recording': self.start_video_recording,
                'stop_video_recording': self.stop_video_recording,
            })

        # Add IMU callbacks (hardware only)
        if self.operation_mode == 'real':
            callbacks.update({
                'imu_tilt_correction_toggle': self.on_imu_tilt_correction_toggle,
                'imu_kalman_param_change': self.on_imu_kalman_param_change,
                'imu_mag_toggle': self.on_imu_mag_toggle,
                'imu_yaw_tracking_toggle': self.on_imu_yaw_tracking_toggle,
                'imu_set_yaw_reference': self.on_imu_set_yaw_reference,
                'imu_yaw_offset_change': self.on_imu_yaw_offset_change,
            })

        # Add LQR-specific callbacks
        if self.controller_type_selection == 'LQR':
            callbacks['show_gain_matrix'] = self.show_gain_matrix

        # Add PID-specific callbacks
        if self.controller_type_selection == 'PID':
            callbacks['kalman_derivative_toggle'] = self.on_kalman_derivative_toggle

        return callbacks

    def get_layout_config(self) -> Dict[str, Any]:
        """Return layout configuration based on mode and controller.

        Returns:
            Dictionary containing the layout configuration for the GUI.
        """
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
                                       'min_freq': ControlLoopConfig.MIN_FREQUENCY_HZ,
                                       'max_freq': ControlLoopConfig.MAX_FREQUENCY_HZ}})

        # Simulation control
        left_modules.append({'type': 'simulation_control'})

        # Trajectory pattern
        left_modules.append({'type': 'trajectory_pattern', 'args': {'pattern_var': self.pattern_type}})

        # Performance data collection (both sim and hardware)
        left_modules.append({'type': 'performance_data',
                           'args': {'controller_type': self.controller_type_selection}})

        # Hardware-only: Video recording
        if self.operation_mode == 'real':
            left_modules.append({'type': 'video_record'})

        # Simulation-only: Ball control
        if self.operation_mode == 'sim':
            left_modules.append({'type': 'ball_control'})

        # Simulation-only: Camera noise
        if self.operation_mode == 'sim':
            left_modules.append({'type': 'pixy2_camera',
                               'args': {'camera': getattr(self, 'pixy_camera', None)}})

        # Kalman filter
        left_modules.append({'type': 'kalman_filter',
                           'args': {'kalman_filter': getattr(self, 'kalman_filter', None),
                                   'enabled': self.kalman_enabled,
                                   'camera_type': self.camera_type}})

        # Hardware-only: IMU modules
        if self.operation_mode == 'real':
            left_modules.append({'type': 'imu_kalman_parameters',
                               'args': {'orientation_kalman': getattr(self, 'orientation_kalman', None)}})

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

    def _build_modular_gui(self) -> None:
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
            'ik_z_optimization': gm.IKZOptimizationModule,
            'performance_data': gm.PerformanceDataCollectionModule,
            'video_record': gm.VideoRecordModule,
        }

        layout_config = self.get_layout_config()
        callbacks = self._create_callbacks()

        self.gui_builder = GUIBuilder(self.central_widget, module_registry)
        self.gui_modules = self.gui_builder.build(layout_config, self.colors, callbacks)

        if 'plot_panel' in self.gui_modules:
            self._create_plot(self.gui_modules['plot_panel'])

        # Add controller-specific widgets
        self._add_controller_specific_widgets()

    def _add_controller_specific_widgets(self) -> None:
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

    def connect_serial(self) -> None:
        """Establish connection to hardware (serial for servos/Pixy2, or ZED camera)."""
        if self.connected:
            self.log("Connection already established")
            return

        if 'serial_connection' in self.gui_modules:
            port = self.gui_modules['serial_connection'].port_combo.currentText()
        else:
            return

        if not port:
            QMessageBox.critical(self, "Error", "No port selected")
            return

        # Connect to servo controller via serial
        self.serial_controller = SerialController(port)
        success, message = self.serial_controller.connect()

        if not success:
            QMessageBox.critical(self, "Error", message)
            self.log(f"Error: {message}")
            return

        self.connected = True
        self.log(f"Connected to {port}")

        time.sleep(HardwareConnectionConfig.POST_CONNECTION_DELAY_S)
        self.serial_controller.set_servo_speed(0)
        time.sleep(HardwareConnectionConfig.POST_SERVO_SPEED_DELAY_S)
        self.serial_controller.set_servo_acceleration(0)
        time.sleep(HardwareConnectionConfig.POST_SERVO_ACCEL_DELAY_S)
        self.log("Servo parameters configured: Speed=0, Acceleration=0")

        # Connect camera based on type
        if self.camera_type == 'STEREO':
            from ball_detection.integration.stereo_controller import StereoCameraController
            self.camera_controller = StereoCameraController()
            cam_success, cam_message = self.camera_controller.connect()

            if cam_success:
                self.log(f"Camera: {cam_message}")
            else:
                QMessageBox.critical(self, "Camera Error", cam_message)
                self.log(f"Camera Error: {cam_message}")
                self.disconnect_serial()
                return

        success_timer, msg_timer = self.timer_manager.set_high_resolution()
        self.log(msg_timer)

        self.initialize_ik_cache()
        self.start_imu_initialization()

        if 'simulation_control' in self.gui_modules:
            self.gui_modules['simulation_control'].start_btn.setEnabled(True)

        if 'serial_connection' in self.gui_modules:
            self.gui_modules['serial_connection'].update({'connected': True})

    def disconnect_serial(self) -> None:
        """Disconnect from hardware."""
        if self.simulation_running:
            self.stop_simulation()

        if self.serial_controller:
            self.serial_controller.disconnect()

        if self.camera_controller:
            self.camera_controller.disconnect()

        self.connected = False

        if 'simulation_control' in self.gui_modules:
            self.gui_modules['simulation_control'].start_btn.setEnabled(False)

        if 'serial_connection' in self.gui_modules:
            self.gui_modules['serial_connection'].update({'connected': False})

        self.log("Disconnected")

    def on_frequency_change(self, frequency: int) -> None:
        """Handle control frequency change.

        Args:
            frequency: New control frequency in Hz.
        """
        self.control_frequency = frequency
        self.control_interval = 1.0 / frequency

        # Update Kalman filter dt
        if hasattr(self, 'kalman_filter') and self.kalman_filter:
            self.kalman_filter.set_dt(self.control_interval)

        # Update window title
        self.setWindowTitle(f"Stewart Platform - {self.controller_type_selection} [{self.operation_mode.upper()}] @ {frequency}Hz")
        self.log(f"Control frequency: {frequency}Hz")

    def on_plot_enable_change(self, enabled: bool) -> None:
        """Handle plot enable/disable.

        Args:
            enabled: True to enable plot updates, False to disable.
        """
        self.plot_enabled = enabled
        status = "ENABLED" if enabled else "DISABLED"
        self.log(f"Plot updates: {status}")

    def on_plot_rate_change(self, rate: int) -> None:
        """Handle plot refresh rate change.

        Args:
            rate: New plot refresh rate in Hz.
        """
        self.plot_rate_hz = rate
        self.log(f"Plot refresh rate: {rate} Hz")

    def show_timing_stats(self) -> None:
        """Show performance timing statistics dialog."""
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
        text_edit.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL))
        layout.addWidget(text_edit)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.setLayout(layout)
        dialog.exec()

    # ============================================================================
    # PID-specific methods
    # ============================================================================

    def on_kalman_derivative_toggle(self) -> None:
        """Toggle Kalman derivative option for PID controller."""
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

    def show_gain_matrix(self) -> None:
        """Show LQR gain matrix in a dialog window."""
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
        text_edit.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL))
        layout.addWidget(text_edit)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.setLayout(layout)
        dialog.exec()

    # ============================================================================
    # Controller parameter methods
    # ============================================================================

    def on_controller_param_change(self) -> None:
        """Update controller when parameters change via GUI sliders."""
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

        elif self.controller_type_selection == 'MPC':
            Q_pos = self.controller_config.get_scaled_param('Q_pos', sliders, scalar_vars)
            Q_vel = self.controller_config.get_scaled_param('Q_vel', sliders, scalar_vars)
            R_pos = self.controller_config.get_scaled_param('R_pos', sliders, scalar_vars)
            R_vel = self.controller_config.get_scaled_param('R_vel', sliders, scalar_vars)

            self.controller.set_weights(Q_pos=Q_pos, Q_vel=Q_vel, R_pos=R_pos, R_vel=R_vel)

            if self.controller_enabled:
                self.log(f"MPC weights updated: Q_pos={Q_pos:.2e}, Q_vel={Q_vel:.2e}, R_pos={R_pos:.2e}, R_vel={R_vel:.2e}")

    # ============================================================================
    # Mode switching
    # ============================================================================

    def on_mode_change(self, mode: str) -> None:
        """Handle mode change between simulation and hardware.

        Args:
            mode: New operation mode ('sim' or 'real').
        """
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
        self.setWindowTitle(f"Stewart Platform - {self.controller_type_selection} [{self.operation_mode.upper()}]")

    def on_controller_type_change(self, controller_type: str) -> None:
        """Handle controller type change.

        Args:
            controller_type: New controller type ('PID', 'LQR', or 'Manual').
        """
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

        # Update window title
        self.setWindowTitle(f"Stewart Platform - {self.controller_type_selection} [{self.operation_mode.upper()}]")

    # ============================================================================
    # Kalman filter methods
    # ============================================================================

    def on_kalman_enable_change(self, enabled: bool) -> None:
        """Handle Kalman filter enable/disable.

        Args:
            enabled: True to enable Kalman filter, False to disable.
        """
        self.kalman_enabled = enabled
        if enabled:
            self.kalman_filter.reset(self.ball_pos_mm)
        else:
            # Disable Kalman derivative if Kalman is disabled
            if self.use_kalman_derivative:
                self.use_kalman_derivative = False
        self.log(f"Kalman filter: {'ENABLED' if enabled else 'DISABLED'}")

    def on_kalman_param_change(self, param_name: str, value: float) -> None:
        """Handle Kalman filter parameter change.

        Args:
            param_name: Name of the parameter being changed.
            value: New parameter value.
        """
        param_labels = {
            'process_noise': 'Process noise',
            'measurement_noise': 'Measurement noise'
        }
        label = param_labels.get(param_name, param_name)
        self.log(f"Kalman {label}: {value:.2f}")

    def on_kalman_reset(self) -> None:
        """Reset Kalman filter to current ball position."""
        self.kalman_filter.reset(self.ball_pos_mm)
        self.log("Kalman filter reset")

    def on_z_optimization_toggle(self, enabled: bool) -> None:
        """Handle Z optimization enable/disable toggle."""
        self.z_optimization_enabled = enabled
        status = "ENABLED" if enabled else "DISABLED"
        self.log(f"Z optimization: {status}")

    def start_recording(self) -> None:
        """Start recording performance data to CSV file."""
        if self.recording:
            self.log("Already recording!")
            return

        # Create output directory
        output_dir = Path('data/performance')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_dir / f"performance_{self.controller_type_selection}_{timestamp}.csv"

        try:
            self.csv_file = open(filename, 'w', newline='')
            self.csv_writer = csv.DictWriter(
                self.csv_file,
                fieldnames=['timestamp', 'elapsed_time', 'ball_x', 'ball_y', 'ball_detected',
                           'kalman_x', 'kalman_y', 'kalman_vx', 'kalman_vy',
                           'platform_rx', 'platform_ry',
                           'target_x', 'target_y', 'error_x', 'error_y', 'error_magnitude',
                           'controller_rx', 'controller_ry', 'tilt_magnitude', 'loop_time_ms']
            )
            self.csv_writer.writeheader()

            self.recording = True
            self.recording_start_time = time.time()
            self.sample_count = 0
            self.recording_filename = str(filename)

            self.log(f"Started recording to {filename.name}")
        except Exception as e:
            self.log(f"ERROR: Failed to start recording: {e}")

    def stop_recording(self) -> None:
        """Stop recording performance data and close CSV file."""
        if not self.recording:
            self.log("Not currently recording")
            return

        try:
            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None
            self.csv_writer = None

            elapsed = time.time() - self.recording_start_time
            sample_rate = self.sample_count / elapsed if elapsed > 0 else 0

            self.log(f"Recording stopped. {self.sample_count} samples in {elapsed:.1f}s ({sample_rate:.1f} Hz)")
            self.log(f"Data saved to {Path(self.recording_filename).name}")

            self.recording = False
            self.recording_filename = None
        except Exception as e:
            self.log(f"ERROR: Failed to stop recording: {e}")

    def start_video_recording(self) -> str:
        """Start recording video from stereo camera."""
        if not hasattr(self, 'camera_controller') or self.camera_controller is None:
            self.log("ERROR: No camera connected")
            return ""
        filepath = self.camera_controller.start_recording()
        self.log(f"Video recording started: {filepath}")
        return filepath

    def stop_video_recording(self) -> None:
        """Stop recording video from stereo camera."""
        if not hasattr(self, 'camera_controller') or self.camera_controller is None:
            return
        self.camera_controller.stop_recording()
        self.log("Video recording stopped")

    def _control_thread_func(self) -> None:
        """Hardware control thread with IMU integration."""
        self.log("Control thread started")

        while self.simulation_running:
            loop_start = time.perf_counter()

            # Handle IMU calibration phases (from mixin)
            if self._process_imu_calibration_phase():
                time.sleep(self.control_interval)
                continue

            # Update IMU orientation (from mixin)
            self._update_imu_orientation()

            # Get ball data from camera (Pixy2 via serial, or ZED direct)
            ball_data = self.get_ball_data()

            if ball_data is not None:
                self.last_ball_update = self.simulation_time

                # Handle coordinate conversion based on camera type
                if self.camera_type == 'PIXY2':
                    # Pixy2 returns pixel coordinates, convert to mm
                    pixy_x = ball_data['x']
                    pixy_y = ball_data['y']

                    # Camera dimensions: 316×208 pixels, origin at top-left
                    ball_x_mm = (pixy_x - Pixy2CameraConfig.CENTER_X) * Pixy2CameraConfig.PIXELS_TO_MM_X
                    ball_y_mm = (Pixy2CameraConfig.RESOLUTION_HEIGHT_PX - pixy_y - Pixy2CameraConfig.CENTER_Y) * Pixy2CameraConfig.PIXELS_TO_MM_Y
                else:  # STEREO camera
                    # Stereo returns 3D coordinates in platform frame (mm)
                    # X, Y used for control; Z logged only
                    ball_x_mm = ball_data['x']
                    ball_y_mm = ball_data['y']
                    # Z available via ball_data.get('z', 0.0) for logging

                self.ball_pos_mm = np.array([ball_x_mm, ball_y_mm])
                self.ball_detected = ball_data.get('detected', False)

                # Update ball_pos tensor for plotting (convert mm to m)
                self.ball_pos[0, 0] = ball_x_mm / 1000.0
                self.ball_pos[0, 1] = ball_y_mm / 1000.0
                # Update Z from stereo camera if available
                if 'z' in ball_data:
                    self.ball_pos[0, 2] = ball_data['z'] / 1000.0

                # Track ball history for trail (only in hardware mode, only if detected)
                # In simulation mode, base_simulator.py handles trail updates
                if self.operation_mode == 'real' and self.ball_detected:
                    self.ball_history_x.append(ball_x_mm)
                    self.ball_history_y.append(ball_y_mm)
                    if len(self.ball_history_x) > self.max_history:
                        self.ball_history_x.pop(0)
                        self.ball_history_y.pop(0)

            # Kalman filter (only if enabled)
            if self.kalman_enabled:
                # Predict ball dynamics using effective platform tilt (gravity-relative)
                # IMU compensation is excluded to maintain consistent physics model
                rx_deg = self.prev_effective_angles.get('rx', 0.0)
                ry_deg = self.prev_effective_angles.get('ry', 0.0)
                self.kalman_filter.predict([rx_deg, ry_deg])

                # Update step (only when ball is detected)
                if ball_data is not None and self.ball_detected:
                    # Convert mm to meters for Kalman filter
                    ball_pos_m = [self.ball_pos_mm[0] / 1000.0, self.ball_pos_mm[1] / 1000.0]
                    self.kalman_filter.update(ball_pos_m, self.simulation_time)

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

            # Initialize controller output values (for CSV recording)
            rx_ctrl = 0.0
            ry_ctrl = 0.0

            # Update controller if enabled and ball is detected
            if self.controller_enabled and self.controller is not None and self.ball_detected:
                control_output = self._update_controller(
                    ball_pos_mm, ball_vel_mm_s, target_pos_mm, self.control_interval
                )

                if control_output is not None:
                    rx_ctrl, ry_ctrl = control_output
                    rx_ctrl, ry_ctrl, _ = clip_tilt_vector(rx_ctrl, ry_ctrl, MAX_CONTROLLER_OUTPUT_DEG)

                    # Store controller output (gravity-relative) for Kalman physics prediction
                    self.prev_effective_angles['rx'] = rx_ctrl
                    self.prev_effective_angles['ry'] = ry_ctrl

                    # Apply IMU compensation to achieve desired effective tilt
                    rx, ry = self._apply_imu_tilt_correction(rx_ctrl, ry_ctrl)

                    # Update dof_values so GUI reflects final combined state
                    self.dof_values['rx'] = rx
                    self.dof_values['ry'] = ry

                    # Determine yaw angle (from IMU if 6-DOF enabled, otherwise manual)
                    if self.orientation_kalman.enable_yaw_tracking and self.imu_tilt_correction_enabled:
                        rz = np.clip(self.current_yaw_imu, -MAX_YAW_ANGLE_DEG, MAX_YAW_ANGLE_DEG)
                    else:
                        rz = self.dof_values['rz']

                    # Calculate servo angles
                    translation = np.array([0.0, 0.0, self.ik.home_height_top_surface])
                    rotation = np.array([rx, ry, rz])

                    # Apply Z optimization if enabled
                    if self.z_optimization_enabled:
                        optimized_translation, angles, success = self.ik.optimize_z_offset(
                            translation, rotation,
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
                            # Fallback to standard IK
                            angles = self.ik.calculate_servo_angles(translation, rotation, self.use_top_surface_offset)
                            if angles is not None:
                                max_angle = np.max(angles)
                                min_angle = np.min(angles)
                                self.servo_balance = (max_angle, min_angle)
                                self.z_offset = 0.0
                    else:
                        # Standard IK (with cache if available)
                        if self.ik_cache:
                            angles = self.ik_cache.get(translation, rotation)
                            if angles is None:
                                angles = self.ik.calculate_servo_angles(translation, rotation, self.use_top_surface_offset)
                                if angles is not None:
                                    self.ik_cache.put(translation, rotation, angles)
                        else:
                            angles = self.ik.calculate_servo_angles(translation, rotation, self.use_top_surface_offset)

                        # Update servo balance for display
                        if angles is not None:
                            max_angle = np.max(angles)
                            min_angle = np.min(angles)
                            self.servo_balance = (max_angle, min_angle)
                            self.z_offset = 0.0

                    if angles is not None:
                        self.serial_controller.send_servo_angles(angles)
                        self.last_cmd_angles = angles

                        self.prev_platform_angles['rx'] = rx
                        self.prev_platform_angles['ry'] = ry

            # Manual control when controller disabled
            elif not self.controller_enabled:
                # Manual DOF values represent desired effective tilt (gravity-relative)
                rx_effective = self.dof_values['rx']
                ry_effective = self.dof_values['ry']

                # Store effective angles for Kalman physics prediction
                self.prev_effective_angles['rx'] = rx_effective
                self.prev_effective_angles['ry'] = ry_effective

                # Apply IMU compensation to achieve desired effective tilt
                rx, ry = self._apply_imu_tilt_correction(rx_effective, ry_effective)

                # Determine yaw angle (from IMU if 6-DOF enabled, otherwise manual)
                if self.orientation_kalman.enable_yaw_tracking and self.imu_tilt_correction_enabled:
                    rz = np.clip(self.current_yaw_imu, -MAX_YAW_ANGLE_DEG, MAX_YAW_ANGLE_DEG)
                else:
                    rz = self.dof_values['rz']

                translation = np.array([self.dof_values['x'], self.dof_values['y'], self.dof_values['z']])
                rotation = np.array([rx, ry, rz])

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

            # Record performance data if recording (regardless of controller state or ball detection)
            if self.recording and self.csv_writer:
                try:
                    current_time = time.time()
                    elapsed_time = current_time - self.recording_start_time

                    # Use RAW camera measurements (unfiltered) for CSV recording
                    raw_ball_x = self.ball_pos_mm[0]
                    raw_ball_y = self.ball_pos_mm[1]

                    # Get Kalman filter estimates if enabled (for offline comparison)
                    if self.kalman_enabled:
                        kalman_x, kalman_y = self.kalman_filter.get_position_mm()
                        kalman_vx, kalman_vy = self.kalman_filter.get_velocity_mm_s()
                    else:
                        kalman_x, kalman_y = 0.0, 0.0
                        kalman_vx, kalman_vy = 0.0, 0.0

                    # Get actual platform angles sent to hardware (for Kalman prediction replay)
                    platform_rx = self.dof_values['rx']
                    platform_ry = self.dof_values['ry']

                    # Calculate error from raw position
                    error_x = raw_ball_x - target_pos_mm[0]
                    error_y = raw_ball_y - target_pos_mm[1]
                    error_magnitude = np.sqrt(error_x**2 + error_y**2)

                    # Calculate tilt magnitude from controller output
                    tilt_magnitude = np.sqrt(rx_ctrl**2 + ry_ctrl**2)

                    # Calculate loop time before writing
                    loop_time_current = (time.perf_counter() - loop_start) * 1000

                    self.csv_writer.writerow({
                        'timestamp': datetime.fromtimestamp(current_time).strftime('%H:%M:%S.%f')[:-3],
                        'elapsed_time': f'{elapsed_time:.3f}',
                        'ball_x': f'{raw_ball_x:.3f}',
                        'ball_y': f'{raw_ball_y:.3f}',
                        'ball_detected': self.ball_detected,
                        'kalman_x': f'{kalman_x:.3f}',
                        'kalman_y': f'{kalman_y:.3f}',
                        'kalman_vx': f'{kalman_vx:.3f}',
                        'kalman_vy': f'{kalman_vy:.3f}',
                        'platform_rx': f'{platform_rx:.3f}',
                        'platform_ry': f'{platform_ry:.3f}',
                        'target_x': f'{target_pos_mm[0]:.3f}',
                        'target_y': f'{target_pos_mm[1]:.3f}',
                        'error_x': f'{error_x:.3f}',
                        'error_y': f'{error_y:.3f}',
                        'error_magnitude': f'{error_magnitude:.3f}',
                        'controller_rx': f'{rx_ctrl:.3f}',
                        'controller_ry': f'{ry_ctrl:.3f}',
                        'tilt_magnitude': f'{tilt_magnitude:.3f}',
                        'loop_time_ms': f'{loop_time_current:.3f}'
                    })
                    self.sample_count += 1

                    # Flush periodically to ensure data is written
                    if self.sample_count % PerformanceConfig.CSV_FLUSH_INTERVAL_SAMPLES == 0:
                        self.csv_file.flush()
                except Exception as e:
                    self.log(f"CSV write error: {e}")
                    self.recording = False

            # Track performance
            loop_time = (time.perf_counter() - loop_start) * 1000
            if len(self.performance_data['loop_times']) < PerformanceConfig.LOOP_TIME_HISTORY_LIMIT:
                self.performance_data['loop_times'].append(loop_time)

            # Timing control
            sleep_time = self.control_interval - (loop_time / 1000)
            if sleep_time > 0:
                time.sleep(sleep_time)

            self.simulation_time += self.control_interval

    def _gui_update_loop(self) -> None:
        """Update GUI at lower rate while control thread runs."""
        if not self.simulation_running:
            return

        # Update GUI modules
        self.update_gui_modules()

        # Update plot based on active tab
        if self.plot_enabled and hasattr(self, 'plot_tab_widget'):
            if self.plot_tab_widget.currentIndex() == 0:
                # 2D View tab active
                self.update_plot()

                # Update ball trail from history
                if hasattr(self, 'ball_trail') and self.ball_history_x and len(self.ball_history_x) > 1:
                    self.ball_trail.setData(self.ball_history_x, self.ball_history_y)
            else:
                # 3D View tab active
                self._update_3d_view()

        # Calculate plot update rate
        plot_interval_ms = int(1000 / self.plot_rate_hz) if self.plot_enabled else GUIConfig.PLOT_DISABLED_INTERVAL_MS

        # Schedule next update
        QTimer.singleShot(plot_interval_ms, self._gui_update_loop)

    def _update_3d_view(self) -> None:
        """Update 3D visualization with current ball and platform state."""
        if not hasattr(self, 'plot_3d') or self.plot_3d is None:
            return

        # Get ball position (ball_pos stores [x, y, z] in meters)
        # Z is updated by control thread from stereo camera or simulation physics
        ball_x_mm = self.ball_pos[0, 0].item() * 1000
        ball_y_mm = self.ball_pos[0, 1].item() * 1000
        ball_z_mm = self.ball_pos[0, 2].item() * 1000

        # Get platform pose from FK
        platform_x, platform_y, platform_z = 0.0, 0.0, self.ik.home_height_top_surface
        platform_rx, platform_ry = 0.0, 0.0
        if hasattr(self, 'last_fk_translation') and self.last_fk_translation is not None:
            platform_x = self.last_fk_translation[0]
            platform_y = self.last_fk_translation[1]
            platform_z = self.last_fk_translation[2]
        if hasattr(self, 'last_fk_rotation') and self.last_fk_rotation is not None:
            platform_rx = self.last_fk_rotation[0]
            platform_ry = self.last_fk_rotation[1]
        elif hasattr(self, 'dof_values'):
            platform_rx = self.dof_values.get('rx', 0.0)
            platform_ry = self.dof_values.get('ry', 0.0)

        # Get current target position from trajectory pattern
        target_x, target_y = 0.0, 0.0
        if hasattr(self, 'current_pattern') and self.current_pattern is not None:
            pattern_time = self.simulation_time - self.pattern_start_time
            target_x, target_y = self.current_pattern.get_position(pattern_time)

        # Get pattern type and parameters for visualization
        pattern_type = getattr(self, 'pattern_type', 'static').capitalize()
        pattern_params = {}
        if hasattr(self, 'current_pattern'):
            pattern = self.current_pattern
            if hasattr(pattern, 'radius'):
                pattern_params['radius'] = pattern.radius
            if hasattr(pattern, 'width'):
                pattern_params['width'] = pattern.width
            if hasattr(pattern, 'height'):
                pattern_params['height'] = pattern.height

        # Update 3D module
        state = {
            'ball_x': ball_x_mm,
            'ball_y': ball_y_mm,
            'ball_z': ball_z_mm,
            'platform_x': platform_x,
            'platform_y': platform_y,
            'platform_z': platform_z,
            'platform_rx': platform_rx,
            'platform_ry': platform_ry,
            'target_x': target_x,
            'target_y': target_y,
            'pattern_type': pattern_type,
            'pattern_params': pattern_params,
        }
        self.plot_3d.update(state)

    def _create_plot(self, parent: QWidget) -> None:
        """Create plot panel with 2D/3D tab views.

        Args:
            parent: Parent widget to contain the plot.
        """
        # Create tab widget
        self.plot_tab_widget = QTabWidget()

        # === 2D View Tab ===
        plot_2d_container = QWidget()
        plot_2d_layout = QVBoxLayout(plot_2d_container)
        plot_2d_layout.setContentsMargins(0, 0, 0, 0)

        # Create PyQtGraph 2D plot widget
        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(self.colors['widget_bg'])
        self.plot_widget.setMinimumSize(600, 600)
        plot_2d_layout.addWidget(self.plot_widget)

        self.plot_tab_widget.addTab(plot_2d_container, "2D View")

        # === 3D View Tab ===
        platform_home_z = self.ik.home_height_top_surface
        self.plot_3d = gm.Plot3DModule(
            parent, self.colors, self._create_callbacks(),
            platform_home_z=platform_home_z
        )
        plot_3d_widget = self.plot_3d.create()
        if plot_3d_widget:
            self.plot_tab_widget.addTab(plot_3d_widget, "3D View")

        # Add tab widget to parent layout
        if hasattr(parent, 'layout') and parent.layout() is not None:
            parent.layout().addWidget(self.plot_tab_widget)

        # Setup 2D plot (platform boundary, markers, etc.)
        self.setup_plot()

        # Add ball trail plot item (dashed red line)
        plot_item = self.plot_widget.getPlotItem()
        self.ball_trail = plot_item.plot([], [], pen=pg.mkPen(color='#ff8888', width=2, style=Qt.PenStyle.DashLine))

    def update_gui_modules(self) -> None:
        """Override to provide hardware-specific state to GUI modules."""
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

            # Add IMU orientation state (handles both 4-DOF and 6-DOF modes)
            if self.orientation_kalman.enable_yaw_tracking:
                # 6-DOF mode: roll, pitch, yaw
                state['imu_orientation'] = (self.current_rx_imu, self.current_ry_imu, self.current_yaw_imu)
                state['imu_bias'] = (self.orientation_kalman.state[3], self.orientation_kalman.state[4], self.orientation_kalman.state[5])
            else:
                # 4-DOF mode: roll, pitch only
                state['imu_orientation'] = (self.current_rx_imu, self.current_ry_imu)
                state['imu_bias'] = (self.orientation_kalman.state[2], self.orientation_kalman.state[3])

            # Add IMU calibration status
            state['imu_initializing'] = self.imu_initializing
            state['imu_calibrating'] = self.imu_calibrating
            state['initialization_time_remaining'] = self.initialization_time_remaining
            state['calibration_time_remaining'] = self.calibration_time_remaining

            # Add magnetometer statistics
            state['imu_mag_updates'] = self.orientation_kalman.mag_update_count

            # Add recording state
            state['recording'] = self.recording
            if self.recording:
                elapsed = time.time() - self.recording_start_time
                sample_rate = self.sample_count / elapsed if elapsed > 0 else 0
                state['recording_duration'] = elapsed
                state['recording_samples'] = self.sample_count
                state['recording_rate'] = sample_rate
                state['recording_filename'] = Path(self.recording_filename).name if self.recording_filename else None
            else:
                state['recording_duration'] = 0.0
                state['recording_samples'] = 0
                state['recording_rate'] = 0.0
                state['recording_filename'] = None

            # Update all modules (skip plot_panel which is a Qt widget, not a GUIModule)
            for key, module in self.gui_modules.items():
                if key == 'plot_panel':
                    continue
                if module and hasattr(module, 'update') and hasattr(module, 'widget'):
                    module.update(state)
        else:
            # Simulation mode: use parent implementation first
            super().update_gui_modules()

            # Add recording state for performance data module
            recording_state = {
                'recording': self.recording
            }
            if self.recording:
                elapsed = time.time() - self.recording_start_time
                sample_rate = self.sample_count / elapsed if elapsed > 0 else 0
                recording_state['recording_duration'] = elapsed
                recording_state['recording_samples'] = self.sample_count
                recording_state['recording_rate'] = sample_rate
                recording_state['recording_filename'] = Path(self.recording_filename).name if self.recording_filename else None
            else:
                recording_state['recording_duration'] = 0.0
                recording_state['recording_samples'] = 0
                recording_state['recording_rate'] = 0.0
                recording_state['recording_filename'] = None

            if 'performance_data' in self.gui_modules:
                self.gui_modules['performance_data'].update(recording_state)

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


def main() -> None:
    """Launch the Stewart Platform controller application."""
    app = QApplication(sys.argv)
    controller = StewartController(app)
    controller.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
