#!/usr/bin/env python3
"""
Stewart Platform Simulator with PID Ball Balancing Control

Features:
- Kalman filter for ball state estimation
- Optional Kalman velocity for PID derivative term
- Pixy2 camera noise model

Usage:
    python PID_sim.py
"""

import sys
from PyQt6.QtWidgets import QApplication, QWidget, QHBoxLayout, QCheckBox, QLabel
from PyQt6.QtGui import QFont

from setup.base_simulator import ControllerConfig, BaseStewartSimulator
from core.control_core import PIDController, KalmanFilter
from gui.gui_builder import create_standard_layout
from core.utils import (get_controller_defaults,
                        BallPhysicsConfig, KalmanFilterConfig, ControlLoopConfig)


class PIDControllerConfig(ControllerConfig):
    """Configuration for PID controller."""

    def __init__(self, mode='simulation'):

        config = get_controller_defaults('PID', mode)
        self.scalar_values = config['scalar_values']
        self.default_gains = config['gains']
        self.default_scalar_indices = config['scalar_indices']
        self.output_limit = config['output_limit']
        self.derivative_filter_alpha = config['derivative_filter']
        self.controller_ref = None
        self.mode = mode

    def get_controller_name(self) -> str:
        return "PID"

    def create_controller(self, **kwargs):
        return PIDController(
            kp=kwargs.get('kp', self.default_gains['kp']),
            ki=kwargs.get('ki', self.default_gains['ki']),
            kd=kwargs.get('kd', self.default_gains['kd']),
            output_limit=kwargs.get('output_limit', self.output_limit),
            derivative_filter_alpha=kwargs.get('derivative_filter_alpha',
                                               self.derivative_filter_alpha)
        )

    def get_scalar_values(self) -> list:
        return self.scalar_values


class PIDStewartSimulator(BaseStewartSimulator):
    """PID-specific Stewart Platform Simulator with Kalman filter support."""

    def __init__(self, app):

        # Ball physics parameters from centralized config
        ball_physics_params = BallPhysicsConfig.as_dict()

        # Initialize Kalman filter with centralized defaults
        self.kalman_filter = KalmanFilter(
            process_noise_scale=KalmanFilterConfig.DEFAULT_PROCESS_NOISE_SCALE,
            measurement_noise_scale=KalmanFilterConfig.DEFAULT_MEASUREMENT_NOISE_SCALE,
            ball_physics_params=ball_physics_params,
            dt=ControlLoopConfig.INTERVAL_S
        )
        self.kalman_enabled = False

        # PID-specific: Option to use Kalman velocity for derivative
        self.use_kalman_derivative = False

        config = PIDControllerConfig(mode='simulation')
        super().__init__(app, config)

    def get_layout_config(self):
        """Define GUI layout for PID simulator with Kalman filter."""
        layout = create_standard_layout(scrollable_columns=False, include_plot=True)

        layout['columns'][0]['modules'] = [
            {'type': 'simulation_control'},
            {'type': 'trajectory_pattern',
             'args': {'pattern_var': self.pattern_type}},
            {'type': 'ball_control'},
            {'type': 'ball_state'},
            {'type': 'configuration',
             'args': {'use_offset_var': self.use_top_surface_offset}},
            {'type': 'pixy2_camera',
             'args': {'camera': self.pixy_camera}},
            {'type': 'kalman_filter',
             'args': {'kalman_filter': self.kalman_filter}},
        ]

        layout['columns'][1]['modules'] = [
            {'type': 'controller',
             'args': {'controller_config': self.controller_config,
                      'controller_widgets': self.controller_widgets}},
            {'type': 'servo_angles', 'args': {'show_actual': True}},
            {'type': 'platform_pose'},
            {'type': 'controller_output', 'args': {'controller_name': 'PID'}},
            {'type': 'manual_pose', 'args': {'dof_config': self.dof_config}},
            {'type': 'debug_log', 'args': {'height': 8}},
        ]

        return layout

    def _create_callbacks(self):
        """Override to add Kalman filter callbacks."""
        callbacks = super()._create_callbacks()
        callbacks.update({
            'kalman_enable_change': self.on_kalman_enable_change,
            'kalman_param_change': self.on_kalman_param_change,
            'kalman_reset': self.on_kalman_reset,
        })
        return callbacks

    def _build_modular_gui(self):
        """Override to add PID Kalman derivative option."""
        super()._build_modular_gui()

        if 'controller' in self.gui_modules:
            controller_widget = self.gui_modules['controller'].widget

            derivative_widget = QWidget()
            derivative_layout = QHBoxLayout()
            derivative_layout.setContentsMargins(0, 10, 0, 0)

            self.kalman_deriv_checkbox = QCheckBox("Use Kalman Velocity for Derivative")
            self.kalman_deriv_checkbox.setChecked(self.use_kalman_derivative)
            self.kalman_deriv_checkbox.stateChanged.connect(self.on_kalman_derivative_toggle)
            derivative_layout.addWidget(self.kalman_deriv_checkbox)

            self.derivative_status = QLabel("[OFF]")
            font = QFont('Segoe UI', 10)
            self.derivative_status.setFont(font)
            self.derivative_status.setStyleSheet(f"color: {self.colors['border']};")
            derivative_layout.addWidget(self.derivative_status)

            derivative_layout.addStretch()
            derivative_widget.setLayout(derivative_layout)

            # Add to controller widget's layout
            if hasattr(controller_widget, 'layout') and controller_widget.layout() is not None:
                controller_widget.layout().addWidget(derivative_widget)

    def _initialize_controller(self):
        """Initialize PID controller with parameters from widgets."""
        sliders = self.controller_widgets['sliders']
        scalar_vars = self.controller_widgets['scalar_vars']

        kp = self.controller_config.get_scaled_param('kp', sliders, scalar_vars)
        ki = self.controller_config.get_scaled_param('ki', sliders, scalar_vars)
        kd = self.controller_config.get_scaled_param('kd', sliders, scalar_vars)

        self.controller = self.controller_config.create_controller(
            kp=kp, ki=ki, kd=kd, output_limit=15.0
        )

        self.controller_config.controller_ref = self.controller
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
            # Reset filter with current ball position
            ball_x_mm = self.ball_pos[0, 0].item() * 1000
            ball_y_mm = self.ball_pos[0, 1].item() * 1000
            self.kalman_filter.reset((ball_x_mm, ball_y_mm))
        else:
            # Disable Kalman derivative if Kalman is disabled
            if self.use_kalman_derivative:
                self.use_kalman_derivative = False
                if hasattr(self, 'kalman_deriv_checkbox'):
                    self.kalman_deriv_checkbox.setChecked(False)
                self.on_kalman_derivative_toggle()
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
        ball_x_mm = self.ball_pos[0, 0].item() * 1000
        ball_y_mm = self.ball_pos[0, 1].item() * 1000
        self.kalman_filter.reset((ball_x_mm, ball_y_mm))
        self.log("Kalman filter reset")

    def on_kalman_derivative_toggle(self):
        """Handle PID derivative mode toggle."""
        enabled = self.use_kalman_derivative if hasattr(self, 'kalman_deriv_checkbox') else self.kalman_deriv_checkbox.isChecked()
        self.use_kalman_derivative = enabled

        if enabled and not self.kalman_enabled:
            # Can't use Kalman derivative without Kalman enabled
            self.use_kalman_derivative = False
            if hasattr(self, 'kalman_deriv_checkbox'):
                self.kalman_deriv_checkbox.setChecked(False)
            self.log("Enable Kalman filter first to use Kalman derivative")
            return

        mode = "Kalman velocity" if enabled else "finite difference"
        self.log(f"PID derivative: {mode}")

        if hasattr(self, 'derivative_status'):
            self.derivative_status.setText("[ON]" if enabled else "[OFF]")
            self.derivative_status.setStyleSheet(
                f"color: {self.colors['success'] if enabled else self.colors['border']};"
            )

    def update_gui_modules(self):
        """Override to add Kalman filter state."""
        super().update_gui_modules()

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

    def _update_controller(self, ball_pos_mm, ball_vel_mm_s, target_pos_mm, dt):
        """
        Update PID controller with optional Kalman filtering.

        Uses Kalman filter if enabled for:
        - Position filtering (smoothing quantized camera measurements)
        - Velocity estimation (for derivative term if use_kalman_derivative is True)
        """
        import numpy as np

        if self.kalman_enabled:
            # Update Kalman dt to match actual simulation timestep
            self.kalman_filter.set_dt(dt)

            # Kalman predict step (using actual platform angles from previous timestep)
            # Ball motion was caused by actual servo angles, not commanded angles
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

            filtered_pos = (filtered_x, filtered_y)
            filtered_vel = (filtered_vx, filtered_vy)
        else:
            # No filtering - use raw measurements
            filtered_pos = ball_pos_mm
            filtered_vel = ball_vel_mm_s

        # PID control with optional Kalman derivative
        if self.use_kalman_derivative and self.kalman_enabled:
            # Use Kalman velocity for derivative term
            error_x = filtered_pos[0] - target_pos_mm[0]
            error_y = filtered_pos[1] - target_pos_mm[1]

            error_dot_x = filtered_vel[0]
            error_dot_y = filtered_vel[1]

            # Manual PID computation with Kalman velocity
            output_x = (self.controller.kp * error_x +
                        self.controller.ki * self.controller.integral_x +
                        self.controller.kd * error_dot_x)
            output_y = (self.controller.kp * error_y +
                        self.controller.ki * self.controller.integral_y +
                        self.controller.kd * error_dot_y)

            # Update integrals
            self.controller.integral_x += error_x * dt
            self.controller.integral_y += error_y * dt

            # Apply limits
            from core.utils import MAX_TILT_ANGLE_DEG
            output_x = np.clip(output_x, -MAX_TILT_ANGLE_DEG, MAX_TILT_ANGLE_DEG)
            output_y = np.clip(output_y, -MAX_TILT_ANGLE_DEG, MAX_TILT_ANGLE_DEG)

            rx = output_y
            ry = -output_x
        else:
            # Standard PID (finite difference derivative)
            rx, ry = self.controller.update(filtered_pos, target_pos_mm, dt)

        return rx, ry


def main():
    """Launch PID Stewart Platform Simulator."""
    app = QApplication(sys.argv)
    simulator = PIDStewartSimulator(app)
    simulator.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
