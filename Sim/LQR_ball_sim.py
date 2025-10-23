#!/usr/bin/env python3
"""
Stewart Platform Simulator with LQR Ball Balancing Control

Features:
- Kalman filter for ball state estimation
- LQR optimal control with velocity feedback
- Pixy2 camera noise model

Usage:
    python LQR_ball_sim.py
"""

import sys
from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout,
                              QHBoxLayout, QTextEdit, QMessageBox, QDialog)
from PyQt6.QtGui import QFont

from setup.base_simulator import ControllerConfig, BaseStewartSimulator
from core.control_core import LQRController, KalmanFilter
from gui.gui_builder import create_standard_layout
from core.utils import (ControlLoopConfig, LQRConfig, BallPhysicsConfig, get_controller_defaults,
                        BallPhysicsConfig, KalmanFilterConfig, ControlLoopConfig)


class LQRControllerConfig(ControllerConfig):
    """Configuration for LQR controller."""

    def __init__(self, mode='simulation'):
        config = get_controller_defaults('LQR', mode)
        self.scalar_values = config['scalar_values']
        self.default_weights = config['weights']
        self.default_scalar_indices = config['scalar_indices']
        self.output_limit = config['output_limit']
        self.ball_physics_params = BallPhysicsConfig.as_dict()
        self.controller_ref = None
        self.mode = mode

    def get_controller_name(self) -> str:
        return "LQR"

    def create_controller(self, **kwargs):
        return LQRController(
            Q_pos=kwargs.get('Q_pos', self.default_weights['Q_pos']),
            Q_vel=kwargs.get('Q_vel', self.default_weights['Q_vel']),
            R=kwargs.get('R', self.default_weights['R']),
            output_limit=kwargs.get('output_limit', self.output_limit),
            ball_physics_params=self.ball_physics_params
        )

    def get_scalar_values(self) -> list:
        return self.scalar_values


class LQRStewartSimulator(BaseStewartSimulator):
    """LQR-specific Stewart Platform Simulator with Kalman filter support."""

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

        config = LQRControllerConfig(mode='simulation')
        super().__init__(app, config)

    def get_layout_config(self):
        """Define GUI layout for LQR simulator with Kalman filter."""
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
            {'type': 'controller_output', 'args': {'controller_name': 'LQR'}},
            {'type': 'manual_pose', 'args': {'dof_config': self.dof_config}},
            {'type': 'debug_log', 'args': {'height': 8}},
        ]

        return layout

    def _create_callbacks(self):
        """Override to add LQR-specific callbacks."""
        callbacks = super()._create_callbacks()
        callbacks.update({
            'show_gain_matrix': self.show_gain_matrix,
            'kalman_enable_change': self.on_kalman_enable_change,
            'kalman_param_change': self.on_kalman_param_change,
            'kalman_reset': self.on_kalman_reset,
        })
        return callbacks

    def _build_modular_gui(self):
        """Override to add gain matrix button after GUI is built."""
        super()._build_modular_gui()

        if 'controller' in self.gui_modules:
            controller_widget = self.gui_modules['controller'].widget

            info_widget = QWidget()
            info_layout = QHBoxLayout()
            info_layout.setContentsMargins(0, 10, 0, 0)

            gain_btn = QPushButton("Show Gain Matrix")
            gain_btn.clicked.connect(self.show_gain_matrix)
            gain_btn.setMinimumWidth(150)
            info_layout.addWidget(gain_btn)
            info_layout.addStretch()

            info_widget.setLayout(info_layout)

            # Add to controller widget's layout
            if hasattr(controller_widget, 'layout') and controller_widget.layout() is not None:
                controller_widget.layout().addWidget(info_widget)

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
        popup.resize(500, 300)

        layout = QVBoxLayout()

        text = QTextEdit()
        text.setReadOnly(True)
        text.setFont(QFont('Consolas', 9))
        text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {self.colors['widget_bg']};
                color: {self.colors['fg']};
                border: 1px solid {self.colors['border']};
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

        text.setPlainText(content)
        layout.addWidget(text)

        popup.setLayout(layout)
        popup.exec()

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
            # Reset filter with current ball position
            ball_x_mm = self.ball_pos[0, 0].item() * 1000
            ball_y_mm = self.ball_pos[0, 1].item() * 1000
            self.kalman_filter.reset((ball_x_mm, ball_y_mm))
            self.log("Kalman filter: ENABLED (velocity feedback active)")
        else:
            self.log("Kalman filter: DISABLED (velocity = 0, Q_vel has no effect)")

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
        Update LQR controller - MATCHES HARDWARE BEHAVIOR EXACTLY.

        LQR requires velocity feedback:
        - Without Kalman: velocity = (0.0, 0.0) → position-only control, Q_vel ignored
        - With Kalman: velocity from filter estimates → full state feedback

        This matches the hardware implementation in LQR_ball_real.py line 250-264.
        """
        if self.kalman_enabled:
            # Kalman predict step (using platform angles from FK)
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
            # No Kalman filter - no velocity available (EXACTLY like hardware)
            ball_pos_filtered = ball_pos_mm
            ball_vel_filtered = (0.0, 0.0)  # No velocity without Kalman

        # LQR control update
        return self.controller.update(ball_pos_filtered, ball_vel_filtered, target_pos_mm)


def main():
    """Launch LQR Stewart Platform Simulator."""
    app = QApplication(sys.argv)
    simulator = LQRStewartSimulator(app)
    simulator.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
