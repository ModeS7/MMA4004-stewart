#!/usr/bin/env python3
"""
PID Controller Configuration for Stewart Platform Simulator
"""

import tkinter as tk
from base_simulator import ControllerConfig, BaseStewartSimulator
from control_core import PIDController


class PIDControllerConfig(ControllerConfig):
    """Configuration for PID controller."""

    def __init__(self):
        self.scalar_values = [0.0000001, 0.000001, 0.00001, 0.0001,
                              0.001, 0.01, 0.1, 1.0, 10.0]
        self.default_gains = {'kp': 3.0, 'ki': 1.0, 'kd': 3.0}
        self.default_scalar_idx = 4  # 0.001

    def get_controller_name(self) -> str:
        return "PID"

    def create_controller(self, **kwargs):
        return PIDController(
            kp=kwargs.get('kp', 0.003),
            ki=kwargs.get('ki', 0.001),
            kd=kwargs.get('kd', 0.003),
            output_limit=kwargs.get('output_limit', 15.0),
            derivative_filter_alpha=kwargs.get('derivative_filter_alpha', 0.0)
        )

    def create_parameter_widgets(self, parent_frame, colors, on_param_change_callback):
        """Create PID gain parameter widgets."""
        gains = [
            ('kp', 'P (Proportional)', self.default_gains['kp']),
            ('ki', 'I (Integral)', self.default_gains['ki']),
            ('kd', 'D (Derivative)', self.default_gains['kd'])
        ]

        sliders = {}
        value_labels = {}
        scalar_vars = {}

        for param_name, label, default in gains:
            self.create_parameter_slider(
                parent_frame, param_name, label, default,
                sliders, value_labels, scalar_vars,
                on_param_change_callback
            )

        return {
            'sliders': sliders,
            'value_labels': value_labels,
            'scalar_vars': scalar_vars,
            'update_fn': lambda: None
        }

    def get_scalar_values(self) -> list:
        return self.scalar_values

    def create_info_widgets(self, parent_frame, colors, controller_instance):
        """No additional info widgets for PID."""
        pass


class PIDStewartSimulator(BaseStewartSimulator):
    """PID-specific Stewart Platform Simulator."""

    def __init__(self, root):
        config = PIDControllerConfig()
        super().__init__(root, config)

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

        if self.controller_enabled.get():
            self.log(f"PID gains updated: Kp={kp:.6f}, Ki={ki:.6f}, Kd={kd:.6f}")

    def _update_controller(self, ball_pos_mm, ball_vel_mm_s, target_pos_mm, dt):
        """Update PID controller."""
        return self.controller.update(ball_pos_mm, target_pos_mm, dt)


def main():
    root = tk.Tk()
    app = PIDStewartSimulator(root)
    root.mainloop()


if __name__ == "__main__":
    main()