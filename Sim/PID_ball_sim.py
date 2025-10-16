#!/usr/bin/env python3
"""
Stewart Platform Simulator with PID Ball Balancing Control

Usage:
    python PID_ball_sim.py
"""

import tkinter as tk

from Sim.setup.base_simulator import ControllerConfig, BaseStewartSimulator
from Sim.core.control_core import PIDController
from Sim.gui.gui_builder import create_standard_layout


class PIDControllerConfig(ControllerConfig):
    """Configuration for PID controller."""

    def __init__(self):
        self.scalar_values = [0.0000001, 0.000001, 0.00001, 0.0001,
                              0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        self.default_gains = {'kp': 3.0, 'ki': 1.0, 'kd': 3.0}
        self.default_scalar_indices = {'kp': 4, 'ki': 4, 'kd': 4}
        self.controller_ref = None

    def get_controller_name(self) -> str:
        return "PID"

    def create_controller(self, **kwargs):
        return PIDController(
            kp=kwargs.get('kp', 0.003),
            ki=kwargs.get('ki', 0.001),
            kd=kwargs.get('kd', 0.003),
            output_limit=kwargs.get('output_limit', 15.0),
            derivative_filter_alpha=0.0
        )

    def get_scalar_values(self) -> list:
        return self.scalar_values


class PIDStewartSimulator(BaseStewartSimulator):
    """PID-specific Stewart Platform Simulator with modular GUI."""

    def __init__(self, root):
        config = PIDControllerConfig()
        super().__init__(root, config)

    def get_layout_config(self):
        """Define GUI layout for PID simulator."""
        layout = create_standard_layout(scrollable_columns=True, include_plot=True)

        layout['columns'][0]['modules'] = [
            {'type': 'simulation_control'},
            {'type': 'controller',
             'args': {'controller_config': self.controller_config,
                      'controller_widgets': self.controller_widgets}},
            {'type': 'trajectory_pattern',
             'args': {'pattern_var': self.pattern_type}},
            {'type': 'ball_control'},
            {'type': 'ball_state'},
            {'type': 'configuration',
             'args': {'use_offset_var': self.use_top_surface_offset}},
        ]

        layout['columns'][1]['modules'] = [
            {'type': 'servo_angles', 'args': {'show_actual': True}},
            {'type': 'platform_pose'},
            {'type': 'controller_output', 'args': {'controller_name': 'PID'}},
            {'type': 'manual_pose', 'args': {'dof_config': self.dof_config}},
            {'type': 'debug_log', 'args': {'height': 8}},
        ]

        return layout

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

        if self.controller_enabled.get():
            self.log(f"PID gains updated: Kp={kp:.6f}, Ki={ki:.6f}, Kd={kd:.6f}")

    def _update_controller(self, ball_pos_mm, ball_vel_mm_s, target_pos_mm, dt):
        """Update PID controller."""
        return self.controller.update(ball_pos_mm, target_pos_mm, dt)


def main():
    """Launch PID Stewart Platform Simulator."""
    root = tk.Tk()
    app = PIDStewartSimulator(root)

    app.log("=" * 50)
    app.log("PID Ball Balancing Control - Ready")
    app.log("=" * 50)
    app.log("")
    app.log("Quick Start:")
    app.log("1. Click 'Enable PID Control' to activate automatic balancing")
    app.log("2. Click 'Start' to begin simulation")
    app.log("3. Use 'Push Ball' to test disturbance rejection")
    app.log("4. Select different trajectory patterns to track")
    app.log("5. Adjust pattern size/speed with sliders")
    app.log("")
    app.log("Tuning Tips:")
    app.log("- Increase Kp for faster position correction")
    app.log("- Increase Kd for more damping (reduce oscillation)")
    app.log("- Increase Ki to eliminate steady-state error")
    app.log("- Start with Ki=0 and tune Kp/Kd first")
    app.log("")

    root.mainloop()


if __name__ == "__main__":
    main()