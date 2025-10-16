#!/usr/bin/env python3
"""
Stewart Platform Simulator with LQR Ball Balancing Control

Usage:
    python LQR_ball_sim.py
"""

import tkinter as tk
from tkinter import ttk
import numpy as np

from setup.base_simulator import ControllerConfig, BaseStewartSimulator
from core.control_core import LQRController
from gui.gui_builder import create_standard_layout


class LQRControllerConfig(ControllerConfig):
    """Configuration for LQR controller."""

    def __init__(self, ball_physics_params):
        self.scalar_values = [0.0000001, 0.000001, 0.00001, 0.0001,
                              0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        self.default_weights = {'Q_pos': 1.0, 'Q_vel': 1.0, 'R': 1.0}
        self.default_scalar_indices = {'Q_pos': 7, 'Q_vel': 6, 'R': 5}
        self.ball_physics_params = ball_physics_params
        self.controller_ref = None

    def get_controller_name(self) -> str:
        return "LQR"

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


class LQRStewartSimulator(BaseStewartSimulator):
    """LQR-specific Stewart Platform Simulator with modular GUI."""

    def __init__(self, root):
        ball_physics_params = {
            'radius': 0.04,
            'mass': 0.0027,
            'gravity': 9.81,
            'mass_factor': 1.667
        }

        config = LQRControllerConfig(ball_physics_params)
        super().__init__(root, config)

    def get_layout_config(self):
        """Define GUI layout for LQR simulator."""
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
            {'type': 'controller_output', 'args': {'controller_name': 'LQR'}},
            {'type': 'manual_pose', 'args': {'dof_config': self.dof_config}},
            {'type': 'debug_log', 'args': {'height': 8}},
        ]

        return layout

    def _create_callbacks(self):
        """Override to add LQR-specific callbacks."""
        callbacks = super()._create_callbacks()
        callbacks['show_gain_matrix'] = self.show_gain_matrix
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
            print("Error: Controller not initialized")
            return

        K = self.controller.get_gain_matrix()
        if K is None:
            print("Error: LQR gain matrix not computed")
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

    def _update_controller(self, ball_pos_mm, ball_vel_mm_s, target_pos_mm, dt):
        """Update LQR controller."""
        return self.controller.update(ball_pos_mm, ball_vel_mm_s, target_pos_mm)


def main():
    """Launch LQR Stewart Platform Simulator."""
    root = tk.Tk()
    app = LQRStewartSimulator(root)

    app.log("=" * 50)
    app.log("LQR Ball Balancing Control - Ready")
    app.log("=" * 50)
    app.log("")
    app.log("Quick Start:")
    app.log("1. Click 'Enable LQR Control' to activate automatic balancing")
    app.log("2. Click 'Start' to begin simulation")
    app.log("3. Use 'Push Ball' to test disturbance rejection")
    app.log("4. Select different trajectory patterns to track")
    app.log("5. Adjust pattern size/speed with sliders")
    app.log("")
    app.log("Tuning Tips:")
    app.log("- Increase Q_pos for tighter position control")
    app.log("- Increase Q_vel for more damping")
    app.log("- Decrease R for more aggressive control")
    app.log("- Click 'Show Gain Matrix' to see computed gains")
    app.log("")

    root.mainloop()


if __name__ == "__main__":
    main()