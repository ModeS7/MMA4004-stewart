#!/usr/bin/env python3
"""
Stewart Platform Simulator with LQR Ball Balancing Control

Usage:
    python LQR_ball_sim.py
"""

import tkinter as tk
from tkinter import ttk
import numpy as np

from base_simulator import ControllerConfig, BaseStewartSimulator
from control_core import LQRController


class LQRControllerConfig(ControllerConfig):
    """Configuration for LQR controller."""

    def __init__(self, ball_physics_params):
        self.scalar_values = [0.0000001, 0.000001, 0.00001, 0.0001,
                              0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        self.default_weights = {'Q_pos': 1.0, 'Q_vel': 1.0, 'R': 1.0}
        self.default_scalar_indices = {
            'Q_pos': 7,  # 1.0
            'Q_vel': 6,  # 0.1
            'R': 5  # 0.01
        }
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

    def create_parameter_widgets(self, parent_frame, colors, on_param_change_callback):
        """Create LQR weight parameter widgets."""
        weights = [
            ('Q_pos', 'Q Position Weight', self.default_weights['Q_pos']),
            ('Q_vel', 'Q Velocity Weight', self.default_weights['Q_vel']),
            ('R', 'R Control Weight', self.default_weights['R'])
        ]

        sliders = {}
        value_labels = {}
        scalar_vars = {}

        for param_name, label, default in weights:
            self.default_scalar_idx = self.default_scalar_indices[param_name]
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
        """Create 'Show Gain Matrix' button for LQR."""
        info_btn_frame = ttk.Frame(parent_frame)
        info_btn_frame.pack(fill='x', pady=(10, 0))

        ttk.Button(info_btn_frame, text="Show Gain Matrix",
                   command=lambda: self._show_gain_matrix(parent_frame, colors),
                   width=20).pack(side='left', padx=5)

    def _show_gain_matrix(self, parent_widget, colors):
        """Display LQR gain matrix in popup."""
        if self.controller_ref is None:
            print("Error: Controller not initialized")
            return

        K = self.controller_ref.get_gain_matrix()
        if K is None:
            print("Error: LQR gain matrix not computed")
            return

        popup = tk.Toplevel(parent_widget)
        popup.title("LQR Gain Matrix")
        popup.configure(bg=colors['bg'])
        popup.geometry("500x300")

        text = tk.Text(popup,
                       bg=colors['widget_bg'],
                       fg=colors['fg'],
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


class LQRStewartSimulator(BaseStewartSimulator):
    """LQR-specific Stewart Platform Simulator."""

    def __init__(self, root):
        ball_physics_params = {
            'radius': 0.04,
            'mass': 0.0027,
            'gravity': 9.81,
            'mass_factor': 1.667
        }

        config = LQRControllerConfig(ball_physics_params)
        super().__init__(root, config)

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