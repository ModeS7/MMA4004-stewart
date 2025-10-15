#!/usr/bin/env python3
"""
Stewart Platform Simulator with LQR Ball Balancing Control
Refactored version using base simulator architecture

Usage:
    python LQR_ball_sim_refactored.py
"""

import tkinter as tk
from tkinter import ttk
import numpy as np

# Import required base classes
from base_simulator import ControllerConfig, BaseStewartSimulator
from control_core import LQRController


class LQRControllerConfig(ControllerConfig):
    """Configuration for LQR controller."""

    def __init__(self, ball_physics_params):
        self.scalar_values = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        self.default_weights = {'Q_pos': 1.0, 'Q_vel': 1.0, 'R': 1.0}
        self.default_scalar_indices = {
            'Q_pos': 7,  # 1.0
            'Q_vel': 6,  # 0.1
            'R': 5  # 0.01
        }
        self.ball_physics_params = ball_physics_params
        self.controller_ref = None  # Will be set after controller creation

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

        for weight_name, label, default in weights:
            frame = ttk.Frame(parent_frame)
            frame.pack(fill='x', pady=5)

            ttk.Label(frame, text=label, font=('Segoe UI', 9)).grid(
                row=0, column=0, sticky='w', pady=2
            )

            slider = ttk.Scale(frame, from_=0.0, to=10.0, orient='horizontal')
            slider.grid(row=0, column=1, sticky='ew', padx=10)
            slider.set(default)
            sliders[weight_name] = slider

            value_label = ttk.Label(frame, text=f"{default:.2f}", width=6, font=('Consolas', 9))
            value_label.grid(row=0, column=2)
            value_labels[weight_name] = value_label

            default_idx = self.default_scalar_indices[weight_name]
            scalar_var = tk.IntVar(value=default_idx)
            scalar_vars[weight_name] = scalar_var

            scalar_combo = ttk.Combobox(
                frame, width=12, state='readonly',
                values=[f'×{s:.7g}' for s in self.scalar_values]
            )
            scalar_combo.grid(row=0, column=3, padx=(5, 0))
            scalar_combo.current(default_idx)

            # Bind events
            slider.config(command=lambda val, w=weight_name: self._on_slider_change(
                w, val, sliders, value_labels, on_param_change_callback
            ))
            scalar_combo.bind('<<ComboboxSelected>>', lambda e, combo=scalar_combo, var=scalar_var, w=weight_name:
            self._on_scalar_change(combo, var, w, on_param_change_callback))

            frame.columnconfigure(1, weight=1)

        return {
            'sliders': sliders,
            'value_labels': value_labels,
            'scalar_vars': scalar_vars,
            'update_fn': lambda: None  # Not needed, handled by callback
        }

    def _on_slider_change(self, weight_name, value, sliders, value_labels, callback):
        """Handle slider value change."""
        val = float(value)
        value_labels[weight_name].config(text=f"{val:.2f}")
        callback()

    def _on_scalar_change(self, combo, var, weight_name, callback):
        """Handle scalar selection change."""
        var.set(combo.current())
        callback()

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

        # Create popup window
        popup = tk.Toplevel(parent_widget)
        popup.title("LQR Gain Matrix")
        popup.configure(bg=colors['bg'])
        popup.geometry("500x300")

        # Create text widget
        text = tk.Text(popup,
                       bg=colors['widget_bg'],
                       fg=colors['fg'],
                       font=('Consolas', 9),
                       wrap='none')
        text.pack(fill='both', expand=True, padx=10, pady=10)

        # Format gain matrix
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
        # Ball physics parameters needed for LQR linearization
        ball_physics_params = {
            'radius': 0.04,
            'mass': 0.0027,
            'gravity': 9.81,
            'mass_factor': 1.667  # For hollow sphere: I = (2/3)*m*r²
        }

        config = LQRControllerConfig(ball_physics_params)
        super().__init__(root, config)

    def _initialize_controller(self):
        """Initialize LQR controller with parameters from widgets."""
        sliders = self.controller_widgets['sliders']
        scalar_vars = self.controller_widgets['scalar_vars']

        # Get raw slider values
        Q_pos_raw = float(sliders['Q_pos'].get())
        Q_vel_raw = float(sliders['Q_vel'].get())
        R_raw = float(sliders['R'].get())

        # Get scalar multipliers
        Q_pos_scalar = self.controller_config.scalar_values[scalar_vars['Q_pos'].get()]
        Q_vel_scalar = self.controller_config.scalar_values[scalar_vars['Q_vel'].get()]
        R_scalar = self.controller_config.scalar_values[scalar_vars['R'].get()]

        # Compute final weights
        Q_pos = Q_pos_raw * Q_pos_scalar
        Q_vel = Q_vel_raw * Q_vel_scalar
        R = R_raw * R_scalar

        # Create controller
        self.controller = self.controller_config.create_controller(
            Q_pos=Q_pos, Q_vel=Q_vel, R=R, output_limit=15.0
        )

        # Store reference for gain matrix display
        self.controller_config.controller_ref = self.controller

        # Log initial weights
        self.log(f"LQR initialized: Q_pos={Q_pos:.6f}, Q_vel={Q_vel:.6f}, R={R:.6f}")

    def on_controller_param_change(self):
        """Update controller when parameters change."""
        if self.controller is None:
            return

        sliders = self.controller_widgets['sliders']
        scalar_vars = self.controller_widgets['scalar_vars']

        # Get raw slider values
        Q_pos_raw = float(sliders['Q_pos'].get())
        Q_vel_raw = float(sliders['Q_vel'].get())
        R_raw = float(sliders['R'].get())

        # Get scalar multipliers
        Q_pos_scalar = self.controller_config.scalar_values[scalar_vars['Q_pos'].get()]
        Q_vel_scalar = self.controller_config.scalar_values[scalar_vars['Q_vel'].get()]
        R_scalar = self.controller_config.scalar_values[scalar_vars['R'].get()]

        # Compute final weights
        Q_pos = Q_pos_raw * Q_pos_scalar
        Q_vel = Q_vel_raw * Q_vel_scalar
        R = R_raw * R_scalar

        # Update controller weights (recomputes gain matrix)
        self.controller.set_weights(Q_pos=Q_pos, Q_vel=Q_vel, R=R)

        # Log update
        if self.controller_enabled.get():
            self.log(f"LQR weights updated: Q_pos={Q_pos:.6f}, Q_vel={Q_vel:.6f}, R={R:.6f}")

    def _update_controller(self, ball_pos_mm, ball_vel_mm_s, target_pos_mm, dt):
        """
        Update LQR controller.

        LQR requires both position AND velocity for optimal control.
        Unlike PID, LQR doesn't need dt since it's a state-feedback controller.

        Args:
            ball_pos_mm: (x, y) ball position in mm
            ball_vel_mm_s: (vx, vy) ball velocity in mm/s
            target_pos_mm: (x, y) target position in mm
            dt: timestep (not used by LQR)

        Returns:
            (rx, ry): platform tilt angles in degrees
        """
        return self.controller.update(ball_pos_mm, ball_vel_mm_s, target_pos_mm)


def main():
    """Launch LQR Stewart Platform Simulator."""
    root = tk.Tk()
    app = LQRStewartSimulator(root)

    # Display startup message in log
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