#!/usr/bin/env python3
"""
PID Controller Configuration for Stewart Platform Simulator
"""

import tkinter as tk
from tkinter import ttk
from base_simulator import ControllerConfig, BaseStewartSimulator
from control_core import PIDController


class PIDControllerConfig(ControllerConfig):
    """Configuration for PID controller."""

    def __init__(self):
        self.scalar_values = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
        self.default_gains = {'kp': 3.0, 'ki': 1.0, 'kd': 3.0}
        self.default_scalar_idx = 4  # 0.001

    def get_controller_name(self) -> str:
        return "PID"

    def create_controller(self, **kwargs):
        return PIDController(
            kp=kwargs.get('kp', 0.003),
            ki=kwargs.get('ki', 0.001),
            kd=kwargs.get('kd', 0.003),
            output_limit=kwargs.get('output_limit', 15.0)
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

        for gain_name, label, default in gains:
            frame = ttk.Frame(parent_frame)
            frame.pack(fill='x', pady=5)

            ttk.Label(frame, text=label, font=('Segoe UI', 9)).grid(
                row=0, column=0, sticky='w', pady=2
            )

            slider = ttk.Scale(frame, from_=0.0, to=10.0, orient='horizontal')
            slider.grid(row=0, column=1, sticky='ew', padx=10)
            slider.set(default)
            sliders[gain_name] = slider

            value_label = ttk.Label(frame, text=f"{default:.2f}", width=6, font=('Consolas', 9))
            value_label.grid(row=0, column=2)
            value_labels[gain_name] = value_label

            scalar_var = tk.IntVar(value=self.default_scalar_idx)
            scalar_vars[gain_name] = scalar_var

            scalar_combo = ttk.Combobox(
                frame, width=12, state='readonly',
                values=[f'×{s:.7g}' for s in self.scalar_values]
            )
            scalar_combo.grid(row=0, column=3, padx=(5, 0))
            scalar_combo.current(self.default_scalar_idx)

            # Bind events
            slider.config(command=lambda val, g=gain_name: self._on_slider_change(
                g, val, sliders, value_labels, on_param_change_callback
            ))
            scalar_combo.bind('<<ComboboxSelected>>', lambda e, combo=scalar_combo, var=scalar_var, g=gain_name:
            self._on_scalar_change(combo, var, g, on_param_change_callback))

            frame.columnconfigure(1, weight=1)

        # Return widget references and update function
        return {
            'sliders': sliders,
            'value_labels': value_labels,
            'scalar_vars': scalar_vars,
            'update_fn': lambda: self._update_gains(sliders, scalar_vars, on_param_change_callback)
        }

    def _on_slider_change(self, gain_name, value, sliders, value_labels, callback):
        """Handle slider value change."""
        val = float(value)
        value_labels[gain_name].config(text=f"{val:.2f}")
        callback()

    def _on_scalar_change(self, combo, var, gain_name, callback):
        """Handle scalar selection change."""
        var.set(combo.current())
        callback()

    def _update_gains(self, sliders, scalar_vars, callback):
        """Update PID gains (called by callback mechanism)."""
        # This is handled by the parent callback mechanism
        pass

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

        kp_raw = float(sliders['kp'].get())
        ki_raw = float(sliders['ki'].get())
        kd_raw = float(sliders['kd'].get())

        kp_scalar = self.controller_config.scalar_values[scalar_vars['kp'].get()]
        ki_scalar = self.controller_config.scalar_values[scalar_vars['ki'].get()]
        kd_scalar = self.controller_config.scalar_values[scalar_vars['kd'].get()]

        kp = kp_raw * kp_scalar
        ki = ki_raw * ki_scalar
        kd = kd_raw * kd_scalar

        self.controller = self.controller_config.create_controller(
            kp=kp, ki=ki, kd=kd, output_limit=15.0
        )

    def on_controller_param_change(self):
        """Update controller when parameters change."""
        if self.controller is None:
            return

        sliders = self.controller_widgets['sliders']
        scalar_vars = self.controller_widgets['scalar_vars']

        kp_raw = float(sliders['kp'].get())
        ki_raw = float(sliders['ki'].get())
        kd_raw = float(sliders['kd'].get())

        kp_scalar = self.controller_config.scalar_values[scalar_vars['kp'].get()]
        ki_scalar = self.controller_config.scalar_values[scalar_vars['ki'].get()]
        kd_scalar = self.controller_config.scalar_values[scalar_vars['kd'].get()]

        kp = kp_raw * kp_scalar
        ki = ki_raw * ki_scalar
        kd = kd_raw * kd_scalar

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