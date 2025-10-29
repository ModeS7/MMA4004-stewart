#!/usr/bin/env python3
"""
Unified Stewart Platform Controller
Supports 4 modes: PID/LQR × Simulation/Real Hardware
Clean minimal GUI with essential controls only
"""

import sys
import gc
from PyQt6.QtWidgets import QApplication, QWidget

from setup.base_simulator import BaseStewartSimulator
from setup.hardware_controller_config import (SerialController, IKCache, WindowsTimerManager,
                                               ThreadPriorityManager, HardwareControllerConfig,
                                               LQRControllerConfig as HardwareLQRConfig)
from LQR_sim import LQRControllerConfig as SimLQRConfig
from gui import gui_modules as gm
from gui.gui_builder import create_standard_layout, GUIBuilder
from core.control_core import PIDController, LQRController


DEFAULT_HW_FREQUENCY_HZ = 250


class UnifiedStewartController(BaseStewartSimulator):
    """Unified controller with minimal GUI supporting PID/LQR and Sim/Real modes."""

    def __init__(self, app):
        # Mode selection
        self.operation_mode = 'sim'  # 'sim' or 'real'
        self.controller_type_selection = 'Manual'  # 'PID', 'LQR', or 'Manual'

        # Hardware components
        self.serial_controller = None
        self.connected = False
        self.port_var = ''
        self.ik_cache = None
        self.timer_manager = WindowsTimerManager()
        self.priority_manager = ThreadPriorityManager()
        self.control_frequency = DEFAULT_HW_FREQUENCY_HZ
        self.use_kalman_derivative = False

        # Performance tracking
        self.performance_data = {
            'loop_times': [],
            'ik_times': [],
            'serial_times': []
        }

        # Create controller config
        controller_config = self._create_controller_config()

        # Call parent constructor
        super().__init__(app, controller_config)

        # Window title
        self.setWindowTitle("Stewart Platform - Unified Controller")

    def _get_controller_type(self):
        """Override to return selected controller type."""
        return self.controller_type_selection

    def _create_controller_config(self):
        """Create controller config based on mode and controller type."""
        controller_name = self.controller_type_selection
        if "PID" in controller_name:
            return HardwareControllerConfig()
        elif "LQR" in controller_name:
            if self.operation_mode == 'real':
                return HardwareLQRConfig()
            else:
                return SimLQRConfig(mode='simulation')
        else:
            return HardwareControllerConfig()

    def _initialize_controller(self):
        """Initialize controller based on current type."""
        if self.controller_type_selection == 'Manual':
            self.controller = None
            return

        sliders = self.controller_widgets['sliders']
        scalar_vars = self.controller_widgets['scalar_vars']

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
                self.controller = None

    def _update_controller(self, ball_pos_mm, ball_vel_mm_s, target_pos_mm, dt):
        """Update controller and return control output."""
        if self.controller is None:
            return None

        controller_name = self.controller_config.get_controller_name()
        if "PID" in controller_name:
            rx, ry = self.controller.update(ball_pos_mm, target_pos_mm, dt)
        elif "LQR" in controller_name:
            rx, ry = self.controller.update(ball_pos_mm, ball_vel_mm_s, target_pos_mm)
        else:
            return None

        return rx, ry

    def get_layout_config(self):
        """Return minimal layout configuration."""
        # Custom layout with narrow left column for controls, plot takes remaining space
        layout = {
            'columns': [
                {
                    'width': 350,  # Narrower control column
                    'scrollable': True,
                    'modules': []
                }
            ],
            'plot': {
                'enabled': True,
                'title': 'Ball Position (Top View)'
            }
        }

        # LEFT COLUMN: Minimal essential controls only
        left_modules = [
            {'type': 'mode_selector', 'args': {'current_mode': self.operation_mode}},
            {'type': 'controller_selector', 'args': {'current_controller': self.controller_type_selection}},
        ]

        # Hardware: Serial connection
        if self.operation_mode == 'real':
            left_modules.append({'type': 'serial_connection', 'args': {'port_var': self.port_var}})

        # Simulation control
        left_modules.append({'type': 'simulation_control'})

        # Trajectory pattern
        left_modules.append({'type': 'trajectory_pattern', 'args': {'pattern_var': self.pattern_type}})

        # Controller parameters (if not Manual)
        if self.controller_type_selection != 'Manual':
            left_modules.append({'type': 'controller',
                                'args': {'controller_config': self.controller_config,
                                        'controller_widgets': self.controller_widgets}})

        # Manual pose control
        left_modules.append({'type': 'manual_pose', 'args': {'dof_config': self.dof_config}})

        # Ball control (sim only)
        if self.operation_mode == 'sim':
            left_modules.append({'type': 'ball_control'})

        layout['columns'][0]['modules'] = left_modules

        return layout

    def _create_callbacks(self):
        """Create callback dictionary."""
        callbacks = super()._create_callbacks()

        callbacks.update({
            'mode_change': self.on_mode_change,
            'controller_type_change': self.on_controller_type_change,
        })

        if self.operation_mode == 'real':
            callbacks.update({
                'connect': self.connect_serial,
                'disconnect': self.disconnect_serial,
            })

        return callbacks

    def _build_modular_gui(self):
        """Build GUI using modular system."""
        module_registry = {
            'mode_selector': gm.ModeSelectionModule,
            'controller_selector': gm.ControllerSelectionModule,
            'simulation_control': gm.SimulationControlModule,
            'controller': gm.ControllerModule,
            'trajectory_pattern': gm.TrajectoryPatternModule,
            'ball_control': gm.BallControlModule,
            'manual_pose': gm.ManualPoseControlModule,
            'serial_connection': gm.SerialConnectionModule,
        }

        layout_config = self.get_layout_config()
        callbacks = self._create_callbacks()

        self.gui_builder = GUIBuilder(self.central_widget, module_registry)
        self.gui_modules = self.gui_builder.build(layout_config, self.colors, callbacks)

        if 'plot_panel' in self.gui_modules:
            self._create_plot(self.gui_modules['plot_panel'])

        self.update_gui_modules()

    def on_mode_change(self, mode):
        """Handle mode change between sim and real."""
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

        # Update controller config and rebuild
        self.controller_config = self._create_controller_config()
        self._create_controller_param_widgets()
        self._rebuild_gui()

        QApplication.processEvents()

        self._initialize_controller()

        # Update window title
        self.setWindowTitle(f"Stewart Platform - Unified Controller [{mode.upper()}]")

    def on_controller_type_change(self, controller_type):
        """Handle controller type change between PID/LQR/Manual."""
        if controller_type == self.controller_type_selection:
            return

        # Stop simulation if running
        if self.simulation_running:
            self.stop_simulation()

        # If controller is currently enabled, disable it before switching
        if self.controller_enabled:
            self.controller_enabled = False
            self.log("Auto-disabling controller for switch")

        self.controller_type_selection = controller_type
        self.log(f"Controller changed to: {controller_type}")

        self.controller_config = self._create_controller_config()
        self._create_controller_param_widgets()
        self._rebuild_gui()

        QApplication.processEvents()

        self._initialize_controller()

    def _rebuild_gui(self):
        """Rebuild GUI after mode/controller change."""
        # Clear old GUI
        if hasattr(self, 'gui_modules'):
            for module in self.gui_modules.values():
                if module and hasattr(module, 'widget'):
                    try:
                        module.widget.deleteLater()
                    except:
                        pass

        old_central = self.central_widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        old_central.deleteLater()

        gc.collect()

        # Rebuild GUI
        self._build_modular_gui()

        QApplication.processEvents()

    def connect_serial(self):
        """Connect to serial hardware."""
        self.log("Serial connection not yet implemented")

    def disconnect_serial(self):
        """Disconnect from serial hardware."""
        self.log("Serial disconnection not yet implemented")


def main():
    """Launch unified controller."""
    app = QApplication(sys.argv)
    controller = UnifiedStewartController(app)
    controller.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
