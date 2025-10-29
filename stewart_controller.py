#!/usr/bin/env python3
"""
Comprehensive Stewart Platform Controller
Unified interface supporting all features from PID_sim, LQR_sim, PID_real, LQR_real

Inherits from BaseStewartSimulator and adds:
- Mode switching: Simulation / Hardware
- Controller switching: PID / LQR / Manual
- Conditional GUI module loading based on mode and controller
"""

from PyQt6.QtWidgets import QMessageBox, QDialog, QVBoxLayout, QLabel, QTextEdit, QPushButton
from PyQt6.QtGui import QFont
import time

from setup.base_simulator import BaseStewartSimulator
from setup.hardware_controller_config import (SerialController, IKCache, WindowsTimerManager,
                                               ThreadPriorityManager, HardwareControllerConfig,
                                               LQRControllerConfig as HardwareLQRConfig)
from gui import gui_modules as gm
from core.utils import ControlLoopConfig

# Control configurations
DEFAULT_HW_FREQUENCY_HZ = 250


class ComprehensiveStewartController(BaseStewartSimulator):
    """Comprehensive unified Stewart Platform controller with all features."""

    def __init__(self, app):
        # Mode selection
        self.operation_mode = 'real'  # 'sim' or 'real'
        self.controller_type_selection = 'LQR'  # 'PID', 'LQR', or 'Manual'

        # Hardware-specific components (initialize before super().__init__)
        self.serial_controller = None
        self.connected = False
        self.port_var = ''
        self.ik_cache = None
        self.timer_manager = WindowsTimerManager()
        self.priority_manager = ThreadPriorityManager()
        self.control_frequency = DEFAULT_HW_FREQUENCY_HZ
        self.use_kalman_derivative = False  # PID-specific

        # Create controller config based on mode and controller type
        controller_config = self._create_controller_config()

        # Call parent constructor
        super().__init__(app, controller_config)

        # Override window title
        self.setWindowTitle(f"Stewart Platform - {self.controller_type_selection} [{self.operation_mode.upper()}]")

    def _get_controller_type(self):
        """Override to return selected controller type."""
        return self.controller_type_selection

    def _create_controller_config(self):
        """Create appropriate controller config based on mode and controller type."""
        if self.controller_type_selection == 'PID':
            return HardwareControllerConfig()
        elif self.controller_type_selection == 'LQR':
            if self.operation_mode == 'real':
                return HardwareLQRConfig()
            else:
                from setup.base_simulator import LQRControllerConfig as SimLQRConfig
                return SimLQRConfig()
        else:  # Manual
            # For Manual mode, use a dummy config (won't be used)
            return HardwareControllerConfig()

    def _create_callbacks(self):
        """Create callback dictionary with all features."""
        callbacks = super()._create_callbacks()

        # Add mode/controller switching
        callbacks.update({
            'mode_change': self.on_mode_change,
            'controller_type_change': self.on_controller_type_change,
        })

        # Add hardware callbacks
        if self.operation_mode == 'real':
            callbacks.update({
                'connect': self.connect_serial,
                'disconnect': self.disconnect_serial,
                'show_stats': self.show_timing_stats,
                'frequency_change': self.on_frequency_change,
            })

        # Add LQR-specific callbacks
        if self.controller_type_selection == 'LQR':
            callbacks['show_gain_matrix'] = self.show_gain_matrix

        # Add PID-specific callbacks
        if self.controller_type_selection == 'PID':
            callbacks['kalman_derivative_toggle'] = self.on_kalman_derivative_toggle

        return callbacks

    def get_layout_config(self):
        """Return layout configuration based on mode and controller."""
        from gui.gui_builder import create_standard_layout

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
            left_modules.append({'type': 'serial_connection', 'args': {'port_var': self.port_var}})
            left_modules.append({'type': 'control_frequency',
                               'args': {'frequency_var': self.control_frequency,
                                       'min_freq': 50, 'max_freq': 500}})

        # Simulation control
        left_modules.append({'type': 'simulation_control'})

        # Trajectory pattern
        left_modules.append({'type': 'trajectory_pattern', 'args': {'pattern_var': self.pattern_type}})

        # Simulation-only: Ball control
        if self.operation_mode == 'sim':
            left_modules.append({'type': 'ball_control'})

        # Simulation-only: Camera noise
        if self.operation_mode == 'sim':
            left_modules.append({'type': 'pixy2_camera',
                               'args': {'camera': getattr(self, 'pixy_camera', None)}})

        # Kalman filter
        left_modules.append({'type': 'kalman_filter',
                           'args': {'kalman_filter': getattr(self, 'kalman_filter', None)}})

        # Configuration
        left_modules.append({'type': 'configuration',
                           'args': {'use_offset_var': self.use_top_surface_offset}})

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

        # Hardware-only: Performance stats
        if self.operation_mode == 'real':
            right_modules.append({'type': 'performance_stats'})

        # Debug log
        right_modules.append({'type': 'debug_log', 'args': {'height': 8}})

        layout['columns'][1]['modules'] = right_modules

        return layout

    def _build_modular_gui(self):
        """Override to add mode/controller selector modules and build GUI."""
        from gui.gui_builder import GUIBuilder

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
        }

        layout_config = self.get_layout_config()
        callbacks = self._create_callbacks()

        self.gui_builder = GUIBuilder(self.central_widget, module_registry)
        self.gui_modules = self.gui_builder.build(layout_config, self.colors, callbacks)

        if 'plot_panel' in self.gui_modules:
            self._create_plot(self.gui_modules['plot_panel'])

        # Add controller-specific widgets
        self._add_controller_specific_widgets()

    def _add_controller_specific_widgets(self):
        """Add controller-specific widgets (PID Kalman derivative, LQR gain matrix)."""
        if self.controller_type_selection == 'PID' and 'controller' in self.gui_modules:
            # Add "Use Kalman Velocity for Derivative" checkbox
            from PyQt6.QtWidgets import QWidget, QHBoxLayout, QCheckBox, QLabel

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
            from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton

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

    def connect_serial(self):
        """Connect to hardware."""
        if 'serial_connection' in self.gui_modules:
            port = self.gui_modules['serial_connection'].port_combo.currentText()
        else:
            return

        if not port:
            QMessageBox.critical(self, "Error", "No port selected")
            return

        self.serial_controller = SerialController(port)
        success, message = self.serial_controller.connect()

        if success:
            self.connected = True
            self.log(f"Connected to {port}")

            time.sleep(0.5)
            self.serial_controller.set_servo_speed(0)
            time.sleep(0.1)
            self.serial_controller.set_servo_acceleration(0)
            time.sleep(0.2)
            self.log("Servos: Speed=0, Accel=0")

            success_timer, msg_timer = self.timer_manager.set_high_resolution()
            self.log(msg_timer)

            self.prewarm_ik_cache()

            if 'simulation_control' in self.gui_modules:
                self.gui_modules['simulation_control'].start_btn.setEnabled(True)

            if 'serial_connection' in self.gui_modules:
                self.gui_modules['serial_connection'].update({'connected': True})
        else:
            QMessageBox.critical(self, "Error", message)
            self.log(f"Error: {message}")

    def disconnect_serial(self):
        """Disconnect from hardware."""
        if self.simulation_running:
            self.stop_simulation()

        if self.serial_controller:
            self.serial_controller.disconnect()

        self.connected = False

        if 'simulation_control' in self.gui_modules:
            self.gui_modules['simulation_control'].start_btn.setEnabled(False)

        if 'serial_connection' in self.gui_modules:
            self.gui_modules['serial_connection'].update({'connected': False})

        self.log("Disconnected")

    def prewarm_ik_cache(self):
        """Pre-calculate common IK solutions."""
        if not hasattr(self, 'ik_cache') or self.ik_cache is None:
            self.ik_cache = IKCache(max_size=5000)

        self.log("Pre-warming IK cache...")
        import numpy as np

        tilts = np.arange(-15, 16, 2)
        count = 0
        for rx in tilts:
            for ry in tilts:
                translation = np.array([0.0, 0.0, self.ik.home_height_top_surface])
                rotation = np.array([float(rx), float(ry), 0.0])
                angles = self.ik.calculate_servo_angles(translation, rotation, self.use_top_surface_offset)
                if angles is not None:
                    self.ik_cache.put(translation, rotation, angles)
                    count += 1

        self.log(f"IK cache pre-warmed: {count} poses")

    def on_frequency_change(self, frequency):
        """Handle control frequency change."""
        self.control_frequency = frequency
        self.control_interval = 1.0 / frequency

        # Update Kalman filter dt
        if hasattr(self, 'kalman_filter') and self.kalman_filter:
            self.kalman_filter.set_dt(self.control_interval)

        # Update window title
        self.setWindowTitle(f"Stewart Platform - {self.controller_type_selection} [{self.operation_mode.upper()}] @ {frequency}Hz")
        self.log(f"Control frequency: {frequency}Hz")

    def show_timing_stats(self):
        """Show performance timing statistics."""
        if not hasattr(self, 'performance_data') or not self.performance_data:
            QMessageBox.information(self, "Performance Stats", "No data available. Start the controller first.")
            return

        # Calculate statistics
        import numpy as np

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
        text_edit.setFont(QFont("Consolas", 10))
        layout.addWidget(text_edit)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.setLayout(layout)
        dialog.exec()

    # ============================================================================
    # PID-specific methods
    # ============================================================================

    def on_kalman_derivative_toggle(self):
        """Toggle Kalman derivative option for PID."""
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

    def show_gain_matrix(self):
        """Show LQR gain matrix dialog."""
        if not hasattr(self, 'controller') or self.controller is None:
            QMessageBox.warning(self, "Gain Matrix", "Controller not initialized")
            return

        from core.control_core import LQRController
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
        text_edit.setFont(QFont("Consolas", 10))
        layout.addWidget(text_edit)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.setLayout(layout)
        dialog.exec()

    # ============================================================================
    # Mode/controller switching
    # ============================================================================

    def on_mode_change(self, mode):
        """Handle mode change (sim/real)."""
        if mode == self.operation_mode:
            return

        # Stop simulation if running
        if self.simulation_running:
            self.stop_simulation()

        # Clean up hardware resources
        if self.operation_mode == 'real' and self.connected:
            self.disconnect_serial()

        self.operation_mode = mode
        self.log(f"Mode changed to: {mode.upper()}")

        # Rebuild GUI dynamically
        self._rebuild_gui()

        # Update window title
        self.setWindowTitle(f"Stewart Platform - {self.controller_type_selection} [{mode.upper()}]")

    def on_controller_type_change(self, controller_type):
        """Handle controller type change (PID/LQR/Manual)."""
        if controller_type == self.controller_type_selection:
            return

        # Stop simulation if running
        if self.simulation_running:
            self.stop_simulation()

        self.controller_type_selection = controller_type
        self.log(f"Controller changed to: {controller_type}")

        # Update controller config
        self.controller_config = self._create_controller_config()

        # Rebuild GUI dynamically
        self._rebuild_gui()

        # Reinitialize controller
        self._initialize_controller()

        # Update window title
        self.setWindowTitle(f"Stewart Platform - {controller_type} [{self.operation_mode.upper()}]")

    def _rebuild_gui(self):
        """Rebuild entire GUI with new mode/controller configuration."""
        import gc
        from PyQt6.QtWidgets import QWidget

        # Clear old GUI modules
        if hasattr(self, 'gui_modules'):
            for module in self.gui_modules.values():
                if module and hasattr(module, 'widget'):
                    try:
                        module.widget.deleteLater()
                    except:
                        pass

        # Clear central widget
        old_central = self.central_widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        old_central.deleteLater()

        # Force garbage collection
        gc.collect()

        # Rebuild GUI
        self._build_modular_gui()

        # Process events to ensure GUI is updated
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()


def main():
    """Launch comprehensive controller."""
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    controller = ComprehensiveStewartController(app)
    controller.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
