#!/usr/bin/env python3
"""
Modular GUI Components for Stewart Platform Simulators

Each module is a self-contained, reusable widget panel.
Modules communicate with the main simulator through callbacks.
PyQt6 implementation for high-performance rendering.
"""

from typing import Dict, Any, Optional, Tuple
import serial.tools.list_ports

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                              QLabel, QPushButton, QSlider, QCheckBox, QComboBox,
                              QGroupBox, QTextEdit, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QMatrix4x4, QQuaternion, QVector3D
import numpy as np

try:
    import pyqtgraph.opengl as gl
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False
    print("Warning: pyqtgraph.opengl not available. Install PyOpenGL for 3D visualization.")

from core.utils import (
    format_time, format_vector_2d, MAX_TILT_ANGLE_DEG,
    IMUKalmanConfig, TrajectoryPatternConfig,
    KalmanFilterConfig, Pixy2CameraConfig, StereoCameraConfig, CAMERA_TYPE, GUIConfig,
    # GUI constants
    GUI_FONT_MONOSPACE, GUI_FONT_SANS,
    GUI_FONT_SIZE_TINY, GUI_FONT_SIZE_SMALL, GUI_FONT_SIZE_NORMAL, GUI_FONT_SIZE_LARGE,
    GUI_BUTTON_WIDTH_SMALL, GUI_BUTTON_WIDTH_MEDIUM, GUI_BUTTON_WIDTH_LARGE,
    GUI_BUTTON_HEIGHT_NORMAL, GUI_BUTTON_HEIGHT_LARGE,
    GUI_VALUE_LABEL_WIDTH_SMALL, GUI_VALUE_LABEL_WIDTH_MEDIUM, GUI_VALUE_LABEL_WIDTH_LARGE,
    GUI_SLIDER_SCALE_COARSE, GUI_SLIDER_SCALE_FINE,
    GUI_COLOR_ORANGE, GUI_LOG_HEIGHT_MULTIPLIER
)


class GUIModule:
    """Base class for all GUI modules."""

    def __init__(self, parent: QWidget, colors: Dict[str, str], callbacks: Optional[Dict[str, Any]] = None) -> None:
        """
        Args:
            parent: Parent QWidget
            colors: Color scheme dict
            callbacks: Dict of callback functions to simulator
        """
        self.parent = parent
        self.colors = colors
        self.callbacks = callbacks or {}
        self.widget: Optional[QWidget] = None

    def create(self) -> Optional[QWidget]:
        """Create and return the module's widget."""
        raise NotImplementedError

    def update(self, state: Dict[str, Any]) -> None:
        """Update module with new state."""
        pass


class SimulationControlModule(GUIModule):
    """Start/Stop/Reset simulation controls."""

    def create(self) -> QWidget:
        """Create simulation control widget with start/stop/reset buttons and time display."""
        group = QGroupBox("Simulation Control")
        layout = QVBoxLayout()

        btn_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.callbacks.get('start'))
        self.start_btn.setMinimumWidth(GUI_BUTTON_WIDTH_MEDIUM)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.callbacks.get('stop'))
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumWidth(GUI_BUTTON_WIDTH_MEDIUM)
        btn_layout.addWidget(self.stop_btn)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.callbacks.get('reset'))
        self.reset_btn.setMinimumWidth(GUI_BUTTON_WIDTH_MEDIUM)
        btn_layout.addWidget(self.reset_btn)

        layout.addLayout(btn_layout)

        self.time_label = QLabel("Time: 0.00s")
        font = QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_LARGE)
        font.setBold(True)
        self.time_label.setFont(font)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.time_label)

        # Calibration timer
        self.calibration_label = QLabel("")
        calib_font = QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_LARGE)
        calib_font.setBold(True)
        self.calibration_label.setFont(calib_font)
        self.calibration_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.calibration_label)

        group.setLayout(layout)
        self.widget = group
        return self.widget

    def update(self, state: Dict[str, Any]) -> None:
        """Update simulation time display, button states, and calibration timer."""
        if 'simulation_time' in state:
            self.time_label.setText(f"Time: {format_time(state['simulation_time'])}")

        # Update button states based on simulation running status
        if 'simulation_running' in state:
            running = state['simulation_running']
            self.start_btn.setEnabled(not running)
            self.stop_btn.setEnabled(running)

        # Update calibration timer
        if state.get('imu_initializing', False):
            time_remaining = state.get('initialization_time_remaining', 0.0)
            self.calibration_label.setText(f"Initialization: {time_remaining:.1f}s")
            self.calibration_label.setStyleSheet(f"color: {self.colors['warning']};")
        elif state.get('imu_calibrating', False):
            time_remaining = state.get('calibration_time_remaining', 0.0)
            self.calibration_label.setText(f"Calibration: {time_remaining:.1f}s")
            self.calibration_label.setStyleSheet(f"color: {self.colors['warning']};")
        else:
            self.calibration_label.setText("")


class ControllerModule(GUIModule):
    """Controller enable/disable and parameter tuning."""

    def __init__(self, parent: QWidget, colors: Dict[str, str], callbacks: Dict[str, Any],
                 controller_config: Any, controller_widgets: Dict[str, Any]) -> None:
        """
        Initialize controller module.

        Args:
            parent: Parent QWidget
            colors: Color scheme dict
            callbacks: Callback functions dict
            controller_config: Controller configuration object
            controller_widgets: Dict containing parameter slider widgets
        """
        super().__init__(parent, colors, callbacks)
        self.controller_config = controller_config
        self.controller_widgets = controller_widgets
        self.controller_enabled = callbacks.get('controller_enabled_var')

    def create(self) -> QWidget:
        """Create controller widget with enable toggle and parameter sliders."""
        controller_name = self.controller_config.get_controller_name()
        group = QGroupBox(f"{controller_name} Ball Balancing")
        layout = QVBoxLayout()

        enable_layout = QHBoxLayout()

        self.enable_checkbox = QCheckBox(f"Enable {controller_name} Control")
        self.enable_checkbox.setChecked(self.controller_enabled)
        self.enable_checkbox.stateChanged.connect(self._on_enable_toggle)
        enable_layout.addWidget(self.enable_checkbox)

        self.status_label = QLabel("[OFF]")
        font = QFont(GUI_FONT_SANS, GUI_FONT_SIZE_LARGE)
        self.status_label.setFont(font)
        self.status_label.setStyleSheet(f"color: {self.colors['border']};")
        enable_layout.addWidget(self.status_label)
        enable_layout.addStretch()

        layout.addLayout(enable_layout)

        # Create a container for parameter sliders (so they can be rebuilt independently)
        self.params_container = QWidget()
        self.params_layout = QVBoxLayout()
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        self.params_container.setLayout(self.params_layout)
        layout.addWidget(self.params_container)

        # Add initial parameter sliders to the container
        self._build_param_sliders()

        group.setLayout(layout)
        self.widget = group
        return self.widget

    def _build_param_sliders(self) -> None:
        """Build parameter sliders in the params_container."""
        if 'param_definitions' in self.controller_widgets:
            param_definitions = self.controller_widgets['param_definitions']
            sliders = self.controller_widgets['sliders']
            value_labels = self.controller_widgets['value_labels']
            scalar_vars = self.controller_widgets['scalar_vars']

            for param_def in param_definitions:
                if len(param_def) == 4:
                    param_name, label, default, default_scalar_idx = param_def
                else:
                    param_name, label, default = param_def
                    default_scalar_idx = 4

                old_idx = getattr(self.controller_config, 'default_scalar_idx', 4)
                self.controller_config.default_scalar_idx = default_scalar_idx

                self.controller_config.create_parameter_slider(
                    self.params_layout, param_name, label, default,
                    sliders, value_labels, scalar_vars,
                    self.callbacks.get('param_change')
                )

                self.controller_config.default_scalar_idx = old_idx

    def _on_enable_toggle(self) -> None:
        """Handle enable checkbox toggle - update state then call callback."""
        self.controller_enabled = self.enable_checkbox.isChecked()
        if self.callbacks.get('toggle_controller'):
            self.callbacks['toggle_controller']()

    def rebuild_params(self, param_definitions: Any, controller_config: Any) -> None:
        """Rebuild parameter sliders for new controller type."""
        self.controller_config = controller_config
        self.controller_widgets['param_definitions'] = param_definitions

        # Update group box title
        controller_name = self.controller_config.get_controller_name()
        if hasattr(self.widget, 'setTitle'):
            self.widget.setTitle(f"{controller_name} Ball Balancing")

        # Update enable checkbox label
        self.enable_checkbox.setText(f"Enable {controller_name} Control")

        # Clear the params_layout (remove all parameter sliders)
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    child = item.layout().takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()

        # Clear slider references
        self.controller_widgets['sliders'].clear()
        self.controller_widgets['value_labels'].clear()
        self.controller_widgets['scalar_vars'].clear()

        # Rebuild parameter sliders
        self._build_param_sliders()

    def update(self, state: Dict[str, Any]) -> None:
        """Update controller enable status and status indicator display."""
        if 'controller_enabled' in state:
            enabled = state['controller_enabled']
            # Update checkbox if state changed externally (block signals to prevent double-toggle)
            if self.enable_checkbox.isChecked() != enabled:
                self.enable_checkbox.blockSignals(True)
                self.enable_checkbox.setChecked(enabled)
                self.enable_checkbox.blockSignals(False)
            # Update status indicator
            if enabled:
                self.status_label.setStyleSheet(f"color: {self.colors['success']};")
            else:
                self.status_label.setStyleSheet(f"color: {self.colors['border']};")


class TrajectoryPatternModule(GUIModule):
    """Trajectory pattern selection with dynamic parameter controls."""

    def __init__(self, parent: QWidget, colors: Dict[str, str], callbacks: Dict[str, Any],
                 pattern_var: str) -> None:
        """
        Initialize trajectory pattern module.

        Args:
            parent: Parent QWidget
            colors: Color scheme dict
            callbacks: Callback functions dict
            pattern_var: Initial pattern selection
        """
        super().__init__(parent, colors, callbacks)
        self.pattern_var = pattern_var
        self.param_widgets: Dict[str, Any] = {}
        self.params_layout: Optional[QVBoxLayout] = None

    def create(self) -> QWidget:
        """Create trajectory pattern widget with pattern selector and dynamic parameter sliders."""
        group = QGroupBox("Trajectory Pattern")
        layout = QVBoxLayout()

        selector_layout = QHBoxLayout()

        selector_layout.addWidget(QLabel("Pattern:"))

        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(['static', 'circle', 'figure8', 'star'])
        self.pattern_combo.setCurrentText(self.pattern_var)
        self.pattern_combo.currentTextChanged.connect(self._on_pattern_change)
        selector_layout.addWidget(self.pattern_combo)

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.callbacks.get('pattern_reset'))
        reset_btn.setMinimumWidth(GUI_BUTTON_WIDTH_SMALL)
        selector_layout.addWidget(reset_btn)
        selector_layout.addStretch()

        layout.addLayout(selector_layout)

        self.info_label = QLabel("Tracking: Center (0, 0)")
        font = QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_SMALL)
        self.info_label.setFont(font)
        self.info_label.setStyleSheet(f"color: {self.colors['success']};")
        layout.addWidget(self.info_label)

        self.params_widget = QWidget()
        self.params_layout = QVBoxLayout()
        self.params_layout.setContentsMargins(0, 10, 0, 0)
        self.params_widget.setLayout(self.params_layout)
        layout.addWidget(self.params_widget)

        self._update_pattern_params()

        group.setLayout(layout)
        self.widget = group
        return self.widget

    def _on_pattern_change(self, pattern: str) -> None:
        """Handle pattern selection change and rebuild parameter sliders."""
        self.pattern_var = pattern
        self._update_pattern_params()
        if self.callbacks.get('pattern_change'):
            self.callbacks['pattern_change'](pattern)

    def _update_pattern_params(self) -> None:
        """Update parameter sliders based on selected pattern."""
        # Clear existing widgets
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.param_widgets.clear()

        pattern_type = self.pattern_var if isinstance(self.pattern_var, str) else self.pattern_combo.currentText()

        pattern_params = {
            'static': [],
            'circle': [
                ('radius', 'Radius (mm)', *TrajectoryPatternConfig.CIRCLE_RADIUS_RANGE_MM, TrajectoryPatternConfig.CIRCLE_DEFAULT_RADIUS_MM, 1.0),
                ('period', 'Period (s)', *TrajectoryPatternConfig.CIRCLE_PERIOD_RANGE_S, TrajectoryPatternConfig.CIRCLE_DEFAULT_PERIOD_S, 0.5)
            ],
            'figure8': [
                ('width', 'Width (mm)', *TrajectoryPatternConfig.FIGURE8_WIDTH_RANGE_MM, TrajectoryPatternConfig.FIGURE8_DEFAULT_WIDTH_MM, 1.0),
                ('height', 'Height (mm)', *TrajectoryPatternConfig.FIGURE8_HEIGHT_RANGE_MM, TrajectoryPatternConfig.FIGURE8_DEFAULT_HEIGHT_MM, 1.0),
                ('period', 'Period (s)', *TrajectoryPatternConfig.FIGURE8_PERIOD_RANGE_S, TrajectoryPatternConfig.FIGURE8_DEFAULT_PERIOD_S, 0.5)
            ],
            'star': [
                ('radius', 'Radius (mm)', *TrajectoryPatternConfig.STAR_RADIUS_RANGE_MM, TrajectoryPatternConfig.STAR_DEFAULT_RADIUS_MM, 1.0),
                ('period', 'Period (s)', *TrajectoryPatternConfig.STAR_PERIOD_RANGE_S, TrajectoryPatternConfig.STAR_DEFAULT_PERIOD_S, 0.5)
            ]
        }

        params = pattern_params.get(pattern_type, [])

        if not params:
            no_params_label = QLabel("No adjustable parameters")
            font = QFont(GUI_FONT_SANS, GUI_FONT_SIZE_SMALL)
            font.setItalic(True)
            no_params_label.setFont(font)
            no_params_label.setStyleSheet(f"color: {self.colors['border']};")
            self.params_layout.addWidget(no_params_label)
            return

        for param_name, label, min_val, max_val, default, resolution in params:
            self._create_param_slider(param_name, label, min_val, max_val, default, resolution)

    def _create_param_slider(self, param_name: str, label: str, min_val: float, max_val: float,
                             default: float, resolution: float) -> None:
        """Create a parameter slider for pattern configuration."""
        param_widget = QWidget()
        grid = QGridLayout()
        grid.setContentsMargins(0, 3, 0, 3)

        label_widget = QLabel(label)
        label_widget.setFont(QFont(GUI_FONT_SANS, GUI_FONT_SIZE_NORMAL))
        grid.addWidget(label_widget, 0, 0)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(int(min_val / resolution))
        slider.setMaximum(int(max_val / resolution))
        slider.setValue(int(default / resolution))
        grid.addWidget(slider, 0, 1)

        value_label = QLabel(f"{default:.1f}")
        value_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL))
        value_label.setMinimumWidth(GUI_VALUE_LABEL_WIDTH_SMALL)
        grid.addWidget(value_label, 0, 2)

        def on_change(val):
            value = val * resolution
            value_label.setText(f"{value:.1f}")
            if self.callbacks.get('pattern_param_change'):
                self.callbacks['pattern_param_change'](param_name, value)

        slider.valueChanged.connect(on_change)
        grid.setColumnStretch(1, 1)

        param_widget.setLayout(grid)
        self.params_layout.addWidget(param_widget)

        self.param_widgets[param_name] = {
            'slider': slider,
            'label': value_label,
            'widget': param_widget,
            'resolution': resolution
        }

    def update(self, state: Dict[str, Any]) -> None:
        """Update trajectory pattern info label with current target position."""
        if 'pattern_info' in state:
            self.info_label.setText(state['pattern_info'])


class BallControlModule(GUIModule):
    """Ball reset and push buttons."""

    def create(self) -> QWidget:
        """Create ball control widget with reset and push buttons."""
        group = QGroupBox("Ball Control")
        layout = QHBoxLayout()

        reset_btn = QPushButton("Reset Ball")
        reset_btn.clicked.connect(self.callbacks.get('reset_ball'))
        reset_btn.setMinimumWidth(GUI_BUTTON_WIDTH_LARGE)
        layout.addWidget(reset_btn)

        push_btn = QPushButton("Push Ball")
        push_btn.clicked.connect(self.callbacks.get('push_ball'))
        push_btn.setMinimumWidth(GUI_BUTTON_WIDTH_LARGE)
        layout.addWidget(push_btn)

        group.setLayout(layout)
        self.widget = group
        return self.widget


class BallStateModule(GUIModule):
    """Ball position and velocity display."""

    def create(self) -> QWidget:
        """Create ball state widget with position and velocity labels."""
        group = QGroupBox("Ball State")
        layout = QVBoxLayout()

        self.pos_label = QLabel("Position: (0.0, 0.0) mm")
        self.pos_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL))
        layout.addWidget(self.pos_label)

        self.vel_label = QLabel("Velocity: (0.0, 0.0) mm/s")
        self.vel_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL))
        layout.addWidget(self.vel_label)

        group.setLayout(layout)
        self.widget = group
        return self.widget

    def update(self, state: Dict[str, Any]) -> None:
        """Update ball position and velocity display."""
        if 'ball_pos' in state:
            self.pos_label.setText(f"Position: {format_vector_2d(state['ball_pos'])}")
        if 'ball_vel' in state:
            if isinstance(state['ball_vel'], tuple) and len(state['ball_vel']) == 2:
                if isinstance(state['ball_vel'][0], str):
                    self.vel_label.setText(f"Status: {state['ball_vel'][0]}")
                else:
                    self.vel_label.setText(f"Velocity: {format_vector_2d(state['ball_vel'], 'mm/s')}")
            elif isinstance(state['ball_vel'], str):
                self.vel_label.setText(f"Status: {state['ball_vel']}")


class ConfigurationModule(GUIModule):
    """Configuration options."""

    def __init__(self, parent: QWidget, colors: Dict[str, str], callbacks: Dict[str, Any],
                 use_offset_var: bool) -> None:
        """
        Initialize configuration module.

        Args:
            parent: Parent QWidget
            colors: Color scheme dict
            callbacks: Callback functions dict
            use_offset_var: Initial state of offset checkbox
        """
        super().__init__(parent, colors, callbacks)
        self.use_offset_var = use_offset_var

    def create(self) -> QWidget:
        """Create configuration widget with top surface offset checkbox."""
        group = QGroupBox("Configuration")
        layout = QVBoxLayout()

        self.offset_checkbox = QCheckBox("Use Top Surface Offset")
        self.offset_checkbox.setChecked(self.use_offset_var)
        self.offset_checkbox.stateChanged.connect(self._on_offset_toggle)
        layout.addWidget(self.offset_checkbox)

        group.setLayout(layout)
        self.widget = group
        return self.widget

    def _on_offset_toggle(self) -> None:
        """Handle offset checkbox toggle."""
        self.use_offset_var = self.offset_checkbox.isChecked()
        if self.callbacks.get('toggle_offset'):
            self.callbacks['toggle_offset']()


class ManualPoseControlModule(GUIModule):
    """6 DOF manual control sliders."""

    def __init__(self, parent: QWidget, colors: Dict[str, str], callbacks: Dict[str, Any],
                 dof_config: Dict[str, Tuple[float, float, float, float, str]]) -> None:
        """
        Initialize manual pose control module.

        Args:
            parent: Parent QWidget
            colors: Color scheme dict
            callbacks: Callback functions dict
            dof_config: Dict of DOF configs (min, max, resolution, default, label)
        """
        super().__init__(parent, colors, callbacks)
        self.dof_config = dof_config
        self.sliders: Dict[str, QSlider] = {}
        self.value_labels: Dict[str, QLabel] = {}

    def create(self) -> QWidget:
        """Create manual pose control widget with 6 DOF sliders and tilt indicator."""
        group = QGroupBox("Manual Pose Control (6 DOF)")
        main_layout = QVBoxLayout()

        # Home button at the top
        home_btn_layout = QHBoxLayout()
        home_btn = QPushButton("Home Position")
        home_btn.clicked.connect(self.callbacks.get('go_home'))
        home_btn.setMinimumWidth(GUI_BUTTON_WIDTH_LARGE)
        home_btn_layout.addWidget(home_btn)
        home_btn_layout.addStretch()
        main_layout.addLayout(home_btn_layout)

        # DOF sliders
        grid = QGridLayout()

        for idx, (dof, (min_val, max_val, res, default, label)) in enumerate(self.dof_config.items()):
            label_widget = QLabel(label)
            label_widget.setFont(QFont(GUI_FONT_SANS, GUI_FONT_SIZE_NORMAL))
            grid.addWidget(label_widget, idx, 0)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(int(min_val / res))
            slider.setMaximum(int(max_val / res))
            slider.setValue(int(default / res))
            grid.addWidget(slider, idx, 1)

            value_label = QLabel(f"{default:.2f}")
            value_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL))
            value_label.setMinimumWidth(GUI_VALUE_LABEL_WIDTH_LARGE)
            grid.addWidget(value_label, idx, 2)

            def on_change(val, dof_name=dof, resolution=res):
                value = val * resolution
                self.value_labels[dof_name].setText(f"{value:.2f}")
                if self.callbacks.get('slider_change'):
                    self.callbacks['slider_change'](dof_name, value)

            slider.valueChanged.connect(on_change)
            self.sliders[dof] = slider
            self.value_labels[dof] = value_label

        grid.setColumnStretch(1, 1)

        # Tilt vector indicator
        tilt_layout = QHBoxLayout()
        tilt_layout.addWidget(QLabel("Tilt Vector:"))
        self.tilt_magnitude_label = QLabel("0.00° (0.0%)")
        self.tilt_magnitude_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL))
        self.tilt_magnitude_label.setStyleSheet(f"color: {self.colors['success']};")
        tilt_layout.addWidget(self.tilt_magnitude_label)
        tilt_layout.addStretch()

        main_layout.addLayout(grid)
        main_layout.addLayout(tilt_layout)

        group.setLayout(main_layout)
        self.widget = group
        return self.widget

    def update(self, state: Dict[str, Any]) -> None:
        """Update DOF slider positions and tilt magnitude indicator."""
        if 'dof_values' in state:
            for dof, val in state['dof_values'].items():
                if dof in self.value_labels:
                    self.value_labels[dof].setText(f"{val:.2f}")
                    # Update slider position to match (for IMU tilt correction feedback)
                    if dof in self.sliders:
                        min_val, max_val, res, _, _ = self.dof_config[dof]
                        self.sliders[dof].blockSignals(True)  # Prevent recursion
                        self.sliders[dof].setValue(int(val / res))
                        self.sliders[dof].blockSignals(False)

        if 'tilt_magnitude' in state:
            mag = state['tilt_magnitude']
            percent = (mag / MAX_TILT_ANGLE_DEG) * 100

            if percent > 80:
                color = self.colors['warning']
            elif percent > 60:
                color = GUI_COLOR_ORANGE
            else:
                color = self.colors['success']

            self.tilt_magnitude_label.setText(f"{mag:.2f}° ({percent:.1f}%)")
            self.tilt_magnitude_label.setStyleSheet(f"color: {color};")


class ServoAnglesModule(GUIModule):
    """Display commanded and actual servo angles."""

    def __init__(self, parent: QWidget, colors: Dict[str, str], callbacks: Dict[str, Any],
                 show_actual: bool = True) -> None:
        """
        Initialize servo angles module.

        Args:
            parent: Parent QWidget
            colors: Color scheme dict
            callbacks: Callback functions dict
            show_actual: Whether to show actual angles (for hardware mode)
        """
        super().__init__(parent, colors, callbacks)
        self.show_actual = show_actual

    def create(self) -> QWidget:
        """Create servo angles widget showing commanded and optionally actual servo angles."""
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        cmd_group = QGroupBox("Commanded Servo Angles (IK)")
        cmd_grid = QGridLayout()

        self.cmd_labels = []
        for i in range(6):
            label = QLabel(f"S{i + 1}: 0.00°")
            label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL))
            cmd_grid.addWidget(label, i // 3, i % 3)
            self.cmd_labels.append(label)

        cmd_group.setLayout(cmd_grid)
        layout.addWidget(cmd_group)

        if self.show_actual:
            actual_group = QGroupBox("Actual Servo Angles")
            actual_grid = QGridLayout()

            self.actual_labels = []
            for i in range(6):
                label = QLabel(f"S{i + 1}: 0.00°")
                label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL))
                actual_grid.addWidget(label, i // 3, i % 3)
                self.actual_labels.append(label)

            actual_group.setLayout(actual_grid)
            layout.addWidget(actual_group)

        container.setLayout(layout)
        self.widget = container
        return self.widget

    def update(self, state: Dict[str, Any]) -> None:
        """Update commanded and actual servo angle displays."""
        if 'cmd_angles' in state:
            for i, angle in enumerate(state['cmd_angles']):
                self.cmd_labels[i].setText(f"S{i + 1}: {angle:6.2f}°")

        if self.show_actual and 'actual_angles' in state:
            for i, angle in enumerate(state['actual_angles']):
                self.actual_labels[i].setText(f"S{i + 1}: {angle:6.2f}°")


class PlatformPoseModule(GUIModule):
    """Display platform pose from forward kinematics."""

    def create(self) -> QWidget:
        """Create platform pose widget showing translation and rotation."""
        group = QGroupBox("Platform Pose (FK)")
        layout = QVBoxLayout()

        self.pos_label = QLabel("X: 0.00  Y: 0.00  Z: 0.00 mm")
        self.pos_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL))
        layout.addWidget(self.pos_label)

        self.rot_label = QLabel("Roll: 0.00  Pitch: 0.00  Yaw: 0.00°")
        self.rot_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL))
        layout.addWidget(self.rot_label)

        group.setLayout(layout)
        self.widget = group
        return self.widget

    def update(self, state: Dict[str, Any]) -> None:
        """Update platform translation and rotation from FK."""
        if 'fk_translation' in state:
            translation = state['fk_translation']
            self.pos_label.setText(f"X: {translation[0]:6.2f}  Y: {translation[1]:6.2f}  Z: {translation[2]:6.2f} mm")

        if 'fk_rotation' in state:
            rotation = state['fk_rotation']
            self.rot_label.setText(f"Roll: {rotation[0]:6.2f}  Pitch: {rotation[1]:6.2f}  Yaw: {rotation[2]:6.2f}°")


class ControllerOutputModule(GUIModule):
    """Display controller output and error."""

    def __init__(self, parent: QWidget, colors: Dict[str, str], callbacks: Dict[str, Any],
                 controller_name: str) -> None:
        """
        Initialize controller output module.

        Args:
            parent: Parent QWidget
            colors: Color scheme dict
            callbacks: Callback functions dict
            controller_name: Name of controller (PID/LQR)
        """
        super().__init__(parent, colors, callbacks)
        self.controller_name = controller_name

    def create(self) -> QWidget:
        """Create controller output widget showing tilt output, magnitude, and error."""
        group = QGroupBox(f"{self.controller_name} Output")
        layout = QVBoxLayout()

        self.output_label = QLabel("Tilt: rx=0.00°  ry=0.00°")
        self.output_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL))
        layout.addWidget(self.output_label)

        self.magnitude_label = QLabel("Magnitude: 0.00° (0%)")
        self.magnitude_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL))
        layout.addWidget(self.magnitude_label)

        self.error_label = QLabel("Error: (0.0, 0.0) mm")
        self.error_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL))
        layout.addWidget(self.error_label)

        group.setLayout(layout)
        self.widget = group
        return self.widget

    def set_controller_name(self, new_name: str) -> None:
        """Update controller name in the group box title."""
        self.controller_name = new_name
        if hasattr(self.widget, 'setTitle'):
            self.widget.setTitle(f"{new_name} Output")

    def update(self, state: Dict[str, Any]) -> None:
        """Update controller output tilt angles, magnitude, and error display."""
        if 'controller_output' in state:
            rx, ry = state['controller_output']
            self.output_label.setText(f"Tilt: rx={rx:.2f}°  ry={ry:.2f}°")

        if 'controller_magnitude' in state:
            mag, percent = state['controller_magnitude']
            self.magnitude_label.setText(f"Magnitude: {mag:.2f}° ({percent:.1f}%)")

        if 'controller_error' in state:
            error = state['controller_error']
            self.error_label.setText(f"Error: {format_vector_2d(error)}")


class DebugLogModule(GUIModule):
    """Debug log display."""

    def __init__(self, parent: QWidget, colors: Dict[str, str], callbacks: Dict[str, Any],
                 height: int = 10) -> None:
        """
        Initialize debug log module.

        Args:
            parent: Parent QWidget
            colors: Color scheme dict
            callbacks: Callback functions dict
            height: Height multiplier for log text area
        """
        super().__init__(parent, colors, callbacks)
        self.height = height

    def create(self) -> QWidget:
        """Create debug log widget with scrollable text area."""
        group = QGroupBox("Debug Log")
        layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_SMALL))
        self.log_text.setMinimumHeight(self.height * GUI_LOG_HEIGHT_MULTIPLIER)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {self.colors['widget_bg']};
                color: {self.colors['fg']};
                border: none;
            }}
        """)
        layout.addWidget(self.log_text)

        group.setLayout(layout)
        self.widget = group
        return self.widget

    def log(self, message: str, timestamp: Optional[float] = None) -> None:
        """
        Append a message to the debug log.

        Args:
            message: Log message text
            timestamp: Optional timestamp to prepend
        """
        if timestamp is not None:
            msg = f"[{format_time(timestamp)}] {message}"
        else:
            msg = f"{message}"
        self.log_text.append(msg)


class SerialConnectionModule(GUIModule):
    """Serial port connection for hardware."""

    def __init__(self, parent: QWidget, colors: Dict[str, str], callbacks: Dict[str, Any],
                 port_var: str, connected_var: bool = False) -> None:
        """
        Initialize serial connection module.

        Args:
            parent: Parent QWidget
            colors: Color scheme dict
            callbacks: Callback functions dict
            port_var: Initial selected port
            connected_var: Initial connection status
        """
        super().__init__(parent, colors, callbacks)
        self.port_var = port_var
        self.connected_var = connected_var
        self.port_combo: Optional[QComboBox] = None
        self.port_status_label: Optional[QLabel] = None

    def create(self) -> QWidget:
        """Create serial connection widget with port selector and connect/disconnect buttons."""
        group = QGroupBox("Serial Connection")
        layout = QVBoxLayout()

        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("Port:"))

        self.port_combo = QComboBox()
        port_layout.addWidget(self.port_combo)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_ports)
        refresh_btn.setMinimumWidth(GUI_BUTTON_WIDTH_SMALL)
        port_layout.addWidget(refresh_btn)
        port_layout.addStretch()

        layout.addLayout(port_layout)

        self.port_status_label = QLabel("")
        self.port_status_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_SMALL))
        self.port_status_label.setStyleSheet(f"color: {self.colors['fg']};")
        layout.addWidget(self.port_status_label)

        btn_layout = QHBoxLayout()

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.callbacks.get('connect'))
        self.connect_btn.setMinimumWidth(GUI_BUTTON_WIDTH_MEDIUM)
        btn_layout.addWidget(self.connect_btn)

        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.callbacks.get('disconnect'))
        self.disconnect_btn.setEnabled(False)
        self.disconnect_btn.setMinimumWidth(GUI_BUTTON_WIDTH_MEDIUM)
        btn_layout.addWidget(self.disconnect_btn)

        layout.addLayout(btn_layout)

        self.status_label = QLabel("Not connected")
        self.status_label.setStyleSheet(f"color: {self.colors['border']};")
        layout.addWidget(self.status_label)

        self._refresh_ports()

        # Set initial connection state if already connected
        if self.connected_var:
            self.update({'connected': True})

        group.setLayout(layout)
        self.widget = group
        return self.widget

    def _refresh_ports(self) -> None:
        """Scan for available serial ports and update port selector dropdown."""
        try:
            ports = list(serial.tools.list_ports.comports())
            port_names = [port.device for port in ports]

            self.port_combo.clear()
            self.port_combo.addItems(port_names)

            if port_names:
                self.port_status_label.setText(f"Found {len(port_names)} port(s)")
                self.port_status_label.setStyleSheet(f"color: {self.colors['success']};")
                self.connect_btn.setEnabled(True)
            else:
                self.port_status_label.setText("No serial ports found")
                self.port_status_label.setStyleSheet(f"color: {self.colors['warning']};")
                self.connect_btn.setEnabled(False)

        except Exception as e:
            self.port_status_label.setText(f"Error: {str(e)}")
            self.port_status_label.setStyleSheet(f"color: {self.colors['warning']};")
            self.connect_btn.setEnabled(False)

    def update(self, state: Dict[str, Any]) -> None:
        """Update connection status and button states."""
        if 'connected' in state:
            if state['connected']:
                self.status_label.setText("Connected")
                self.status_label.setStyleSheet(f"color: {self.colors['success']};")
                self.connect_btn.setEnabled(False)
                self.disconnect_btn.setEnabled(True)
            else:
                self.status_label.setText("Not connected")
                self.status_label.setStyleSheet(f"color: {self.colors['border']};")
                self.connect_btn.setEnabled(True)
                self.disconnect_btn.setEnabled(False)


class PerformanceStatsModule(GUIModule):
    """Performance statistics for hardware mode."""

    def create(self) -> QWidget:
        """Create performance statistics widget showing control loop frequency and IK cache stats."""
        self.group = QGroupBox("Hardware Mode")
        layout = QVBoxLayout()

        self.fps_label = QLabel("Control Loop: 0 Hz")
        font = QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_LARGE)
        font.setBold(True)
        self.fps_label.setFont(font)
        self.fps_label.setStyleSheet(f"color: {self.colors['success']};")
        layout.addWidget(self.fps_label)

        self.cache_label = QLabel("IK Cache: 0.0%")
        self.cache_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL))
        layout.addWidget(self.cache_label)

        self.timeout_label = QLabel("IK Timeouts: 0")
        self.timeout_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL))
        layout.addWidget(self.timeout_label)

        stats_btn = QPushButton("Show Statistics")
        stats_btn.clicked.connect(self.callbacks.get('show_stats'))
        layout.addWidget(stats_btn)

        self.group.setLayout(layout)
        self.widget = self.group
        return self.widget

    def update(self, state: Dict[str, Any]) -> None:
        """Update control loop frequency, IK cache hit rate, and timeout count."""
        # Accept both 'frequency' and 'fps' keys
        freq = state.get('frequency', state.get('fps', 0))
        if freq > 0:
            self.fps_label.setText(f"Control: {freq:.0f} Hz")
            self.group.setTitle(f"{freq:.0f}Hz Mode")

        if 'cache_hit_rate' in state:
            self.cache_label.setText(f"IK Cache: {state['cache_hit_rate'] * 100:.1f}%")

        if 'ik_timeouts' in state:
            self.timeout_label.setText(f"IK Timeouts: {state['ik_timeouts']}")


class CameraModelModule(GUIModule):
    """Camera noise model configuration and control."""

    def __init__(self, parent: QWidget, colors: Dict[str, str], callbacks: Dict[str, Any],
                 camera: Optional[Any] = None) -> None:
        """
        Initialize camera model module.

        Args:
            parent: Parent QWidget
            colors: Color scheme dict
            callbacks: Callback functions dict
            camera: Optional CameraModel instance to control
        """
        super().__init__(parent, colors, callbacks)
        self.camera = camera
        self.sliders: Dict[str, QSlider] = {}
        self.value_labels: Dict[str, QLabel] = {}
        self.camera_enabled = True

    def create(self) -> QWidget:
        """Create camera configuration widget with noise parameters and statistics."""
        # Detect camera type and use appropriate config
        camera_type = CAMERA_TYPE
        if camera_type == 'STEREO':
            camera_name = "Stereo Camera"
            noise_x_range = StereoCameraConfig.NOISE_XY_RANGE
            noise_y_range = StereoCameraConfig.NOISE_XY_RANGE
            noise_x_default = StereoCameraConfig.NOISE_STD_XY_MM
            noise_y_default = StereoCameraConfig.NOISE_STD_XY_MM
            detection_range = StereoCameraConfig.DETECTION_RATE_RANGE
            detection_default = StereoCameraConfig.DETECTION_RATE
            sample_range = StereoCameraConfig.SAMPLE_RATE_RANGE
            sample_default = StereoCameraConfig.DEFAULT_SAMPLE_RATE_HZ
        else:  # PIXY2
            camera_name = "Pixy2 Camera"
            noise_x_range = Pixy2CameraConfig.NOISE_RANGE
            noise_y_range = Pixy2CameraConfig.NOISE_RANGE
            noise_x_default = Pixy2CameraConfig.SUBPIXEL_NOISE_STD_MM
            noise_y_default = Pixy2CameraConfig.SUBPIXEL_NOISE_STD_MM
            detection_range = Pixy2CameraConfig.DETECTION_RATE_RANGE
            detection_default = Pixy2CameraConfig.DEFAULT_DETECTION_RATE
            sample_range = Pixy2CameraConfig.SAMPLE_RATE_RANGE
            sample_default = Pixy2CameraConfig.DEFAULT_SAMPLE_RATE_HZ

        group = QGroupBox(f"{camera_name} Model")
        layout = QVBoxLayout()

        # Enable/Disable camera noise
        enable_layout = QHBoxLayout()

        self.enable_checkbox = QCheckBox("Enable Camera Noise")
        self.enable_checkbox.setChecked(True)
        self.enable_checkbox.stateChanged.connect(self._on_enable_toggle)
        enable_layout.addWidget(self.enable_checkbox)

        self.status_indicator = QLabel("[ON]")
        self.status_indicator.setFont(QFont(GUI_FONT_SANS, GUI_FONT_SIZE_LARGE))
        self.status_indicator.setStyleSheet(f"color: {self.colors['success']};")
        enable_layout.addWidget(self.status_indicator)
        enable_layout.addStretch()

        layout.addLayout(enable_layout)

        # Parameters (camera-specific)
        if camera_type == 'PIXY2':
            # Pixy2 has pixel quantization
            self._create_parameter_row(layout, "Pixel Size:", *Pixy2CameraConfig.PIXEL_SIZE_RANGE, Pixy2CameraConfig.PIXEL_SIZE_MM, 0.1, 'pixel_size', 'mm')
            self._create_parameter_row(layout, "Sub-pixel Noise:", *noise_x_range, noise_x_default, 0.01, 'noise_std', 'mm')
        else:
            # Stereo uses triangulation - noise is similar in X and Y
            self._create_parameter_row(layout, "Noise X-axis:", *noise_x_range, noise_x_default, 0.001, 'noise_x', 'mm', "{:.4f}")
            self._create_parameter_row(layout, "Noise Y-axis:", *noise_y_range, noise_y_default, 0.0001, 'noise_y', 'mm', "{:.4f}")

        self._create_parameter_row(layout, "Detection Rate:", *detection_range, detection_default, 0.001, 'detection_rate', '', "{:.3f}")
        self._create_parameter_row(layout, "Sample Rate:", *sample_range, sample_default, 0.1, 'sample_rate', 'Hz', "{:.1f}")

        info_label = QLabel("(0 Hz = sample every frame)")
        font = QFont(GUI_FONT_SANS, GUI_FONT_SIZE_TINY)
        font.setItalic(True)
        info_label.setFont(font)
        info_label.setStyleSheet(f"color: {self.colors['border']};")
        layout.addWidget(info_label)

        # Camera statistics
        stats_layout = QVBoxLayout()
        stats_label = QLabel("Camera Stats:")
        font = QFont(GUI_FONT_SANS, GUI_FONT_SIZE_NORMAL)
        font.setBold(True)
        stats_label.setFont(font)
        stats_layout.addWidget(stats_label)

        self.measurement_label = QLabel("Raw: (0.0, 0.0) mm")
        self.measurement_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_SMALL))
        stats_layout.addWidget(self.measurement_label)

        self.quantized_label = QLabel("Quantized: (0.0, 0.0) mm")
        self.quantized_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_SMALL))
        stats_layout.addWidget(self.quantized_label)

        self.sample_status_label = QLabel("Last sample: never")
        self.sample_status_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_SMALL))
        stats_layout.addWidget(self.sample_status_label)

        layout.addLayout(stats_layout)

        # Control buttons
        btn_layout = QHBoxLayout()

        reset_btn = QPushButton("Reset Camera")
        reset_btn.clicked.connect(self._on_reset_camera)
        reset_btn.setMinimumWidth(GUI_BUTTON_WIDTH_LARGE)
        btn_layout.addWidget(reset_btn)

        preset_btn = QPushButton("Preset: Real")
        preset_btn.clicked.connect(self._load_real_preset)
        preset_btn.setMinimumWidth(GUI_BUTTON_WIDTH_LARGE)
        btn_layout.addWidget(preset_btn)

        layout.addLayout(btn_layout)

        preset_info = QLabel("Real preset: measured values from actual Pixy2")
        font = QFont(GUI_FONT_SANS, GUI_FONT_SIZE_TINY)
        font.setItalic(True)
        preset_info.setFont(font)
        preset_info.setStyleSheet(f"color: {self.colors['border']};")
        layout.addWidget(preset_info)

        group.setLayout(layout)
        self.widget = group
        return self.widget

    def _create_parameter_row(self, parent_layout: QVBoxLayout, label: str, min_val: float,
                              max_val: float, default: float, resolution: float,
                              param_name: str, unit: str = '', format_str: str = "{:.2f}") -> None:
        """Create a parameter slider row with label, slider, and value display."""
        grid = QGridLayout()

        label_widget = QLabel(label)
        label_widget.setFont(QFont(GUI_FONT_SANS, GUI_FONT_SIZE_NORMAL))
        grid.addWidget(label_widget, 0, 0)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(int(min_val / resolution))
        slider.setMaximum(int(max_val / resolution))
        slider.setValue(int(default / resolution))
        grid.addWidget(slider, 0, 1)

        value_text = format_str.format(default) + (f" {unit}" if unit else "")
        value_label = QLabel(value_text)
        value_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL))
        value_label.setStyleSheet(f"color: {self.colors['highlight']};")
        value_label.setMinimumWidth(GUI_VALUE_LABEL_WIDTH_LARGE)
        grid.addWidget(value_label, 0, 2)

        self.sliders[param_name] = slider
        self.value_labels[param_name] = value_label

        def on_change(val, param=param_name, res=resolution, fmt=format_str, unit_str=unit):
            value = val * res
            value_text = fmt.format(value) + (f" {unit_str}" if unit_str else "")
            self.value_labels[param].setText(value_text)
            self._on_param_change(param, value)

        slider.valueChanged.connect(on_change)
        grid.setColumnStretch(1, 1)

        parent_layout.addLayout(grid)

    def _on_enable_toggle(self) -> None:
        """Handle camera noise enable/disable toggle."""
        enabled = self.enable_checkbox.isChecked()
        self.camera_enabled = enabled

        if enabled:
            self.status_indicator.setText("[ON]")
            self.status_indicator.setStyleSheet(f"color: {self.colors['success']};")
            for slider in self.sliders.values():
                slider.setEnabled(True)
        else:
            self.status_indicator.setText("[OFF]")
            self.status_indicator.setStyleSheet(f"color: {self.colors['border']};")
            for slider in self.sliders.values():
                slider.setEnabled(False)

        if self.callbacks.get('camera_enable_change'):
            self.callbacks['camera_enable_change'](enabled)

    def _on_param_change(self, param_name: str, value: float) -> None:
        """Apply parameter change to camera model."""
        if not self.camera or not self.camera_enabled:
            return

        if param_name == 'pixel_size':
            self.camera.pixel_size = value
        elif param_name == 'noise_std':
            self.camera.set_noise_level(value)
        elif param_name == 'detection_rate':
            self.camera.detection_rate = value
        elif param_name == 'sample_rate':
            self.camera.set_sample_rate(value)

        if self.callbacks.get('camera_param_change'):
            self.callbacks['camera_param_change'](param_name, value)

    def _on_reset_camera(self) -> None:
        """Reset camera to default state."""
        if self.camera:
            self.camera.reset()

        if self.callbacks.get('camera_reset'):
            self.callbacks['camera_reset']()

    def _load_real_preset(self) -> None:
        """Load measured values from actual Pixy2 hardware."""
        presets = {
            'pixel_size': 1.4,
            'noise_std': 0.4,
            'detection_rate': 0.999,
            'sample_rate': 19.3
        }

        for param_name, value in presets.items():
            if param_name in self.sliders:
                # Get resolution from slider
                slider = self.sliders[param_name]
                resolution = (slider.maximum() - slider.minimum()) / 100.0  # Approximate
                if param_name == 'detection_rate':
                    resolution = 0.001
                elif param_name == 'noise_std':
                    resolution = 0.01
                elif param_name == 'pixel_size':
                    resolution = 0.1
                elif param_name == 'sample_rate':
                    resolution = 0.1

                slider.setValue(int(value / resolution))
                self._on_param_change(param_name, value)

        if self.callbacks.get('log'):
            self.callbacks['log']("Loaded real Pixy2 camera preset")

    def update(self, state: Dict[str, Any]) -> None:
        """Update camera display with current measurements and status."""
        if 'camera_raw_measurement' in state:
            x, y = state['camera_raw_measurement']
            if x is not None and y is not None:
                self.measurement_label.setText(f"Raw: ({x:.2f}, {y:.2f}) mm")

        if 'camera_quantized_measurement' in state:
            x, y = state['camera_quantized_measurement']
            if x is not None and y is not None:
                self.quantized_label.setText(f"Quantized: ({x:.2f}, {y:.2f}) mm")

        if 'camera_last_sample_time' in state:
            time_val = state['camera_last_sample_time']
            if time_val >= 0:
                self.sample_status_label.setText(f"Last sample: {time_val:.3f}s")


# Backward compatibility alias
Pixy2CameraModule = CameraModelModule


class KalmanFilterModule(GUIModule):
    """Kalman filter configuration and monitoring."""

    def __init__(self, parent: QWidget, colors: Dict[str, str], callbacks: Dict[str, Any],
                 kalman_filter: Optional[Any] = None, enabled: bool = False, camera_type: Optional[str] = None) -> None:
        """
        Initialize Kalman filter module.

        Args:
            parent: Parent QWidget
            colors: Color scheme dict
            callbacks: Callback functions dict
            kalman_filter: Optional KalmanFilter instance to monitor
            enabled: Initial enabled state
            camera_type: Camera type for selecting appropriate defaults ('PIXY2' or 'STEREO')
        """
        super().__init__(parent, colors, callbacks)
        self.kalman_filter = kalman_filter
        self.sliders: Dict[str, QSlider] = {}
        self.value_labels: Dict[str, QLabel] = {}
        self.filter_enabled = enabled
        self.camera_type = camera_type if camera_type is not None else CAMERA_TYPE

    def create(self) -> QWidget:
        """Create Kalman filter control panel with parameter sliders and state display."""
        group = QGroupBox("Kalman Filter")
        layout = QVBoxLayout()

        # Enable/Disable control
        enable_layout = QHBoxLayout()

        self.enable_checkbox = QCheckBox("Enable Kalman Filter")
        self.enable_checkbox.setChecked(self.filter_enabled)
        self.enable_checkbox.stateChanged.connect(self._on_enable_toggle)
        enable_layout.addWidget(self.enable_checkbox)

        status_text = "[ON]" if self.filter_enabled else "[OFF]"
        status_color = self.colors['success'] if self.filter_enabled else self.colors['border']
        self.status_indicator = QLabel(status_text)
        self.status_indicator.setFont(QFont(GUI_FONT_SANS, GUI_FONT_SIZE_LARGE))
        self.status_indicator.setStyleSheet(f"color: {status_color};")
        enable_layout.addWidget(self.status_indicator)
        enable_layout.addStretch()

        layout.addLayout(enable_layout)

        # Parameters (defaults from KalmanFilterConfig, camera-specific for measurement noise)
        # Select measurement noise default based on camera type
        if self.camera_type == 'STEREO':
            measurement_noise_default = KalmanFilterConfig.DEFAULT_MEASUREMENT_NOISE_STEREO
        else:
            measurement_noise_default = KalmanFilterConfig.DEFAULT_MEASUREMENT_NOISE_PIXY2

        self._create_parameter_slider(layout, "Process Noise (Q):", *KalmanFilterConfig.PROCESS_NOISE_RANGE, KalmanFilterConfig.DEFAULT_PROCESS_NOISE, 'process_noise')
        self._create_parameter_slider(layout, "Measurement Noise (R):", *KalmanFilterConfig.MEASUREMENT_NOISE_RANGE, measurement_noise_default, 'measurement_noise')

        # Filter state display
        state_layout = QVBoxLayout()
        state_label = QLabel("Filter State:")
        font = QFont(GUI_FONT_SANS, GUI_FONT_SIZE_NORMAL)
        font.setBold(True)
        state_label.setFont(font)
        state_layout.addWidget(state_label)

        self.position_label = QLabel("Position: (0.0, 0.0) mm")
        self.position_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_SMALL))
        state_layout.addWidget(self.position_label)

        self.velocity_label = QLabel("Velocity: (0.0, 0.0) mm/s")
        self.velocity_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_SMALL))
        state_layout.addWidget(self.velocity_label)

        self.uncertainty_label = QLabel("Uncertainty: ±0.0 mm")
        self.uncertainty_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_SMALL))
        state_layout.addWidget(self.uncertainty_label)

        layout.addLayout(state_layout)

        # Statistics
        self.stats_label = QLabel("Updates: 0/0 (0.0%)")
        self.stats_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_SMALL))
        self.stats_label.setStyleSheet(f"color: {self.colors['border']};")
        layout.addWidget(self.stats_label)

        # Control buttons
        btn_layout = QHBoxLayout()

        reset_btn = QPushButton("Reset Filter")
        reset_btn.clicked.connect(self._on_reset_filter)
        reset_btn.setMinimumWidth(GUI_BUTTON_WIDTH_LARGE)
        btn_layout.addWidget(reset_btn)

        details_btn = QPushButton("Show Details")
        details_btn.clicked.connect(self._on_show_details)
        details_btn.setMinimumWidth(GUI_BUTTON_WIDTH_LARGE)
        btn_layout.addWidget(details_btn)

        layout.addLayout(btn_layout)

        group.setLayout(layout)
        self.widget = group
        return self.widget

    def _create_parameter_slider(self, parent_layout: QVBoxLayout, label: str, min_val: float,
                                  max_val: float, default: float, param_name: str) -> None:
        """Create a parameter slider for filter tuning."""
        grid = QGridLayout()

        label_widget = QLabel(label)
        label_widget.setFont(QFont(GUI_FONT_SANS, GUI_FONT_SIZE_NORMAL))
        grid.addWidget(label_widget, 0, 0)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(int(min_val * GUI_SLIDER_SCALE_COARSE))
        slider.setMaximum(int(max_val * GUI_SLIDER_SCALE_COARSE))
        slider.setValue(int(default * GUI_SLIDER_SCALE_COARSE))
        slider.setEnabled(False)
        grid.addWidget(slider, 0, 1)

        value_label = QLabel(f"{default:.2f}")
        value_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL))
        value_label.setStyleSheet(f"color: {self.colors['highlight']};")
        value_label.setMinimumWidth(GUI_VALUE_LABEL_WIDTH_SMALL)
        grid.addWidget(value_label, 0, 2)

        self.sliders[param_name] = slider
        self.value_labels[param_name] = value_label

        def on_change(val, param=param_name):
            value = val / 100.0
            self.value_labels[param].setText(f"{value:.2f}")
            self._on_param_change(param, value)

        slider.valueChanged.connect(on_change)
        grid.setColumnStretch(1, 1)

        parent_layout.addLayout(grid)

    def _on_enable_toggle(self) -> None:
        """Handle Kalman filter enable/disable toggle."""
        enabled = self.enable_checkbox.isChecked()
        self.filter_enabled = enabled

        if enabled:
            self.status_indicator.setText("[ON]")
            self.status_indicator.setStyleSheet(f"color: {self.colors['success']};")
            for slider in self.sliders.values():
                slider.setEnabled(True)
        else:
            self.status_indicator.setText("[OFF]")
            self.status_indicator.setStyleSheet(f"color: {self.colors['border']};")
            for slider in self.sliders.values():
                slider.setEnabled(False)

        if self.callbacks.get('kalman_enable_change'):
            self.callbacks['kalman_enable_change'](enabled)

    def _on_param_change(self, param_name: str, value: float) -> None:
        """Apply parameter change to Kalman filter."""
        if not self.kalman_filter or not self.filter_enabled:
            return

        if param_name == 'process_noise':
            self.kalman_filter.set_process_noise(value)
        elif param_name == 'measurement_noise':
            self.kalman_filter.set_measurement_noise(value)

        if self.callbacks.get('kalman_param_change'):
            self.callbacks['kalman_param_change'](param_name, value)

    def _on_reset_filter(self) -> None:
        """Reset Kalman filter to initial state."""
        if self.kalman_filter:
            self.kalman_filter.reset()

        if self.callbacks.get('kalman_reset'):
            self.callbacks['kalman_reset']()

    def _on_show_details(self) -> None:
        """Display detailed Kalman filter statistics in a message box."""
        if not self.kalman_filter:
            return

        stats = self.kalman_filter.get_statistics()
        pos, vel, _ = self.kalman_filter.get_state()
        std_pos = self.kalman_filter.get_position_uncertainty()
        std_vel = self.kalman_filter.get_velocity_uncertainty()

        details = "Kalman Filter Details\n"
        details += "=" * 50 + "\n\n"

        details += "Current State:\n"
        details += f"  Position: ({pos[0]:.2f}, {pos[1]:.2f}) mm\n"
        details += f"  Velocity: ({vel[0]:.2f}, {vel[1]:.2f}) mm/s\n\n"

        details += "Uncertainty (1σ):\n"
        details += f"  Position: (±{std_pos[0]:.3f}, ±{std_pos[1]:.3f}) mm\n"
        details += f"  Velocity: (±{std_vel[0]:.3f}, ±{std_vel[1]:.3f}) mm/s\n\n"

        details += "Statistics:\n"
        details += f"  Predictions: {stats['predictions']}\n"
        details += f"  Updates: {stats['updates']}\n"
        details += f"  Update ratio: {stats['update_ratio'] * 100:.1f}%\n"
        details += f"  Last measurement: {stats['last_measurement_time']:.3f}s\n\n"

        details += "Parameters:\n"
        details += f"  Process noise scale: {stats['process_noise_scale']:.3f}\n"
        details += f"  Measurement noise scale: {stats['measurement_noise_scale']:.3f}\n"

        msg_box = QMessageBox()
        msg_box.setWindowTitle("Kalman Filter Details")
        msg_box.setText(details)
        msg_box.exec()

    def update(self, state: Dict[str, Any]) -> None:
        """Update Kalman filter display with current state and statistics."""
        if 'kalman_position' in state:
            pos = state['kalman_position']
            self.position_label.setText(f"Position: ({pos[0]:.2f}, {pos[1]:.2f}) mm")

        if 'kalman_velocity' in state:
            vel = state['kalman_velocity']
            self.velocity_label.setText(f"Velocity: ({vel[0]:.2f}, {vel[1]:.2f}) mm/s")

        if 'kalman_uncertainty' in state:
            std = state['kalman_uncertainty']
            avg_std = (std[0] + std[1]) / 2.0
            self.uncertainty_label.setText(f"Uncertainty: ±{avg_std:.3f} mm")

        if 'kalman_stats' in state:
            stats = state['kalman_stats']
            self.stats_label.setText(
                f"Updates: {stats['updates']}/{stats['predictions']} "
                f"({stats['update_ratio'] * 100:.1f}%)"
            )


class PlotControlModule(GUIModule):
    """Control plot updates and refresh rate."""

    def __init__(self, parent: QWidget, colors: Dict[str, str], callbacks: Dict[str, Any],
                 plot_enabled_var: bool, plot_rate_var: int) -> None:
        """
        Initialize plot control module.

        Args:
            parent: Parent QWidget
            colors: Color scheme dict
            callbacks: Callback functions dict
            plot_enabled_var: Initial plot enabled state
            plot_rate_var: Initial plot refresh rate in Hz
        """
        super().__init__(parent, colors, callbacks)
        self.plot_enabled_var = plot_enabled_var
        self.plot_rate_var = plot_rate_var

    def create(self) -> QWidget:
        """Create plot control panel with enable toggle and refresh rate slider."""
        group = QGroupBox("Plot Control")
        layout = QVBoxLayout()

        # Enable toggle
        enable_layout = QHBoxLayout()

        self.enable_checkbox = QCheckBox("Enable Live Plot Updates")
        self.enable_checkbox.setChecked(self.plot_enabled_var)
        self.enable_checkbox.stateChanged.connect(self._on_enable_toggle)
        enable_layout.addWidget(self.enable_checkbox)

        status_text = "[ON]" if self.plot_enabled_var else "[OFF]"
        color = self.colors['success'] if self.plot_enabled_var else self.colors['border']
        self.status_indicator = QLabel(status_text)
        self.status_indicator.setFont(QFont(GUI_FONT_SANS, GUI_FONT_SIZE_LARGE))
        self.status_indicator.setStyleSheet(f"color: {color};")
        enable_layout.addWidget(self.status_indicator)
        enable_layout.addStretch()

        layout.addLayout(enable_layout)

        # Rate slider (defaults from GUIConfig)
        rate_grid = QGridLayout()

        rate_grid.addWidget(QLabel("Plot Refresh Rate:"), 0, 0)

        self.rate_slider = QSlider(Qt.Orientation.Horizontal)
        self.rate_slider.setMinimum(GUIConfig.MIN_PLOT_RATE_HZ)
        self.rate_slider.setMaximum(GUIConfig.MAX_PLOT_RATE_HZ)
        self.rate_slider.setValue(self.plot_rate_var)
        self.rate_slider.valueChanged.connect(self._on_rate_change)
        rate_grid.addWidget(self.rate_slider, 0, 1)

        self.rate_label = QLabel(f"{self.plot_rate_var} Hz")
        self.rate_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL))
        self.rate_label.setStyleSheet(f"color: {self.colors['highlight']};")
        self.rate_label.setMinimumWidth(GUI_VALUE_LABEL_WIDTH_MEDIUM)
        rate_grid.addWidget(self.rate_label, 0, 2)

        rate_grid.setColumnStretch(1, 1)
        layout.addLayout(rate_grid)

        info_label = QLabel("Lower rate = better control performance")
        font = QFont(GUI_FONT_SANS, GUI_FONT_SIZE_TINY)
        font.setItalic(True)
        info_label.setFont(font)
        info_label.setStyleSheet(f"color: {self.colors['border']};")
        layout.addWidget(info_label)

        self.perf_label = QLabel("")
        self.perf_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_SMALL))
        self.perf_label.setStyleSheet(f"color: {self.colors['border']};")
        layout.addWidget(self.perf_label)

        group.setLayout(layout)
        self.widget = group
        return self.widget

    def _on_enable_toggle(self) -> None:
        """Handle plot enable/disable toggle."""
        enabled = self.enable_checkbox.isChecked()
        self.plot_enabled_var = enabled
        status_text = "[ON]" if enabled else "[OFF]"
        color = self.colors['success'] if enabled else self.colors['border']
        self.status_indicator.setText(status_text)
        self.status_indicator.setStyleSheet(f"color: {color};")
        self.rate_slider.setEnabled(enabled)
        if self.callbacks.get('plot_enable_change'):
            self.callbacks['plot_enable_change'](enabled)

    def _on_rate_change(self, value: int) -> None:
        """Handle plot refresh rate change."""
        self.plot_rate_var = value
        self.rate_label.setText(f"{value} Hz")
        if self.callbacks.get('plot_rate_change'):
            self.callbacks['plot_rate_change'](value)

    def update(self, state: Dict[str, Any]) -> None:
        """Update plot performance display with frame drop statistics."""
        if 'plot_drops' in state:
            drops = state['plot_drops']
            if drops > 0:
                self.perf_label.setText(f"{drops} frames dropped")
                self.perf_label.setStyleSheet(f"color: {self.colors['warning']};")
            else:
                self.perf_label.setText("No frame drops")
                self.perf_label.setStyleSheet(f"color: {self.colors['success']};")


class ModeSelectionModule(GUIModule):
    """Real/Simulation mode toggle selector."""

    def __init__(self, parent: QWidget, colors: Dict[str, str], callbacks: Dict[str, Any],
                 current_mode: str = 'sim') -> None:
        """
        Initialize mode selection module.

        Args:
            parent: Parent QWidget
            colors: Color scheme dict
            callbacks: Callback functions dict
            current_mode: Initial operation mode ('sim' or 'real')
        """
        super().__init__(parent, colors, callbacks)
        self.current_mode = current_mode

    def create(self) -> QWidget:
        """Create operation mode selector with Simulation/Real Hardware buttons."""
        group = QGroupBox("Operation Mode")
        layout = QHBoxLayout()

        self.sim_btn = QPushButton("Simulation")
        self.sim_btn.setCheckable(True)
        self.sim_btn.setMinimumWidth(GUI_BUTTON_WIDTH_LARGE)
        self.sim_btn.setMinimumHeight(GUI_BUTTON_HEIGHT_LARGE)
        self.sim_btn.clicked.connect(lambda: self._on_mode_select('sim'))
        layout.addWidget(self.sim_btn)

        self.real_btn = QPushButton("Real Hardware")
        self.real_btn.setCheckable(True)
        self.real_btn.setMinimumWidth(GUI_BUTTON_WIDTH_LARGE)
        self.real_btn.setMinimumHeight(GUI_BUTTON_HEIGHT_LARGE)
        self.real_btn.clicked.connect(lambda: self._on_mode_select('real'))
        layout.addWidget(self.real_btn)

        # Set initial state
        if self.current_mode == 'sim':
            self.sim_btn.setChecked(True)
        else:
            self.real_btn.setChecked(True)

        group.setLayout(layout)
        self.widget = group
        return self.widget

    def _on_mode_select(self, mode: str) -> None:
        """Handle mode selection change."""
        if mode == self.current_mode:
            return

        self.current_mode = mode

        # Update button states
        self.sim_btn.setChecked(mode == 'sim')
        self.real_btn.setChecked(mode == 'real')

        if self.callbacks.get('mode_change'):
            self.callbacks['mode_change'](mode)


class ControllerSelectionModule(GUIModule):
    """PID/LQR/Manual controller toggle selector."""

    def __init__(self, parent: QWidget, colors: Dict[str, str], callbacks: Dict[str, Any],
                 current_controller: str = 'PID') -> None:
        """
        Initialize controller selection module.

        Args:
            parent: Parent QWidget
            colors: Color scheme dict
            callbacks: Callback functions dict
            current_controller: Initial controller type ('PID', 'LQR', or 'Manual')
        """
        super().__init__(parent, colors, callbacks)
        self.current_controller = current_controller

    def create(self) -> QWidget:
        """Create controller selection panel with PID/LQR/Manual buttons."""
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Controller selector buttons
        btn_layout = QHBoxLayout()

        self.pid_btn = QPushButton("PID")
        self.pid_btn.setCheckable(True)
        self.pid_btn.setMinimumWidth(GUI_BUTTON_WIDTH_MEDIUM)
        self.pid_btn.setMinimumHeight(GUI_BUTTON_HEIGHT_NORMAL)
        self.pid_btn.clicked.connect(lambda: self._on_controller_select('PID'))
        btn_layout.addWidget(self.pid_btn)

        self.lqr_btn = QPushButton("LQR")
        self.lqr_btn.setCheckable(True)
        self.lqr_btn.setMinimumWidth(GUI_BUTTON_WIDTH_MEDIUM)
        self.lqr_btn.setMinimumHeight(GUI_BUTTON_HEIGHT_NORMAL)
        self.lqr_btn.clicked.connect(lambda: self._on_controller_select('LQR'))
        btn_layout.addWidget(self.lqr_btn)

        self.manual_btn = QPushButton("Manual")
        self.manual_btn.setCheckable(True)
        self.manual_btn.setMinimumWidth(GUI_BUTTON_WIDTH_MEDIUM)
        self.manual_btn.setMinimumHeight(GUI_BUTTON_HEIGHT_NORMAL)
        self.manual_btn.clicked.connect(lambda: self._on_controller_select('Manual'))
        btn_layout.addWidget(self.manual_btn)

        btn_layout.addStretch()

        layout.addLayout(btn_layout)

        # Set initial state
        if self.current_controller == 'PID':
            self.pid_btn.setChecked(True)
        elif self.current_controller == 'LQR':
            self.lqr_btn.setChecked(True)
        else:
            self.manual_btn.setChecked(True)

        container.setLayout(layout)
        self.widget = container
        return self.widget

    def _on_controller_select(self, controller: str) -> None:
        """Handle controller type selection change."""
        if controller == self.current_controller:
            return

        self.current_controller = controller

        # Update button states
        self.pid_btn.setChecked(controller == 'PID')
        self.lqr_btn.setChecked(controller == 'LQR')
        self.manual_btn.setChecked(controller == 'Manual')

        if self.callbacks.get('controller_type_change'):
            self.callbacks['controller_type_change'](controller)


class ControlFrequencyModule(GUIModule):
    """Control loop frequency adjustment (hardware only)."""

    def __init__(self, parent: QWidget, colors: Dict[str, str], callbacks: Dict[str, Any],
                 frequency_var: int, min_freq: int = 50, max_freq: int = 500) -> None:
        """
        Initialize control frequency module.

        Args:
            parent: Parent QWidget
            colors: Color scheme dict
            callbacks: Callback functions dict
            frequency_var: Initial control frequency in Hz
            min_freq: Minimum allowed frequency
            max_freq: Maximum allowed frequency
        """
        super().__init__(parent, colors, callbacks)
        self.frequency_var = frequency_var
        self.min_freq = min_freq
        self.max_freq = max_freq

    def create(self) -> QWidget:
        """Create control frequency panel with slider and diagnostic info."""
        group = QGroupBox("Control Frequency")
        layout = QVBoxLayout()

        freq_grid = QGridLayout()

        freq_grid.addWidget(QLabel("Loop Frequency:"), 0, 0)

        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_slider.setMinimum(self.min_freq)
        self.freq_slider.setMaximum(self.max_freq)
        self.freq_slider.setValue(self.frequency_var)
        self.freq_slider.setTickInterval(50)
        self.freq_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.freq_slider.valueChanged.connect(self._on_freq_change)
        freq_grid.addWidget(self.freq_slider, 0, 1)

        self.freq_label = QLabel(f"{self.frequency_var} Hz")
        freq_font = QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_LARGE)
        freq_font.setBold(True)
        self.freq_label.setFont(freq_font)
        self.freq_label.setStyleSheet(f"color: {self.colors['highlight']};")
        self.freq_label.setMinimumWidth(GUI_VALUE_LABEL_WIDTH_LARGE)
        freq_grid.addWidget(self.freq_label, 0, 2)

        freq_grid.setColumnStretch(1, 1)
        layout.addLayout(freq_grid)

        # Info labels
        info_layout = QVBoxLayout()
        self.period_label = QLabel(f"Period: {1000.0/self.frequency_var:.2f} ms")
        self.period_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_SMALL))
        self.period_label.setStyleSheet(f"color: {self.colors['fg']};")
        info_layout.addWidget(self.period_label)

        self.camera_ratio_label = QLabel(f"Camera ratio: {self.frequency_var/50:.1f}× (50Hz)")
        self.camera_ratio_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_SMALL))
        self.camera_ratio_label.setStyleSheet(f"color: {self.colors['fg']};")
        info_layout.addWidget(self.camera_ratio_label)

        layout.addLayout(info_layout)

        warning_label = QLabel("Higher frequencies increase CPU load")
        warning_label.setFont(QFont(GUI_FONT_SANS, GUI_FONT_SIZE_SMALL))
        warning_label.setStyleSheet(f"color: {self.colors['warning']};")
        layout.addWidget(warning_label)

        group.setLayout(layout)
        self.widget = group
        return self.widget

    def _on_freq_change(self, value: int) -> None:
        """Handle control frequency change and update diagnostic labels."""
        self.frequency_var = value
        self.freq_label.setText(f"{value} Hz")
        self.period_label.setText(f"Period: {1000.0/value:.2f} ms")
        self.camera_ratio_label.setText(f"Camera ratio: {value/50:.1f}× (50Hz)")

        if self.callbacks.get('frequency_change'):
            self.callbacks['frequency_change'](value)

    def update(self, state: Dict[str, Any]) -> None:
        """Update frequency display with actual measured frequency."""
        if 'actual_frequency' in state:
            actual = state['actual_frequency']
            if abs(actual - self.frequency_var) > 5:
                self.freq_label.setStyleSheet(f"color: {self.colors['warning']};")
            else:
                self.freq_label.setStyleSheet(f"color: {self.colors['highlight']};")


class IMUKalmanParametersModule(GUIModule):
    """IMU Kalman filter parameters for orientation tracking."""

    def __init__(self, parent: QWidget, colors: Dict[str, str], callbacks: Dict[str, Any],
                 orientation_kalman: Optional[Any] = None) -> None:
        """
        Initialize IMU Kalman parameters module.

        Args:
            parent: Parent QWidget
            colors: Color scheme dict
            callbacks: Callback functions dict
            orientation_kalman: Optional orientation Kalman filter instance
        """
        super().__init__(parent, colors, callbacks)
        self.orientation_kalman = orientation_kalman
        self.sliders: Dict[str, QSlider] = {}
        self.value_labels: Dict[str, QLabel] = {}

    def create(self) -> QWidget:
        """Create IMU Kalman filter configuration panel with noise parameters."""
        group = QGroupBox("IMU Kalman Filter Parameters")
        layout = QVBoxLayout()

        # Enable/Disable IMU tilt correction
        enable_layout = QHBoxLayout()

        self.enable_checkbox = QCheckBox("Enable IMU Tilt Correction")
        self.enable_checkbox.setChecked(False)
        self.enable_checkbox.stateChanged.connect(self._on_enable_toggle)
        enable_layout.addWidget(self.enable_checkbox)

        self.status_indicator = QLabel("[OFF]")
        self.status_indicator.setFont(QFont(GUI_FONT_SANS, GUI_FONT_SIZE_LARGE))
        self.status_indicator.setStyleSheet(f"color: {self.colors['border']};")
        enable_layout.addWidget(self.status_indicator)
        enable_layout.addStretch()

        layout.addLayout(enable_layout)

        # Yaw tracking toggle (6-DOF mode with magnetometer)
        yaw_layout = QHBoxLayout()

        self.yaw_tracking_checkbox = QCheckBox("Enable Yaw Tracking (6-DOF)")
        self.yaw_tracking_checkbox.setChecked(IMUKalmanConfig.ENABLE_YAW_TRACKING)
        self.yaw_tracking_checkbox.stateChanged.connect(self._on_yaw_tracking_toggle)
        yaw_layout.addWidget(self.yaw_tracking_checkbox)

        self.yaw_status_indicator = QLabel("[4-DOF]" if not IMUKalmanConfig.ENABLE_YAW_TRACKING else "[6-DOF]")
        self.yaw_status_indicator.setFont(QFont(GUI_FONT_SANS, GUI_FONT_SIZE_NORMAL))
        self.yaw_status_indicator.setStyleSheet(f"color: {self.colors['border'] if not IMUKalmanConfig.ENABLE_YAW_TRACKING else self.colors['success']};")
        yaw_layout.addWidget(self.yaw_status_indicator)
        yaw_layout.addStretch()

        layout.addLayout(yaw_layout)

        # Kalman filter noise parameters (defaults from utils.py)
        self._create_parameter_slider(layout, "Accel Noise:", 0.1, 5.0, IMUKalmanConfig.DEFAULT_ACCEL_NOISE, 'accel_noise')
        self._create_parameter_slider(layout, "Gyro Noise:", 0.001, 0.1, IMUKalmanConfig.DEFAULT_GYRO_NOISE, 'gyro_noise')
        self._create_parameter_slider(layout, "Process Noise Angle:", 0.000001, 0.1, IMUKalmanConfig.DEFAULT_PROCESS_NOISE_ANGLE, 'process_noise_angle')
        self._create_parameter_slider(layout, "Process Noise Bias:", 0.000001, 0.01, IMUKalmanConfig.DEFAULT_PROCESS_NOISE_BIAS, 'process_noise_bias')

        # Magnetometer noise (only relevant when yaw tracking enabled)
        self._create_parameter_slider(layout, "Mag Noise (yaw):", 0.01, 1.0, IMUKalmanConfig.DEFAULT_MAG_NOISE, 'mag_noise')

        # Gyro scale multiplier (for tilt correction tuning, default from utils.py)
        self._create_parameter_slider(layout, "Gyro Scale Multiplier:", 0.1, 15.0, IMUKalmanConfig.DEFAULT_GYRO_SCALE_MULTIPLIER, 'gyro_scale')

        # Yaw reference and control
        yaw_control_layout = QVBoxLayout()
        yaw_control_label = QLabel("Yaw Control:")
        font = QFont(GUI_FONT_SANS, GUI_FONT_SIZE_NORMAL)
        font.setBold(True)
        yaw_control_label.setFont(font)
        yaw_control_layout.addWidget(yaw_control_label)

        # Button to set current orientation as reference
        button_layout = QHBoxLayout()
        self.set_yaw_ref_button = QPushButton("Set Current as 0°")
        self.set_yaw_ref_button.clicked.connect(self._on_set_yaw_reference)
        button_layout.addWidget(self.set_yaw_ref_button)
        button_layout.addStretch()
        yaw_control_layout.addLayout(button_layout)

        # Slider for manual yaw offset adjustment
        self._create_yaw_offset_slider(yaw_control_layout)

        layout.addLayout(yaw_control_layout)

        # Current orientation display
        state_layout = QVBoxLayout()
        state_label = QLabel("IMU Orientation:")
        font = QFont(GUI_FONT_SANS, GUI_FONT_SIZE_NORMAL)
        font.setBold(True)
        state_label.setFont(font)
        state_layout.addWidget(state_label)

        self.orientation_label = QLabel("RX: 0.0°, RY: 0.0°")
        self.orientation_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL))
        state_layout.addWidget(self.orientation_label)

        self.bias_label = QLabel("Bias: (0.00, 0.00, 0.00) rad/s")
        self.bias_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_SMALL))
        self.bias_label.setStyleSheet(f"color: {self.colors['border']};")
        state_layout.addWidget(self.bias_label)

        self.mag_count_label = QLabel("Mag updates: 0")
        self.mag_count_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_SMALL))
        self.mag_count_label.setStyleSheet(f"color: {self.colors['border']};")
        state_layout.addWidget(self.mag_count_label)

        layout.addLayout(state_layout)

        group.setLayout(layout)
        self.widget = group
        return self.widget

    def _create_parameter_slider(self, parent_layout: QVBoxLayout, label: str, min_val: float,
                                  max_val: float, default: float, param_name: str) -> None:
        """Create a parameter slider with automatic scale selection."""
        grid = QGridLayout()

        label_widget = QLabel(label)
        label_widget.setFont(QFont(GUI_FONT_SANS, GUI_FONT_SIZE_NORMAL))
        grid.addWidget(label_widget, 0, 0)

        slider = QSlider(Qt.Orientation.Horizontal)
        # Use logarithmic scale for very small values
        if max_val < 1.0:
            slider.setMinimum(int(min_val * GUI_SLIDER_SCALE_FINE))
            slider.setMaximum(int(max_val * GUI_SLIDER_SCALE_FINE))
            slider.setValue(int(default * GUI_SLIDER_SCALE_FINE))
        else:
            slider.setMinimum(int(min_val * GUI_SLIDER_SCALE_COARSE))
            slider.setMaximum(int(max_val * GUI_SLIDER_SCALE_COARSE))
            slider.setValue(int(default * GUI_SLIDER_SCALE_COARSE))
        grid.addWidget(slider, 0, 1)

        value_label = QLabel(f"{default:.4f}")
        value_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL))
        value_label.setStyleSheet(f"color: {self.colors['highlight']};")
        value_label.setMinimumWidth(GUI_VALUE_LABEL_WIDTH_MEDIUM)
        grid.addWidget(value_label, 0, 2)

        self.sliders[param_name] = slider
        self.value_labels[param_name] = value_label

        def on_change(val, param=param_name):
            if max_val < 1.0:
                value = val / GUI_SLIDER_SCALE_FINE
            else:
                value = val / GUI_SLIDER_SCALE_COARSE
            self.value_labels[param].setText(f"{value:.4f}")
            self._on_param_change(param, value)

        slider.valueChanged.connect(on_change)
        grid.setColumnStretch(1, 1)

        parent_layout.addLayout(grid)

    def _on_enable_toggle(self) -> None:
        """Handle IMU tilt correction enable/disable toggle."""
        enabled = self.enable_checkbox.isChecked()

        if enabled:
            self.status_indicator.setText("[ON]")
            self.status_indicator.setStyleSheet(f"color: {self.colors['success']};")
        else:
            self.status_indicator.setText("[OFF]")
            self.status_indicator.setStyleSheet(f"color: {self.colors['border']};")

        if self.callbacks.get('imu_tilt_correction_toggle'):
            self.callbacks['imu_tilt_correction_toggle'](enabled)

    def _on_yaw_tracking_toggle(self) -> None:
        """Handle yaw tracking (6-DOF) enable/disable toggle."""
        enabled = self.yaw_tracking_checkbox.isChecked()

        if enabled:
            self.yaw_status_indicator.setText("[6-DOF]")
            self.yaw_status_indicator.setStyleSheet(f"color: {self.colors['success']};")
        else:
            self.yaw_status_indicator.setText("[4-DOF]")
            self.yaw_status_indicator.setStyleSheet(f"color: {self.colors['border']};")

        if self.callbacks.get('imu_yaw_tracking_toggle'):
            self.callbacks['imu_yaw_tracking_toggle'](enabled)

    def _create_yaw_offset_slider(self, parent_layout: QVBoxLayout) -> None:
        """Create yaw offset adjustment slider for reference frame calibration."""
        grid = QGridLayout()

        label_widget = QLabel("Yaw Offset:")
        label_widget.setFont(QFont(GUI_FONT_SANS, GUI_FONT_SIZE_NORMAL))
        grid.addWidget(label_widget, 0, 0)

        self.yaw_offset_slider = QSlider(Qt.Orientation.Horizontal)
        self.yaw_offset_slider.setMinimum(-180 * 10)
        self.yaw_offset_slider.setMaximum(180 * 10)
        self.yaw_offset_slider.setValue(0)
        grid.addWidget(self.yaw_offset_slider, 0, 1)

        self.yaw_offset_label = QLabel("0.0°")
        self.yaw_offset_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL))
        self.yaw_offset_label.setStyleSheet(f"color: {self.colors['highlight']};")
        self.yaw_offset_label.setFixedWidth(80)
        grid.addWidget(self.yaw_offset_label, 0, 2)

        def on_change(val):
            offset_deg = val / 10.0
            self.yaw_offset_label.setText(f"{offset_deg:.1f}°")
            if self.callbacks.get('imu_yaw_offset_change'):
                self.callbacks['imu_yaw_offset_change'](offset_deg)

        self.yaw_offset_slider.valueChanged.connect(on_change)
        grid.setColumnStretch(1, 1)

        parent_layout.addLayout(grid)

    def _on_set_yaw_reference(self) -> None:
        """Set current yaw as reference (0°)."""
        if self.callbacks.get('imu_set_yaw_reference'):
            self.callbacks['imu_set_yaw_reference']()

    def _on_param_change(self, param_name: str, value: float) -> None:
        """Apply IMU Kalman parameter change."""
        if self.callbacks.get('imu_kalman_param_change'):
            self.callbacks['imu_kalman_param_change'](param_name, value)

    def update(self, state: Dict[str, Any]) -> None:
        """Update IMU orientation and bias display."""
        if 'imu_orientation' in state:
            orientation = state['imu_orientation']
            if len(orientation) == 3:
                # 6-DOF mode (roll, pitch, yaw)
                rx, ry, yaw = orientation
                self.orientation_label.setText(f"RX: {rx:.2f}°, RY: {ry:.2f}°, Yaw: {yaw:.2f}°")
            else:
                # 4-DOF mode (roll, pitch only)
                rx, ry = orientation
                self.orientation_label.setText(f"RX: {rx:.2f}°, RY: {ry:.2f}°")

        if 'imu_bias' in state:
            bias = state['imu_bias']
            if len(bias) == 3:
                # 6-DOF mode (bx, by, bz)
                bias_x, bias_y, bias_z = bias
                self.bias_label.setText(f"Bias: ({bias_x:.4f}, {bias_y:.4f}, {bias_z:.4f}) rad/s")
            else:
                # 4-DOF mode (bx, by only)
                bias_x, bias_y = bias
                self.bias_label.setText(f"Bias: ({bias_x:.4f}, {bias_y:.4f}) rad/s")

        if 'imu_mag_updates' in state:
            mag_count = state['imu_mag_updates']
            self.mag_count_label.setText(f"Mag updates: {mag_count}")


class IKZOptimizationModule(GUIModule):
    """IK Z offset optimization control."""

    def __init__(self, parent: QWidget, colors: Dict[str, str], callbacks: Dict[str, Any]) -> None:
        """
        Initialize IK Z optimization module.

        Args:
            parent: Parent QWidget
            colors: Color scheme dict
            callbacks: Callback functions dict
        """
        super().__init__(parent, colors, callbacks)
        self.enabled = False

    def create(self) -> QWidget:
        """Create Z optimization control panel with status display."""
        group = QGroupBox("IK Z Optimization")
        layout = QVBoxLayout()

        # Enable/Disable control
        enable_layout = QHBoxLayout()

        self.enable_checkbox = QCheckBox("Enable Z Optimization")
        self.enable_checkbox.setChecked(False)
        self.enable_checkbox.stateChanged.connect(self._on_enable_toggle)
        enable_layout.addWidget(self.enable_checkbox)

        self.status_indicator = QLabel("[OFF]")
        self.status_indicator.setFont(QFont(GUI_FONT_SANS, GUI_FONT_SIZE_LARGE))
        self.status_indicator.setStyleSheet(f"color: {self.colors['border']};")
        enable_layout.addWidget(self.status_indicator)
        enable_layout.addStretch()

        layout.addLayout(enable_layout)

        # Description
        description = QLabel(
            "Dynamically adjusts Z height to balance servo angles around neutral (0°).\n"
            "Keeps servos centered and maximizes available range."
        )
        description.setFont(QFont(GUI_FONT_SANS, GUI_FONT_SIZE_SMALL))
        description.setStyleSheet(f"color: {self.colors['border']};")
        description.setWordWrap(True)
        layout.addWidget(description)

        # Statistics display
        stats_layout = QVBoxLayout()

        self.z_offset_label = QLabel("Z Offset: 0.0 mm")
        self.z_offset_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_SMALL))
        stats_layout.addWidget(self.z_offset_label)

        self.servo_balance_label = QLabel("Servo Balance: Max=0.0°, Min=0.0°")
        self.servo_balance_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_SMALL))
        stats_layout.addWidget(self.servo_balance_label)

        self.imbalance_label = QLabel("Imbalance: 0.0°")
        self.imbalance_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_SMALL))
        stats_layout.addWidget(self.imbalance_label)

        layout.addLayout(stats_layout)

        group.setLayout(layout)
        self.widget = group
        return self.widget

    def _on_enable_toggle(self, state: int) -> None:
        """Handle Z optimization enable/disable toggle."""
        self.enabled = state == Qt.CheckState.Checked.value
        if self.enabled:
            self.status_indicator.setText("[ON]")
            self.status_indicator.setStyleSheet(f"color: {self.colors['highlight']};")
        else:
            self.status_indicator.setText("[OFF]")
            self.status_indicator.setStyleSheet(f"color: {self.colors['border']};")

        if 'z_optimization_toggle' in self.callbacks:
            self.callbacks['z_optimization_toggle'](self.enabled)

    def update(self, state: Dict[str, Any]) -> None:
        """Update Z offset and servo balance statistics."""
        if not hasattr(self, 'widget'):
            return

        if 'z_optimization_enabled' in state:
            enabled = state['z_optimization_enabled']
            if enabled != self.enabled:
                self.enable_checkbox.setChecked(enabled)

        if 'z_offset' in state:
            z_offset = state['z_offset']
            self.z_offset_label.setText(f"Z Offset: {z_offset:.2f} mm")

        if 'servo_balance' in state:
            max_angle, min_angle = state['servo_balance']
            self.servo_balance_label.setText(f"Servo Balance: Max={max_angle:.1f}°, Min={min_angle:.1f}°")

            imbalance = max_angle + min_angle
            self.imbalance_label.setText(f"Imbalance: {imbalance:.2f}°")

            if abs(imbalance) < 1.0:
                self.imbalance_label.setStyleSheet(f"color: {self.colors['highlight']};")
            else:
                self.imbalance_label.setStyleSheet(f"color: {self.colors['fg']};")


class PerformanceDataCollectionModule(GUIModule):
    """Performance data collection for PID vs LQR comparison."""

    def __init__(self, parent: QWidget, colors: Dict[str, str], callbacks: Dict[str, Any],
                 controller_type: str = 'PID') -> None:
        """
        Initialize performance data collection module.

        Args:
            parent: Parent QWidget
            colors: Color scheme dict
            callbacks: Callback functions dict
            controller_type: Current controller type ('PID' or 'LQR')
        """
        super().__init__(parent, colors, callbacks)
        self.controller_type = controller_type
        self.recording = False

    def create(self) -> QWidget:
        """Create performance data collection panel with recording controls."""
        group = QGroupBox("Performance Data Collection")
        layout = QVBoxLayout()

        # Recording control
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Recording")
        self.start_btn.clicked.connect(self._on_start_recording)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Recording")
        self.stop_btn.clicked.connect(self._on_stop_recording)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)

        layout.addLayout(control_layout)

        # Status display
        status_layout = QVBoxLayout()

        self.status_label = QLabel("Status: Not recording")
        self.status_label.setStyleSheet(f"color: {self.colors['fg']}; font-weight: bold;")
        status_layout.addWidget(self.status_label)

        self.stats_label = QLabel("Duration: 0.0s | Samples: 0 | Rate: 0.0 Hz")
        self.stats_label.setStyleSheet(f"color: {self.colors['fg']};")
        status_layout.addWidget(self.stats_label)

        self.filename_label = QLabel("File: None")
        self.filename_label.setStyleSheet(f"color: {self.colors['fg']};")
        status_layout.addWidget(self.filename_label)

        layout.addLayout(status_layout)

        # Instructions
        instructions_label = QLabel(
            "Instructions:\n"
            "1. Enable controller and let system stabilize\n"
            "2. Click 'Start Recording' to begin data collection\n"
            "3. Disturb the ball or change target position\n"
            "4. Wait for ball to settle (~10-20 seconds)\n"
            "5. Click 'Stop Recording' to save data\n\n"
            "Repeat for both PID and LQR controllers, then use:\n"
            "  python scripts/plot_comp.py \\\n"
            "    --pid data/performance/performance_PID_*.csv \\\n"
            "    --lqr data/performance/performance_LQR_*.csv"
        )
        instructions_label.setStyleSheet(f"color: {self.colors['border']}; font-size: 9pt;")
        instructions_label.setWordWrap(True)
        layout.addWidget(instructions_label)

        group.setLayout(layout)
        self.widget = group
        return self.widget

    def _on_start_recording(self) -> None:
        """Handle start recording button click."""
        callback = self.callbacks.get('start_recording')
        if callback:
            callback()

    def _on_stop_recording(self) -> None:
        """Handle stop recording button click."""
        callback = self.callbacks.get('stop_recording')
        if callback:
            callback()

    def update(self, state: Dict[str, Any]) -> None:
        """Update module display with current state."""
        recording = state.get('recording', False)

        if recording:
            self.status_label.setText("Status: Recording")
            self.status_label.setStyleSheet(f"color: {self.colors['highlight']}; font-weight: bold;")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

            duration = state.get('recording_duration', 0.0)
            samples = state.get('recording_samples', 0)
            rate = state.get('recording_rate', 0.0)
            self.stats_label.setText(f"Duration: {duration:.1f}s | Samples: {samples} | Rate: {rate:.1f} Hz")

            filename = state.get('recording_filename', 'None')
            self.filename_label.setText(f"File: {filename}")
        else:
            self.status_label.setText("Status: Not recording")
            self.status_label.setStyleSheet(f"color: {self.colors['fg']}; font-weight: bold;")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.stats_label.setText("Duration: 0.0s | Samples: 0 | Rate: 0.0 Hz")
            self.filename_label.setText("File: None")



class VideoRecordModule(GUIModule):
    """Video recording toggle for stereo camera."""

    def __init__(self, parent=None, colors=None, callbacks=None):
        super().__init__(parent, colors, callbacks)
        self.is_recording = False

    def create(self) -> QWidget:
        """Create video recording toggle panel."""
        group = QGroupBox("Video Recording")
        layout = QVBoxLayout()

        # Toggle button
        self.record_btn = QPushButton("Start Video Recording")
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(self._on_toggle)
        self.record_btn.setStyleSheet("""
            QPushButton { padding: 8px; font-weight: bold; }
            QPushButton:checked { background-color: #cc4444; color: white; }
        """)
        layout.addWidget(self.record_btn)

        # Status label
        self.status_label = QLabel("Status: Not recording")
        self.status_label.setStyleSheet(f"color: {self.colors['fg']};")
        layout.addWidget(self.status_label)

        # Filename label
        self.filename_label = QLabel("")
        self.filename_label.setStyleSheet(f"color: {self.colors['fg']}; font-size: 10px;")
        layout.addWidget(self.filename_label)

        group.setLayout(layout)
        self.widget = group
        return group

    def _on_toggle(self) -> None:
        """Handle record button toggle."""
        if self.record_btn.isChecked():
            callback = self.callbacks.get('start_video_recording')
            if callback:
                filepath = callback()
                self.is_recording = True
                self.record_btn.setText("Stop Video Recording")
                self.status_label.setText("Status: Recording...")
                self.filename_label.setText(f"File: {filepath}")
        else:
            callback = self.callbacks.get('stop_video_recording')
            if callback:
                callback()
            self.is_recording = False
            self.record_btn.setText("Start Video Recording")
            self.status_label.setText("Status: Stopped")

    def update(self, state: dict) -> None:
        """Update display from state."""
        pass


class Plot3DModule(GUIModule):
    """3D visualization of ball and platform using PyQtGraph OpenGL."""

    # Constants
    BALL_RADIUS_MM = 20.0  # Ping pong ball radius
    PLATFORM_DIAMETER_MM = 300.0  # Platform disc diameter
    PLATFORM_THICKNESS_MM = 10.0  # Platform thickness
    TRAIL_LENGTH = 50  # Number of trail points
    TARGET_MARKER_RADIUS_MM = 5.0  # Target marker size
    TRAJECTORY_LINE_WIDTH = 2.0  # Trajectory path line width

    def __init__(self, parent: QWidget, colors: Dict[str, str],
                 callbacks: Optional[Dict[str, Any]] = None,
                 platform_home_z: float = 180.0) -> None:
        """
        Initialize 3D visualization module.

        Args:
            parent: Parent QWidget
            colors: Color scheme dict
            callbacks: Callback functions dict
            platform_home_z: Platform home height in mm
        """
        super().__init__(parent, colors, callbacks)
        self.platform_home_z = platform_home_z
        self.trail_positions: list = []
        self.view: Optional[gl.GLViewWidget] = None
        self.current_pattern_type: Optional[str] = None
        self.current_pattern_params: Dict[str, float] = {}

    def create(self) -> Optional[QWidget]:
        """Create and return the 3D view widget."""
        if not HAS_OPENGL:
            # Return placeholder if OpenGL not available
            placeholder = QLabel("3D View requires PyOpenGL.\nInstall with: pip install PyOpenGL")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.widget = placeholder
            return self.widget

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(self.colors.get('bg', '#1e1e1e'))

        # Setup camera - elevated view looking down at platform
        self.view.setCameraPosition(distance=600, elevation=45, azimuth=-135)
        self.view.pan(0, 0, self.platform_home_z)  # Center view on platform height

        # Add grid at Z=0
        self._setup_grid()

        # Add platform disc
        self._setup_platform()

        # Add ball sphere
        self._setup_ball()

        # Add trail scatter
        self._setup_trail()

        # Add trajectory path and target marker
        self._setup_trajectory()
        self._setup_target_marker()

        # Add coordinate axes
        self._setup_axes()

        self.widget = self.view
        return self.widget

    def _setup_axes(self) -> None:
        """Add X and Y coordinate axis arrows at edges with mm labels."""
        edge_offset = -180.0  # Position at edge of grid (grid is 400x400, so ±200)
        axis_length = 60.0  # Length of axis arrows in mm
        arrow_head_size = 8.0  # Size of arrow head
        z = self.platform_home_z + 2  # Slightly above platform

        # X axis (red) at bottom edge, pointing in +X direction
        x_start = edge_offset
        x_axis_points = np.array([
            [x_start, edge_offset, z],
            [x_start + axis_length, edge_offset, z],
        ])
        x_arrow_head = np.array([
            [x_start + axis_length, edge_offset, z],
            [x_start + axis_length - arrow_head_size, edge_offset + arrow_head_size * 0.4, z],
            [x_start + axis_length, edge_offset, z],
            [x_start + axis_length - arrow_head_size, edge_offset - arrow_head_size * 0.4, z],
        ])
        self.x_axis = gl.GLLinePlotItem(
            pos=x_axis_points,
            color=(1.0, 0.3, 0.3, 1.0),  # Red
            width=2.5,
            antialias=True
        )
        self.x_arrow = gl.GLLinePlotItem(
            pos=x_arrow_head,
            color=(1.0, 0.3, 0.3, 1.0),  # Red
            width=2.5,
            antialias=True
        )
        self.view.addItem(self.x_axis)
        self.view.addItem(self.x_arrow)

        # Y axis (blue) at left edge, pointing in +Y direction
        y_axis_points = np.array([
            [edge_offset, edge_offset, z],
            [edge_offset, edge_offset + axis_length, z],
        ])
        y_arrow_head = np.array([
            [edge_offset, edge_offset + axis_length, z],
            [edge_offset + arrow_head_size * 0.4, edge_offset + axis_length - arrow_head_size, z],
            [edge_offset, edge_offset + axis_length, z],
            [edge_offset - arrow_head_size * 0.4, edge_offset + axis_length - arrow_head_size, z],
        ])
        self.y_axis = gl.GLLinePlotItem(
            pos=y_axis_points,
            color=(0.3, 0.5, 1.0, 1.0),  # Blue
            width=2.5,
            antialias=True
        )
        self.y_arrow = gl.GLLinePlotItem(
            pos=y_arrow_head,
            color=(0.3, 0.5, 1.0, 1.0),  # Blue
            width=2.5,
            antialias=True
        )
        self.view.addItem(self.y_axis)
        self.view.addItem(self.y_arrow)

        # Add axis labels with mm units (GLTextItem if available)
        try:
            self.x_label = gl.GLTextItem(
                pos=(x_start + axis_length + 10, edge_offset, z),
                text='X (mm)',
                color=(1.0, 0.3, 0.3, 1.0)
            )
            self.y_label = gl.GLTextItem(
                pos=(edge_offset, edge_offset + axis_length + 10, z),
                text='Y (mm)',
                color=(0.3, 0.5, 1.0, 1.0)
            )
            self.view.addItem(self.x_label)
            self.view.addItem(self.y_label)
        except (AttributeError, TypeError):
            # GLTextItem may not be available in all pyqtgraph versions
            pass

    def _setup_grid(self) -> None:
        """Add reference grid at platform level."""
        grid = gl.GLGridItem()
        grid.setSize(400, 400)  # 400mm x 400mm
        grid.setSpacing(50, 50)  # 50mm grid spacing
        grid.translate(0, 0, self.platform_home_z)
        self.view.addItem(grid)

    def _setup_platform(self) -> None:
        """Create platform disc mesh."""
        # Create cylinder vertices for disc
        n_segments = 32
        radius = self.PLATFORM_DIAMETER_MM / 2
        thickness = self.PLATFORM_THICKNESS_MM

        # Generate cylinder mesh data
        theta = np.linspace(0, 2 * np.pi, n_segments + 1)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        # Top circle vertices
        top_verts = np.column_stack([x, y, np.full_like(x, 0)])
        # Bottom circle vertices
        bottom_verts = np.column_stack([x, y, np.full_like(x, -thickness)])

        # Create faces for cylinder
        faces = []

        # Top face (fan from center)
        top_center_idx = len(top_verts) + len(bottom_verts)
        bottom_center_idx = top_center_idx + 1

        # All vertices: top circle, bottom circle, top center, bottom center
        all_verts = np.vstack([
            top_verts,
            bottom_verts,
            [[0, 0, 0]],  # top center
            [[0, 0, -thickness]]  # bottom center
        ])

        # Top cap faces
        for i in range(n_segments):
            faces.append([top_center_idx, i, (i + 1) % (n_segments + 1)])

        # Bottom cap faces
        for i in range(n_segments):
            faces.append([bottom_center_idx, n_segments + 1 + (i + 1) % (n_segments + 1), n_segments + 1 + i])

        # Side faces
        for i in range(n_segments):
            i1 = i
            i2 = (i + 1) % (n_segments + 1)
            i3 = n_segments + 1 + i
            i4 = n_segments + 1 + (i + 1) % (n_segments + 1)
            faces.append([i1, i2, i4])
            faces.append([i1, i4, i3])

        faces = np.array(faces)

        # Platform color (dark gray)
        colors_arr = np.ones((len(faces), 4)) * [0.3, 0.3, 0.35, 0.9]

        self.platform_mesh = gl.GLMeshItem(
            vertexes=all_verts,
            faces=faces,
            faceColors=colors_arr,
            smooth=False,
            drawEdges=True,
            edgeColor=(0.5, 0.5, 0.5, 1)
        )
        self.platform_mesh.translate(0, 0, self.platform_home_z)
        self.view.addItem(self.platform_mesh)

        # Store initial transform
        self.platform_base_transform = self.platform_mesh.transform()

    def _setup_ball(self) -> None:
        """Create ball sphere mesh."""
        # Create sphere mesh data
        md = gl.MeshData.sphere(rows=16, cols=16, radius=self.BALL_RADIUS_MM)

        # Soft red ball color
        self.ball_mesh = gl.GLMeshItem(
            meshdata=md,
            smooth=True,
            color=(0.9, 0.4, 0.4, 1.0),  # Soft red
            shader='shaded'
        )
        # Initial position at platform center
        self.ball_mesh.translate(0, 0, self.platform_home_z + self.BALL_RADIUS_MM)
        self.view.addItem(self.ball_mesh)

    def _setup_trail(self) -> None:
        """Create trail line and scatter plot for ball trajectory."""
        # Trail line connecting ball positions
        self.trail_line = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)),
            color=(1.0, 0.6, 0.2, 0.9),  # Orange
            width=2.5,
            antialias=True
        )
        self.view.addItem(self.trail_line)

        # Trail scatter points (endpoint markers)
        self.trail_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)),
            color=(1, 0.5, 0, 0.5),
            size=5,
            pxMode=True
        )
        self.view.addItem(self.trail_scatter)

    def _setup_trajectory(self) -> None:
        """Create trajectory path line on platform surface."""
        # Initialize with empty line (will be updated when pattern is set)
        self.trajectory_line = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)),
            color=(0.0, 0.8, 0.2, 0.8),  # Green
            width=self.TRAJECTORY_LINE_WIDTH,
            antialias=True
        )
        self.view.addItem(self.trajectory_line)

    def _setup_target_marker(self) -> None:
        """Create target position marker."""
        # Small sphere for current target position
        md = gl.MeshData.sphere(rows=8, cols=8, radius=self.TARGET_MARKER_RADIUS_MM)
        self.target_marker = gl.GLMeshItem(
            meshdata=md,
            smooth=True,
            color=(0.0, 1.0, 0.0, 1.0),  # Bright green
            shader='shaded'
        )
        # Start at platform center
        self.target_marker.translate(0, 0, self.platform_home_z)
        self.view.addItem(self.target_marker)

    def _generate_pattern_points(self, pattern_type: str, params: Dict[str, float],
                                  n_points: int = 64) -> np.ndarray:
        """Generate points for trajectory pattern visualization."""
        z = self.platform_home_z  # Pattern is on platform surface

        if pattern_type == 'Static':
            # Single point at center
            return np.array([[0, 0, z]])

        elif pattern_type == 'Circle':
            radius = params.get('radius', 50.0)
            theta = np.linspace(0, 2 * np.pi, n_points)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points = np.column_stack([x, y, np.full(n_points, z)])
            return points

        elif pattern_type == 'Figure-8':
            width = params.get('width', 60.0)
            height = params.get('height', 40.0)
            t = np.linspace(0, 2 * np.pi, n_points)
            x = width * np.sin(t)
            y = height * np.sin(2 * t)
            points = np.column_stack([x, y, np.full(n_points, z)])
            return points

        elif pattern_type == 'Star':
            radius = params.get('radius', 60.0)
            # Star pattern: 5 points on a circle, visited in order 0→2→4→1→3
            # This creates the star shape when connecting the points
            num_vertices = 5
            visit_order = [0, 2, 4, 1, 3, 0]  # Close the star

            # Compute vertex positions (evenly spaced, starting at top)
            vertex_positions = []
            for i in range(num_vertices):
                angle = np.pi / 2 - i * (2 * np.pi / num_vertices)  # Start at top, go CCW
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                vertex_positions.append([x, y, z])

            # Build path in visit order
            star_points = [vertex_positions[i] for i in visit_order]
            points = np.array(star_points)
            return points

        else:
            # Default: single point
            return np.array([[0, 0, z]])

    def _update_trajectory(self, pattern_type: str, params: Dict[str, float],
                           rx_deg: float, ry_deg: float) -> None:
        """Update trajectory path with current pattern and platform rotation."""
        # Check if pattern changed
        if pattern_type != self.current_pattern_type or params != self.current_pattern_params:
            self.current_pattern_type = pattern_type
            self.current_pattern_params = params.copy()

        # Generate pattern points
        points = self._generate_pattern_points(pattern_type, params)

        if len(points) < 2:
            # Hide line for static pattern
            self.trajectory_line.setData(pos=np.zeros((2, 3)))
            return

        # Apply platform rotation to points
        rotated_points = self._rotate_points(points, rx_deg, ry_deg)
        self.trajectory_line.setData(pos=rotated_points)

    def _update_target_marker(self, target_x: float, target_y: float,
                               rx_deg: float, ry_deg: float) -> None:
        """Update target marker position with platform rotation."""
        # Target is on platform surface
        z = self.platform_home_z

        # Create point and apply rotation
        point = np.array([[target_x, target_y, z]])
        rotated = self._rotate_points(point, rx_deg, ry_deg)

        self.target_marker.resetTransform()
        self.target_marker.translate(rotated[0, 0], rotated[0, 1], rotated[0, 2])

    def _rotate_points(self, points: np.ndarray, rx_deg: float, ry_deg: float) -> np.ndarray:
        """Rotate points around platform center based on tilt angles."""
        if rx_deg == 0 and ry_deg == 0:
            return points

        # Convert to radians
        rx = np.radians(rx_deg)
        ry = np.radians(ry_deg)

        # Rotation matrices
        cos_rx, sin_rx = np.cos(rx), np.sin(rx)
        cos_ry, sin_ry = np.cos(ry), np.sin(ry)

        # Translate to origin (relative to platform center at home_z)
        centered = points.copy()
        centered[:, 2] -= self.platform_home_z

        # Apply rotations (Rx then Ry)
        rotated = np.zeros_like(centered)
        for i, p in enumerate(centered):
            x, y, z = p
            # Rotate around X axis
            y1 = y * cos_rx - z * sin_rx
            z1 = y * sin_rx + z * cos_rx
            # Rotate around Y axis
            x2 = x * cos_ry + z1 * sin_ry
            z2 = -x * sin_ry + z1 * cos_ry
            rotated[i] = [x2, y1, z2]

        # Translate back
        rotated[:, 2] += self.platform_home_z

        return rotated

    def update(self, state: Dict[str, Any]) -> None:
        """
        Update 3D visualization with current state.

        Expected state keys:
            ball_x, ball_y, ball_z: Ball position in mm (platform frame)
            platform_rx, platform_ry: Platform tilt angles in degrees
            target_x, target_y: Current target position in mm
            pattern_type: Trajectory pattern type (Circle, Figure-8, Star, Static)
            pattern_params: Dict with pattern parameters (radius, width, height, etc.)
        """
        if not HAS_OPENGL or self.view is None:
            return

        ball_x = state.get('ball_x', 0.0)
        ball_y = state.get('ball_y', 0.0)
        ball_z = state.get('ball_z', self.platform_home_z + self.BALL_RADIUS_MM)
        platform_x = state.get('platform_x', 0.0)  # mm
        platform_y = state.get('platform_y', 0.0)  # mm
        platform_z = state.get('platform_z', self.platform_home_z)  # mm
        platform_rx = state.get('platform_rx', 0.0)  # degrees
        platform_ry = state.get('platform_ry', 0.0)  # degrees
        ball_detected = state.get('ball_detected', True)

        # Trajectory info
        target_x = state.get('target_x', 0.0)
        target_y = state.get('target_y', 0.0)
        pattern_type = state.get('pattern_type', 'Static')
        pattern_params = state.get('pattern_params', {})

        # Update ball position (ball_z is already the ball center position)
        if ball_detected:
            self.ball_mesh.resetTransform()
            self.ball_mesh.translate(ball_x, ball_y, ball_z)

            # Update trail
            self._update_trail(ball_x, ball_y, ball_z)

        # Update platform pose (position and rotation)
        self._update_platform_pose(platform_x, platform_y, platform_z, platform_rx, platform_ry)

        # Update trajectory path (rotates with platform)
        self._update_trajectory(pattern_type, pattern_params, platform_rx, platform_ry)

        # Update target marker (rotates with platform)
        self._update_target_marker(target_x, target_y, platform_rx, platform_ry)

    def _update_trail(self, x: float, y: float, z: float) -> None:
        """Update ball trail with new position."""
        self.trail_positions.append([x, y, z])

        # Keep only last N positions
        if len(self.trail_positions) > self.TRAIL_LENGTH:
            self.trail_positions = self.trail_positions[-self.TRAIL_LENGTH:]

        if len(self.trail_positions) > 1:
            pos = np.array(self.trail_positions)

            # Update trail line
            self.trail_line.setData(pos=pos)

            # Create fading colors for scatter (newer = brighter)
            n = len(pos)
            alphas = np.linspace(0.1, 0.8, n)
            colors = np.zeros((n, 4))
            colors[:, 0] = 1.0  # Red
            colors[:, 1] = 0.5  # Green (orange)
            colors[:, 2] = 0.0  # Blue
            colors[:, 3] = alphas  # Alpha (fading)

            # Sizes decrease with age
            sizes = np.linspace(2, 8, n)

            self.trail_scatter.setData(pos=pos, color=colors, size=sizes)

    def _update_platform_pose(self, x: float, y: float, z: float,
                               rx_deg: float, ry_deg: float) -> None:
        """Update platform position and orientation."""
        # Reset to base transform
        self.platform_mesh.resetTransform()

        # Apply rotations (rx = roll around X, ry = pitch around Y)
        self.platform_mesh.rotate(rx_deg, 1, 0, 0)  # Rotate around X axis
        self.platform_mesh.rotate(ry_deg, 0, 1, 0)  # Rotate around Y axis

        # Translate to position (x, y relative to center, z is absolute height)
        self.platform_mesh.translate(x, y, z)

    def get_widget(self) -> Optional[QWidget]:
        """Return the 3D view widget."""
        return self.view if self.view else self.widget

    def clear_trail(self) -> None:
        """Clear the ball trail."""
        self.trail_positions = []
        if HAS_OPENGL:
            if hasattr(self, 'trail_line'):
                self.trail_line.setData(pos=np.zeros((2, 3)))
            if hasattr(self, 'trail_scatter'):
                self.trail_scatter.setData(pos=np.zeros((1, 3)))
