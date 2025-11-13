#!/usr/bin/env python3
"""
Stewart Platform Simulator - Modular Base Class

Reusable simulator with pluggable controller support and modular GUI.
PyQt6 + PyQtGraph implementation.
"""

from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QGroupBox, QGridLayout, QLabel, QSlider, QComboBox, QApplication
from PyQt6.QtCore import QTimer, Qt, QEvent
from PyQt6.QtGui import QFont
import pyqtgraph as pg
import numpy as np
import time
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List

from core.core import FirstOrderServo, StewartPlatformIK, SimpleBallPhysics2D, PatternFactory, Pixy2Camera
from core.control_core import clip_tilt_vector
from core.utils import (
    MAX_TILT_ANGLE_DEG, MAX_CONTROLLER_OUTPUT_DEG, PLATFORM_RADIUS_MM, PLATFORM_HALF_SIZE_MM, PLATFORM_VERSION,
    SimulationConfig, IKZOptimizationConfig, Pixy2CameraConfig,
    StewartPlatformConfig, ColorScheme, BallPhysicsConfig, VisualizationConfig,
    PIDConfig, LQRConfig, ManualPoseControlConfig, BallControlConfig, GUIConfig,
    format_time, format_error_context,
    GUI_FONT_SANS, GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL,
    GUI_VALUE_LABEL_WIDTH_SMALL
)
from gui.gui_builder import GUIBuilder
from gui import gui_modules as gm

# UI color constants for theme
BUTTON_PRESSED_COLOR = '#005a9e'
CHECKBOX_HOVER_COLOR = '#0088dd'
BALL_COLOR = '#ff4444'

# Plot visualization constants
BALL_MARKER_SIZE = 20
TARGET_MARKER_SIZE = 15


class ControllerConfig(ABC):
    """Abstract base for controller-specific configuration."""

    @abstractmethod
    def get_controller_name(self) -> str:
        """Return display name for controller."""
        pass

    @abstractmethod
    def create_controller(self, **kwargs) -> Any:
        """Create and return controller instance."""
        pass

    @abstractmethod
    def get_scalar_values(self) -> List[float]:
        """Return list of scalar multipliers for parameters."""
        pass

    def get_scaled_param(self, param_name: str, sliders: Dict[str, QSlider],
                        scalar_vars: Dict[str, int]) -> float:
        """Extract and scale a parameter value from widgets."""
        raw = sliders[param_name].value() / 100.0  # QSlider uses integers
        scalar = self.get_scalar_values()[scalar_vars[param_name]]
        return raw * scalar

    def create_parameter_slider(self, parent_layout: QVBoxLayout, param_name: str, label: str,
                                default: float, sliders: Dict[str, QSlider],
                                value_labels: Dict[str, QLabel], scalar_vars: Dict[str, int],
                                on_change_callback: Any) -> None:
        """Create standard parameter slider with scalar multiplier."""
        grid = QGridLayout()

        label_widget = QLabel(label)
        label_widget.setFont(QFont(GUI_FONT_SANS, GUI_FONT_SIZE_NORMAL))
        grid.addWidget(label_widget, 0, 0, Qt.AlignmentFlag.AlignLeft)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(1000)
        slider.setValue(int(default * 100))
        grid.addWidget(slider, 0, 1)
        sliders[param_name] = slider

        value_label = QLabel(f"{default:.2f}")
        value_label.setFont(QFont(GUI_FONT_MONOSPACE, GUI_FONT_SIZE_NORMAL))
        value_label.setMinimumWidth(GUI_VALUE_LABEL_WIDTH_SMALL)
        grid.addWidget(value_label, 0, 2)
        value_labels[param_name] = value_label

        scalar_combo = QComboBox()
        scalar_combo.addItems([f'×{scalar:.7g}' for scalar in self.get_scalar_values()])
        scalar_combo.setCurrentIndex(getattr(self, 'default_scalar_idx', 4))
        scalar_combo.setMinimumWidth(120)  # Scalar selector width
        grid.addWidget(scalar_combo, 0, 3)

        scalar_vars[param_name] = scalar_combo.currentIndex()

        def on_slider_change(val):
            value = val / 100.0
            value_labels[param_name].setText(f"{value:.2f}")
            on_change_callback()

        def on_scalar_change(idx):
            scalar_vars[param_name] = idx
            on_change_callback()

        slider.valueChanged.connect(on_slider_change)
        scalar_combo.currentIndexChanged.connect(on_scalar_change)

        grid.setColumnStretch(1, 1)

        # Add to parent layout (now receives layout directly)
        parent_layout.addLayout(grid)


class BaseStewartSimulator(QMainWindow):
    """
    Base Stewart Platform Simulator with modular GUI.

    Subclasses define layout via get_layout_config().
    """

    def __init__(self, app: QApplication, controller_config: ControllerConfig) -> None:
        """
        Initialize Stewart Platform Simulator.

        Args:
            app: QApplication instance
            controller_config: Controller configuration instance
        """
        super().__init__()
        self.app: QApplication = app
        self.controller_config: ControllerConfig = controller_config

        controller_name = controller_config.get_controller_name()
        self.setWindowTitle(f"Stewart Platform - {controller_name} Ball Balancing Control")
        self.resize(1400, 900)

        self.colors = ColorScheme.as_dict()

        self.setup_dark_theme()

        self.platform_params = StewartPlatformConfig.as_dict()
        self.ik = StewartPlatformIK(**self.platform_params)
        self.servos = [
            FirstOrderServo(
                K=1.0,
                tau=SimulationConfig.DEFAULT_SERVO_TAU,
                delay=SimulationConfig.DEFAULT_SERVO_DELAY,
                max_velocity=SimulationConfig.DEFAULT_SERVO_MAX_VELOCITY
            )
            for _ in range(6)
        ]

        self.ball_physics = SimpleBallPhysics2D(**BallPhysicsConfig.for_physics_sim())

        self.pixy_camera = Pixy2Camera(
            pixel_size_mm=Pixy2CameraConfig.PIXEL_SIZE_MM,
            subpixel_noise_std=Pixy2CameraConfig.SUBPIXEL_NOISE_STD_MM,
            detection_rate=Pixy2CameraConfig.DEFAULT_DETECTION_RATE,
            sample_rate_hz=Pixy2CameraConfig.DEFAULT_SAMPLE_RATE_HZ
        )
        self.camera_enabled = True

        self.controller = None
        self.controller_enabled = False

        self.current_pattern = PatternFactory.create('static', x=0.0, y=0.0)
        self.pattern_type = 'static'
        self.pattern_start_time = 0.0
        self.pattern_params = {}

        ball_start_height = (self.ik.home_height_top_surface / 1000) + self.ball_physics.radius
        self.ball_pos = np.array([[0.0, 0.0, ball_start_height]], dtype=np.float32)
        self.ball_vel = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        self.ball_omega = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        # Ball trail history
        self.max_history = VisualizationConfig.BALL_TRAIL_MAX_HISTORY
        self.ball_history_x = []
        self.ball_history_y = []

        self.simulation_running = False
        self.simulation_time = 0.0
        self.last_update_time = None
        self.update_rate_ms = 10  # 100 Hz target (10ms), more reliable than 2ms

        self.use_top_surface_offset = True
        self.dof_values = {
            'x': 0.0, 'y': 0.0, 'z': self.ik.home_height_top_surface,
            'rx': 0.0, 'ry': 0.0, 'rz': 0.0
        }

        # Manual pose control configuration (from ManualPoseControlConfig)
        self.dof_config = {
            'x': (*ManualPoseControlConfig.X_RANGE_MM, ManualPoseControlConfig.X_RESOLUTION_MM, ManualPoseControlConfig.X_DEFAULT_MM, "X Position (mm)"),
            'y': (*ManualPoseControlConfig.Y_RANGE_MM, ManualPoseControlConfig.Y_RESOLUTION_MM, ManualPoseControlConfig.Y_DEFAULT_MM, "Y Position (mm)"),
            'z': (self.ik.home_height_top_surface + ManualPoseControlConfig.Z_OFFSET_RANGE_MM[0],
                  self.ik.home_height_top_surface + ManualPoseControlConfig.Z_OFFSET_RANGE_MM[1],
                  ManualPoseControlConfig.Z_RESOLUTION_MM, self.ik.home_height_top_surface, "Z Height (mm)"),
            'rx': (*ManualPoseControlConfig.RX_RANGE_DEG, ManualPoseControlConfig.RX_RESOLUTION_DEG, ManualPoseControlConfig.RX_DEFAULT_DEG, "Roll (°)"),
            'ry': (*ManualPoseControlConfig.RY_RANGE_DEG, ManualPoseControlConfig.RY_RESOLUTION_DEG, ManualPoseControlConfig.RY_DEFAULT_DEG, "Pitch (°)"),
            'rz': (*ManualPoseControlConfig.RZ_RANGE_DEG, ManualPoseControlConfig.RZ_RESOLUTION_DEG, ManualPoseControlConfig.RZ_DEFAULT_DEG, "Yaw (°)")
        }

        self.prev_platform_angles = {'rx': 0.0, 'ry': 0.0}
        self.platform_angular_vel = {'rx': 0.0, 'ry': 0.0}
        self.platform_angular_accel = {'rx': 0.0, 'ry': 0.0}

        self.last_cmd_angles = np.zeros(6)
        self.last_fk_translation = np.zeros(3)
        self.last_fk_rotation = np.zeros(3)

        # Z optimization state
        self.z_optimization_enabled = IKZOptimizationConfig.ENABLED
        self.z_offset = 0.0
        self.servo_balance = (0.0, 0.0)

        # Initialize servo balance with home position
        home_translation = np.array([0.0, 0.0, self.ik.home_height_top_surface])
        home_rotation = np.array([0.0, 0.0, 0.0])
        home_angles = self.ik.calculate_servo_angles(home_translation, home_rotation, True)
        if home_angles is not None:
            self.servo_balance = (np.max(home_angles), np.min(home_angles))

        self.update_timer = None
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self.simulation_loop)

        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self._create_controller_param_widgets()
        self._build_modular_gui()
        self._initialize_controller()

    def setup_dark_theme(self) -> None:
        """Configure PyQt6 dark theme using QSS."""
        stylesheet = f"""
            QMainWindow {{
                background-color: {self.colors['bg']};
            }}
            QWidget {{
                background-color: {self.colors['bg']};
                color: {self.colors['fg']};
            }}
            QGroupBox {{
                background-color: {self.colors['panel_bg']};
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: {self.colors['highlight']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
            QLabel {{
                color: {self.colors['fg']};
            }}
            QPushButton {{
                background-color: {self.colors['button_bg']};
                color: {self.colors['button_fg']};
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background-color: {self.colors['highlight']};
            }}
            QPushButton:pressed {{
                background-color: {BUTTON_PRESSED_COLOR};
            }}
            QPushButton:disabled {{
                background-color: {self.colors['widget_bg']};
                color: {self.colors['border']};
            }}
            QSlider::groove:horizontal {{
                background: {self.colors['widget_bg']};
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {self.colors['highlight']};
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }}
            QCheckBox {{
                color: {self.colors['fg']};
                spacing: 8px;
                font-size: 10pt;
            }}
            QCheckBox::indicator {{
                width: 20px;
                height: 20px;
                border: 2px solid {self.colors['border']};
                border-radius: 4px;
                background-color: {self.colors['widget_bg']};
            }}
            QCheckBox::indicator:hover {{
                border-color: {self.colors['highlight']};
                background-color: {self.colors['panel_bg']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {self.colors['highlight']};
                border-color: {self.colors['highlight']};
            }}
            QCheckBox::indicator:checked:hover {{
                background-color: {CHECKBOX_HOVER_COLOR};
                border-color: {CHECKBOX_HOVER_COLOR};
            }}
            QComboBox {{
                background-color: {self.colors['widget_bg']};
                color: {self.colors['fg']};
                border: 1px solid {self.colors['border']};
                padding: 3px 5px;
                border-radius: 3px;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {self.colors['fg']};
                margin-right: 5px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {self.colors['widget_bg']};
                color: {self.colors['fg']};
                selection-background-color: {self.colors['highlight']};
                selection-color: {self.colors['button_fg']};
            }}
            QTextEdit {{
                background-color: {self.colors['widget_bg']};
                color: {self.colors['fg']};
                border: 1px solid {self.colors['border']};
                border-radius: 3px;
            }}
            QScrollBar:vertical {{
                background: {self.colors['widget_bg']};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: {self.colors['border']};
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """
        self.app.setStyleSheet(stylesheet)

    def _create_controller_param_widgets(self) -> None:
        """Create controller parameter widgets."""
        controller_name = self.controller_config.get_controller_name()

        # Check if controller is PID (handles both "PID" and "PID (Hardware)")
        if "PID" in controller_name:
            # Determine if hardware or simulation mode
            is_hardware = "(Hardware)" in controller_name

            # Get defaults from PIDConfig
            if is_hardware:
                gains = PIDConfig.HW_DEFAULT_GAINS
                scalar_indices = PIDConfig.HW_SCALAR_INDICES
            else:
                gains = PIDConfig.SIM_DEFAULT_GAINS
                scalar_indices = PIDConfig.SIM_SCALAR_INDICES

            self.param_definitions = [
                ('kp', 'P (Proportional)', gains['kp'], scalar_indices['kp']),
                ('ki', 'I (Integral)', gains['ki'], scalar_indices['ki']),
                ('kd', 'D (Derivative)', gains['kd'], scalar_indices['kd'])
            ]
        elif "LQR" in controller_name:
            # Get defaults from LQRConfig
            weights = LQRConfig.DEFAULT_WEIGHTS
            scalar_indices = LQRConfig.DEFAULT_SCALAR_INDICES

            self.param_definitions = [
                ('Q_pos', 'Q Position Weight', weights['Q_pos'], scalar_indices['Q_pos']),
                ('Q_vel', 'Q Velocity Weight', weights['Q_vel'], scalar_indices['Q_vel']),
                ('R', 'R Control Weight', weights['R'], scalar_indices['R'])
            ]
        else:
            self.param_definitions = []

        # Update existing dict instead of creating new one (to preserve GUI module references)
        if not hasattr(self, 'controller_widgets'):
            self.controller_widgets = {
                'sliders': {},
                'value_labels': {},
                'scalar_vars': {},
                'update_fn': lambda: None,
                'param_definitions': self.param_definitions
            }
        else:
            # Clear and update existing dict
            self.controller_widgets['sliders'].clear()
            self.controller_widgets['value_labels'].clear()
            self.controller_widgets['scalar_vars'].clear()
            self.controller_widgets['param_definitions'] = self.param_definitions

    def _build_modular_gui(self) -> None:
        """Build GUI using modular system."""
        module_registry = {
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
            'plot_control': gm.PlotControlModule,
            'control_frequency': gm.ControlFrequencyModule,
            'performance_data': gm.PerformanceDataCollectionModule,
        }

        layout_config = self.get_layout_config()
        callbacks = self._create_callbacks()

        self.gui_builder = GUIBuilder(self.central_widget, module_registry)
        self.gui_modules = self.gui_builder.build(layout_config, self.colors, callbacks)

        if 'plot_panel' in self.gui_modules:
            self._create_plot(self.gui_modules['plot_panel'])

    def _create_callbacks(self) -> Dict[str, Any]:
        """Create callback dictionary for modules."""
        return {
            'start': self.start_simulation,
            'stop': self.stop_simulation,
            'reset': self.reset_simulation,
            'controller_enabled_var': self.controller_enabled,
            'toggle_controller': self.on_controller_toggle,
            'param_change': self.on_controller_param_change,
            'pattern_change': self.on_pattern_change,
            'pattern_reset': self.reset_pattern,
            'pattern_param_change': self.on_pattern_param_change,
            'reset_ball': self.reset_ball,
            'push_ball': self.push_ball,
            'toggle_offset': self.on_offset_toggle,
            'slider_change': self.on_slider_change,
            'go_home': self.go_home,
            'camera_enable_change': self.on_camera_enable_change,
            'camera_param_change': self.on_camera_param_change,
            'camera_reset': self.on_camera_reset,
            'z_optimization_toggle': self.on_z_optimization_toggle,
            'log': self.log,
        }

    def on_pattern_param_change(self, param_name: str, value: float) -> None:
        """Update pattern with new parameters."""
        pattern_type = self.pattern_type

        self.pattern_params[param_name] = value

        # If period is changing, preserve the current phase (position in cycle)
        if param_name == 'period' and hasattr(self.current_pattern, 'period'):
            old_period = self.current_pattern.period
            current_pattern_time = self.simulation_time - self.pattern_start_time
            # Calculate current phase (0 to 1)
            current_phase = (current_pattern_time % old_period) / old_period
            # Adjust start time to maintain phase with new period
            new_pattern_time = current_phase * value
            self.pattern_start_time = self.simulation_time - new_pattern_time

        if pattern_type == 'circle':
            radius = self.pattern_params.get('radius', 50.0)
            period = self.pattern_params.get('period', 10.0)
            self.current_pattern = PatternFactory.create('circle',
                                                         radius=radius,
                                                         period=period,
                                                         clockwise=True)

        elif pattern_type == 'figure8':
            width = self.pattern_params.get('width', 60.0)
            height = self.pattern_params.get('height', 40.0)
            period = self.pattern_params.get('period', 12.0)
            self.current_pattern = PatternFactory.create('figure8',
                                                         width=width,
                                                         height=height,
                                                         period=period)

        elif pattern_type == 'star':
            radius = self.pattern_params.get('radius', 60.0)
            period = self.pattern_params.get('period', 15.0)
            self.current_pattern = PatternFactory.create('star',
                                                         radius=radius,
                                                         period=period)

        # Reset pattern internal state but keep timing to continue on path
        self.current_pattern.reset()
        self.update_plot()

    @abstractmethod
    def get_layout_config(self) -> Dict[str, Any]:
        """Return layout configuration for this simulator."""
        raise NotImplementedError

    def _create_plot(self, parent: QWidget) -> None:
        """Create PyQtGraph plot."""
        plot_group = QGroupBox("Ball Position (Top View)")
        plot_layout = QVBoxLayout()

        # Create PyQtGraph plot widget
        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(self.colors['widget_bg'])
        self.plot_widget.setMinimumSize(600, 600)

        plot_layout.addWidget(self.plot_widget)
        plot_group.setLayout(plot_layout)

        # Add to parent layout
        if hasattr(parent, 'layout') and parent.layout() is not None:
            parent.layout().addWidget(plot_group)

        self.setup_plot()

    def setup_plot(self) -> None:
        """Setup PyQtGraph plot with platform boundary, ball, and trajectory markers."""
        plot_item = self.plot_widget.getPlotItem()
        plot_item.setXRange(-180, 180)
        plot_item.setYRange(-180, 180)
        plot_item.setLabel('bottom', 'X (mm)', color=self.colors['fg'])
        plot_item.setLabel('left', 'Y (mm)', color=self.colors['fg'])
        plot_item.setTitle('Ball Position (Top View)', color=self.colors['fg'])
        plot_item.showGrid(x=True, y=True, alpha=0.2)
        plot_item.setAspectLocked(True)

        # Platform boundary (shape depends on platform version)
        pen = pg.mkPen(color=self.colors['fg'], width=2, style=Qt.PenStyle.DashLine)
        if PLATFORM_VERSION == 'V1':
            # V1: Square platform (200x200mm)
            self.platform_boundary = pg.QtWidgets.QGraphicsRectItem(
                -PLATFORM_HALF_SIZE_MM, -PLATFORM_HALF_SIZE_MM,
                PLATFORM_HALF_SIZE_MM * 2, PLATFORM_HALF_SIZE_MM * 2
            )
        else:
            # V2: Circular platform
            self.platform_boundary = pg.QtWidgets.QGraphicsEllipseItem(
                -PLATFORM_RADIUS_MM, -PLATFORM_RADIUS_MM,
                PLATFORM_RADIUS_MM * 2, PLATFORM_RADIUS_MM * 2
            )
        self.platform_boundary.setPen(pen)
        plot_item.addItem(self.platform_boundary)

        # Trajectory line
        self.trajectory_line = plot_item.plot([], [], pen=pg.mkPen(color=self.colors['highlight'],
                                                                     width=1, style=Qt.PenStyle.DashLine))

        # Target marker
        self.target_marker = pg.ScatterPlotItem([0], [0], symbol='x', size=TARGET_MARKER_SIZE,
                                                 pen=pg.mkPen(color=self.colors['success'], width=2))
        plot_item.addItem(self.target_marker)

        # Ball
        self.ball_scatter = pg.ScatterPlotItem([0], [0], symbol='o', size=BALL_MARKER_SIZE,
                                                pen=pg.mkPen(None),
                                                brush=pg.mkBrush(BALL_COLOR))
        plot_item.addItem(self.ball_scatter)

        # Tilt arrow (will be added dynamically)
        self.tilt_arrow = None

    def update_plot(self) -> None:
        """Update plot with current state (ball position, trajectory, tilt arrow)."""
        # Check if plot items still exist (window might be closing)
        if not hasattr(self, 'ball_scatter') or self.ball_scatter is None:
            return

        try:
            ball_x = self.ball_pos[0, 0].item() * 1000
            ball_y = self.ball_pos[0, 1].item() * 1000
            self.ball_scatter.setData([ball_x], [ball_y])
        except RuntimeError:
            # Plot items have been deleted, stop trying to update
            return

        # Update ball trail if it exists (only in simulation mode)
        # In hardware mode, the control thread handles trail updates
        if hasattr(self, 'ball_trail') and hasattr(self, 'ball_history_x') and hasattr(self, 'ball_pos_measured_mm'):
            # Only update trail in simulation mode (check if we're running simulation loop)
            if hasattr(self, 'operation_mode'):
                # full_c.py/min_c.py with operation_mode attribute
                if self.operation_mode == 'sim':
                    measured_x, measured_y = self.ball_pos_measured_mm
                    # Skip if None values
                    if measured_x is not None and measured_y is not None:
                        # Ensure numeric values
                        measured_x = float(measured_x)
                        measured_y = float(measured_y)
                        self.ball_history_x.append(measured_x)
                        self.ball_history_y.append(measured_y)
                        if len(self.ball_history_x) > self.max_history:
                            self.ball_history_x.pop(0)
                            self.ball_history_y.pop(0)
                        # Update the trail plot
                        if len(self.ball_history_x) > 1:
                            self.ball_trail.setData(np.array(self.ball_history_x, dtype=np.float64),
                                                   np.array(self.ball_history_y, dtype=np.float64))
            else:
                # Simple simulator without operation_mode (always simulation)
                measured_x, measured_y = self.ball_pos_measured_mm
                # Skip if None values
                if measured_x is not None and measured_y is not None:
                    # Ensure numeric values
                    measured_x = float(measured_x)
                    measured_y = float(measured_y)
                    self.ball_history_x.append(measured_x)
                    self.ball_history_y.append(measured_y)
                    if len(self.ball_history_x) > self.max_history:
                        self.ball_history_x.pop(0)
                        self.ball_history_y.pop(0)
                    # Update the trail plot
                    if len(self.ball_history_x) > 1:
                        self.ball_trail.setData(np.array(self.ball_history_x, dtype=np.float64),
                                               np.array(self.ball_history_y, dtype=np.float64))

        try:
            if self.pattern_type != 'static':
                # Use actual period from pattern object instead of hardcoded defaults
                period = getattr(self.current_pattern, 'period', 10.0)

                # Period-adaptive sampling: sample every 0.05 seconds for smooth curves
                # More samples for longer periods, ensuring smooth visualization
                num_samples = max(100, int(period / 0.05))
                t_samples = np.linspace(0, period, num_samples, endpoint=True)
                path_x, path_y = [], []
                for time_sample in t_samples:
                    x, y = self.current_pattern.get_position(time_sample)
                    path_x.append(x)
                    path_y.append(y)

                self.trajectory_line.setData(path_x, path_y)

                pattern_time = self.simulation_time - self.pattern_start_time
                target_x, target_y = self.current_pattern.get_position(pattern_time)
                self.target_marker.setData([target_x], [target_y])
            else:
                self.trajectory_line.setData([], [])
                self.target_marker.setData([0], [0])

            # Update tilt arrow
            if self.tilt_arrow is not None:
                self.plot_widget.getPlotItem().removeItem(self.tilt_arrow)
                self.tilt_arrow = None

            rx = self.dof_values['rx']
            ry = self.dof_values['ry']

            if abs(rx) > 0.5 or abs(ry) > 0.5:
                dx = -np.sin(np.radians(ry))
                dy = -np.sin(np.radians(rx))
                magnitude = np.sqrt(dx ** 2 + dy ** 2)

                if magnitude > 0:
                    dx = (dx / magnitude) * 30
                    dy = (dy / magnitude) * 30
                    color = self.colors['success'] if self.controller_enabled else self.colors['highlight']

                    self.tilt_arrow = pg.ArrowItem(angle=np.degrees(np.arctan2(dy, dx)),
                                                    tipAngle=30, headLen=15, tailLen=25,
                                                    pen=pg.mkPen(color), brush=pg.mkBrush(color))
                    self.tilt_arrow.setPos(0, 0)
                    self.plot_widget.getPlotItem().addItem(self.tilt_arrow)
        except RuntimeError:
            # Plot items deleted during update, skip
            pass

    def log(self, message: str) -> None:
        """Add message to debug log."""
        if 'debug_log' in self.gui_modules:
            self.gui_modules['debug_log'].log(message, self.simulation_time)

    def update_gui_modules(self) -> None:
        """Update all GUI modules with current state."""
        ball_x_mm = self.ball_pos[0, 0].item() * 1000
        ball_y_mm = self.ball_pos[0, 1].item() * 1000
        vel_x_mm = self.ball_vel[0, 0].item() * 1000
        vel_y_mm = self.ball_vel[0, 1].item() * 1000

        state = {
            'simulation_time': self.simulation_time,
            'simulation_running': self.simulation_running,
            'controller_enabled': self.controller_enabled,
            'ball_pos': (ball_x_mm, ball_y_mm),
            'ball_vel': (vel_x_mm, vel_y_mm),
            'dof_values': self.dof_values,
            'cmd_angles': self.last_cmd_angles,
            'actual_angles': [servo.get_angle() for servo in self.servos],
            'fk_translation': self.last_fk_translation,
            'fk_rotation': self.last_fk_rotation,
            'z_optimization_enabled': self.z_optimization_enabled,
            'z_offset': self.z_offset,
            'servo_balance': self.servo_balance,
        }

        if self.controller_enabled:
            rx = self.dof_values['rx']
            ry = self.dof_values['ry']
            magnitude = np.sqrt(rx ** 2 + ry ** 2)
            magnitude_percent = (magnitude / MAX_TILT_ANGLE_DEG) * 100

            state['controller_output'] = (rx, ry)
            state['controller_magnitude'] = (magnitude, magnitude_percent)

            pattern_time = self.simulation_time - self.pattern_start_time
            target_x, target_y = self.current_pattern.get_position(pattern_time)
            error_x = ball_x_mm - target_x
            error_y = ball_y_mm - target_y
            state['controller_error'] = (error_x, error_y)

        rx = self.dof_values['rx']
        ry = self.dof_values['ry']
        _, _, magnitude = clip_tilt_vector(rx, ry, MAX_TILT_ANGLE_DEG)
        state['tilt_magnitude'] = magnitude

        pattern_configs = {
            'static': "Tracking: Center (0, 0)",
            'circle': "Tracking: Circle (r=50mm, T=10s)",
            'figure8': "Tracking: Figure-8 (60×40mm, T=12s)",
            'star': "Tracking: 5-Point Star (r=60mm, T=15s)"
        }
        state['pattern_info'] = pattern_configs.get(self.pattern_type, "")

        self.gui_builder.update_modules(state)

    def on_controller_toggle(self) -> None:
        """Handle controller enable/disable."""
        # Check if controller exists
        if self.controller is None:
            self.log("Controller not initialized - cannot enable")
            return

        # Toggle the state
        self.controller_enabled = not self.controller_enabled
        enabled = self.controller_enabled

        if enabled:
            self.controller.reset()
            self.reset_pattern()

            controller_name = self.controller_config.get_controller_name()
            self.log(f"{controller_name} control ENABLED")

            if 'manual_pose' in self.gui_modules:
                manual_pose = self.gui_modules['manual_pose']
                manual_pose.sliders['rx'].setEnabled(False)
                manual_pose.sliders['ry'].setEnabled(False)

            self.dof_values['rx'] = 0.0
            self.dof_values['ry'] = 0.0

            self.calculate_ik()
        else:
            controller_name = self.controller_config.get_controller_name()
            self.log(f"{controller_name} control DISABLED")

            if 'manual_pose' in self.gui_modules:
                manual_pose = self.gui_modules['manual_pose']
                manual_pose.sliders['rx'].setEnabled(True)
                manual_pose.sliders['ry'].setEnabled(True)

    def on_pattern_change(self, pattern_type: Optional[str] = None) -> None:
        """Handle pattern selection change."""
        if pattern_type is not None:
            self.pattern_type = pattern_type
        else:
            pattern_type = self.pattern_type

        self.pattern_params.clear()

        pattern_configs = {
            'static': ('static', {'x': 0.0, 'y': 0.0}),
            'circle': ('circle', {'radius': 50.0, 'period': 10.0, 'clockwise': True}),
            'figure8': ('figure8', {'width': 60.0, 'height': 40.0, 'period': 12.0}),
            'star': ('star', {'radius': 60.0, 'period': 15.0})
        }

        if pattern_type in pattern_configs:
            pattern_name, params = pattern_configs[pattern_type]

            for key, value in params.items():
                if key != 'clockwise':
                    self.pattern_params[key] = value

            self.current_pattern = PatternFactory.create(pattern_name, **params)
            self.reset_pattern()
            self.update_plot()
            self.log(f"Pattern changed to: {pattern_type}")

    def reset_pattern(self) -> None:
        """Reset pattern timing and controller state."""
        self.pattern_start_time = self.simulation_time
        self.current_pattern.reset()
        self.log(f"Pattern reset at t={format_time(self.simulation_time)}")

        if self.controller_enabled and self.controller is not None:
            self.controller.reset()

    def reset_ball(self) -> None:
        """Reset ball to center position with zero velocity."""
        home_z = self.ik.home_height_top_surface if self.use_top_surface_offset else self.ik.home_height
        ball_start_height = (home_z / 1000) + self.ball_physics.radius

        self.ball_pos = np.array([[0.0, 0.0, ball_start_height]], dtype=np.float32)
        self.ball_vel = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        self.ball_omega = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        if self.controller_enabled and self.controller is not None:
            self.controller.reset()

        self.update_plot()
        self.log("Ball reset to center")

    def push_ball(self) -> None:
        """Apply random velocity to ball (magnitude from BallControlConfig)."""
        push_vel = BallControlConfig.PUSH_VELOCITY_MS
        vx = np.random.uniform(-push_vel, push_vel)
        vy = np.random.uniform(-push_vel, push_vel)
        self.ball_vel = np.array([[vx, vy, 0.0]], dtype=np.float32)
        self.log(f"Ball pushed: vx={vx:.3f}, vy={vy:.3f} m/s")

    def on_offset_toggle(self) -> None:
        """Handle top surface offset toggle between anchor center and top surface."""
        # Toggle the state
        self.use_top_surface_offset = not self.use_top_surface_offset
        enabled = self.use_top_surface_offset
        home_z = self.ik.home_height_top_surface if enabled else self.ik.home_height

        if 'manual_pose' in self.gui_modules:
            manual_pose = self.gui_modules['manual_pose']
            z_config = self.dof_config['z']
            new_config = (home_z - 90, home_z + 90, z_config[2], home_z, z_config[4])
            self.dof_config['z'] = new_config

            slider = manual_pose.sliders['z']
            res = z_config[2]
            slider.setMinimum(int((home_z - 90) / res))
            slider.setMaximum(int((home_z + 90) / res))

        self.dof_values['z'] = home_z

        ball_start_height = (home_z / 1000) + self.ball_physics.radius
        self.ball_pos[0, 2] = ball_start_height
        self.log(f"Offset: {'Top Surface' if enabled else 'Anchor Center'}")

    def on_slider_change(self, dof: str, value: float) -> None:
        """Handle manual DOF slider change."""
        val = float(value)
        self.dof_values[dof] = val

        if self.update_timer is not None:
            self.update_timer.stop()
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.calculate_ik)
        self.update_timer.start(50)

    def go_home(self) -> None:
        """Return platform to home position (all DOFs to default)."""
        if self.controller_enabled:
            self.log("Disable controller to use manual control")
            return

        # Set all DOF values to home position
        home_z = (self.ik.home_height_top_surface if self.use_top_surface_offset
                  else self.ik.home_height)

        self.dof_values['x'] = 0.0
        self.dof_values['y'] = 0.0
        self.dof_values['z'] = home_z
        self.dof_values['rx'] = 0.0
        self.dof_values['ry'] = 0.0
        self.dof_values['rz'] = 0.0

        # Update sliders to reflect home position
        if 'manual_pose' in self.gui_modules:
            manual_pose = self.gui_modules['manual_pose']
            for dof, value in self.dof_values.items():
                if dof in manual_pose.sliders:
                    res = self.dof_config[dof][2]
                    manual_pose.sliders[dof].blockSignals(True)
                    manual_pose.sliders[dof].setValue(int(value / res))
                    manual_pose.sliders[dof].blockSignals(False)
                    manual_pose.value_labels[dof].setText(f"{value:.2f}")

        # Calculate and send IK to hardware/simulation
        self.calculate_ik()
        self.update_plot()
        self.log("Platform moved to home position")

    def on_camera_enable_change(self, enabled: bool) -> None:
        """Handle camera enable/disable."""
        self.camera_enabled = enabled
        self.log(f"Camera noise: {'ENABLED' if enabled else 'DISABLED'}")

    def on_camera_param_change(self, param_name: str, value: float) -> None:
        """Handle camera parameter change."""
        self.log(f"Camera {param_name}: {value}")

    def on_camera_reset(self) -> None:
        """Handle camera reset to default state."""
        self.pixy_camera.reset()
        self.log("Camera reset")

    def on_z_optimization_toggle(self, enabled: bool) -> None:
        """Handle Z optimization enable/disable."""
        self.z_optimization_enabled = enabled
        self.log(f"Z Optimization: {'ENABLED' if enabled else 'DISABLED'}")

        # Immediately recalculate IK with new optimization setting
        if not self.controller_enabled:  # Only in manual mode
            self.calculate_ik()

    def calculate_ik(self) -> None:
        """Calculate inverse kinematics for current pose and update servos."""
        translation = np.array([self.dof_values['x'],
                                self.dof_values['y'],
                                self.dof_values['z']])

        # In manual mode, allow full slider range (no clipping)
        # Clipping is handled in control logic (controller + IMU compensation)
        rotation = np.array([self.dof_values['rx'],
                            self.dof_values['ry'],
                            self.dof_values['rz']])

        # Apply Z optimization if enabled
        if self.z_optimization_enabled:
            # Use CURRENT Z as search center (not home Z)
            # This allows optimization at extreme angles where home Z is invalid
            search_translation = translation.copy()

            optimized_translation, angles, success = self.ik.optimize_z_offset(
                search_translation, rotation,
                use_top_surface_offset=self.use_top_surface_offset,
                z_search_range=IKZOptimizationConfig.Z_SEARCH_RANGE_MM,
                max_iterations=IKZOptimizationConfig.MAX_ITERATIONS,
                tolerance=IKZOptimizationConfig.TOLERANCE_DEG,
                ik_cache=getattr(self, 'ik_cache', None)
            )

            if success and angles is not None:
                # Z offset from initial search position
                self.z_offset = optimized_translation[2] - translation[2]
                max_angle = np.max(angles)
                min_angle = np.min(angles)
                self.servo_balance = (max_angle, min_angle)
                # Use optimized angles for servos (not the regular IK)
            else:
                # Fallback to regular IK
                self.log(f"Z opt FAILED at rx={rotation[0]:.1f}°, ry={rotation[1]:.1f}° - using regular IK")
                angles = self.ik.calculate_servo_angles(translation, rotation,
                                                        self.use_top_surface_offset)
                if angles is not None:
                    max_angle = np.max(angles)
                    min_angle = np.min(angles)
                    self.servo_balance = (max_angle, min_angle)
                    self.z_offset = 0.0
        else:
            angles = self.ik.calculate_servo_angles(translation, rotation,
                                                    self.use_top_surface_offset)
            if angles is not None:
                max_angle = np.max(angles)
                min_angle = np.min(angles)
                self.servo_balance = (max_angle, min_angle)
                self.z_offset = 0.0

        if angles is not None:
            self.last_cmd_angles = angles
            if self.simulation_running:
                for i, servo in enumerate(self.servos):
                    servo.send_command(angles[i], self.simulation_time)

        # Update GUI to reflect Z optimization changes
        self.update_gui_modules()

    def start_simulation(self) -> None:
        """Start simulation loop and enable controls."""
        self.simulation_running = True
        self.last_update_time = time.time()
        self.log("Simulation started")

        if 'simulation_control' in self.gui_modules:
            sim_ctrl = self.gui_modules['simulation_control']
            sim_ctrl.start_btn.setEnabled(False)
            sim_ctrl.stop_btn.setEnabled(True)

        self.simulation_timer.start(self.update_rate_ms)

    def stop_simulation(self) -> None:
        """Stop simulation loop and disable controls."""
        self.simulation_running = False
        self.simulation_timer.stop()

        if 'simulation_control' in self.gui_modules:
            sim_ctrl = self.gui_modules['simulation_control']
            sim_ctrl.start_btn.setEnabled(True)
            sim_ctrl.stop_btn.setEnabled(False)

        self.log("Simulation stopped")

    def reset_simulation(self) -> None:
        """Reset simulation to initial state and clear all data."""
        was_running = self.simulation_running
        if was_running:
            self.stop_simulation()

        for servo in self.servos:
            servo.reset()

        self.simulation_time = 0.0
        self.last_update_time = None

        for dof, (_, _, _, default, _) in self.dof_config.items():
            if dof == 'z':
                home_z = (self.ik.home_height_top_surface if self.use_top_surface_offset
                          else self.ik.home_height)
                self.dof_values[dof] = home_z
            else:
                self.dof_values[dof] = default

        self.reset_ball()

        if self.controller_enabled and self.controller is not None:
            self.controller.reset()

        self.log("Simulation reset")

        if was_running:
            self.start_simulation()

    def simulation_loop(self) -> None:
        """Main simulation update loop - physics, controller, and visualization."""
        if not self.simulation_running:
            return

        # Safety check: stop if window is closing
        if not self.isVisible():
            self.simulation_running = False
            return

        current_time = time.time()
        if self.last_update_time is not None:
            # Use actual elapsed time for real-time simulation
            dt = current_time - self.last_update_time
            # Cap dt to prevent huge jumps if timer was delayed
            dt = min(dt, 0.1)  # Max 100ms per step
            self.simulation_time += dt

            # Get true ball position for camera measurement (always needed for visualization)
            ball_x_mm_true = self.ball_pos[0, 0].item() * 1000
            ball_y_mm_true = self.ball_pos[0, 1].item() * 1000

            if self.controller_enabled:
                try:
                    ball_vx_mm_s = self.ball_vel[0, 0].item() * 1000
                    ball_vy_mm_s = self.ball_vel[0, 1].item() * 1000

                    # Apply camera noise if enabled (for ball trail visualization)
                    if self.camera_enabled:
                        measured_x, measured_y, detected, is_new = self.pixy_camera.measure(
                            (ball_x_mm_true, ball_y_mm_true),
                            self.simulation_time
                        )

                        # Update camera stats for GUI
                        camera_state = {
                            'camera_raw_measurement': (ball_x_mm_true, ball_y_mm_true),
                            'camera_quantized_measurement': (measured_x, measured_y) if detected else (None, None),
                            'camera_last_sample_time': self.pixy_camera.last_sample_time,
                            'camera_is_new_sample': is_new
                        }
                        if 'pixy2_camera' in self.gui_modules:
                            self.gui_modules['pixy2_camera'].update(camera_state)

                        if not detected:
                            # Use last known measurement or true position
                            if self.pixy_camera.cached_measurement[0] is not None:
                                ball_x_mm = self.pixy_camera.cached_measurement[0]
                                ball_y_mm = self.pixy_camera.cached_measurement[1]
                            else:
                                ball_x_mm = ball_x_mm_true
                                ball_y_mm = ball_y_mm_true
                        else:
                            # Use raw measured position
                            ball_x_mm = measured_x
                            ball_y_mm = measured_y

                        ball_x_mm_measured = measured_x
                        ball_y_mm_measured = measured_y
                    else:
                        # Camera disabled - use true position
                        ball_x_mm = ball_x_mm_true
                        ball_y_mm = ball_y_mm_true
                        ball_x_mm_measured = ball_x_mm_true
                        ball_y_mm_measured = ball_y_mm_true

                    # Store measured position for ball trail (position after camera noise)
                    self.ball_pos_measured_mm = (ball_x_mm_measured, ball_y_mm_measured)

                    pattern_time = self.simulation_time - self.pattern_start_time
                    target_x, target_y = self.current_pattern.get_position(pattern_time)
                    target_pos_mm = (target_x, target_y)

                    controller_output = self._update_controller(
                        (ball_x_mm, ball_y_mm),  # Use filtered position
                        (ball_vx_mm_s, ball_vy_mm_s),
                        target_pos_mm,
                        dt
                    )

                    if controller_output is None:
                        self.log("Controller returned None - disabling controller")
                        self.controller_enabled = False
                    else:
                        rx_raw, ry_raw = controller_output
                        rx, ry, tilt_mag = clip_tilt_vector(rx_raw, ry_raw, MAX_CONTROLLER_OUTPUT_DEG)

                        if tilt_mag > MAX_CONTROLLER_OUTPUT_DEG:
                            controller_name = self.controller_config.get_controller_name()
                            self.log(f"{controller_name} output clipped: "
                                     f"({rx_raw:.2f}, {ry_raw:.2f}) → ({rx:.2f}, {ry:.2f})")

                        self.dof_values['rx'] = rx
                        self.dof_values['ry'] = ry

                        # Record performance data if recording (for full_c.py/min_c.py)
                        if hasattr(self, 'recording') and self.recording and hasattr(self, 'csv_writer') and self.csv_writer:
                            try:
                                current_time = time.time()
                                elapsed_time = current_time - self.recording_start_time

                                error_x = ball_x_mm - target_pos_mm[0]
                                error_y = ball_y_mm - target_pos_mm[1]
                                error_magnitude = np.sqrt(error_x**2 + error_y**2)
                                tilt_magnitude = np.sqrt(rx**2 + ry**2)

                                self.csv_writer.writerow({
                                    'timestamp': datetime.fromtimestamp(current_time).strftime('%H:%M:%S.%f')[:-3],
                                    'elapsed_time': f'{elapsed_time:.3f}',
                                    'ball_x': f'{ball_x_mm:.3f}',
                                    'ball_y': f'{ball_y_mm:.3f}',
                                    'target_x': f'{target_pos_mm[0]:.3f}',
                                    'target_y': f'{target_pos_mm[1]:.3f}',
                                    'error_x': f'{error_x:.3f}',
                                    'error_y': f'{error_y:.3f}',
                                    'error_magnitude': f'{error_magnitude:.3f}',
                                    'controller_rx': f'{rx:.3f}',
                                    'controller_ry': f'{ry:.3f}',
                                    'tilt_magnitude': f'{tilt_magnitude:.3f}'
                                })
                                self.sample_count += 1

                                # Flush every 100 samples to ensure data is written
                                if self.sample_count % 100 == 0:
                                    self.csv_file.flush()
                            except Exception as e:
                                if hasattr(self, 'log'):
                                    self.log(f"CSV write error: {e}")
                                self.recording = False

                        translation = np.array([self.dof_values['x'],
                                                self.dof_values['y'],
                                                self.dof_values['z']])
                        rotation = np.array([rx, ry, self.dof_values['rz']])

                        # Apply Z optimization if enabled
                        if self.z_optimization_enabled:
                            optimized_translation, angles, success = self.ik.optimize_z_offset(
                                translation, rotation,
                                use_top_surface_offset=self.use_top_surface_offset,
                                z_search_range=IKZOptimizationConfig.Z_SEARCH_RANGE_MM,
                                max_iterations=IKZOptimizationConfig.MAX_ITERATIONS,
                                tolerance=IKZOptimizationConfig.TOLERANCE_DEG
                            )

                            if success and angles is not None:
                                self.z_offset = optimized_translation[2] - translation[2]
                                max_angle = np.max(angles)
                                min_angle = np.min(angles)
                                self.servo_balance = (max_angle, min_angle)
                                translation = optimized_translation
                            else:
                                angles = self.ik.calculate_servo_angles(translation, rotation,
                                                                        self.use_top_surface_offset)
                                if angles is not None:
                                    max_angle = np.max(angles)
                                    min_angle = np.min(angles)
                                    self.servo_balance = (max_angle, min_angle)
                        else:
                            angles = self.ik.calculate_servo_angles(translation, rotation,
                                                                    self.use_top_surface_offset)
                            if angles is not None:
                                max_angle = np.max(angles)
                                min_angle = np.min(angles)
                                self.servo_balance = (max_angle, min_angle)
                                self.z_offset = 0.0

                        if angles is not None:
                            self.last_cmd_angles = angles
                            for i in range(6):
                                self.servos[i].send_command(angles[i], self.simulation_time)
                        else:
                            controller_name = self.controller_config.get_controller_name()
                            self.log(f"{controller_name}: IK solution out of range")

                except Exception as e:
                    controller_name = self.controller_config.get_controller_name()
                    error_msg = format_error_context(
                        self.simulation_time,
                        (ball_x_mm, ball_y_mm),
                        (ball_vx_mm_s, ball_vy_mm_s),
                        str(e)
                    )
                    self.log(f"{controller_name} error:\n{error_msg}")
                    self.controller_enabled = False
                    self.on_controller_toggle()

            for servo in self.servos:
                servo.update(dt, self.simulation_time)

            actual_angles = np.array([servo.get_angle() for servo in self.servos])

            initial_guess = None
            if self.last_fk_translation is not None and self.last_fk_rotation is not None:
                initial_guess = (self.last_fk_translation, self.last_fk_rotation)

            translation, rotation, success, iterations = self.ik.calculate_forward_kinematics(
                actual_angles,
                initial_guess=initial_guess,
                use_top_surface_offset=self.use_top_surface_offset
            )

            # Fallback: if FK fails with initial guess, retry from home position
            if not success and initial_guess is not None:
                translation, rotation, success, iterations = self.ik.calculate_forward_kinematics(
                    actual_angles,
                    initial_guess=None,
                    use_top_surface_offset=self.use_top_surface_offset
                )
                if success:
                    self.log(f"FK recovered using home position guess")

            if not success:
                self.log(f"FK FAILED after retry, angles={actual_angles}")

            if success:
                self.last_fk_translation = translation
                self.last_fk_rotation = rotation

                try:
                    platform_pose = np.array([[
                        translation[0] / 1000, translation[1] / 1000, translation[2] / 1000,
                        rotation[0], rotation[1], rotation[2]
                    ]], dtype=np.float32)

                    self.ball_pos, self.ball_vel, self.ball_omega, contact_info = \
                        self.ball_physics.step(
                            self.ball_pos, self.ball_vel, self.ball_omega, platform_pose, dt,
                            platform_angular_accel=self.platform_angular_accel
                        )

                    if contact_info.get('fell_off', False):
                        self.log("Ball fell off platform")

                except Exception as e:
                    self.log(format_error_context(
                        self.simulation_time,
                        self.ball_pos[0, :2],
                        self.ball_vel[0, :2] * 1000,
                        f"Physics error: {str(e)}"
                    ))
                    self.reset_ball()

                rx_now = rotation[0]
                ry_now = rotation[1]

                omega_rx = (rx_now - self.prev_platform_angles['rx']) / dt
                omega_ry = (ry_now - self.prev_platform_angles['ry']) / dt

                alpha_rx = (omega_rx - self.platform_angular_vel['rx']) / dt
                alpha_ry = (omega_ry - self.platform_angular_vel['ry']) / dt

                self.platform_angular_vel['rx'] = omega_rx
                self.platform_angular_vel['ry'] = omega_ry
                self.platform_angular_accel['rx'] = alpha_rx
                self.platform_angular_accel['ry'] = alpha_ry

                self.prev_platform_angles['rx'] = rx_now
                self.prev_platform_angles['ry'] = ry_now

            self.update_gui_modules()
            self.update_plot()

        self.last_update_time = current_time

    @abstractmethod
    def _initialize_controller(self) -> None:
        """Initialize controller (implemented by subclass)."""
        pass

    @abstractmethod
    def _update_controller(self, ball_pos_mm: Tuple[float, float],
                          ball_vel_mm_s: Tuple[float, float],
                          target_pos_mm: Tuple[float, float],
                          dt: float) -> Optional[Tuple[float, float]]:
        """Update controller and return control output (implemented by subclass)."""
        pass

    def on_controller_param_change(self) -> None:
        """Callback when controller parameters change."""
        pass

    def closeEvent(self, event: QEvent) -> None:
        """Clean shutdown when window is closed."""
        # Stop simulation first
        self.simulation_running = False

        # Stop all timers
        if hasattr(self, 'simulation_timer'):
            self.simulation_timer.stop()

        if hasattr(self, 'update_timer') and self.update_timer is not None:
            self.update_timer.stop()

        # Clear plot references to prevent access after deletion
        if hasattr(self, 'ball_scatter'):
            self.ball_scatter = None
        if hasattr(self, 'trajectory_line'):
            self.trajectory_line = None
        if hasattr(self, 'target_marker'):
            self.target_marker = None
        if hasattr(self, 'tilt_arrow'):
            self.tilt_arrow = None

        event.accept()
