#!/usr/bin/env python3
"""
Stewart Platform Simulator - Modular Base Class

Reusable simulator with pluggable controller support and modular GUI.
PyQt6 + PyQtGraph implementation.
"""

from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QGroupBox, QGridLayout, QLabel, QSlider, QComboBox
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont
import pyqtgraph as pg
import numpy as np
import torch
import time
from abc import ABC, abstractmethod

from core.core import FirstOrderServo, StewartPlatformIK, SimpleBallPhysics2D, PatternFactory, Pixy2Camera
from core.control_core import clip_tilt_vector
from core.utils import (
    MAX_TILT_ANGLE_DEG, PLATFORM_HALF_SIZE_MM,
    SimulationConfig, format_time, format_error_context
)
from gui.gui_builder import GUIBuilder
from gui import gui_modules as gm


class ControllerConfig(ABC):
    """Abstract base for controller-specific configuration."""

    @abstractmethod
    def get_controller_name(self) -> str:
        """Return display name for controller."""
        pass

    @abstractmethod
    def create_controller(self, **kwargs):
        """Create and return controller instance."""
        pass

    @abstractmethod
    def get_scalar_values(self) -> list:
        """Return list of scalar multipliers for parameters."""
        pass

    def get_scaled_param(self, param_name, sliders, scalar_vars):
        """Extract and scale a parameter value from widgets."""
        raw = sliders[param_name].value() / 100.0  # QSlider uses integers
        scalar = self.get_scalar_values()[scalar_vars[param_name]]
        return raw * scalar

    def create_parameter_slider(self, parent_layout, param_name, label, default,
                                sliders, value_labels, scalar_vars,
                                on_change_callback):
        """Create standard parameter slider with scalar multiplier."""
        grid = QGridLayout()

        label_widget = QLabel(label)
        label_widget.setFont(QFont('Segoe UI', 9))
        grid.addWidget(label_widget, 0, 0, Qt.AlignmentFlag.AlignLeft)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(1000)
        slider.setValue(int(default * 100))
        grid.addWidget(slider, 0, 1)
        sliders[param_name] = slider

        value_label = QLabel(f"{default:.2f}")
        value_label.setFont(QFont('Consolas', 9))
        value_label.setMinimumWidth(60)
        grid.addWidget(value_label, 0, 2)
        value_labels[param_name] = value_label

        scalar_combo = QComboBox()
        scalar_combo.addItems([f'×{s:.7g}' for s in self.get_scalar_values()])
        scalar_combo.setCurrentIndex(getattr(self, 'default_scalar_idx', 4))
        scalar_combo.setMinimumWidth(120)
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

    def __init__(self, app, controller_config: ControllerConfig):
        super().__init__()
        self.app = app
        self.controller_config = controller_config

        controller_name = controller_config.get_controller_name()
        self.setWindowTitle(f"Stewart Platform - {controller_name} Ball Balancing Control")
        self.resize(1400, 900)

        self.colors = {
            'bg': '#1e1e1e',
            'panel_bg': '#2d2d2d',
            'widget_bg': '#3d3d3d',
            'fg': '#e0e0e0',
            'highlight': '#007acc',
            'button_bg': '#0e639c',
            'button_fg': '#ffffff',
            'entry_bg': '#3d3d3d',
            'border': '#555555',
            'success': '#4ec9b0',
            'warning': '#ce9178'
        }

        self.setup_dark_theme()

        self.platform_params = {
            "horn_length": 31.75,
            "rod_length": 145.0,
            "base": 73.025,
            "base_anchors": 36.8893,
            "platform": 67.775,
            "platform_anchors": 12.7,
            "top_surface_offset": 26.0
        }
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

        self.ball_physics = SimpleBallPhysics2D(
            ball_radius=0.02,
            ball_mass=0.0027,
            gravity=9.81,
            rolling_friction=0.0225,
            sphere_type='hollow'
        )

        self.pixy_camera = Pixy2Camera(
            pixel_size_mm=1.4,
            subpixel_noise_std=0.4,
            detection_rate=0.999,
            sample_rate_hz=19.3
        )
        self.camera_enabled = True

        self.controller = None
        self.controller_enabled = False

        self.current_pattern = PatternFactory.create('static', x=0.0, y=0.0)
        self.pattern_type = 'static'
        self.pattern_start_time = 0.0
        self.pattern_params = {}

        ball_start_height = (self.ik.home_height_top_surface / 1000) + self.ball_physics.radius
        self.ball_pos = torch.tensor([[0.0, 0.0, ball_start_height]], dtype=torch.float32)
        self.ball_vel = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        self.ball_omega = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

        self.simulation_running = False
        self.simulation_time = 0.0
        self.last_update_time = None
        self.update_rate_ms = 10  # 100 Hz target (10ms), more reliable than 2ms

        self.use_top_surface_offset = True
        self.dof_values = {
            'x': 0.0, 'y': 0.0, 'z': self.ik.home_height_top_surface,
            'rx': 0.0, 'ry': 0.0, 'rz': 0.0
        }

        self.dof_config = {
            'x': (-30.0, 30.0, 0.1, 0.0, "X Position (mm)"),
            'y': (-30.0, 30.0, 0.1, 0.0, "Y Position (mm)"),
            'z': (self.ik.home_height_top_surface - 30,
                  self.ik.home_height_top_surface + 30,
                  0.1, self.ik.home_height_top_surface, "Z Height (mm)"),
            'rx': (-MAX_TILT_ANGLE_DEG, MAX_TILT_ANGLE_DEG, 0.1, 0.0, "Roll (°)"),
            'ry': (-MAX_TILT_ANGLE_DEG, MAX_TILT_ANGLE_DEG, 0.1, 0.0, "Pitch (°)"),
            'rz': (-MAX_TILT_ANGLE_DEG, MAX_TILT_ANGLE_DEG, 0.1, 0.0, "Yaw (°)")
        }

        self.prev_platform_angles = {'rx': 0.0, 'ry': 0.0}
        self.platform_angular_vel = {'rx': 0.0, 'ry': 0.0}
        self.platform_angular_accel = {'rx': 0.0, 'ry': 0.0}

        self.last_cmd_angles = np.zeros(6)
        self.last_fk_translation = np.zeros(3)
        self.last_fk_rotation = np.zeros(3)

        self.update_timer = None
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self.simulation_loop)

        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self._create_controller_param_widgets()
        self._build_modular_gui()
        self._initialize_controller()

    def setup_dark_theme(self):
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
                background-color: #005a9e;
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
                background-color: #0088dd;
                border-color: #0088dd;
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

    def _create_controller_param_widgets(self):
        """Create controller parameter widgets."""
        controller_name = self.controller_config.get_controller_name()

        if controller_name == "PID":
            self.param_definitions = [
                ('kp', 'P (Proportional)', 3.0, 4),
                ('ki', 'I (Integral)', 1.0, 4),
                ('kd', 'D (Derivative)', 3.0, 4)
            ]
        elif controller_name == "LQR":
            self.param_definitions = [
                ('Q_pos', 'Q Position Weight', 1.0, 7),
                ('Q_vel', 'Q Velocity Weight', 1.0, 6),
                ('R', 'R Control Weight', 1.0, 5)
            ]
        else:
            self.param_definitions = []

        self.controller_widgets = {
            'sliders': {},
            'value_labels': {},
            'scalar_vars': {},
            'update_fn': lambda: None,
            'param_definitions': self.param_definitions
        }

    def _build_modular_gui(self):
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
            'ball_filter': gm.BallFilterModule,
            'pixy2_camera': gm.Pixy2CameraModule,
            'kalman_filter': gm.KalmanFilterModule,
            'plot_control': gm.PlotControlModule,
        }

        layout_config = self.get_layout_config()
        callbacks = self._create_callbacks()

        self.gui_builder = GUIBuilder(self.central_widget, module_registry)
        self.gui_modules = self.gui_builder.build(layout_config, self.colors, callbacks)

        if 'plot_panel' in self.gui_modules:
            self._create_plot(self.gui_modules['plot_panel'])

    def _create_callbacks(self):
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
            'camera_enable_change': self.on_camera_enable_change,
            'camera_param_change': self.on_camera_param_change,
            'camera_reset': self.on_camera_reset,
            'log': self.log,
        }

    def on_pattern_param_change(self, param_name, value):
        """Update pattern with new parameters."""
        pattern_type = self.pattern_type

        self.pattern_params[param_name] = value

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

        self.reset_pattern()
        self.update_plot()

    @abstractmethod
    def get_layout_config(self):
        """Return layout configuration for this simulator."""
        raise NotImplementedError

    def _create_plot(self, parent):
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

    def setup_plot(self):
        """Setup PyQtGraph plot."""
        plot_item = self.plot_widget.getPlotItem()
        plot_item.setXRange(-120, 120)
        plot_item.setYRange(-120, 120)
        plot_item.setLabel('bottom', 'X (mm)', color=self.colors['fg'])
        plot_item.setLabel('left', 'Y (mm)', color=self.colors['fg'])
        plot_item.setTitle('Ball Position (Top View)', color=self.colors['fg'])
        plot_item.showGrid(x=True, y=True, alpha=0.2)
        plot_item.setAspectLocked(True)

        # Platform boundary
        self.platform_rect = pg.QtWidgets.QGraphicsRectItem(
            -PLATFORM_HALF_SIZE_MM, -PLATFORM_HALF_SIZE_MM,
            PLATFORM_HALF_SIZE_MM * 2, PLATFORM_HALF_SIZE_MM * 2
        )
        pen = pg.mkPen(color=self.colors['fg'], width=2, style=Qt.PenStyle.DashLine)
        self.platform_rect.setPen(pen)
        plot_item.addItem(self.platform_rect)

        # Trajectory line
        self.trajectory_line = plot_item.plot([], [], pen=pg.mkPen(color=self.colors['highlight'],
                                                                     width=1, style=Qt.PenStyle.DashLine))

        # Target marker
        self.target_marker = pg.ScatterPlotItem([0], [0], symbol='x', size=15,
                                                 pen=pg.mkPen(color=self.colors['success'], width=2))
        plot_item.addItem(self.target_marker)

        # Ball
        self.ball_scatter = pg.ScatterPlotItem([0], [0], symbol='o', size=20,
                                                pen=pg.mkPen(None),
                                                brush=pg.mkBrush('#ff4444'))
        plot_item.addItem(self.ball_scatter)

        # Tilt arrow (will be added dynamically)
        self.tilt_arrow = None

    def update_plot(self):
        """Update plot with current state."""
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

        try:
            if self.pattern_type != 'static':
                pattern_periods = {'circle': 10.0, 'figure8': 12.0, 'star': 15.0}
                period = pattern_periods.get(self.pattern_type, 10.0)

                t_samples = np.linspace(0, period, 100)
                path_x, path_y = [], []
                for t in t_samples:
                    x, y = self.current_pattern.get_position(t)
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

    def log(self, message):
        """Add message to debug log."""
        if 'debug_log' in self.gui_modules:
            self.gui_modules['debug_log'].log(message, self.simulation_time)

    def update_gui_modules(self):
        """Update all GUI modules with current state."""
        ball_x_mm = self.ball_pos[0, 0].item() * 1000
        ball_y_mm = self.ball_pos[0, 1].item() * 1000
        vel_x_mm = self.ball_vel[0, 0].item() * 1000
        vel_y_mm = self.ball_vel[0, 1].item() * 1000

        state = {
            'simulation_time': self.simulation_time,
            'controller_enabled': self.controller_enabled,
            'ball_pos': (ball_x_mm, ball_y_mm),
            'ball_vel': (vel_x_mm, vel_y_mm),
            'dof_values': self.dof_values,
            'cmd_angles': self.last_cmd_angles,
            'actual_angles': [s.get_angle() for s in self.servos],
            'fk_translation': self.last_fk_translation,
            'fk_rotation': self.last_fk_rotation,
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

    def on_controller_toggle(self):
        """Handle controller enable/disable."""
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

    def on_pattern_change(self, pattern_type=None):
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

    def reset_pattern(self):
        """Reset pattern timing."""
        self.pattern_start_time = self.simulation_time
        self.current_pattern.reset()
        self.log(f"Pattern reset at t={format_time(self.simulation_time)}")

        if self.controller_enabled:
            self.controller.reset()

    def reset_ball(self):
        """Reset ball to center."""
        home_z = self.ik.home_height_top_surface if self.use_top_surface_offset else self.ik.home_height
        ball_start_height = (home_z / 1000) + self.ball_physics.radius

        self.ball_pos = torch.tensor([[0.0, 0.0, ball_start_height]], dtype=torch.float32)
        self.ball_vel = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        self.ball_omega = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

        if self.controller_enabled:
            self.controller.reset()

        self.update_plot()
        self.log("Ball reset to center")

    def push_ball(self):
        """Apply random velocity to ball."""
        vx = np.random.uniform(-0.05, 0.05)
        vy = np.random.uniform(-0.05, 0.05)
        self.ball_vel = torch.tensor([[vx, vy, 0.0]], dtype=torch.float32)
        self.log(f"Ball pushed: vx={vx:.3f}, vy={vy:.3f} m/s")

    def on_offset_toggle(self):
        """Handle top surface offset toggle."""
        # Toggle the state
        self.use_top_surface_offset = not self.use_top_surface_offset
        enabled = self.use_top_surface_offset
        home_z = self.ik.home_height_top_surface if enabled else self.ik.home_height

        if 'manual_pose' in self.gui_modules:
            manual_pose = self.gui_modules['manual_pose']
            z_config = self.dof_config['z']
            new_config = (home_z - 30, home_z + 30, z_config[2], home_z, z_config[4])
            self.dof_config['z'] = new_config

            slider = manual_pose.sliders['z']
            res = z_config[2]
            slider.setMinimum(int((home_z - 30) / res))
            slider.setMaximum(int((home_z + 30) / res))

        self.dof_values['z'] = home_z

        ball_start_height = (home_z / 1000) + self.ball_physics.radius
        self.ball_pos[0, 2] = ball_start_height
        self.log(f"Offset: {'Top Surface' if enabled else 'Anchor Center'}")

    def on_slider_change(self, dof, value):
        """Handle manual DOF slider change."""
        val = float(value)
        self.dof_values[dof] = val

        if self.update_timer is not None:
            self.update_timer.stop()
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.calculate_ik)
        self.update_timer.start(50)

    def on_camera_enable_change(self, enabled):
        """Handle camera enable/disable."""
        self.camera_enabled = enabled
        self.log(f"Camera noise: {'ENABLED' if enabled else 'DISABLED'}")

    def on_camera_param_change(self, param_name, value):
        """Handle camera parameter change."""
        self.log(f"Camera {param_name}: {value}")

    def on_camera_reset(self):
        """Handle camera reset."""
        self.pixy_camera.reset()
        self.log("Camera reset")

    def calculate_ik(self):
        """Calculate inverse kinematics for current pose."""
        translation = np.array([self.dof_values['x'],
                                self.dof_values['y'],
                                self.dof_values['z']])

        rx_limited, ry_limited, tilt_mag = clip_tilt_vector(
            self.dof_values['rx'],
            self.dof_values['ry'],
            MAX_TILT_ANGLE_DEG
        )

        if tilt_mag > MAX_TILT_ANGLE_DEG and not self.controller_enabled:
            self.dof_values['rx'] = rx_limited
            self.dof_values['ry'] = ry_limited

        rotation = np.array([rx_limited, ry_limited, self.dof_values['rz']])

        angles = self.ik.calculate_servo_angles(translation, rotation,
                                                self.use_top_surface_offset)

        if angles is not None:
            self.last_cmd_angles = angles
            if self.simulation_running:
                for i, servo in enumerate(self.servos):
                    servo.send_command(angles[i], self.simulation_time)

    def start_simulation(self):
        """Start simulation loop."""
        self.simulation_running = True
        self.last_update_time = time.time()
        self.log("Simulation started")

        if 'simulation_control' in self.gui_modules:
            sim_ctrl = self.gui_modules['simulation_control']
            sim_ctrl.start_btn.setEnabled(False)
            sim_ctrl.stop_btn.setEnabled(True)

        self.simulation_timer.start(self.update_rate_ms)

    def stop_simulation(self):
        """Stop simulation loop."""
        self.simulation_running = False
        self.simulation_timer.stop()

        if 'simulation_control' in self.gui_modules:
            sim_ctrl = self.gui_modules['simulation_control']
            sim_ctrl.start_btn.setEnabled(True)
            sim_ctrl.stop_btn.setEnabled(False)

        self.log("Simulation stopped")

    def reset_simulation(self):
        """Reset simulation to initial state."""
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

        if self.controller_enabled:
            self.controller.reset()

        self.log("Simulation reset")

        if was_running:
            self.start_simulation()

    def simulation_loop(self):
        """Main simulation update loop."""
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

            if self.controller_enabled:
                try:
                    # Get true ball position
                    ball_x_mm_true = self.ball_pos[0, 0].item() * 1000
                    ball_y_mm_true = self.ball_pos[0, 1].item() * 1000
                    ball_vx_mm_s = self.ball_vel[0, 0].item() * 1000
                    ball_vy_mm_s = self.ball_vel[0, 1].item() * 1000

                    # Apply camera noise if enabled
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
                    else:
                        # Camera disabled - use true position
                        ball_x_mm = ball_x_mm_true
                        ball_y_mm = ball_y_mm_true

                    pattern_time = self.simulation_time - self.pattern_start_time
                    target_x, target_y = self.current_pattern.get_position(pattern_time)
                    target_pos_mm = (target_x, target_y)

                    rx_raw, ry_raw = self._update_controller(
                        (ball_x_mm, ball_y_mm),  # Use filtered position
                        (ball_vx_mm_s, ball_vy_mm_s),
                        target_pos_mm,
                        dt
                    )

                    rx, ry, tilt_mag = clip_tilt_vector(rx_raw, ry_raw, MAX_TILT_ANGLE_DEG)

                    if tilt_mag > MAX_TILT_ANGLE_DEG:
                        controller_name = self.controller_config.get_controller_name()
                        self.log(f"{controller_name} output clipped: "
                                 f"({rx_raw:.2f}, {ry_raw:.2f}) → ({rx:.2f}, {ry:.2f})")

                    self.dof_values['rx'] = rx
                    self.dof_values['ry'] = ry

                    translation = np.array([self.dof_values['x'],
                                            self.dof_values['y'],
                                            self.dof_values['z']])
                    rotation = np.array([rx, ry, self.dof_values['rz']])

                    angles = self.ik.calculate_servo_angles(translation, rotation,
                                                            self.use_top_surface_offset)

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

            translation, rotation, success, _ = self.ik.calculate_forward_kinematics(
                actual_angles, use_top_surface_offset=self.use_top_surface_offset
            )

            if success:
                self.last_fk_translation = translation
                self.last_fk_rotation = rotation

                try:
                    platform_pose = torch.tensor([[
                        translation[0] / 1000, translation[1] / 1000, translation[2] / 1000,
                        rotation[0], rotation[1], rotation[2]
                    ]], dtype=torch.float32)

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
    def _initialize_controller(self):
        """Initialize controller (implemented by subclass)."""
        pass

    @abstractmethod
    def _update_controller(self, ball_pos_mm, ball_vel_mm_s, target_pos_mm, dt):
        """Update controller and return control output (implemented by subclass)."""
        pass

    def on_controller_param_change(self):
        """Callback when controller parameters change."""
        pass

    def closeEvent(self, event):
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
