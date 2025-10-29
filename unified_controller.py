#!/usr/bin/env python3
"""
Unified Stewart Platform Controller
Supports 4 modes: PID/LQR × Simulation/Real Hardware
Clean minimal GUI with essential controls only
"""

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QGroupBox, QMessageBox, QScrollArea)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont
import pyqtgraph as pg
import numpy as np
import torch
import time
import threading
import gc
import sys
import ctypes
from queue import Queue, Empty

from core.core import FirstOrderServo, StewartPlatformIK, SimpleBallPhysics2D, PatternFactory, Pixy2Camera
from core.control_core import PIDController, LQRController, clip_tilt_vector, KalmanFilter
from core.utils import (MAX_TILT_ANGLE_DEG, PLATFORM_RADIUS_MM, SimulationConfig,
                        format_time, format_vector_2d)
from setup.hardware_controller_config import (SerialController, IKCache, WindowsTimerManager,
                                               ThreadPriorityManager, HardwareControllerConfig,
                                               LQRControllerConfig)
from gui import gui_modules as gm

# Fixed configuration
CONTROL_FREQUENCY_HZ = 250
PLOT_REFRESH_HZ = 10
THREAD_PRIORITY_TIME_CRITICAL = 15


class UnifiedStewartController(QMainWindow):
    """Unified controller supporting PID/LQR and Sim/Real modes."""

    def __init__(self, app):
        super().__init__()
        self.app = app

        # Mode state
        self.operation_mode = 'sim'  # 'sim' or 'real'
        self.controller_type = 'Manual'  # 'PID', 'LQR', or 'Manual'

        self.setWindowTitle(f"Stewart Platform - Unified Controller")
        self.resize(1400, 900)

        # Colors
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

        # Platform configuration
        self.platform_params = {
            "horn_length": 45.3722,
            "rod_length": 205.0,
            "base": 86.6025 + 18.75 + 11,
            "base_anchors": 64.75,
            "platform": 84.0759,
            "platform_anchors": 12.5,
            "top_surface_offset": 38.0
        }
        self.ik = StewartPlatformIK(**self.platform_params)

        # Simulation components
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
            rolling_friction=0.01,
            sphere_type='hollow'
        )

        # Camera with fixed parameters
        self.pixy_camera = Pixy2Camera(
            pixel_size_mm=1.4,
            subpixel_noise_std=0.4,
            detection_rate=0.999,
            sample_rate_hz=19.3
        )
        self.camera_enabled = True  # Always on in sim mode

        # Ball state
        ball_start_height = (self.ik.home_height_top_surface / 1000) + self.ball_physics.radius
        self.ball_pos = torch.tensor([[0.0, 0.0, ball_start_height]], dtype=torch.float32)
        self.ball_vel = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        self.ball_omega = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

        # Hardware components
        self.serial_controller = None
        self.connected = False
        self.port_var = ''

        # Hardware camera calibration
        self.pixy_width_mm = 558.0
        self.pixy_height_mm = 424.0
        self.pixels_to_mm_x = self.pixy_width_mm / 316.0
        self.pixels_to_mm_y = self.pixy_height_mm / 208.0
        self.ball_pos_mm = (0.0, 0.0)
        self.ball_detected = False
        self.last_ball_update = 0
        self.ball_history_x = []
        self.ball_history_y = []
        self.max_history = 100

        # IK cache for hardware mode
        self.ik_cache = IKCache(max_size=5000)
        self._translation_buffer = np.zeros(3, dtype=np.float64)
        self._rotation_buffer = np.zeros(3, dtype=np.float64)

        # Controllers (will be initialized based on mode)
        self.controller = None
        self.controller_enabled = False

        # Kalman filter (always on)
        ball_physics_params = {
            'radius': 0.02,
            'mass': 0.0027,
            'gravity': 9.81,
            'mass_factor': 1.667
        }
        self.control_interval = 1.0 / CONTROL_FREQUENCY_HZ
        self.kalman_filter = KalmanFilter(
            process_noise_scale=1.0,
            measurement_noise_scale=1.0,
            ball_physics_params=ball_physics_params,
            dt=self.control_interval
        )
        self.kalman_enabled = True  # Always on

        # Trajectory pattern
        self.current_pattern = PatternFactory.create('static', x=0.0, y=0.0)
        self.pattern_type = 'static'
        self.pattern_start_time = 0.0
        self.pattern_params = {}

        # DOF state (offset always enabled)
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

        # Platform angular state (for simulation)
        self.prev_platform_angles = {'rx': 0.0, 'ry': 0.0}
        self.platform_angular_vel = {'rx': 0.0, 'ry': 0.0}
        self.platform_angular_accel = {'rx': 0.0, 'ry': 0.0}

        # Servo state
        self.last_cmd_angles = np.zeros(6)
        self.last_fk_translation = np.zeros(3)
        self.last_fk_rotation = np.zeros(3)
        self.last_sent_angles = None
        self.angle_change_threshold = 0.2

        # Control loop state
        self.simulation_running = False
        self.simulation_time = 0.0
        self.last_update_time = None
        self.update_timer = None

        # Threading (for hardware mode)
        self.control_thread = None
        self.control_thread_id = None
        self.priority_manager = ThreadPriorityManager()
        self.timer_manager = WindowsTimerManager()

        # GUI update
        self.last_gui_update = time.time()
        self.gui_update_count = 0

        # Simulation timer (for sim mode)
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self.simulation_loop)
        self.update_rate_ms = 10  # 100 Hz target for sim mode

        # Build GUI
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self._create_controller_param_widgets()
        self._build_gui()
        self._initialize_controller()

    def setup_dark_theme(self):
        """Configure dark theme."""
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
            QPushButton:checked {{
                background-color: {self.colors['success']};
                color: {self.colors['button_fg']};
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
            QCheckBox::indicator:checked {{
                background-color: {self.colors['highlight']};
                border-color: {self.colors['highlight']};
            }}
            QComboBox {{
                background-color: {self.colors['widget_bg']};
                color: {self.colors['fg']};
                border: 1px solid {self.colors['border']};
                padding: 3px 5px;
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
        """
        self.app.setStyleSheet(stylesheet)

    def _create_controller_param_widgets(self):
        """Create controller parameter widgets for PID and LQR."""
        # PID parameters
        self.pid_param_definitions = [
            ('kp', 'P (Proportional)', 1.0, 6),
            ('ki', 'I (Integral)', 1.0, 6),
            ('kd', 'D (Derivative)', 4.0, 5)
        ]

        # LQR parameters
        self.lqr_param_definitions = [
            ('Q_pos', 'Q Position Weight', 1.0, 9),
            ('Q_vel', 'Q Velocity Weight', 1.0, 5),
            ('R', 'R Control Weight', 1.0, 5)
        ]

        self.controller_widgets = {
            'sliders': {},
            'value_labels': {},
            'scalar_vars': {},
            'param_definitions': self.pid_param_definitions  # Start with PID
        }

    def _build_gui(self):
        """Build the unified GUI."""
        main_layout = QHBoxLayout(self.central_widget)

        # Left side: Scrollable controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(450)
        scroll.setMaximumWidth(500)

        scroll_content = QWidget()
        controls_layout = QVBoxLayout(scroll_content)
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Create modules
        callbacks = self._create_callbacks()

        # Mode selector
        self.mode_selector = gm.ModeSelectionModule(
            self, self.colors, callbacks, current_mode=self.operation_mode
        )
        controls_layout.addWidget(self.mode_selector.create())

        # Serial connection (only visible in hardware mode)
        self.serial_module = gm.SerialConnectionModule(
            self, self.colors, callbacks, port_var=self.port_var
        )
        self.serial_widget = self.serial_module.create()
        controls_layout.addWidget(self.serial_widget)
        # Hide serial connection in sim mode (default)
        if self.operation_mode == 'sim':
            self.serial_widget.setVisible(False)

        # Simulation control
        self.sim_control = gm.SimulationControlModule(
            self, self.colors, callbacks
        )
        controls_layout.addWidget(self.sim_control.create())

        # Trajectory pattern
        self.trajectory_module = gm.TrajectoryPatternModule(
            self, self.colors, callbacks, pattern_var=self.pattern_type
        )
        controls_layout.addWidget(self.trajectory_module.create())

        # Controller selector + controller panel container
        controller_container = QWidget()
        controller_container_layout = QVBoxLayout(controller_container)
        controller_container_layout.setContentsMargins(0, 0, 0, 0)

        # Controller selector (PID/LQR toggle)
        self.controller_selector = gm.ControllerSelectionModule(
            self, self.colors, callbacks, current_controller=self.controller_type
        )
        controller_container_layout.addWidget(self.controller_selector.create())

        # Controller panel (will be dynamically updated)
        self.controller_panel_widget = QWidget()
        self.controller_panel_layout = QVBoxLayout(self.controller_panel_widget)
        self.controller_panel_layout.setContentsMargins(0, 0, 0, 0)
        controller_container_layout.addWidget(self.controller_panel_widget)

        controls_layout.addWidget(controller_container)

        # Build initial controller panel
        self._rebuild_controller_panel()

        # Manual pose control
        self.manual_pose = gm.ManualPoseControlModule(
            self, self.colors, callbacks, dof_config=self.dof_config
        )
        controls_layout.addWidget(self.manual_pose.create())

        # Ball control
        self.ball_control = gm.BallControlModule(
            self, self.colors, callbacks
        )
        controls_layout.addWidget(self.ball_control.create())

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        # Right side: Plot
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        self._create_plot(plot_layout)
        main_layout.addWidget(plot_widget)

        # Store GUI modules for updates
        self.gui_modules = {
            'mode_selector': self.mode_selector,
            'serial_connection': self.serial_module,
            'simulation_control': self.sim_control,
            'trajectory_pattern': self.trajectory_module,
            'controller_selector': self.controller_selector,
            'controller': self.controller_module,
            'manual_pose': self.manual_pose,
            'ball_control': self.ball_control
        }

    def _rebuild_controller_panel(self):
        """Rebuild the controller panel based on current controller type."""
        # Clear existing panel
        widgets_to_delete = []
        while self.controller_panel_layout.count():
            item = self.controller_panel_layout.takeAt(0)
            if item.widget():
                widget = item.widget()
                widgets_to_delete.append(widget)
                widget.setParent(None)
                widget.deleteLater()

        # Process pending events multiple times to ensure widgets are deleted
        for _ in range(3):
            QApplication.processEvents()
            time.sleep(0.01)

        # Explicitly delete widget references
        for widget in widgets_to_delete:
            try:
                del widget
            except:
                pass

        # Force garbage collection
        gc.collect()

        # If Manual mode, don't create controller panel
        if self.controller_type == 'Manual':
            # Clear old slider references
            self.controller_widgets['sliders'].clear()
            self.controller_widgets['value_labels'].clear()
            self.controller_widgets['scalar_vars'].clear()

            # Set controller_module to None and update gui_modules
            self.controller_module = None
            if hasattr(self, 'gui_modules'):
                self.gui_modules['controller'] = None
            return

        # Update param definitions for PID/LQR
        if self.controller_type == 'PID':
            self.controller_widgets['param_definitions'] = self.pid_param_definitions
            controller_name = "PID"
        else:
            self.controller_widgets['param_definitions'] = self.lqr_param_definitions
            controller_name = "LQR"

        # Clear old slider references (important!)
        self.controller_widgets['sliders'].clear()
        self.controller_widgets['value_labels'].clear()
        self.controller_widgets['scalar_vars'].clear()

        # Create new controller module
        callbacks = self._create_callbacks()
        controller_module = gm.ControllerModule(
            self, self.colors, callbacks,
            controller_config=self._get_controller_config(),
            controller_widgets=self.controller_widgets
        )
        self.controller_panel_layout.addWidget(controller_module.create())

        # Store reference and update gui_modules (if it exists)
        self.controller_module = controller_module
        if hasattr(self, 'gui_modules'):
            self.gui_modules['controller'] = controller_module

    def _get_controller_config(self):
        """Get controller config object based on current type."""
        if self.controller_type == 'PID':
            return HardwareControllerConfig()
        else:
            return LQRControllerConfig()

    def _create_callbacks(self):
        """Create callback dictionary."""
        return {
            'mode_change': self.on_mode_change,
            'controller_type_change': self.on_controller_type_change,
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
            'slider_change': self.on_slider_change,
            'go_home': self.go_home,
            'connect': self.connect_serial,
            'disconnect': self.disconnect_serial,
        }

    def _create_plot(self, parent_layout):
        """Create PyQtGraph plot."""
        plot_group = QGroupBox("Ball Position (Top View)")
        plot_layout = QVBoxLayout()

        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(self.colors['widget_bg'])
        self.plot_widget.setMinimumSize(600, 600)

        plot_layout.addWidget(self.plot_widget)
        plot_group.setLayout(plot_layout)
        parent_layout.addWidget(plot_group)

        self.setup_plot()

    def setup_plot(self):
        """Setup PyQtGraph plot."""
        plot_item = self.plot_widget.getPlotItem()
        plot_item.setXRange(-180, 180)
        plot_item.setYRange(-180, 180)
        plot_item.setLabel('bottom', 'X (mm)', color=self.colors['fg'])
        plot_item.setLabel('left', 'Y (mm)', color=self.colors['fg'])
        plot_item.setTitle('Ball Position (Top View)', color=self.colors['fg'])
        plot_item.showGrid(x=True, y=True, alpha=0.2)
        plot_item.setAspectLocked(True)

        # Platform boundary
        self.platform_circle = pg.QtWidgets.QGraphicsEllipseItem(
            -PLATFORM_RADIUS_MM, -PLATFORM_RADIUS_MM,
            PLATFORM_RADIUS_MM * 2, PLATFORM_RADIUS_MM * 2
        )
        pen = pg.mkPen(color=self.colors['fg'], width=2, style=Qt.PenStyle.DashLine)
        self.platform_circle.setPen(pen)
        plot_item.addItem(self.platform_circle)

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

        # Ball trail (for hardware mode)
        self.ball_trail = plot_item.plot([], [], pen=pg.mkPen(color='#ff8888', width=2, style=Qt.PenStyle.DashLine))

        # Tilt arrow
        self.tilt_arrow = None

        # Setup plot update timer (10Hz fixed)
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start(int(1000 / PLOT_REFRESH_HZ))

    def update_plot(self):
        """Update plot with current state."""
        if not hasattr(self, 'ball_scatter') or self.ball_scatter is None:
            return

        try:
            if self.operation_mode == 'sim':
                ball_x = self.ball_pos[0, 0].item() * 1000
                ball_y = self.ball_pos[0, 1].item() * 1000
            else:
                ball_x = self.ball_pos_mm[0]
                ball_y = self.ball_pos_mm[1]

            self.ball_scatter.setData([ball_x], [ball_y])

            # Update ball trail in hardware mode
            if self.operation_mode == 'real' and len(self.ball_history_x) > 1:
                self.ball_trail.setData(self.ball_history_x, self.ball_history_y)
            else:
                self.ball_trail.setData([], [])

            # Update trajectory
            if self.pattern_type != 'static':
                period = getattr(self.current_pattern, 'period', 10.0)
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
            pass

    def _initialize_controller(self):
        """Initialize controller based on current type."""
        # Skip initialization for Manual mode
        if self.controller_type == 'Manual':
            self.controller = None
            return

        sliders = self.controller_widgets['sliders']
        scalar_vars = self.controller_widgets['scalar_vars']

        if self.controller_type == 'PID':
            if 'kp' in sliders:
                kp = self._get_controller_config().get_scaled_param('kp', sliders, scalar_vars)
                ki = self._get_controller_config().get_scaled_param('ki', sliders, scalar_vars)
                kd = self._get_controller_config().get_scaled_param('kd', sliders, scalar_vars)

                self.controller = PIDController(
                    kp=kp, ki=ki, kd=kd,
                    output_limit=15.0,
                    derivative_filter_alpha=0.1
                )
        else:  # LQR
            if 'Q_pos' in sliders:
                Q_pos = self._get_controller_config().get_scaled_param('Q_pos', sliders, scalar_vars)
                Q_vel = self._get_controller_config().get_scaled_param('Q_vel', sliders, scalar_vars)
                R = self._get_controller_config().get_scaled_param('R', sliders, scalar_vars)

                self.controller = LQRController(
                    Q_pos=Q_pos,
                    Q_vel=Q_vel,
                    R=R,
                    output_limit=15.0
                )

    def on_controller_param_change(self):
        """Update controller parameters."""
        if self.controller is None:
            return

        sliders = self.controller_widgets['sliders']
        scalar_vars = self.controller_widgets['scalar_vars']
        config = self._get_controller_config()

        if self.controller_type == 'PID':
            if 'kp' in sliders:
                kp = config.get_scaled_param('kp', sliders, scalar_vars)
                ki = config.get_scaled_param('ki', sliders, scalar_vars)
                kd = config.get_scaled_param('kd', sliders, scalar_vars)
                self.controller.set_gains(kp, ki, kd)
        else:  # LQR
            if 'Q_pos' in sliders:
                Q_pos = config.get_scaled_param('Q_pos', sliders, scalar_vars)
                Q_vel = config.get_scaled_param('Q_vel', sliders, scalar_vars)
                R = config.get_scaled_param('R', sliders, scalar_vars)
                # Reinitialize controller with new weights
                self.controller = LQRController(
                    Q_pos=Q_pos,
                    Q_vel=Q_vel,
                    R=R,
                    output_limit=15.0
                )

    def _cleanup_hardware_resources(self):
        """Clean up hardware-specific resources (threads, timers, etc.)."""
        # Stop control thread if running
        if hasattr(self, 'control_thread') and self.control_thread is not None:
            if self.control_thread.is_alive():
                self.simulation_running = False
                self.control_thread.join(timeout=2.0)
            self.control_thread = None
            self.control_thread_id = None

        # Stop GUI update timer for hardware mode
        if hasattr(self, 'gui_update_timer'):
            try:
                self.gui_update_timer.stop()
            except:
                pass

        # Re-enable garbage collection
        gc.enable()
        gc.collect()

        # Restore default timer resolution
        if hasattr(self, 'timer_manager'):
            self.timer_manager.restore_default()

    def on_mode_change(self, mode):
        """Handle mode change (sim/real)."""
        # Stop simulation if running
        was_running = self.simulation_running
        if self.simulation_running:
            self.stop_simulation()

        # Clean up hardware resources before switching
        self._cleanup_hardware_resources()

        # Force garbage collection to free resources
        gc.collect()

        # Process pending events multiple times to ensure cleanup completes
        for _ in range(3):
            QApplication.processEvents()
            time.sleep(0.02)

        self.operation_mode = mode
        self.setWindowTitle(f"Stewart Platform - Unified Controller ({mode.upper()})")

        # Show/hide serial connection based on mode
        if hasattr(self, 'serial_widget'):
            self.serial_widget.setVisible(mode == 'real')

        # Reset everything to clean state
        self._reset_state()

        # Final garbage collection
        gc.collect()

        # Update button states after reset
        if hasattr(self, 'mode_selector'):
            self.mode_selector.sim_btn.setChecked(mode == 'sim')
            self.mode_selector.real_btn.setChecked(mode == 'real')

    def on_controller_type_change(self, controller_type):
        """Handle controller type change (PID/LQR/Manual)."""
        # Stop simulation if running
        was_running = self.simulation_running
        if self.simulation_running:
            self.stop_simulation()

        # Clean up any lingering hardware resources
        self._cleanup_hardware_resources()

        # Explicitly delete old controller to free memory
        if hasattr(self, 'controller') and self.controller is not None:
            del self.controller
            self.controller = None

        # Force garbage collection
        gc.collect()

        # Process pending events to ensure full cleanup
        for _ in range(3):
            QApplication.processEvents()
            time.sleep(0.02)

        self.controller_type = controller_type

        # Rebuild controller panel (will be empty for Manual mode)
        self._rebuild_controller_panel()

        # Process events after GUI rebuild
        QApplication.processEvents()
        time.sleep(0.05)

        # Reinitialize controller (only for PID/LQR, not Manual)
        if controller_type != 'Manual':
            self._initialize_controller()

        # Reset everything to clean state
        self._reset_state()

        # Final garbage collection
        gc.collect()

        # Update button states after reset
        if hasattr(self, 'controller_selector'):
            self.controller_selector.pid_btn.setChecked(controller_type == 'PID')
            self.controller_selector.lqr_btn.setChecked(controller_type == 'LQR')
            self.controller_selector.manual_btn.setChecked(controller_type == 'Manual')

    def _reset_state(self):
        """Reset all simulation state to initial values."""
        # Reset time
        self.simulation_time = 0.0
        self.last_update_time = None

        # Reset servos
        for servo in self.servos:
            servo.reset()

        # Reset DOF values
        for dof in ['x', 'y', 'rx', 'ry', 'rz']:
            self.dof_values[dof] = 0.0
        self.dof_values['z'] = self.ik.home_height_top_surface

        # Reset ball state based on mode
        if self.operation_mode == 'sim':
            ball_start_height = (self.ik.home_height_top_surface / 1000) + self.ball_physics.radius
            self.ball_pos = torch.tensor([[0.0, 0.0, ball_start_height]], dtype=torch.float32)
            self.ball_vel = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
            self.ball_omega = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
            self.camera_enabled = True

            # IMPORTANT: Reset camera to clear cached measurements
            self.pixy_camera.reset()
        else:
            self.ball_pos_mm = (0.0, 0.0)
            self.ball_detected = False
            self.ball_history_x.clear()
            self.ball_history_y.clear()

        # Reset platform angular state
        self.prev_platform_angles = {'rx': 0.0, 'ry': 0.0}
        self.platform_angular_vel = {'rx': 0.0, 'ry': 0.0}
        self.platform_angular_accel = {'rx': 0.0, 'ry': 0.0}

        # Reset FK state
        home_translation = np.array([0.0, 0.0, self.ik.home_height_top_surface])
        home_rotation = np.array([0.0, 0.0, 0.0])
        self.last_fk_translation = home_translation
        self.last_fk_rotation = home_rotation

        # Reset controller
        if self.controller is not None:
            self.controller.reset()

        # Reset Kalman filter and rebuild matrices
        self.kalman_filter.dt = self.control_interval
        self.kalman_filter._build_system_matrices()
        if self.operation_mode == 'sim':
            self.kalman_filter.reset((0.0, 0.0))
        else:
            self.kalman_filter.reset(self.ball_pos_mm)

        # Reset pattern
        self.pattern_start_time = 0.0
        self.current_pattern.reset()

        # Reset servo commands
        self.last_sent_angles = None
        self.last_cmd_angles = np.zeros(6)

        # Disable controller
        self.controller_enabled = False

        # Update GUI elements
        if 'manual_pose' in self.gui_modules:
            manual_pose = self.gui_modules['manual_pose']
            for dof, value in self.dof_values.items():
                if dof in manual_pose.sliders:
                    res = self.dof_config[dof][2]
                    manual_pose.sliders[dof].blockSignals(True)
                    manual_pose.sliders[dof].setValue(int(value / res))
                    manual_pose.sliders[dof].blockSignals(False)
                    manual_pose.value_labels[dof].setText(f"{value:.2f}")
                    manual_pose.sliders[dof].setEnabled(True)

        # Update controller enable checkbox
        if hasattr(self, 'gui_modules') and 'controller' in self.gui_modules:
            try:
                controller_module = self.gui_modules['controller']
                if hasattr(controller_module, 'enable_checkbox'):
                    controller_module.enable_checkbox.blockSignals(True)
                    controller_module.enable_checkbox.setChecked(False)
                    controller_module.enable_checkbox.blockSignals(False)
            except (RuntimeError, AttributeError):
                pass

    def on_controller_toggle(self):
        """Toggle controller enable/disable."""
        self.controller_enabled = not self.controller_enabled

        if self.controller_enabled:
            # Reset controller and Kalman
            self.controller.reset()

            # Reset Kalman filter with current ball position
            if self.operation_mode == 'sim':
                ball_x_mm = self.ball_pos[0, 0].item() * 1000
                ball_y_mm = self.ball_pos[0, 1].item() * 1000
                self.kalman_filter.reset((ball_x_mm, ball_y_mm))
            else:
                self.kalman_filter.reset(self.ball_pos_mm)

            self.reset_pattern()

            # Disable manual sliders
            if 'manual_pose' in self.gui_modules:
                manual_pose = self.gui_modules['manual_pose']
                for slider in manual_pose.sliders.values():
                    slider.setEnabled(False)
        else:
            # Enable manual sliders
            if 'manual_pose' in self.gui_modules:
                manual_pose = self.gui_modules['manual_pose']
                for slider in manual_pose.sliders.values():
                    slider.setEnabled(True)

    def on_pattern_change(self, pattern_type):
        """Handle pattern change."""
        self.pattern_type = pattern_type
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

    def on_pattern_param_change(self, param_name, value):
        """Handle pattern parameter change."""
        self.pattern_params[param_name] = value
        pattern_type = self.pattern_type

        if pattern_type == 'circle':
            radius = self.pattern_params.get('radius', 50.0)
            period = self.pattern_params.get('period', 10.0)
            self.current_pattern = PatternFactory.create('circle', radius=radius, period=period, clockwise=True)
        elif pattern_type == 'figure8':
            width = self.pattern_params.get('width', 60.0)
            height = self.pattern_params.get('height', 40.0)
            period = self.pattern_params.get('period', 12.0)
            self.current_pattern = PatternFactory.create('figure8', width=width, height=height, period=period)
        elif pattern_type == 'star':
            radius = self.pattern_params.get('radius', 60.0)
            period = self.pattern_params.get('period', 15.0)
            self.current_pattern = PatternFactory.create('star', radius=radius, period=period)

        self.current_pattern.reset()

    def reset_pattern(self):
        """Reset pattern timing."""
        self.pattern_start_time = self.simulation_time
        self.current_pattern.reset()
        if self.controller_enabled:
            self.controller.reset()

    def reset_ball(self):
        """Reset ball to center."""
        if self.operation_mode == 'sim':
            ball_start_height = (self.ik.home_height_top_surface / 1000) + self.ball_physics.radius
            self.ball_pos = torch.tensor([[0.0, 0.0, ball_start_height]], dtype=torch.float32)
            self.ball_vel = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
            self.ball_omega = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        else:
            self.ball_pos_mm = (0.0, 0.0)
            self.ball_history_x.clear()
            self.ball_history_y.clear()

        if self.controller_enabled:
            self.controller.reset()

    def push_ball(self):
        """Push ball (sim mode only)."""
        if self.operation_mode == 'sim':
            vx = np.random.uniform(-0.1, 0.1)
            vy = np.random.uniform(-0.1, 0.1)
            self.ball_vel = torch.tensor([[vx, vy, 0.0]], dtype=torch.float32)

    def on_slider_change(self, dof, value):
        """Handle manual slider change."""
        self.dof_values[dof] = float(value)

        if self.update_timer is not None:
            self.update_timer.stop()
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.calculate_ik)
        self.update_timer.start(50)

    def go_home(self):
        """Return platform to home position."""
        if self.controller_enabled:
            return

        self.dof_values['x'] = 0.0
        self.dof_values['y'] = 0.0
        self.dof_values['z'] = self.ik.home_height_top_surface
        self.dof_values['rx'] = 0.0
        self.dof_values['ry'] = 0.0
        self.dof_values['rz'] = 0.0

        # Update sliders
        if 'manual_pose' in self.gui_modules:
            manual_pose = self.gui_modules['manual_pose']
            for dof, value in self.dof_values.items():
                if dof in manual_pose.sliders:
                    res = self.dof_config[dof][2]
                    manual_pose.sliders[dof].blockSignals(True)
                    manual_pose.sliders[dof].setValue(int(value / res))
                    manual_pose.sliders[dof].blockSignals(False)
                    manual_pose.value_labels[dof].setText(f"{value:.2f}")

        self.calculate_ik()

    def calculate_ik(self):
        """Calculate IK and send to hardware/simulation."""
        translation = np.array([self.dof_values['x'], self.dof_values['y'], self.dof_values['z']])
        rotation = np.array([self.dof_values['rx'], self.dof_values['ry'], self.dof_values['rz']])

        rx_limited, ry_limited, tilt_mag = clip_tilt_vector(rotation[0], rotation[1], MAX_TILT_ANGLE_DEG)
        rotation[0] = rx_limited
        rotation[1] = ry_limited

        angles = self.ik.calculate_servo_angles(translation, rotation, self.use_top_surface_offset)

        if angles is not None:
            self.last_cmd_angles = angles

            # Send to servos in simulation mode
            if self.operation_mode == 'sim' and self.simulation_running:
                for i, servo in enumerate(self.servos):
                    servo.send_command(angles[i], self.simulation_time)
            # Send to hardware in real mode
            elif self.operation_mode == 'real' and self.connected and not self.controller_enabled:
                self.serial_controller.send_servo_angles(angles)

                # Update FK to reflect commanded position in hardware mode
                fk_translation, fk_rotation, success, iterations = self.ik.calculate_forward_kinematics(
                    angles,
                    initial_guess=(self.last_fk_translation, self.last_fk_rotation),
                    use_top_surface_offset=self.use_top_surface_offset
                )

                if success:
                    self.last_fk_translation = fk_translation
                    self.last_fk_rotation = fk_rotation
                    # Update manual pose display to show commanded position
                    self._update_manual_pose_display(fk_translation, fk_rotation)

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
            time.sleep(0.5)
            self.serial_controller.set_servo_speed(0)
            time.sleep(0.1)
            self.serial_controller.set_servo_acceleration(0)
            time.sleep(0.2)

            success_timer, msg_timer = self.timer_manager.set_high_resolution()

            # Prewarm IK cache
            tilts = np.arange(-15, 16, 2)
            for rx in tilts:
                for ry in tilts:
                    translation = np.array([0.0, 0.0, self.ik.home_height_top_surface])
                    rotation = np.array([float(rx), float(ry), 0.0])
                    angles = self.ik.calculate_servo_angles(translation, rotation, self.use_top_surface_offset)
                    if angles is not None:
                        self.ik_cache.put(translation, rotation, angles)

            # Update GUI modules
            if 'simulation_control' in self.gui_modules:
                self.gui_modules['simulation_control'].start_btn.setEnabled(True)

            if 'serial_connection' in self.gui_modules:
                self.gui_modules['serial_connection'].update({'connected': True})
        else:
            QMessageBox.critical(self, "Error", message)

    def disconnect_serial(self):
        """Disconnect from hardware."""
        if self.simulation_running:
            self.stop_simulation()

        if self.serial_controller:
            self.serial_controller.disconnect()

        self.connected = False

        # Update GUI modules
        if 'simulation_control' in self.gui_modules:
            self.gui_modules['simulation_control'].start_btn.setEnabled(False)

        if 'serial_connection' in self.gui_modules:
            self.gui_modules['serial_connection'].update({'connected': False})

    def start_simulation(self):
        """Start simulation or hardware control."""
        if self.operation_mode == 'real' and not self.connected:
            QMessageBox.warning(self, "Warning", "Connect to hardware first")
            return

        self.simulation_running = True
        self.simulation_time = 0.0
        self.last_update_time = None

        if 'simulation_control' in self.gui_modules:
            self.gui_modules['simulation_control'].start_btn.setEnabled(False)
            self.gui_modules['simulation_control'].stop_btn.setEnabled(True)

        if self.operation_mode == 'sim':
            # Initialize servo positions to home
            self.calculate_ik()

            # Initialize FK with home position
            home_translation = np.array([0.0, 0.0, self.ik.home_height_top_surface])
            home_rotation = np.array([0.0, 0.0, 0.0])
            self.last_fk_translation = home_translation
            self.last_fk_rotation = home_rotation

            # Simulation mode: use Qt timer
            self.last_update_time = time.time()
            self.simulation_timer.start(self.update_rate_ms)
        else:
            # Hardware mode: use dedicated control thread
            gc.disable()
            self.control_thread = threading.Thread(target=self._control_thread_func, daemon=True)
            self.control_thread.start()

            self.control_thread_id = self.control_thread.ident
            self.priority_manager.set_thread_priority(self.control_thread_id, THREAD_PRIORITY_TIME_CRITICAL)

            # Start GUI update timer for hardware mode
            self.gui_update_timer = QTimer()
            self.gui_update_timer.timeout.connect(self.update_gui_modules)
            self.gui_update_timer.start(100)  # Update GUI at 10Hz

    def stop_simulation(self):
        """Stop simulation or hardware control."""
        self.simulation_running = False

        if self.operation_mode == 'sim':
            self.simulation_timer.stop()
        else:
            # Hardware mode: ensure full cleanup
            if hasattr(self, 'control_thread') and self.control_thread is not None:
                if self.control_thread.is_alive():
                    # Give thread time to see simulation_running = False
                    time.sleep(0.05)
                    self.control_thread.join(timeout=2.0)

            if hasattr(self, 'gui_update_timer'):
                try:
                    self.gui_update_timer.stop()
                except:
                    pass

            # Re-enable GC
            gc.enable()
            gc.collect()

        if 'simulation_control' in self.gui_modules:
            self.gui_modules['simulation_control'].start_btn.setEnabled(True)
            self.gui_modules['simulation_control'].stop_btn.setEnabled(False)

    def reset_simulation(self):
        """Reset simulation state."""
        was_running = self.simulation_running
        if was_running:
            self.stop_simulation()

        # Use centralized reset method
        self._reset_state()

        if was_running:
            self.start_simulation()

    def _update_manual_pose_display(self, translation, rotation):
        """Update manual pose sliders to show controller output (visual only)."""
        if 'manual_pose' not in self.gui_modules:
            return

        manual_pose = self.gui_modules['manual_pose']

        # Update DOF display values (don't trigger callbacks)
        dof_values_display = {
            'x': translation[0],
            'y': translation[1],
            'z': translation[2],
            'rx': rotation[0],
            'ry': rotation[1],
            'rz': rotation[2]
        }

        for dof, value in dof_values_display.items():
            if dof in manual_pose.sliders:
                res = self.dof_config[dof][2]
                # Block signals to prevent triggering slider change callbacks
                manual_pose.sliders[dof].blockSignals(True)
                manual_pose.sliders[dof].setValue(int(value / res))
                manual_pose.sliders[dof].blockSignals(False)
                manual_pose.value_labels[dof].setText(f"{value:.2f}")

    def update_gui_modules(self):
        """Update GUI modules with current state."""
        if 'simulation_control' in self.gui_modules:
            sim_ctrl = self.gui_modules['simulation_control']
            sim_ctrl.time_label.setText(f"Time: {format_time(self.simulation_time)}")

        # Update manual pose display if controller is enabled (both sim and hardware)
        if self.controller_enabled and hasattr(self, 'last_fk_rotation'):
            self._update_manual_pose_display(self.last_fk_translation, self.last_fk_rotation)

    def simulation_loop(self):
        """Main simulation loop (for sim mode)."""
        if not self.simulation_running:
            return

        current_time = time.time()
        if self.last_update_time is not None:
            dt = current_time - self.last_update_time
            dt = min(dt, 0.1)
            self.simulation_time += dt

            if self.controller_enabled:
                # Get ball position
                ball_x_mm_true = self.ball_pos[0, 0].item() * 1000
                ball_y_mm_true = self.ball_pos[0, 1].item() * 1000
                ball_vx_mm_s = self.ball_vel[0, 0].item() * 1000
                ball_vy_mm_s = self.ball_vel[0, 1].item() * 1000

                # Apply camera noise
                if self.camera_enabled:
                    measured_x, measured_y, detected, is_new = self.pixy_camera.measure(
                        (ball_x_mm_true, ball_y_mm_true),
                        self.simulation_time
                    )
                    if detected:
                        ball_x_mm = measured_x
                        ball_y_mm = measured_y
                    else:
                        ball_x_mm = ball_x_mm_true
                        ball_y_mm = ball_y_mm_true
                else:
                    ball_x_mm = ball_x_mm_true
                    ball_y_mm = ball_y_mm_true

                # Kalman filter prediction
                if self.kalman_enabled:
                    # Update Kalman dt to match actual simulation timestep
                    self.kalman_filter.set_dt(dt)
                    rx_deg = self.last_fk_rotation[0] if hasattr(self, 'last_fk_rotation') else self.dof_values['rx']
                    ry_deg = self.last_fk_rotation[1] if hasattr(self, 'last_fk_rotation') else self.dof_values['ry']
                    self.kalman_filter.predict([rx_deg, ry_deg])
                    self.kalman_filter.update((ball_x_mm, ball_y_mm), self.simulation_time)
                    filtered_x, filtered_y = self.kalman_filter.get_position_mm()
                    filtered_vx, filtered_vy = self.kalman_filter.get_velocity_mm_s()
                    ball_pos_mm = (filtered_x, filtered_y)
                    ball_vel_mm_s = (filtered_vx, filtered_vy)
                else:
                    ball_pos_mm = (ball_x_mm, ball_y_mm)
                    ball_vel_mm_s = (ball_vx_mm_s, ball_vy_mm_s)

                # Get target
                pattern_time = self.simulation_time - self.pattern_start_time
                target_x, target_y = self.current_pattern.get_position(pattern_time)

                # Update controller
                if self.controller_type == 'PID':
                    rx, ry = self.controller.update(ball_pos_mm, (target_x, target_y), dt)
                else:  # LQR
                    rx, ry = self.controller.update(ball_pos_mm, ball_vel_mm_s, (target_x, target_y))

                # Clip output
                rx, ry, _ = clip_tilt_vector(rx, ry, MAX_TILT_ANGLE_DEG)

                self.dof_values['rx'] = rx
                self.dof_values['ry'] = ry

                # Calculate IK
                translation = np.array([self.dof_values['x'], self.dof_values['y'], self.dof_values['z']])
                rotation = np.array([rx, ry, self.dof_values['rz']])
                angles = self.ik.calculate_servo_angles(translation, rotation, self.use_top_surface_offset)

                if angles is not None:
                    self.last_cmd_angles = angles
                    for i in range(6):
                        self.servos[i].send_command(angles[i], self.simulation_time)

            # Update servos
            for servo in self.servos:
                servo.update(dt, self.simulation_time)

            # FK
            actual_angles = np.array([servo.get_angle() for servo in self.servos])
            translation, rotation, success, iterations = self.ik.calculate_forward_kinematics(
                actual_angles,
                initial_guess=(self.last_fk_translation, self.last_fk_rotation) if hasattr(self, 'last_fk_translation') else None,
                use_top_surface_offset=self.use_top_surface_offset
            )

            if success:
                self.last_fk_translation = translation
                self.last_fk_rotation = rotation

                # Update ball physics
                platform_pose = torch.tensor([[
                    translation[0] / 1000, translation[1] / 1000, translation[2] / 1000,
                    rotation[0], rotation[1], rotation[2]
                ]], dtype=torch.float32)

                self.ball_pos, self.ball_vel, self.ball_omega, contact_info = \
                    self.ball_physics.step(
                        self.ball_pos, self.ball_vel, self.ball_omega, platform_pose, dt,
                        platform_angular_accel=self.platform_angular_accel
                    )

                # Update platform angular state
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

            # Update GUI
            self.update_gui_modules()

        self.last_update_time = current_time

    def _control_thread_func(self):
        """Dedicated high-frequency control thread (for hardware mode at 250Hz)."""
        while self.simulation_running:
            loop_start = time.perf_counter()

            # Read ball data
            ball_data = self.serial_controller.get_latest_ball_data()

            if ball_data is not None:
                self.last_ball_update = self.simulation_time

                pixy_x = ball_data['x']
                pixy_y = ball_data['y']

                CAMERA_HEIGHT_PIXELS = 208.0
                CAMERA_CENTER_X = 145.0
                CAMERA_CENTER_Y = 102.0

                ball_x_mm = (pixy_x - CAMERA_CENTER_X) * self.pixels_to_mm_x
                ball_y_mm = ((CAMERA_HEIGHT_PIXELS - pixy_y) - CAMERA_CENTER_Y) * self.pixels_to_mm_y

                self.ball_pos_mm = (ball_x_mm, ball_y_mm)
                self.ball_detected = ball_data['detected']

                # Update ball_pos for plotting
                self.ball_pos[0, 0] = ball_x_mm / 1000.0
                self.ball_pos[0, 1] = ball_y_mm / 1000.0

                if self.ball_detected:
                    self.ball_history_x.append(ball_x_mm)
                    self.ball_history_y.append(ball_y_mm)
                    if len(self.ball_history_x) > self.max_history:
                        self.ball_history_x.pop(0)
                        self.ball_history_y.pop(0)

            # Kalman prediction
            if self.kalman_enabled:
                # Update Kalman dt to match control loop rate (hardware runs at fixed rate)
                self.kalman_filter.set_dt(self.control_interval)
                rx_deg = self.last_fk_rotation[0] if hasattr(self, 'last_fk_rotation') else self.dof_values['rx']
                ry_deg = self.last_fk_rotation[1] if hasattr(self, 'last_fk_rotation') else self.dof_values['ry']
                self.kalman_filter.predict([rx_deg, ry_deg])

            # Controller update
            if self.controller_enabled and self.ball_detected:
                if self.kalman_enabled:
                    self.kalman_filter.update(self.ball_pos_mm, self.simulation_time)
                    filtered_x, filtered_y = self.kalman_filter.get_position_mm()
                    filtered_vx, filtered_vy = self.kalman_filter.get_velocity_mm_s()
                    ball_pos_mm = (filtered_x, filtered_y)
                    ball_vel_mm_s = (filtered_vx, filtered_vy)
                else:
                    ball_pos_mm = self.ball_pos_mm
                    ball_vel_mm_s = (0.0, 0.0)

                pattern_time = self.simulation_time - self.pattern_start_time
                target_x, target_y = self.current_pattern.get_position(pattern_time)

                if self.controller_type == 'PID':
                    rx, ry = self.controller.update(ball_pos_mm, (target_x, target_y), self.control_interval)
                else:  # LQR
                    rx, ry = self.controller.update(ball_pos_mm, ball_vel_mm_s, (target_x, target_y))

                self.dof_values['rx'] = rx
                self.dof_values['ry'] = ry

                # IK with cache
                self._translation_buffer[0] = self.dof_values['x']
                self._translation_buffer[1] = self.dof_values['y']
                self._translation_buffer[2] = self.dof_values['z']
                self._rotation_buffer[0] = rx
                self._rotation_buffer[1] = ry
                self._rotation_buffer[2] = self.dof_values['rz']

                angles = self.ik_cache.get(self._translation_buffer, self._rotation_buffer)

                if angles is None:
                    angles = self.ik.calculate_servo_angles(
                        self._translation_buffer,
                        self._rotation_buffer,
                        self.use_top_surface_offset
                    )
                    if angles is not None:
                        self.ik_cache.put(self._translation_buffer, self._rotation_buffer, angles)

                if angles is not None:
                    if (self.last_sent_angles is None or
                            not np.allclose(angles, self.last_sent_angles, atol=self.angle_change_threshold)):
                        success = self.serial_controller.send_servo_angles(angles)
                        if success:
                            self.last_sent_angles = angles.copy()

                    # Update FK to reflect commanded platform state for display
                    fk_translation, fk_rotation, success, iterations = self.ik.calculate_forward_kinematics(
                        angles,
                        initial_guess=(self.last_fk_translation, self.last_fk_rotation),
                        use_top_surface_offset=self.use_top_surface_offset
                    )

                    if success:
                        self.last_fk_translation = fk_translation
                        self.last_fk_rotation = fk_rotation

            self.simulation_time += self.control_interval

            # Sleep to maintain frequency
            elapsed = time.perf_counter() - loop_start
            sleep_time = self.control_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def closeEvent(self, event):
        """Clean shutdown."""
        # Stop any running simulation
        if self.simulation_running:
            self.stop_simulation()

        # Clean up all timers
        if hasattr(self, 'simulation_timer'):
            self.simulation_timer.stop()

        if hasattr(self, 'plot_timer'):
            self.plot_timer.stop()

        # Clean up hardware resources
        self._cleanup_hardware_resources()

        # Disconnect serial if connected
        if self.connected:
            self.disconnect_serial()

        event.accept()


def main():
    """Launch unified controller."""
    app = QApplication(sys.argv)
    controller = UnifiedStewartController(app)
    controller.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
