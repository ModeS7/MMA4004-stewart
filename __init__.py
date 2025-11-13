"""
Stewart Platform Control System

A comprehensive control system for Stewart Platform (6-DOF parallel manipulator)
with ball-on-plate balancing, supporting both simulation and hardware operation.

Main Controllers:
    StewartController: Full-featured controller with complete GUI and data collection
    MinimalController: Lightweight controller for rapid development and testing

Packages:
    core: Core algorithms (IK/FK, physics, control algorithms, utilities)
    gui: Modular GUI components and builder system
    setup: Base classes for simulation and hardware controllers

Usage:
    # Run full controller
    from full_c import main as run_full_controller
    run_full_controller()

    # Run minimal controller
    from min_c import main as run_minimal_controller
    run_minimal_controller()

    # Import specific components
    from core import StewartPlatformIK, PIDController, LQRController
    from gui import GUIBuilder, create_standard_layout
    from setup import BaseStewartSimulator, HardwareControllerBase
"""

# Version information
__version__ = '2.0.0'
__author__ = 'NTNU MMA4004 Project'

# Main controller classes
from full_c import StewartController
from min_c import MinimalController

# Core algorithms and components
from core import (
    # Kinematics and physics
    StewartPlatformIK,
    FirstOrderServo,
    SimpleBallPhysics2D,

    # Trajectory patterns
    PatternFactory,

    # Camera
    Pixy2Camera,

    # Control algorithms
    PIDController,
    LQRController,
    KalmanFilter,
    IMUControllerMixin,
    clip_tilt_vector,

    # Platform constants
    MAX_TILT_ANGLE_DEG,
    MAX_CONTROLLER_OUTPUT_DEG,
    MAX_IMU_CORRECTION_DEG,
    MAX_SERVO_ANGLE_DEG,
    PLATFORM_RADIUS_MM,

    # Configuration classes
    StewartPlatformConfig,
    ColorScheme,
    ControlLoopConfig,
    GUIConfig,
    SimulationConfig,
    PIDConfig,
    LQRConfig,
    KalmanFilterConfig,
    Pixy2CameraConfig,
    BallPhysicsConfig,
    SerialConfig,
    PerformanceConfig,
    IMUKalmanConfig,
    IMUCalibrationConfig,
    HardwareConnectionConfig,
    VisualizationConfig,
    IKZOptimizationConfig,
    TrajectoryPatternConfig,
    ManualPoseControlConfig,
    BallControlConfig,
)

# GUI components
from gui import (
    # Builder system
    GUIBuilder,
    create_standard_layout,

    # Base module
    GUIModule,

    # Control modules
    SimulationControlModule,
    ControllerModule,
    TrajectoryPatternModule,
    BallControlModule,
    ManualPoseControlModule,

    # Display modules
    BallStateModule,
    ServoAnglesModule,
    PlatformPoseModule,
    ControllerOutputModule,
    DebugLogModule,

    # Hardware modules
    SerialConnectionModule,
    PerformanceStatsModule,
    ControlFrequencyModule,

    # Configuration modules
    ConfigurationModule,
    ModeSelectionModule,
    ControllerSelectionModule,
    PlotControlModule,

    # Sensor modules
    Pixy2CameraModule,
    KalmanFilterModule,
    IMUKalmanParametersModule,

    # Optimization modules
    IKZOptimizationModule,

    # Data collection
    PerformanceDataCollectionModule,
)

# Base classes for building controllers
from setup import (
    # Simulator base
    BaseStewartSimulator,
    ControllerConfig,

    # Hardware base
    HardwareControllerBase,
    SerialController,
    WindowsTimerManager,
    ThreadPriorityManager,
    IKCache,
    HardwareControllerConfig,
    LQRControllerConfig,
)

__all__ = [
    # Version
    '__version__',
    '__author__',

    # Main controllers
    'StewartController',
    'MinimalController',

    # Core - Kinematics and physics
    'StewartPlatformIK',
    'FirstOrderServo',
    'SimpleBallPhysics2D',
    'PatternFactory',
    'Pixy2Camera',

    # Core - Control algorithms
    'PIDController',
    'LQRController',
    'KalmanFilter',
    'IMUControllerMixin',
    'clip_tilt_vector',

    # Core - Constants
    'MAX_TILT_ANGLE_DEG',
    'MAX_CONTROLLER_OUTPUT_DEG',
    'MAX_IMU_CORRECTION_DEG',
    'MAX_SERVO_ANGLE_DEG',
    'PLATFORM_RADIUS_MM',

    # Core - Configuration
    'StewartPlatformConfig',
    'ColorScheme',
    'ControlLoopConfig',
    'GUIConfig',
    'SimulationConfig',
    'PIDConfig',
    'LQRConfig',
    'KalmanFilterConfig',
    'Pixy2CameraConfig',
    'BallPhysicsConfig',
    'SerialConfig',
    'PerformanceConfig',
    'IMUKalmanConfig',
    'IMUCalibrationConfig',
    'HardwareConnectionConfig',
    'VisualizationConfig',
    'IKZOptimizationConfig',
    'TrajectoryPatternConfig',
    'ManualPoseControlConfig',
    'BallControlConfig',

    # GUI - Builder
    'GUIBuilder',
    'create_standard_layout',
    'GUIModule',

    # GUI - Control modules
    'SimulationControlModule',
    'ControllerModule',
    'TrajectoryPatternModule',
    'BallControlModule',
    'ManualPoseControlModule',

    # GUI - Display modules
    'BallStateModule',
    'ServoAnglesModule',
    'PlatformPoseModule',
    'ControllerOutputModule',
    'DebugLogModule',

    # GUI - Hardware modules
    'SerialConnectionModule',
    'PerformanceStatsModule',
    'ControlFrequencyModule',

    # GUI - Configuration modules
    'ConfigurationModule',
    'ModeSelectionModule',
    'ControllerSelectionModule',
    'PlotControlModule',

    # GUI - Sensor modules
    'Pixy2CameraModule',
    'KalmanFilterModule',
    'IMUKalmanParametersModule',

    # GUI - Optimization
    'IKZOptimizationModule',

    # GUI - Data collection
    'PerformanceDataCollectionModule',

    # Setup - Base classes
    'BaseStewartSimulator',
    'ControllerConfig',
    'HardwareControllerBase',
    'SerialController',
    'WindowsTimerManager',
    'ThreadPriorityManager',
    'IKCache',
    'HardwareControllerConfig',
    'LQRControllerConfig',
]
