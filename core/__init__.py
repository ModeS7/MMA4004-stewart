"""
Core Stewart Platform functionality.

This package contains the core algorithms and utilities for the Stewart Platform
control system, including inverse/forward kinematics, control algorithms, and
configuration management.

Modules:
    core: Stewart Platform IK/FK, physics simulation, trajectory patterns
    control_core: PID/LQR controllers, Kalman filters, IMU integration
    utils: System configuration, constants, and utility functions
"""

# Core algorithms
from .core import (
    StewartPlatformIK,
    FirstOrderServo,
    SimpleBallPhysics2D,
    PatternFactory,
    Pixy2Camera
)

# Control algorithms
from .control_core import (
    PIDController,
    LQRController,
    KalmanFilter,
    IMUControllerMixin,
    clip_tilt_vector
)

# Configuration and utilities
from .utils import (
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

    # Utility functions
    get_controller_defaults,
    format_vector_2d,
    format_time,
    format_error_context
)

__all__ = [
    # Core
    'StewartPlatformIK',
    'FirstOrderServo',
    'SimpleBallPhysics2D',
    'PatternFactory',
    'Pixy2Camera',

    # Control
    'PIDController',
    'LQRController',
    'KalmanFilter',
    'IMUControllerMixin',
    'clip_tilt_vector',

    # Constants
    'MAX_TILT_ANGLE_DEG',
    'MAX_CONTROLLER_OUTPUT_DEG',
    'MAX_IMU_CORRECTION_DEG',
    'MAX_SERVO_ANGLE_DEG',
    'PLATFORM_RADIUS_MM',

    # Config classes
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

    # Utilities
    'get_controller_defaults',
    'format_vector_2d',
    'format_time',
    'format_error_context',
]
