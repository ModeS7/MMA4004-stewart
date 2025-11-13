"""
GUI components for Stewart Platform controllers.

This package provides a modular GUI system with reusable components for
building Stewart Platform control interfaces.

Modules:
    gui_builder: GUI construction system and layout management
    gui_modules: Modular widget components for different control aspects
"""

# GUI builder and layout
from .gui_builder import (
    GUIBuilder,
    create_standard_layout
)

# GUI modules (all available widgets)
from .gui_modules import (
    # Base
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
    IMUMotionDetectionModule,

    # Optimization modules
    IKZOptimizationModule,

    # Data collection
    PerformanceDataCollectionModule,
)

__all__ = [
    # Builder
    'GUIBuilder',
    'create_standard_layout',

    # Base
    'GUIModule',

    # Control
    'SimulationControlModule',
    'ControllerModule',
    'TrajectoryPatternModule',
    'BallControlModule',
    'ManualPoseControlModule',

    # Display
    'BallStateModule',
    'ServoAnglesModule',
    'PlatformPoseModule',
    'ControllerOutputModule',
    'DebugLogModule',

    # Hardware
    'SerialConnectionModule',
    'PerformanceStatsModule',
    'ControlFrequencyModule',

    # Configuration
    'ConfigurationModule',
    'ModeSelectionModule',
    'ControllerSelectionModule',
    'PlotControlModule',

    # Sensors
    'Pixy2CameraModule',
    'KalmanFilterModule',
    'IMUKalmanParametersModule',
    'IMUMotionDetectionModule',

    # Optimization
    'IKZOptimizationModule',

    # Data
    'PerformanceDataCollectionModule',
]
