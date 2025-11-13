"""
Base classes for Stewart Platform controllers.

This package provides base classes for building Stewart Platform control systems
in both simulation and hardware modes.

Modules:
    base_simulator: Base class for simulation controllers with GUI
    base_hardware: Base class for hardware controllers with serial communication
"""

# Base simulator classes
from .base_simulator import (
    BaseStewartSimulator,
    ControllerConfig
)

# Hardware controller classes
from .base_hardware import (
    HardwareControllerBase,
    SerialController,
    WindowsTimerManager,
    ThreadPriorityManager,
    IKCache,
    HardwareControllerConfig,
    LQRControllerConfig
)

__all__ = [
    # Simulator base
    'BaseStewartSimulator',
    'ControllerConfig',

    # Hardware base
    'HardwareControllerBase',
    'SerialController',
    'WindowsTimerManager',
    'ThreadPriorityManager',
    'IKCache',
    'HardwareControllerConfig',
    'LQRControllerConfig',
]
