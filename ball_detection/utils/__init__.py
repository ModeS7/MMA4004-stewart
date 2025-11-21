"""Camera and shared utilities for ball detection system."""

from .camera import (
    create_camera_capture,
    create_dual_camera_capture,
    apply_camera_settings,
    get_camera_info,
    DEFAULT_CAMERA_SETTINGS
)

__all__ = [
    'create_camera_capture',
    'create_dual_camera_capture',
    'apply_camera_settings',
    'get_camera_info',
    'DEFAULT_CAMERA_SETTINGS'
]
