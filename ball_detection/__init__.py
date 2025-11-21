"""
Ball Detection System

High-level API for ball center detection using CNN + stereo triangulation.

Main Components:
- core: Detection pipeline (detector, roi_extractor, onnx_inference, model, dataset)
- training: Training infrastructure (train, export_onnx)
- apps: End-user applications (stereo_tracker, test_video)
- calibration: Camera calibration tools
- tuning: Interactive parameter tuning
- utils: Shared utilities (camera initialization, config)
- tools: Data collection and analysis
- integration: System integration (camera_controller for Stewart platform)
"""

# Main API exports for backward compatibility
from .core.detector import BallDetector

__version__ = "1.0.0"

__all__ = ['BallDetector']
