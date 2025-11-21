"""
Ball Detection Module for Stewart Platform Ball Tracking

High-speed ball detection using hybrid ROI extraction + CNN refinement.
Two-stage pipeline: HSV color filtering -> MobileNetV3 CNN on 128x128 crop.
"""

__version__ = "1.0.0"

# Lazy imports to avoid loading onnxruntime until needed
# Import detector directly when needed:
#   from ball_detection.detector import BallDetector

__all__ = ['BallDetector']
