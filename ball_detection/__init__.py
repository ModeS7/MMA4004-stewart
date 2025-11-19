"""
Ball Detection Module for Dual Camera 3D Tracking

High-speed ball detection using hybrid ROI extraction + CNN refinement.
Optimized for AMD Ryzen 7 5700U with DirectML GPU acceleration.
"""

__version__ = "1.0.0"

from .model import BallDetectorCNN
from .detector import BallDetector

__all__ = ['BallDetectorCNN', 'BallDetector']
