"""
Main Ball Detector - Hybrid ROI + CNN Pipeline

Combines fast ROI extraction with CNN refinement for high-speed, accurate ball detection.
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple
from pathlib import Path

from .roi_extractor import RedBallROIExtractor
from .onnx_inference import ONNXBallDetector, ONNXStereoDetector


class BallDetector:
    """
    High-speed ball detector combining ROI extraction + CNN.

    Pipeline:
    1. HSV color filtering → quick ROI extraction (~1ms)
    2. CNN inference on crop → sub-pixel refinement (~3-5ms on AMD GPU)

    Total: ~5-6ms per image on Ryzen 7 5700U
    """

    def __init__(self, onnx_model_path, use_gpu=True, crop_size=128, confidence_threshold=0.5):
        """
        Initialize detector.

        Args:
            onnx_model_path: Path to trained ONNX model
            use_gpu: Whether to use DirectML GPU acceleration
            crop_size: Size of crop for CNN
            confidence_threshold: Minimum confidence to accept detection
        """
        self.crop_size = crop_size
        self.confidence_threshold = confidence_threshold

        # Initialize ROI extractor (CPU, fast)
        print("Initializing ROI extractor...")
        self.roi_extractor = RedBallROIExtractor(crop_size=crop_size)

        # Initialize CNN detector (GPU via DirectML)
        print("Initializing CNN detector...")
        self.cnn_detector = ONNXBallDetector(
            model_path=onnx_model_path,
            use_gpu=use_gpu,
            image_size=crop_size
        )

        print("Ball detector ready!\n")

    def detect(self, frame) -> Optional[Tuple[float, float, float]]:
        """
        Detect ball in frame.

        Args:
            frame: Input BGR image

        Returns:
            (x, y, confidence) if detected, else None
                - x, y: Ball center coordinates in frame
                - confidence: Detection confidence [0, 1]
        """
        # Stage 1: Fast ROI extraction
        crop, roi_center, crop_offset = self.roi_extractor.extract_roi(frame)

        if crop is None:
            return None  # No red region detected

        # Stage 2: CNN refinement
        x_norm, y_norm, confidence = self.cnn_detector.detect(crop)

        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return None

        # Convert normalized coordinates to original frame coordinates
        x_offset, y_offset = crop_offset
        x_abs = x_offset + x_norm * self.crop_size
        y_abs = y_offset + y_norm * self.crop_size

        return (x_abs, y_abs, confidence)

    def detect_with_timing(self, frame) -> Tuple[Optional[Tuple[float, float, float]], dict]:
        """
        Detect ball with detailed timing breakdown.

        Returns:
            result: (x, y, confidence) or None
            timing: dict with 'roi_ms', 'preprocess_ms', 'inference_ms', 'total_ms'
        """
        t_start = time.perf_counter()

        # Stage 1: Fast ROI extraction
        t0 = time.perf_counter()
        crop, roi_center, crop_offset = self.roi_extractor.extract_roi(frame)
        t_roi = time.perf_counter()

        timing = {
            'roi_ms': (t_roi - t0) * 1000,
            'preprocess_ms': 0.0,
            'inference_ms': 0.0,
            'total_ms': 0.0
        }

        if crop is None:
            timing['total_ms'] = (time.perf_counter() - t_start) * 1000
            return None, timing

        # Stage 2: CNN refinement with timing
        (x_norm, y_norm, confidence), cnn_timing = self.cnn_detector.detect_with_timing(crop)
        timing['preprocess_ms'] = cnn_timing['preprocess_ms']
        timing['inference_ms'] = cnn_timing['inference_ms']

        # Check confidence threshold
        if confidence < self.confidence_threshold:
            timing['total_ms'] = (time.perf_counter() - t_start) * 1000
            return None, timing

        # Convert normalized coordinates to original frame coordinates
        x_offset, y_offset = crop_offset
        x_abs = x_offset + x_norm * self.crop_size
        y_abs = y_offset + y_norm * self.crop_size

        timing['total_ms'] = (time.perf_counter() - t_start) * 1000

        return (x_abs, y_abs, confidence), timing

    def detect_dual_camera(self, frame1, frame2) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """
        Detect ball in both camera frames.

        Optimized: batches both crops together for single CNN inference.

        Args:
            frame1: Frame from camera 1 (BGR)
            frame2: Frame from camera 2 (BGR)

        Returns:
            result1, result2: Detection results (x, y, confidence) or None for each camera
        """
        # Stage 1: ROI extraction for both cameras
        crop1, center1, offset1 = self.roi_extractor.extract_roi(frame1)
        crop2, center2, offset2 = self.roi_extractor.extract_roi(frame2)

        result1 = None
        result2 = None

        # Stage 2: Batched CNN inference
        crops = []
        valid_indices = []

        if crop1 is not None:
            crops.append(crop1)
            valid_indices.append((0, offset1))

        if crop2 is not None:
            crops.append(crop2)
            valid_indices.append((1, offset2))

        if not crops:
            return None, None

        # Batch inference
        detections = self.cnn_detector.detect_batch(crops)

        # Parse results
        for i, (cam_idx, offset) in enumerate(valid_indices):
            x_norm, y_norm, confidence = detections[i]

            if confidence >= self.confidence_threshold:
                x_offset, y_offset = offset
                x_abs = x_offset + x_norm * self.crop_size
                y_abs = y_offset + y_norm * self.crop_size

                if cam_idx == 0:
                    result1 = (x_abs, y_abs, confidence)
                else:
                    result2 = (x_abs, y_abs, confidence)

        return result1, result2

    def visualize(self, frame, detection_result=None):
        """
        Create visualization of detection.

        Args:
            frame: Input frame
            detection_result: Output from detect() method

        Returns:
            Visualization frame
        """
        vis = frame.copy()

        if detection_result is not None:
            x, y, conf = detection_result

            # Draw detection circle
            cv2.circle(vis, (int(x), int(y)), 8, (0, 255, 0), 2)
            cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

            # Draw crosshair
            cv2.line(vis, (int(x) - 20, int(y)), (int(x) + 20, int(y)), (0, 255, 0), 1)
            cv2.line(vis, (int(x), int(y) - 20), (int(x), int(y) + 20), (0, 255, 0), 1)

        return vis

    def get_statistics(self):
        """Get detection statistics."""
        stats = self.cnn_detector.get_statistics()
        return {
            'cnn_inferences': stats['count'],
            'cnn_avg_time_ms': stats['avg_time_ms'],
            'cnn_fps': stats['fps'],
            'using_gpu': stats['using_gpu']
        }

    def set_hsv_ranges(self, lower1, upper1, lower2, upper2):
        """Set HSV color ranges for ROI extraction (for tuning)."""
        self.roi_extractor.set_hsv_ranges(lower1, upper1, lower2, upper2)


class StereoBallDetector:
    """
    Stereo ball detector using two-stage neural network pipeline.

    Pipeline:
        Stage 1: tiny_stereo (320x180, 6ch) → coarse stereo detection
        Stage 2: crop model (128x128) → per-image refinement (optional)

    Replaces ROI + CNN pipeline with end-to-end neural network detection.
    """

    def __init__(self, stereo_model_path, crop_model_path=None, use_gpu=True,
                 confidence_threshold=0.5, use_refinement=True, convert_to_rgb=False):
        """
        Initialize stereo detector.

        Args:
            stereo_model_path: Path to tiny_stereo ONNX model
            crop_model_path: Path to 128x128 crop model (optional)
            use_gpu: Use DirectML GPU acceleration
            confidence_threshold: Minimum confidence for detection
            use_refinement: Enable stage 2 crop refinement
            convert_to_rgb: Convert BGR to RGB (True if model trained on RGB)
        """
        print("Initializing Stereo Ball Detector...")
        self.stereo_detector = ONNXStereoDetector(
            stereo_model_path=stereo_model_path,
            crop_model_path=crop_model_path,
            use_gpu=use_gpu,
            confidence_threshold=confidence_threshold,
            use_refinement=use_refinement,
            convert_to_rgb=convert_to_rgb
        )
        print("Stereo detector ready!\n")

    def detect_stereo(self, left_frame, right_frame):
        """
        Detect ball in stereo pair.

        Args:
            left_frame: Left camera frame (BGR)
            right_frame: Right camera frame (BGR)

        Returns:
            dict with:
                x_left, y_left: Left camera coords (pixels) or None
                x_right, y_right: Right camera coords (pixels) or None
                confidence: Detection confidence
                detected: Whether ball was detected
                timing: Timing breakdown (ms)
        """
        return self.stereo_detector.detect(left_frame, right_frame)

    def visualize(self, left_frame, right_frame, result):
        """
        Create visualization of stereo detection.

        Args:
            left_frame, right_frame: Input frames
            result: Output from detect_stereo()

        Returns:
            Tuple of (vis_left, vis_right) visualization frames
        """
        vis_left = left_frame.copy()
        vis_right = right_frame.copy()

        if result['detected']:
            # Draw on left frame
            x, y = int(result['x_left']), int(result['y_left'])
            cv2.circle(vis_left, (x, y), 8, (0, 255, 0), 2)
            cv2.circle(vis_left, (x, y), 2, (0, 255, 0), -1)
            cv2.line(vis_left, (x - 20, y), (x + 20, y), (0, 255, 0), 1)
            cv2.line(vis_left, (x, y - 20), (x, y + 20), (0, 255, 0), 1)

            # Draw on right frame
            x, y = int(result['x_right']), int(result['y_right'])
            cv2.circle(vis_right, (x, y), 8, (0, 255, 0), 2)
            cv2.circle(vis_right, (x, y), 2, (0, 255, 0), -1)
            cv2.line(vis_right, (x - 20, y), (x + 20, y), (0, 255, 0), 1)
            cv2.line(vis_right, (x, y - 20), (x, y + 20), (0, 255, 0), 1)

        return vis_left, vis_right

    def get_statistics(self):
        """Get detection statistics."""
        return self.stereo_detector.get_statistics()


def test_detector_webcam(model_path, camera_id=0, use_gpu=True):
    """
    Test detector with webcam.

    Args:
        model_path: Path to ONNX model
        camera_id: Camera device ID
        use_gpu: Whether to use GPU acceleration
    """
    print("Testing Ball Detector with Webcam")
    print("Press 'q' to quit\n")

    # Create detector
    detector = BallDetector(
        onnx_model_path=model_path,
        use_gpu=use_gpu,
        crop_size=128,
        confidence_threshold=0.5
    )

    # Open camera with DirectShow backend (Windows, reduces tearing)
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

    # Reduce buffer size and set resolution
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # FPS calculation
    import time
    fps_start = time.time()
    frame_count = 0

    while True:
        # Flush old buffered frames
        cap.grab()

        ret, frame = cap.read()
        if not ret:
            break

        # Detect ball
        result = detector.detect(frame)

        # Visualize
        vis = detector.visualize(frame, result)

        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start)
            fps_start = time.time()

            cv2.putText(vis, f"FPS: {fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show
        cv2.imshow('Ball Detector Test', vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print final statistics
    stats = detector.get_statistics()
    print("\nDetection Statistics:")
    print(f"  CNN inferences: {stats['cnn_inferences']}")
    print(f"  CNN avg time: {stats['cnn_avg_time_ms']:.2f} ms")
    print(f"  CNN FPS: {stats['cnn_fps']:.1f}")
    print(f"  Using GPU: {stats['using_gpu']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test Ball Detector')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to ONNX model')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration')

    args = parser.parse_args()

    test_detector_webcam(
        model_path=args.model,
        camera_id=args.camera,
        use_gpu=not args.no_gpu
    )
