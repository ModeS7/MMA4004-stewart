"""
ONNX Runtime Inference with DirectML Support

High-speed inference using ONNX Runtime with AMD GPU acceleration via DirectML.
Optimized for Ryzen 7 5700U with Radeon Vega 8 graphics.
"""

import numpy as np
import onnxruntime as ort
import cv2
from pathlib import Path
from typing import Optional, Tuple
import time


class ONNXBallDetector:
    """
    ONNX-based ball detector with DirectML GPU acceleration.

    Provides high-speed inference (~3-5ms on AMD Vega 8) with sub-pixel accuracy.
    """

    def __init__(self, model_path, use_gpu=True, image_size=128):
        """
        Initialize ONNX detector.

        Args:
            model_path: Path to ONNX model file
            use_gpu: Whether to use DirectML GPU acceleration
            image_size: Expected input image size
        """
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.use_gpu = use_gpu

        # Setup execution providers
        if use_gpu:
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            print("ONNX Runtime: Using DirectML (AMD GPU) + CPU fallback")
        else:
            providers = ['CPUExecutionProvider']
            print("ONNX Runtime: Using CPU only")

        # Create inference session
        print(f"Loading ONNX model: {self.model_path}")
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=session_options,
                providers=providers
            )
        except Exception as e:
            print(f"Error loading model with DirectML, falling back to CPU: {e}")
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=session_options,
                providers=['CPUExecutionProvider']
            )
            self.use_gpu = False

        # Get model IO info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"Model loaded successfully")
        print(f"  Input: {self.input_name}")
        print(f"  Output: {self.output_name}")
        print(f"  Expected input shape: (batch, 3, {image_size}, {image_size})")

        # Normalization parameters (BGR order - model trained on BGR)
        self.mean = np.array([0.406, 0.456, 0.485], dtype=np.float32)
        self.std = np.array([0.225, 0.224, 0.229], dtype=np.float32)

        # Statistics
        self.inference_count = 0
        self.total_inference_time = 0.0

    def preprocess(self, image_bgr):
        """
        Preprocess image for model input.

        Args:
            image_bgr: BGR image (H, W, 3), uint8, range [0, 255]

        Returns:
            Preprocessed tensor (1, 3, H, W), float32
        """
        # Resize if needed
        if image_bgr.shape[:2] != (self.image_size, self.image_size):
            image_bgr = cv2.resize(image_bgr, (self.image_size, self.image_size))

        # Convert to float [0, 1]
        image_float = image_bgr.astype(np.float32) / 255.0

        # Normalize
        image_float = (image_float - self.mean) / self.std

        # Transpose to (C, H, W) and add batch dimension
        image_tensor = np.transpose(image_float, (2, 0, 1))
        image_tensor = np.expand_dims(image_tensor, axis=0)

        return image_tensor

    def detect(self, image_bgr) -> Tuple[float, float, float]:
        """
        Detect ball center in image.

        Args:
            image_bgr: BGR image crop (H, W, 3), uint8

        Returns:
            x_normalized: X coordinate normalized to [0, 1]
            y_normalized: Y coordinate normalized to [0, 1]
            confidence: Detection confidence [0, 1]
        """
        # Preprocess
        input_tensor = self.preprocess(image_rgb)

        # Inference
        start_time = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        inference_time = time.time() - start_time

        # Update statistics
        self.inference_count += 1
        self.total_inference_time += inference_time

        # Parse output (handle both 2-output and 3-output models)
        output_data = outputs[0][0]

        if len(output_data) == 2:
            # Model outputs only (x, y)
            x_norm, y_norm = output_data
            confidence = 1.0  # No confidence score
        else:
            # Model outputs (x, y, confidence)
            x_norm, y_norm, confidence = output_data

        return float(x_norm), float(y_norm), float(confidence)

    def detect_with_timing(self, image_rgb) -> Tuple[Tuple[float, float, float], dict]:
        """
        Detect ball center with detailed timing breakdown.

        Returns:
            result: (x_norm, y_norm, confidence)
            timing: dict with 'preprocess_ms' and 'inference_ms'
        """
        # Preprocess with timing
        t0 = time.perf_counter()
        input_tensor = self.preprocess(image_rgb)
        t1 = time.perf_counter()

        # Inference with timing
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        t2 = time.perf_counter()

        # Update statistics
        inference_time = t2 - t1
        self.inference_count += 1
        self.total_inference_time += inference_time

        # Parse output
        output_data = outputs[0][0]
        if len(output_data) == 2:
            x_norm, y_norm = output_data
            confidence = 1.0
        else:
            x_norm, y_norm, confidence = output_data

        timing = {
            'preprocess_ms': (t1 - t0) * 1000,
            'inference_ms': (t2 - t1) * 1000
        }

        return (float(x_norm), float(y_norm), float(confidence)), timing

    def detect_batch(self, images_rgb) -> np.ndarray:
        """
        Detect ball centers in batch of images.

        Args:
            images_rgb: List or array of RGB images

        Returns:
            Array of shape (N, 3) containing (x, y, confidence) for each image
        """
        # Preprocess all images
        batch = []
        for img in images_rgb:
            tensor = self.preprocess(img)
            batch.append(tensor)

        batch_tensor = np.concatenate(batch, axis=0)

        # Batch inference
        start_time = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: batch_tensor})
        inference_time = time.time() - start_time

        # Update statistics
        self.inference_count += len(images_rgb)
        self.total_inference_time += inference_time

        # Handle both 2-output and 3-output models
        raw_outputs = outputs[0]

        if raw_outputs.shape[1] == 2:
            # Model outputs only (x, y), add confidence column
            confidence_col = np.ones((raw_outputs.shape[0], 1), dtype=raw_outputs.dtype)
            return np.concatenate([raw_outputs, confidence_col], axis=1)
        else:
            # Model outputs (x, y, confidence)
            return raw_outputs

    def get_statistics(self):
        """Get inference statistics."""
        if self.inference_count == 0:
            return {
                'count': 0,
                'avg_time_ms': 0.0,
                'fps': 0.0,
                'using_gpu': self.use_gpu
            }

        avg_time = self.total_inference_time / self.inference_count
        return {
            'count': self.inference_count,
            'avg_time_ms': avg_time * 1000,
            'fps': 1.0 / avg_time if avg_time > 0 else 0.0,
            'total_time_s': self.total_inference_time,
            'using_gpu': self.use_gpu
        }

    def reset_statistics(self):
        """Reset inference statistics."""
        self.inference_count = 0
        self.total_inference_time = 0.0


class ONNXStereoDetector:
    """
    Two-stage stereo detector: tiny_stereo + crop refinement.

    Pipeline:
        Stage 1: tiny_stereo (320x180, 6ch) → coarse detection [x_l, y_l, x_r, y_r, conf]
        Stage 2: crop model (128x128) → per-image refinement [x, y] within crop
    """

    def __init__(self, stereo_model_path, crop_model_path=None, use_gpu=True,
                 stereo_size=(320, 180), crop_size=128, frame_size=(1280, 720),
                 confidence_threshold=0.5, use_refinement=True, convert_to_rgb=False):
        """
        Initialize stereo detector.

        Args:
            stereo_model_path: Path to tiny_stereo ONNX model (6ch input)
            crop_model_path: Path to crop refinement model (3ch, 128x128)
            use_gpu: Use DirectML GPU acceleration
            stereo_size: (width, height) for stereo model input
            crop_size: Crop size for refinement model
            frame_size: Original frame size (width, height)
            confidence_threshold: Minimum confidence for detection
            use_refinement: Enable stage 2 crop refinement
            convert_to_rgb: Convert BGR frames to RGB before stereo model
        """
        self.stereo_size = stereo_size  # (320, 180)
        self.crop_size = crop_size
        self.frame_size = frame_size  # (1280, 720)
        self.confidence_threshold = confidence_threshold
        self.use_refinement = use_refinement and crop_model_path is not None
        self.convert_to_rgb = convert_to_rgb

        # Setup providers
        if use_gpu:
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load stereo model (6-channel input)
        print(f"Loading stereo model: {stereo_model_path}")
        try:
            self.stereo_session = ort.InferenceSession(
                str(stereo_model_path), sess_options=session_options, providers=providers
            )
        except Exception as e:
            print(f"DirectML failed, falling back to CPU: {e}")
            self.stereo_session = ort.InferenceSession(
                str(stereo_model_path), sess_options=session_options,
                providers=['CPUExecutionProvider']
            )

        self.stereo_input_name = self.stereo_session.get_inputs()[0].name
        self.stereo_output_name = self.stereo_session.get_outputs()[0].name
        print(f"  Stereo model loaded: input={self.stereo_input_name}, "
              f"shape=(1, 6, {stereo_size[1]}, {stereo_size[0]})")

        # Load crop model (3-channel input) if provided
        self.crop_session = None
        if crop_model_path:
            print(f"Loading crop model: {crop_model_path}")
            try:
                self.crop_session = ort.InferenceSession(
                    str(crop_model_path), sess_options=session_options, providers=providers
                )
                self.crop_input_name = self.crop_session.get_inputs()[0].name
                self.crop_output_name = self.crop_session.get_outputs()[0].name
                print(f"  Crop model loaded: input={self.crop_input_name}, "
                      f"shape=(1, 3, {crop_size}, {crop_size})")
            except Exception as e:
                print(f"Failed to load crop model: {e}")
                self.use_refinement = False

        # Normalization - BGR order for stereo model (no conversion needed)
        self.mean_bgr = np.array([0.406, 0.456, 0.485], dtype=np.float32)
        self.std_bgr = np.array([0.225, 0.224, 0.229], dtype=np.float32)

        # Normalization - RGB order (only used if convert_to_rgb=True)
        self.mean_rgb = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std_rgb = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Statistics
        self.inference_count = 0
        self.total_time = 0.0

    def _preprocess_stereo(self, left_bgr, right_bgr):
        """Preprocess stereo pair for tiny_stereo model."""
        # Resize to stereo input size
        left_small = cv2.resize(left_bgr, self.stereo_size)
        right_small = cv2.resize(right_bgr, self.stereo_size)

        if self.convert_to_rgb:
            # Convert BGR->RGB (for RGB-trained models)
            left_small = cv2.cvtColor(left_small, cv2.COLOR_BGR2RGB)
            right_small = cv2.cvtColor(right_small, cv2.COLOR_BGR2RGB)
            mean, std = self.mean_rgb, self.std_rgb
        else:
            # Keep BGR (for BGR-trained models)
            mean, std = self.mean_bgr, self.std_bgr

        # Normalize
        left_norm = (left_small.astype(np.float32) / 255.0 - mean) / std
        right_norm = (right_small.astype(np.float32) / 255.0 - mean) / std

        # Stack as 6-channel
        # Shape: (H, W, 6) -> (6, H, W) -> (1, 6, H, W)
        stereo = np.concatenate([left_norm, right_norm], axis=2)  # (H, W, 6)
        stereo = np.transpose(stereo, (2, 0, 1))  # (6, H, W)
        stereo = np.expand_dims(stereo, axis=0)  # (1, 6, H, W)

        return stereo.astype(np.float32)

    def _preprocess_crop(self, crop_bgr):
        """Preprocess single crop for refinement model (BGR, no conversion)."""
        if crop_bgr.shape[:2] != (self.crop_size, self.crop_size):
            crop_bgr = cv2.resize(crop_bgr, (self.crop_size, self.crop_size))

        # Normalize with BGR mean/std (same as stereo model)
        crop_norm = (crop_bgr.astype(np.float32) / 255.0 - self.mean_bgr) / self.std_bgr
        crop_tensor = np.transpose(crop_norm, (2, 0, 1))
        crop_tensor = np.expand_dims(crop_tensor, axis=0)

        return crop_tensor.astype(np.float32)

    def _extract_crop(self, frame, x_norm, y_norm):
        """Extract crop centered at normalized coordinates.

        Works with any color format (BGR or RGB).
        """
        h, w = frame.shape[:2]
        cx = int(x_norm * w)
        cy = int(y_norm * h)

        half = self.crop_size // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)

        # Extract crop
        crop = frame[y1:y2, x1:x2].copy()

        # Pad if needed (edge cases)
        if crop.shape[0] != self.crop_size or crop.shape[1] != self.crop_size:
            padded = np.zeros((self.crop_size, self.crop_size, 3), dtype=crop.dtype)
            ph, pw = crop.shape[:2]
            padded[:ph, :pw] = crop
            crop = padded

        # Return crop and offset for coordinate conversion
        return crop, (x1, y1)

    def detect(self, left_frame, right_frame):
        """
        Detect ball in stereo pair.

        Args:
            left_frame: Left camera frame (H, W, 3) BGR from OpenCV
            right_frame: Right camera frame (H, W, 3) BGR from OpenCV

        Returns:
            dict with:
                x_left, y_left: Left camera coordinates (pixels)
                x_right, y_right: Right camera coordinates (pixels)
                confidence: Detection confidence [0, 1]
                detected: Whether ball was detected
                timing: dict with stage timings (ms)
        """
        t_start = time.perf_counter()

        # Work directly with BGR frames (no early conversion)
        frame_h, frame_w = left_frame.shape[:2]

        # Stage 1: Coarse detection with tiny_stereo
        # Preprocessing includes resize (+ BGR->RGB only if convert_to_rgb=True)
        stereo_input = self._preprocess_stereo(left_frame, right_frame)
        t_stereo_prep = time.perf_counter()

        stereo_output = self.stereo_session.run(
            [self.stereo_output_name], {self.stereo_input_name: stereo_input}
        )[0][0]
        t_stereo_end = time.perf_counter()

        # Parse stereo output: [x_l, y_l, x_r, y_r, conf]
        x_l_norm, y_l_norm, x_r_norm, y_r_norm, confidence = stereo_output

        # Check confidence threshold
        if confidence < self.confidence_threshold:
            t_end = time.perf_counter()
            return {
                'x_left': 0, 'y_left': 0,
                'x_right': 0, 'y_right': 0,
                'confidence': float(confidence),
                'detected': False,
                'timing': {
                    'prep_ms': (t_stereo_prep - t_start) * 1000,
                    'stereo_ms': (t_stereo_end - t_stereo_prep) * 1000,
                    'refine_L_ms': 0, 'refine_R_ms': 0,
                    'total_ms': (t_end - t_start) * 1000
                }
            }

        # Stage 2: Refinement (if enabled)
        refine_L_ms = 0
        refine_R_ms = 0

        if self.use_refinement and self.crop_session is not None:
            # Refine left - extract crop from BGR frame
            t_refine_L_start = time.perf_counter()
            left_crop, left_offset = self._extract_crop(left_frame, x_l_norm, y_l_norm)
            left_input = self._preprocess_crop(left_crop)
            left_output = self.crop_session.run(
                [self.crop_output_name], {self.crop_input_name: left_input}
            )[0][0]
            t_refine_L_end = time.perf_counter()
            refine_L_ms = (t_refine_L_end - t_refine_L_start) * 1000

            # Convert crop coords to frame coords
            x_l_crop, y_l_crop = left_output[0], left_output[1]
            x_l_px = left_offset[0] + x_l_crop * self.crop_size
            y_l_px = left_offset[1] + y_l_crop * self.crop_size

            # Refine right - extract crop from BGR frame
            t_refine_R_start = time.perf_counter()
            right_crop, right_offset = self._extract_crop(right_frame, x_r_norm, y_r_norm)
            right_input = self._preprocess_crop(right_crop)
            right_output = self.crop_session.run(
                [self.crop_output_name], {self.crop_input_name: right_input}
            )[0][0]
            t_refine_R_end = time.perf_counter()
            refine_R_ms = (t_refine_R_end - t_refine_R_start) * 1000

            # Convert crop coords to frame coords
            x_r_crop, y_r_crop = right_output[0], right_output[1]
            x_r_px = right_offset[0] + x_r_crop * self.crop_size
            y_r_px = right_offset[1] + y_r_crop * self.crop_size
        else:
            # Use coarse detection directly
            x_l_px = x_l_norm * frame_w
            y_l_px = y_l_norm * frame_h
            x_r_px = x_r_norm * frame_w
            y_r_px = y_r_norm * frame_h

        t_end = time.perf_counter()

        # Update statistics
        self.inference_count += 1
        self.total_time += (t_end - t_start)

        return {
            'x_left': float(x_l_px),
            'y_left': float(y_l_px),
            'x_right': float(x_r_px),
            'y_right': float(y_r_px),
            'confidence': float(confidence),
            'detected': True,
            'timing': {
                'prep_ms': (t_stereo_prep - t_start) * 1000,
                'stereo_ms': (t_stereo_end - t_stereo_prep) * 1000,
                'refine_L_ms': refine_L_ms,
                'refine_R_ms': refine_R_ms,
                'total_ms': (t_end - t_start) * 1000
            }
        }

    def get_statistics(self):
        """Get inference statistics."""
        if self.inference_count == 0:
            return {'count': 0, 'avg_time_ms': 0.0, 'fps': 0.0}

        avg_time = self.total_time / self.inference_count
        return {
            'count': self.inference_count,
            'avg_time_ms': avg_time * 1000,
            'fps': 1.0 / avg_time if avg_time > 0 else 0.0
        }

    def reset_statistics(self):
        """Reset statistics."""
        self.inference_count = 0
        self.total_time = 0.0


def benchmark_onnx_model(model_path, num_iterations=100, batch_sizes=[1, 2, 4], use_gpu=True):
    """
    Benchmark ONNX model performance.

    Args:
        model_path: Path to ONNX model
        num_iterations: Number of iterations for benchmarking
        batch_sizes: List of batch sizes to test
        use_gpu: Whether to use DirectML GPU acceleration
    """
    print("=" * 60)
    print("ONNX Model Benchmark")
    print("=" * 60)

    detector = ONNXBallDetector(model_path, use_gpu=use_gpu)

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        print("-" * 60)

        # Create dummy inputs
        dummy_images = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                        for _ in range(batch_size)]

        # Warmup
        for _ in range(10):
            if batch_size == 1:
                detector.detect(dummy_images[0])
            else:
                detector.detect_batch(dummy_images)

        # Reset statistics
        detector.reset_statistics()

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.time()
            if batch_size == 1:
                detector.detect(dummy_images[0])
            else:
                detector.detect_batch(dummy_images)
            times.append((time.time() - start) * 1000)  # Convert to ms

        # Calculate statistics
        times = np.array(times)
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        p95_time = np.percentile(times, 95)

        print(f"  Average: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  Min: {min_time:.2f} ms")
        print(f"  Max: {max_time:.2f} ms")
        print(f"  P95: {p95_time:.2f} ms")
        print(f"  Throughput: {batch_size * 1000 / avg_time:.1f} images/sec")
        print(f"  FPS (single camera): {1000 / (avg_time / batch_size):.1f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test ONNX Ball Detector')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to ONNX model')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable DirectML GPU acceleration')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run performance benchmark')
    parser.add_argument('--camera', type=int, default=None,
                        help='Test with webcam (specify camera ID)')

    args = parser.parse_args()

    if args.benchmark:
        # Benchmark mode
        benchmark_onnx_model(
            model_path=args.model,
            num_iterations=100,
            batch_sizes=[1, 2, 4],
            use_gpu=not args.no_gpu
        )
    elif args.camera is not None:
        # Webcam test mode
        from roi_extractor import RedBallROIExtractor

        print("Testing ONNX detector with webcam")
        print("Press 'q' to quit\n")

        detector = ONNXBallDetector(args.model, use_gpu=not args.no_gpu)
        roi_extractor = RedBallROIExtractor(crop_size=128)

        # Open camera with DirectShow backend (Windows, reduces tearing)
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(args.camera)

        # Reduce buffer size and set resolution
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        while True:
            # Flush old buffered frames
            cap.grab()

            ret, frame = cap.read()
            if not ret:
                break

            # Extract ROI
            crop, center, offset = roi_extractor.extract_roi(frame)

            if crop is not None:
                # Run CNN
                x_norm, y_norm, conf = detector.detect(crop)

                # Convert to original frame coordinates
                x_offset, y_offset = offset
                x_abs = x_offset + x_norm * roi_extractor.crop_size
                y_abs = y_offset + y_norm * roi_extractor.crop_size

                # Visualize
                cv2.circle(frame, (int(x_abs), int(y_abs)), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"Confidence: {conf:.3f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Show crop in corner
                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                crop_large = cv2.resize(crop_bgr, (128, 128))
                frame[:128, :128] = crop_large

            cv2.imshow('ONNX Detector Test', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Print statistics
        stats = detector.get_statistics()
        print(f"\nStatistics:")
        print(f"  Inferences: {stats['count']}")
        print(f"  Avg time: {stats['avg_time_ms']:.2f} ms")
        print(f"  FPS: {stats['fps']:.1f}")
    else:
        print("Error: Must specify --benchmark or --camera")
        parser.print_help()
