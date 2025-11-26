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

        # Normalization parameters (ImageNet stats)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Statistics
        self.inference_count = 0
        self.total_inference_time = 0.0

    def preprocess(self, image_rgb):
        """
        Preprocess image for model input.

        Args:
            image_rgb: RGB image (H, W, 3), uint8, range [0, 255]

        Returns:
            Preprocessed tensor (1, 3, H, W), float32
        """
        # Resize if needed
        if image_rgb.shape[:2] != (self.image_size, self.image_size):
            image_rgb = cv2.resize(image_rgb, (self.image_size, self.image_size))

        # Convert to float [0, 1]
        image_float = image_rgb.astype(np.float32) / 255.0

        # Normalize
        image_float = (image_float - self.mean) / self.std

        # Transpose to (C, H, W) and add batch dimension
        image_tensor = np.transpose(image_float, (2, 0, 1))
        image_tensor = np.expand_dims(image_tensor, axis=0)

        return image_tensor

    def detect(self, image_rgb) -> Tuple[float, float, float]:
        """
        Detect ball center in image.

        Args:
            image_rgb: RGB image crop (H, W, 3), uint8

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
