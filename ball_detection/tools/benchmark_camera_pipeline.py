"""
Realistic Camera Pipeline Benchmark

Streams frames from the full dataset as a simulated video and benchmarks
different detection pipelines with detailed timing breakdowns.

Modes:
    ROI_NN    - HSV color ROI extraction + 128x128 CNN refinement (old pipeline)
    STEREO_NN - tiny_stereo (320x180, 6ch) + optional crop refinement (new pipeline)

Usage:
    # STEREO_NN without RGB conversion (for BGR-trained models)
    python -m ball_detection.tools.benchmark_camera_pipeline --mode STEREO_NN

    # STEREO_NN with RGB conversion (for RGB-trained models)
    python -m ball_detection.tools.benchmark_camera_pipeline --mode STEREO_NN --convert-rgb

    # ROI_NN pipeline
    python -m ball_detection.tools.benchmark_camera_pipeline --mode ROI_NN

    # List available models
    python -m ball_detection.tools.benchmark_camera_pipeline --list-models
"""

import argparse
import json
import time
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ball_detection.core.onnx_inference import ONNXBallDetector, ONNXStereoDetector
from ball_detection.core.roi_extractor import RedBallROIExtractor


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_DATASET_PATH = "ball_detection/data/full_dataset/training_data_full"

# Model paths (can be overridden via command line)
DEFAULT_MODELS = {
    'stereo': 'ball_detection/models/Tiny_stereo/Tiny_stereo.onnx',
    #'crop': 'ball_detection/models/mobileLiteV3_prunned/mobileLiteV3_pruned.onnx',
    'crop': 'ball_detection/models/shufflenet128x128/shufflenet128x128.onnx',
}


# ============================================================
# FRAME LOADER
# ============================================================

class StereoFrameLoader:
    """Load stereo frame pairs from dataset, simulating camera stream."""

    def __init__(self, dataset_path, max_frames=None, loop=True):
        """
        Initialize frame loader.

        Args:
            dataset_path: Path to dataset with images/ and labels.json
            max_frames: Maximum number of frame pairs to load (None = all)
            loop: Loop frames when reaching end
        """
        self.dataset_path = Path(dataset_path)
        self.loop = loop

        # Find image directory
        if (self.dataset_path / 'images').exists():
            self.image_dir = self.dataset_path / 'images'
        else:
            self.image_dir = self.dataset_path

        # Load labels
        labels_file = self.dataset_path / 'labels.json'
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")

        with open(labels_file, 'r') as f:
            self.labels = json.load(f)

        # Group into stereo pairs
        self.pairs = self._find_stereo_pairs()

        if max_frames:
            self.pairs = self.pairs[:max_frames]

        print(f"Loaded {len(self.pairs)} stereo pairs from {dataset_path}")

        self.current_idx = 0
        self.frame_count = 0

    def _find_stereo_pairs(self):
        """Find matching left/right frame pairs."""
        import re

        left_frames = {}
        right_frames = {}

        for filename, label in self.labels.items():
            # Match patterns like: frame_001234_left.jpg or recording_..._left.jpg
            match = re.match(r'(.+)_(left|right)\.(jpg|jpeg|png)$', filename, re.IGNORECASE)
            if match:
                frame_id = match.group(1)
                side = match.group(2).lower()

                if side == 'left':
                    left_frames[frame_id] = (filename, label)
                else:
                    right_frames[frame_id] = (filename, label)

        # Match pairs
        pairs = []
        for frame_id in left_frames:
            if frame_id in right_frames:
                left_name, left_label = left_frames[frame_id]
                right_name, right_label = right_frames[frame_id]

                left_path = self.image_dir / left_name
                right_path = self.image_dir / right_name

                if left_path.exists() and right_path.exists():
                    pairs.append({
                        'frame_id': frame_id,
                        'left_path': str(left_path),
                        'right_path': str(right_path),
                        'left_label': left_label,
                        'right_label': right_label,
                    })

        # Sort by frame_id for sequential playback
        pairs.sort(key=lambda x: x['frame_id'])

        return pairs

    def get_next_frame(self):
        """
        Get next stereo frame pair.

        Returns:
            dict with left_frame, right_frame (BGR), labels, or None if exhausted
        """
        if self.current_idx >= len(self.pairs):
            if self.loop:
                self.current_idx = 0
            else:
                return None

        pair = self.pairs[self.current_idx]
        self.current_idx += 1
        self.frame_count += 1

        # Load frames (BGR from OpenCV)
        left_frame = cv2.imread(pair['left_path'], cv2.IMREAD_COLOR)
        right_frame = cv2.imread(pair['right_path'], cv2.IMREAD_COLOR)

        if left_frame is None or right_frame is None:
            return self.get_next_frame()  # Skip broken frames

        return {
            'left_frame': left_frame,
            'right_frame': right_frame,
            'left_label': pair['left_label'],
            'right_label': pair['right_label'],
            'frame_id': pair['frame_id'],
        }

    def __len__(self):
        return len(self.pairs)

    def reset(self):
        """Reset to beginning."""
        self.current_idx = 0
        self.frame_count = 0


# ============================================================
# PIPELINE BENCHMARKS
# ============================================================

class ROIPipelineBenchmark:
    """Benchmark ROI_NN pipeline: HSV ROI + CNN refinement."""

    def __init__(self, crop_model_path, use_gpu=True):
        """
        Initialize ROI pipeline.

        Args:
            crop_model_path: Path to 128x128 crop model
            use_gpu: Use DirectML GPU acceleration
        """
        print(f"\n{'='*60}")
        print("ROI_NN Pipeline")
        print(f"{'='*60}")
        print(f"  Crop model: {crop_model_path}")
        print(f"  GPU: {use_gpu}")

        self.roi_extractor = RedBallROIExtractor(crop_size=128)
        self.cnn = ONNXBallDetector(crop_model_path, use_gpu=use_gpu, image_size=128)

        # Timing accumulators
        self.timing = defaultdict(list)

    def process_frame(self, left_frame, right_frame):
        """
        Process stereo pair with ROI pipeline.

        Returns:
            dict with detection results and timing
        """
        t_start = time.perf_counter()

        results = {
            'left': {'detected': False, 'x': 0, 'y': 0, 'confidence': 0},
            'right': {'detected': False, 'x': 0, 'y': 0, 'confidence': 0},
            'timing': {}
        }

        # Process left frame
        t0 = time.perf_counter()
        left_crop, left_center, left_offset = self.roi_extractor.extract_roi(left_frame)
        t_roi_l = time.perf_counter()

        if left_crop is not None:
            x_norm, y_norm, conf = self.cnn.detect(left_crop)
            t_cnn_l = time.perf_counter()

            # Convert to frame coordinates
            x_px = left_offset[0] + x_norm * 128
            y_px = left_offset[1] + y_norm * 128

            results['left'] = {
                'detected': True,
                'x': x_px,
                'y': y_px,
                'confidence': conf
            }

            self.timing['roi_L'].append((t_roi_l - t0) * 1000)
            self.timing['cnn_L'].append((t_cnn_l - t_roi_l) * 1000)

        # Process right frame
        t0 = time.perf_counter()
        right_crop, right_center, right_offset = self.roi_extractor.extract_roi(right_frame)
        t_roi_r = time.perf_counter()

        if right_crop is not None:
            x_norm, y_norm, conf = self.cnn.detect(right_crop)
            t_cnn_r = time.perf_counter()

            x_px = right_offset[0] + x_norm * 128
            y_px = right_offset[1] + y_norm * 128

            results['right'] = {
                'detected': True,
                'x': x_px,
                'y': y_px,
                'confidence': conf
            }

            self.timing['roi_R'].append((t_roi_r - t0) * 1000)
            self.timing['cnn_R'].append((t_cnn_r - t_roi_r) * 1000)

        t_end = time.perf_counter()
        self.timing['total'].append((t_end - t_start) * 1000)

        results['timing'] = {
            'total_ms': (t_end - t_start) * 1000
        }

        return results

    def get_statistics(self):
        """Get timing statistics."""
        stats = {}
        for key, values in self.timing.items():
            if values:
                arr = np.array(values)
                stats[key] = {
                    'mean': np.mean(arr),
                    'std': np.std(arr),
                    'min': np.min(arr),
                    'max': np.max(arr),
                    'p50': np.percentile(arr, 50),
                    'p95': np.percentile(arr, 95),
                    'p99': np.percentile(arr, 99),
                }
        return stats


class StereoPipelineBenchmark:
    """Benchmark STEREO_NN pipeline: tiny_stereo + optional refinement."""

    def __init__(self, stereo_model_path, crop_model_path=None, use_gpu=True,
                 use_refinement=True, convert_to_rgb=False):
        """
        Initialize stereo pipeline.

        Args:
            stereo_model_path: Path to tiny_stereo model (6ch, 320x180)
            crop_model_path: Path to crop refinement model (optional)
            use_gpu: Use DirectML GPU acceleration
            use_refinement: Enable stage 2 crop refinement
            convert_to_rgb: Convert BGR frames to RGB before processing
        """
        self.convert_to_rgb = convert_to_rgb
        self.use_refinement = use_refinement

        print(f"\n{'='*60}")
        print("STEREO_NN Pipeline")
        print(f"{'='*60}")
        print(f"  Stereo model: {stereo_model_path}")
        print(f"  Crop model: {crop_model_path if use_refinement else 'None (disabled)'}")
        print(f"  Convert to RGB: {convert_to_rgb}")
        print(f"  Use refinement: {use_refinement}")
        print(f"  GPU: {use_gpu}")

        self.detector = ONNXStereoDetector(
            stereo_model_path=stereo_model_path,
            crop_model_path=crop_model_path if use_refinement else None,
            use_gpu=use_gpu,
            stereo_size=(320, 180),
            crop_size=128,
            frame_size=(1280, 720),
            confidence_threshold=0.5,
            use_refinement=use_refinement,
            convert_to_rgb=convert_to_rgb
        )

        # Timing accumulators
        self.timing = defaultdict(list)

    def process_frame(self, left_frame, right_frame):
        """
        Process stereo pair with STEREO_NN pipeline.

        Returns:
            dict with detection results and timing
        """
        t_start = time.perf_counter()

        # Run detection (BGR->RGB conversion happens inside if stereo_model_bgr=False)
        result = self.detector.detect(left_frame, right_frame)

        t_end = time.perf_counter()

        # Accumulate timing
        self.timing['total'].append((t_end - t_start) * 1000)
        self.timing['resize'].append(result['timing']['resize_ms'])
        self.timing['convert'].append(result['timing']['convert_ms'])
        self.timing['normalize'].append(result['timing']['normalize_ms'])
        self.timing['stereo_nn'].append(result['timing']['stereo_ms'])
        if self.use_refinement:
            self.timing['refine_L'].append(result['timing']['refine_L_ms'])
            self.timing['refine_R'].append(result['timing']['refine_R_ms'])

        return {
            'left': {
                'detected': result['detected'],
                'x': result['x_left'],
                'y': result['y_left'],
                'confidence': result['confidence']
            },
            'right': {
                'detected': result['detected'],
                'x': result['x_right'],
                'y': result['y_right'],
                'confidence': result['confidence']
            },
            'timing': result['timing']
        }

    def get_statistics(self):
        """Get timing statistics."""
        stats = {}
        for key, values in self.timing.items():
            if values:
                arr = np.array(values)
                stats[key] = {
                    'mean': np.mean(arr),
                    'std': np.std(arr),
                    'min': np.min(arr),
                    'max': np.max(arr),
                    'p50': np.percentile(arr, 50),
                    'p95': np.percentile(arr, 95),
                    'p99': np.percentile(arr, 99),
                }
        return stats


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_detection(left_frame, right_frame, result, frame_id, fps):
    """Create visualization of detection results."""
    # Make copies
    left_vis = left_frame.copy()
    right_vis = right_frame.copy()

    # Draw detections
    if result['left']['detected']:
        x, y = int(result['left']['x']), int(result['left']['y'])
        cv2.circle(left_vis, (x, y), 8, (0, 255, 0), 2)
        cv2.circle(left_vis, (x, y), 2, (0, 255, 0), -1)

    if result['right']['detected']:
        x, y = int(result['right']['x']), int(result['right']['y'])
        cv2.circle(right_vis, (x, y), 8, (0, 255, 0), 2)
        cv2.circle(right_vis, (x, y), 2, (0, 255, 0), -1)

    # Stack horizontally
    combined = np.hstack([left_vis, right_vis])

    # Resize for display
    h, w = combined.shape[:2]
    display_width = 1280
    scale = display_width / w
    combined = cv2.resize(combined, (display_width, int(h * scale)))

    # Add info overlay
    timing = result['timing']
    convert_ms = timing.get('convert_ms', 0)
    info_lines = [
        f"Frame: {frame_id}",
        f"FPS: {fps:.1f}",
        f"Time: {timing.get('total_ms', 0):.1f}ms (cvt:{convert_ms:.1f})",
        f"Conf: {result['left']['confidence']:.2f}",
    ]

    y_offset = 25
    for line in info_lines:
        cv2.putText(combined, line, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25

    return combined


# ============================================================
# MAIN BENCHMARK
# ============================================================

def run_benchmark(args):
    """Run the benchmark."""
    print("\n" + "=" * 60)
    print("CAMERA PIPELINE BENCHMARK")
    print("=" * 60)

    # Load frames
    print(f"\nLoading dataset: {args.dataset}")
    loader = StereoFrameLoader(args.dataset, max_frames=args.max_frames, loop=False)

    if len(loader) == 0:
        print("ERROR: No stereo pairs found in dataset!")
        return

    # Initialize pipeline
    if args.mode == 'ROI_NN':
        pipeline = ROIPipelineBenchmark(
            crop_model_path=args.crop_model,
            use_gpu=args.gpu
        )
    else:  # STEREO_NN
        pipeline = StereoPipelineBenchmark(
            stereo_model_path=args.stereo_model,
            crop_model_path=args.crop_model,
            use_gpu=args.gpu,
            use_refinement=args.refinement,
            convert_to_rgb=args.convert_rgb
        )

    print(f"\n{'='*60}")
    print("Running benchmark...")
    print(f"{'='*60}")
    print(f"  Frames: {len(loader)}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Display: {args.display}")

    # Warmup
    print("\nWarming up...")
    for i in range(args.warmup):
        frame_data = loader.get_next_frame()
        if frame_data:
            pipeline.process_frame(frame_data['left_frame'], frame_data['right_frame'])
    loader.reset()

    # Reset timing after warmup
    pipeline.timing.clear()

    # Main benchmark loop
    print("\nProcessing frames...")
    frame_times = []
    fps_counter = []
    fps_window = 30

    while True:
        frame_data = loader.get_next_frame()
        if frame_data is None:
            break

        t_loop_start = time.perf_counter()

        # Process frame
        result = pipeline.process_frame(
            frame_data['left_frame'],
            frame_data['right_frame']
        )

        t_loop_end = time.perf_counter()
        frame_time = (t_loop_end - t_loop_start) * 1000
        frame_times.append(frame_time)

        # FPS calculation
        fps_counter.append(time.perf_counter())
        if len(fps_counter) > fps_window:
            fps_counter.pop(0)
        if len(fps_counter) > 1:
            fps = len(fps_counter) / (fps_counter[-1] - fps_counter[0])
        else:
            fps = 0

        # Progress with detailed timing
        if loader.frame_count % 100 == 0:
            timing = result['timing']
            if 'resize_ms' in timing:  # STEREO_NN mode
                print(f"  Frame {loader.frame_count}/{len(loader)} | FPS: {fps:.1f} | Total: {frame_time:.1f}ms")
                print(f"    resize: {timing.get('resize_ms', 0):.2f} | cvt: {timing.get('convert_ms', 0):.2f} | "
                      f"norm: {timing.get('normalize_ms', 0):.2f} | stereo: {timing.get('stereo_ms', 0):.2f} | "
                      f"refine_L: {timing.get('refine_L_ms', 0):.2f} | refine_R: {timing.get('refine_R_ms', 0):.2f}")
            else:  # ROI_NN mode
                print(f"  Frame {loader.frame_count}/{len(loader)} | FPS: {fps:.1f} | Total: {frame_time:.1f}ms")

        # Visualization
        if args.display:
            vis = visualize_detection(
                frame_data['left_frame'],
                frame_data['right_frame'],
                result,
                frame_data['frame_id'],
                fps
            )
            cv2.imshow('Benchmark', vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Pause
                cv2.waitKey(0)

    if args.display:
        cv2.destroyAllWindows()

    # Print results
    print_results(pipeline, frame_times, args)


def print_results(pipeline, frame_times, args):
    """Print benchmark results."""
    stats = pipeline.get_statistics()

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\nMode: {args.mode}")
    print(f"Convert to RGB: {args.convert_rgb}")
    print(f"Use refinement: {args.refinement}")
    print(f"GPU: {args.gpu}")
    print(f"Total frames: {len(frame_times)}")

    # Overall stats
    frame_arr = np.array(frame_times)
    print(f"\n{'Overall Loop Time':}")
    print(f"  Mean:  {np.mean(frame_arr):>7.2f} ms")
    print(f"  Std:   {np.std(frame_arr):>7.2f} ms")
    print(f"  Min:   {np.min(frame_arr):>7.2f} ms")
    print(f"  Max:   {np.max(frame_arr):>7.2f} ms")
    print(f"  P50:   {np.percentile(frame_arr, 50):>7.2f} ms")
    print(f"  P95:   {np.percentile(frame_arr, 95):>7.2f} ms")
    print(f"  P99:   {np.percentile(frame_arr, 99):>7.2f} ms")
    print(f"  FPS:   {1000 / np.mean(frame_arr):>7.1f}")

    # Detailed timing breakdown
    print(f"\n{'Timing Breakdown':}")
    print("-" * 50)
    print(f"{'Stage':<20} {'Mean':>8} {'Std':>8} {'P95':>8} {'P99':>8}")
    print("-" * 50)

    for key in ['total', 'resize', 'convert', 'normalize', 'stereo_nn',
                'roi_L', 'roi_R', 'cnn_L', 'cnn_R', 'refine_L', 'refine_R']:
        if key in stats:
            s = stats[key]
            print(f"{key:<20} {s['mean']:>7.2f}ms {s['std']:>7.2f}ms "
                  f"{s['p95']:>7.2f}ms {s['p99']:>7.2f}ms")

    print("-" * 50)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    avg_fps = 1000 / np.mean(frame_arr)
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Target 60 FPS: {'YES' if avg_fps >= 60 else 'NO'} ({avg_fps/60*100:.1f}%)")


def list_available_models():
    """List available ONNX models."""
    print("\nAvailable ONNX Models:")
    print("=" * 60)

    models_dir = Path("ball_detection/models")
    if not models_dir.exists():
        print("  Models directory not found!")
        return

    for model_path in sorted(models_dir.rglob("*.onnx")):
        rel_path = model_path.relative_to(Path.cwd())
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  {rel_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description='Realistic Camera Pipeline Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Benchmark STEREO_NN pipeline (BGR, no conversion)
    python -m ball_detection.tools.benchmark_camera_pipeline --mode STEREO_NN

    # With RGB conversion (for RGB-trained models)
    python -m ball_detection.tools.benchmark_camera_pipeline --mode STEREO_NN --convert-rgb

    # Benchmark ROI_NN pipeline
    python -m ball_detection.tools.benchmark_camera_pipeline --mode ROI_NN

    # Benchmark without refinement stage
    python -m ball_detection.tools.benchmark_camera_pipeline --mode STEREO_NN --no-refinement

    # With visualization
    python -m ball_detection.tools.benchmark_camera_pipeline --mode STEREO_NN --display

    # List available models
    python -m ball_detection.tools.benchmark_camera_pipeline --list-models
        """
    )

    parser.add_argument('--mode', type=str, choices=['ROI_NN', 'STEREO_NN'],
                        default='STEREO_NN', help='Detection mode (default: STEREO_NN)')

    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET_PATH,
                        help='Path to dataset with stereo frames')

    parser.add_argument('--stereo-model', type=str, default=DEFAULT_MODELS['stereo'],
                        help='Path to stereo ONNX model (6ch, 320x180)')

    parser.add_argument('--crop-model', type=str, default=DEFAULT_MODELS['crop'],
                        help='Path to crop refinement model (128x128)')

    parser.add_argument('--convert-rgb', action='store_true',
                        help='Convert BGR frames to RGB before processing')

    parser.add_argument('--no-refinement', action='store_true',
                        help='Disable stage 2 crop refinement')

    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration (use CPU only)')

    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to process')

    parser.add_argument('--warmup', type=int, default=50,
                        help='Number of warmup frames (default: 50)')

    parser.add_argument('--display', action='store_true',
                        help='Show visualization window')

    parser.add_argument('--list-models', action='store_true',
                        help='List available ONNX models and exit')

    args = parser.parse_args()

    # Handle special commands
    if args.list_models:
        list_available_models()
        return

    # Validate paths
    if not Path(args.dataset).exists():
        print(f"ERROR: Dataset not found: {args.dataset}")
        return

    if args.mode == 'STEREO_NN' and not Path(args.stereo_model).exists():
        print(f"ERROR: Stereo model not found: {args.stereo_model}")
        print("Use --list-models to see available models")
        return

    if not Path(args.crop_model).exists():
        print(f"WARNING: Crop model not found: {args.crop_model}")
        if args.mode == 'ROI_NN':
            print("ERROR: Crop model required for ROI_NN mode")
            return
        else:
            print("Disabling refinement stage")
            args.no_refinement = True

    # Set derived args
    args.refinement = not args.no_refinement
    args.gpu = not args.no_gpu

    # Run benchmark
    run_benchmark(args)


if __name__ == "__main__":
    main()
