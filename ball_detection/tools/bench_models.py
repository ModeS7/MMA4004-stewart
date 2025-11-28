#!/usr/bin/env python3
"""
CPU-Only Model Benchmark: FP32 vs Static INT8

Tests multiple CNN architectures for ball detection comparing:
- FP32 (baseline)
- INT8 (static quantization with calibration)

Features:
- Multiple models: BallDetectorCNN, MobileNetV3, MNASNet, SqueezeNet
- Input sizes: 32, 64, 96, 128, 160, 224
- Static INT8 quantization (weights + activations)
- CPU-only with true INT8 compute kernels

All operations are in-memory, no files created (temp files cleaned up).
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
import io
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import model
sys.path.insert(0, str(Path(__file__).parent))
from ball_detection.core.model import (
    BallDetectorCNN,
    BallDetectorFullFrame,
    BallDetectorFullFrameTiny,
    BallDetectorFullFrameUltra,
    BallDetectorFullFrameMobileNet,
    BallDetectorFullFrameMobileNetLite,
)


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class AdaptiveBallDetector(nn.Module):
    """Wrapper to adapt any classifier to ball detection with adaptive pooling."""

    def __init__(self, backbone, num_features):
        super().__init__()
        self.backbone = backbone

        # Replace final classifier with ball detection head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Get features from backbone (remove final classifier)
        features = self.backbone(x)

        # Apply detection head
        output = self.head(features)
        return output


def create_custom_model():
    """Create BallDetectorCNN from codebase."""
    return BallDetectorCNN()


def create_mobilenet_v3_small():
    """Create MobileNetV3-Small adapted for ball detection."""
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

    # Load pretrained backbone
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)

    # Remove classifier, keep features
    backbone = nn.Sequential(*list(model.children())[:-1])

    # Wrap with adaptive detection head
    return AdaptiveBallDetector(backbone, 576)


def create_mobilenet_v3_large():
    """Create MobileNetV3-Large adapted for ball detection."""
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights)

    backbone = nn.Sequential(*list(model.children())[:-1])
    return AdaptiveBallDetector(backbone, 960)


def create_mnasnet0_5():
    """Create MNASNet0.5 adapted for ball detection."""
    from torchvision.models import mnasnet0_5, MNASNet0_5_Weights

    weights = MNASNet0_5_Weights.DEFAULT
    model = mnasnet0_5(weights=weights)

    # MNASNet has layers structure
    backbone = nn.Sequential(*list(model.children())[:-1])
    return AdaptiveBallDetector(backbone, 1280)


def create_squeezenet1_1():
    """Create SqueezeNet1.1 adapted for ball detection."""
    from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights

    weights = SqueezeNet1_1_Weights.DEFAULT
    model = squeezenet1_1(weights=weights)

    # SqueezeNet structure: features -> classifier
    backbone = model.features

    # SqueezeNet outputs (N, 512, H, W) from features
    return AdaptiveBallDetector(backbone, 512)


def create_mobilenet_v2_035():
    """Create MobileNetV2 with width multiplier 0.35 adapted for ball detection."""
    from torchvision.models import mobilenet_v2

    # MobileNetV2 with width_mult=0.35 (smallest variant)
    model = mobilenet_v2(weights=None, width_mult=0.35)

    # Remove classifier, keep features
    backbone = model.features

    # MobileNetV2 0.35 outputs 1280 * 0.35 = 448 channels (rounded to 1280 in torchvision)
    # Actually it's always 1280 for the last layer due to how MobileNetV2 is structured
    return AdaptiveBallDetector(backbone, 1280)


def create_shufflenet_v2_x0_5():
    """Create ShuffleNetV2 x0.5 adapted for ball detection."""
    from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights

    weights = ShuffleNet_V2_X0_5_Weights.DEFAULT
    model = shufflenet_v2_x0_5(weights=weights)

    # ShuffleNet structure: conv1 -> maxpool -> stages -> conv5
    # We need features before the final fc layer
    # Extract everything except the fc layer
    class ShuffleNetBackbone(nn.Module):
        def __init__(self, shufflenet):
            super().__init__()
            self.conv1 = shufflenet.conv1
            self.maxpool = shufflenet.maxpool
            self.stage2 = shufflenet.stage2
            self.stage3 = shufflenet.stage3
            self.stage4 = shufflenet.stage4
            self.conv5 = shufflenet.conv5

        def forward(self, x):
            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.conv5(x)
            return x

    backbone = ShuffleNetBackbone(model)
    # ShuffleNetV2 x0.5 outputs 1024 channels
    return AdaptiveBallDetector(backbone, 1024)


# ============================================================================
# ONNX CONVERSION & QUANTIZATION
# ============================================================================

def export_to_onnx_bytes(model, input_size=64):
    """
    Export PyTorch model to ONNX in-memory.

    Returns:
        bytes: ONNX model as bytes
    """
    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size)

    # Export to bytes buffer
    buffer = io.BytesIO()
    torch.onnx.export(
        model,
        dummy_input,
        buffer,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=14,
        do_constant_folding=True
    )

    buffer.seek(0)
    return buffer.read()


class CalibrationDataReader:
    """
    Calibration data reader for static INT8 quantization.
    Generates random calibration samples to compute activation ranges.
    """
    def __init__(self, input_size, num_samples=100):
        self.input_size = input_size
        self.num_samples = num_samples
        self.data = self._generate_calibration_data()
        self.current_index = 0

    def _generate_calibration_data(self):
        """Generate random calibration images."""
        np.random.seed(42)  # Reproducible
        data = []
        for _ in range(self.num_samples):
            # Random images in [0, 1] range
            img = np.random.rand(1, 3, self.input_size, self.input_size).astype(np.float32)
            data.append({'input': img})
        return data

    def get_next(self):
        """Get next calibration sample."""
        if self.current_index >= len(self.data):
            return None
        sample = self.data[self.current_index]
        self.current_index += 1
        return sample

    def rewind(self):
        """Reset to beginning."""
        self.current_index = 0


def quantize_onnx_model_static(onnx_bytes, input_size, num_calibration_samples=100):
    """
    Static INT8 quantization with calibration data.
    Quantizes both weights AND activations for true INT8 compute.

    Args:
        onnx_bytes: ONNX model as bytes
        input_size: Input image size for calibration data
        num_calibration_samples: Number of samples for calibration

    Returns:
        bytes: Quantized ONNX model, or None if quantization failed
    """
    import tempfile
    import os
    import logging

    # Suppress ONNX quantization warnings
    logging.getLogger('root').setLevel(logging.ERROR)

    try:
        # Create calibration data reader
        calibration_reader = CalibrationDataReader(input_size, num_calibration_samples)

        # Create temp files
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as temp_in:
            temp_in.write(onnx_bytes)
            temp_in_path = temp_in.name

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as temp_out:
            temp_out_path = temp_out.name

        try:
            # Static quantization (weights + activations)
            quantize_static(
                model_input=temp_in_path,
                model_output=temp_out_path,
                calibration_data_reader=calibration_reader,
                quant_format=QuantType.QUInt8,
                per_channel=False,
                reduce_range=False,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QUInt8,
                op_types_to_quantize=['Conv', 'MatMul', 'Gemm'],
            )

            # Read quantized model
            with open(temp_out_path, 'rb') as f:
                quantized_bytes = f.read()

            return quantized_bytes

        finally:
            # Cleanup temp files
            try:
                os.unlink(temp_in_path)
                os.unlink(temp_out_path)
            except:
                pass

    except Exception as e:
        print(f"      Quantization error: {e}")
        return None


def create_inference_session(onnx_bytes):
    """
    Create ONNX Runtime CPU session from bytes.

    Args:
        onnx_bytes: ONNX model as bytes

    Returns:
        ort.InferenceSession or None
    """
    try:
        # Session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Create CPU-only session
        session = ort.InferenceSession(
            onnx_bytes,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )

        return session

    except Exception as e:
        print(f"    Error creating session: {e}")
        return None


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def benchmark_model(session, input_size, num_iterations=100, warmup=10):
    """
    Benchmark ONNX model inference.

    Returns:
        dict: Statistics (mean, min, max, p95, fps)
    """
    if session is None:
        return None

    try:
        # Get input/output names
        input_name = session.get_inputs()[0].name

        # Create dummy input
        dummy_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

        # Warmup
        for _ in range(warmup):
            session.run(None, {input_name: dummy_input})

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            output = session.run(None, {input_name: dummy_input})
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        times = np.array(times)

        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'p95': np.percentile(times, 95),
            'fps': 1000 / np.mean(times)
        }

    except Exception as e:
        print(f"    Benchmark error: {e}")
        return None


def count_parameters(model):
    """Count trainable parameters in PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# MAIN BENCHMARK ORCHESTRATOR
# ============================================================================

def run_comprehensive_benchmark():
    """Run comprehensive benchmark across all models, sizes, and quantizations."""

    print("=" * 80)
    print("CNN ARCHITECTURE BENCHMARK FOR BALL DETECTION")
    print("=" * 80)
    print()

    # Configuration
    models_config = {
        'BallDetectorCNN': create_custom_model,
        'MobileNetV2-0.35': create_mobilenet_v2_035,
        'MobileNetV3-Small': create_mobilenet_v3_small,
        'MobileNetV3-Large': create_mobilenet_v3_large,
        'ShuffleNetV2-0.5x': create_shufflenet_v2_x0_5,
        'MNASNet0.5': create_mnasnet0_5,
        'SqueezeNet1.1': create_squeezenet1_1,
    }

    input_sizes = [128]  # Only test 128x128
    quantizations = ['fp32', 'int8']
    num_iterations = 100

    print(f"Configuration:")
    print(f"  Models: {len(models_config)}")
    print(f"  Input sizes: {input_sizes}")
    print(f"  Quantizations: {quantizations}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Backend: CPU only")
    print()
    print("Note: INT8 uses static quantization (weights + activations, true INT8 compute)")
    print()

    # Results storage
    results = []

    # Test each model
    for model_name, model_fn in models_config.items():
        print(f"Testing {model_name}...")
        print("-" * 80)

        try:
            # Create model
            model = model_fn()
            model.eval()

            # Count parameters
            params = count_parameters(model)
            params_mb = params * 4 / (1024 * 1024)  # Assume FP32

            print(f"  Parameters: {params:,} ({params_mb:.2f} MB)")

            # Test each input size
            for input_size in input_sizes:
                print(f"  Input size: {input_size}x{input_size}")

                # Export to ONNX
                try:
                    onnx_bytes = export_to_onnx_bytes(model, input_size)
                except Exception as e:
                    print(f"    ONNX export failed: {e}")
                    continue

                # Test each quantization
                for quant in quantizations:
                    # Prepare model bytes based on quantization
                    model_bytes = onnx_bytes

                    if quant == 'int8':
                        # Apply static quantization
                        print(f"    {quant.upper()}...", end=' ', flush=True)
                        print("quantizing...", end=' ', flush=True)
                        quantized_bytes = quantize_onnx_model_static(onnx_bytes, input_size)
                        if quantized_bytes is None:
                            print("FAILED")
                            continue
                        model_bytes = quantized_bytes
                    else:
                        print(f"    {quant.upper()}...", end=' ', flush=True)

                    # Create CPU session
                    session = create_inference_session(model_bytes)
                    if not session:
                        print("Session creation FAILED")
                        continue

                    # Benchmark
                    stats = benchmark_model(session, input_size, num_iterations)

                    if stats:
                        print(f"{stats['mean']:.2f}ms ({stats['fps']:.1f} FPS)")

                        results.append({
                            'model': model_name,
                            'params': params,
                            'input_size': input_size,
                            'quant': quant,
                            **stats
                        })
                    else:
                        print("FAILED")

            print()

        except Exception as e:
            print(f"  Error testing {model_name}: {e}")
            print()
            continue

    # Display results
    print()
    print("=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print()

    if not results:
        print("No results collected!")
        return

    # Sort by inference time (fastest first)
    results_sorted = sorted(results, key=lambda x: x['mean'])

    # Print table header
    header = f"{'Model':<20} {'Size':>5} {'Quant':>6} {'Params':>8} {'Mean':>8} {'Min':>8} {'P95':>8} {'FPS':>8}"
    print(header)
    print("-" * len(header))

    # Print results
    for r in results_sorted:
        params_k = r['params'] / 1000
        print(f"{r['model']:<20} {r['input_size']:>5} {r['quant']:>6} "
              f"{params_k:>7.0f}K {r['mean']:>7.2f}ms {r['min']:>7.2f}ms "
              f"{r['p95']:>7.2f}ms {r['fps']:>7.1f}")

    print()

    # Find best performers
    print("TOP PERFORMERS:")
    print()

    # Best FP32
    fp32_results = [r for r in results_sorted if r['quant'] == 'fp32']
    if fp32_results:
        best = fp32_results[0]
        print(f"  Fastest FP32: {best['model']} @ {best['input_size']}x{best['input_size']}")
        print(f"    {best['mean']:.2f}ms ({best['fps']:.1f} FPS) | {best['params']/1000:.0f}K params")

    # Best INT8
    int8_results = [r for r in results_sorted if r['quant'] == 'int8']
    if int8_results:
        best = int8_results[0]
        print(f"  Fastest INT8: {best['model']} @ {best['input_size']}x{best['input_size']}")
        print(f"    {best['mean']:.2f}ms ({best['fps']:.1f} FPS) | {best['params']/1000:.0f}K params")

    # Best efficiency (FPS / param count)
    for r in results_sorted:
        r['efficiency'] = r['fps'] / (r['params'] / 1000000)  # FPS per million params

    efficient = sorted(results_sorted, key=lambda x: x['efficiency'], reverse=True)
    if efficient:
        best = efficient[0]
        print(f"  Most efficient: {best['model']} @ {best['input_size']}x{best['input_size']}")
        print(f"    {best['fps']:.1f} FPS with {best['params']/1000:.0f}K params "
              f"(efficiency: {best['efficiency']:.2f})")

    print()
    print("=" * 80)

    # Analysis
    print()
    print("INT8 QUANTIZATION SPEEDUP ANALYSIS:")
    print()

    # Calculate INT8 speedup for all models
    fp32_results = [r for r in results if r['quant'] == 'fp32']
    int8_results = [r for r in results if r['quant'] == 'int8']

    if fp32_results and int8_results:
        speedups = []
        for fp32_result in fp32_results:
            int8_match = next((r for r in int8_results
                              if r['model'] == fp32_result['model']
                              and r['input_size'] == fp32_result['input_size']), None)
            if int8_match:
                speedup = fp32_result['mean'] / int8_match['mean']
                speedups.append({
                    'model': fp32_result['model'],
                    'input_size': fp32_result['input_size'],
                    'fp32_time': fp32_result['mean'],
                    'int8_time': int8_match['mean'],
                    'speedup': speedup
                })

        if speedups:
            # Show top 5 speedups
            speedups_sorted = sorted(speedups, key=lambda x: x['speedup'], reverse=True)
            print("  Top 5 INT8 speedups:")
            for s in speedups_sorted[:5]:
                print(f"    {s['model']:<20} @ {s['input_size']:>3}x{s['input_size']:<3} "
                      f"FP32: {s['fp32_time']:>6.2f}ms -> INT8: {s['int8_time']:>6.2f}ms "
                      f"({s['speedup']:>4.2f}x)")

            # Average speedup
            avg_speedup = np.mean([s['speedup'] for s in speedups])
            print()
            print(f"  Average INT8 speedup: {avg_speedup:.2f}x")

            # Best and worst
            best_speedup = speedups_sorted[0]
            worst_speedup = speedups_sorted[-1]
            print(f"  Best: {best_speedup['speedup']:.2f}x ({best_speedup['model']} @ {best_speedup['input_size']})")
            print(f"  Worst: {worst_speedup['speedup']:.2f}x ({worst_speedup['model']} @ {worst_speedup['input_size']})")

    print()
    print("=" * 80)
    print()


def benchmark_onnx_file(model_path, input_size=128, num_iterations=100):
    """
    Quick benchmark for existing ONNX model file.
    Tests both CPU and GPU (DirectML) inference.

    Args:
        model_path: Path to ONNX model file
        input_size: Input image size (default: 128)
        num_iterations: Number of iterations (default: 100)
    """
    print(f"Benchmarking: {model_path}")
    print(f"Input size: {input_size}x{input_size}")
    print(f"Iterations: {num_iterations}")
    print()

    # Test data
    dummy_input = np.random.rand(1, 3, input_size, input_size).astype(np.float32)

    # CPU session
    print("CPU Inference:")
    sess_cpu = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = sess_cpu.get_inputs()[0].name

    # Warmup
    for _ in range(10):
        sess_cpu.run(None, {input_name: dummy_input})

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        sess_cpu.run(None, {input_name: dummy_input})
        times.append((time.perf_counter() - start) * 1000)

    mean_time = np.mean(times)
    min_time = np.min(times)
    fps = 1000 / mean_time
    print(f"  Mean: {mean_time:.2f}ms ({fps:.1f} FPS)")
    print(f"  Min:  {min_time:.2f}ms ({1000/min_time:.1f} FPS)")
    print()

    # GPU session (DirectML)
    print("GPU Inference (DirectML):")
    try:
        sess_gpu = ort.InferenceSession(
            model_path,
            providers=['DmlExecutionProvider', 'CPUExecutionProvider']
        )

        # Warmup
        for _ in range(10):
            sess_gpu.run(None, {input_name: dummy_input})

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            sess_gpu.run(None, {input_name: dummy_input})
            times.append((time.perf_counter() - start) * 1000)

        mean_time = np.mean(times)
        min_time = np.min(times)
        fps = 1000 / mean_time
        print(f"  Mean: {mean_time:.2f}ms ({fps:.1f} FPS)")
        print(f"  Min:  {min_time:.2f}ms ({1000/min_time:.1f} FPS)")

    except Exception as e:
        print(f"  GPU not available: {e}")

    print()


def run_onnx_benchmark(model_paths=None):
    """Run quick ONNX model benchmark."""

    if model_paths is None:
        # Default: test models in ball_detection/models/
        models_dir = Path(__file__).parent.parent / "models"
        model_paths = [
            str(models_dir / "CustomCNN.onnx"),
            str(models_dir / "mobileLiteV3.onnx"),
            str(models_dir / "best_pixel_error.onnx")
        ]
        # Filter out models that don't exist
        model_paths = [p for p in model_paths if Path(p).exists()]

        if not model_paths:
            print("ERROR: No ONNX models found in ball_detection/models/")
            print("Expected: CustomCNN.onnx, mobileLiteV3.onnx, or best_pixel_error.onnx")
            return

    print("=" * 80)
    print("ONNX MODEL BENCHMARK")
    print("=" * 80)
    print(f"Testing {len(model_paths)} model(s)")
    print("=" * 80)
    print()

    # Benchmark each model
    for model_path in model_paths:
        model_name = Path(model_path).stem

        print("=" * 80)
        print(f"MODEL: {model_name}")
        print("=" * 80)
        print()
        benchmark_onnx_file(model_path, input_size=128, num_iterations=1000)
        print()

    print("=" * 80)


def run_fullframe_benchmark():
    """Benchmark all models at all resolutions (128x128, 320x180, 1280x720)."""

    print("=" * 80)
    print("ALL MODELS x ALL RESOLUTIONS BENCHMARK")
    print("=" * 80)
    print()

    num_iterations = 100

    # All resolutions to test
    resolutions = [
        (128, 128),    # Crop
        (180, 320),    # Medium
        (720, 1280),   # Full-frame
    ]

    # All models to test
    models = {
        'BallDetectorCNN': BallDetectorCNN,
        'MobileNetV2-0.35': create_mobilenet_v2_035,
        'MobileNetV3-Small': create_mobilenet_v3_small,
        'ShuffleNetV2-0.5x': create_shufflenet_v2_x0_5,
        'FullFrame-Tiny': BallDetectorFullFrameTiny,
        'FullFrame-Ultra': BallDetectorFullFrameUltra,
        'FullFrame-MobileNet-Lite': lambda: BallDetectorFullFrameMobileNetLite(pretrained=False),
    }

    # Build all combinations
    all_models = []
    for model_name, model_fn in models.items():
        for h, w in resolutions:
            all_models.append((f"{model_name} ({w}x{h})", model_name, h, w, model_fn))
    results = []

    for name, model_name, h, w, model_fn in all_models:
        print(f"\nTesting: {name}")
        print("-" * 60)

        try:
            model = model_fn()
            model.eval()
            params = count_parameters(model)

            # Export to ONNX
            print(f"  Params: {params:,}")
            print("  Exporting to ONNX...", end=" ", flush=True)

            buffer = io.BytesIO()
            dummy = torch.randn(1, 3, h, w)
            torch.onnx.export(model, dummy, buffer, input_names=['input'],
                            output_names=['output'], opset_version=14, do_constant_folding=True)
            buffer.seek(0)
            onnx_bytes = buffer.read()
            print(f"OK ({len(onnx_bytes)/1024:.0f} KB)")

            # CPU benchmark
            print("  CPU: ", end="", flush=True)
            try:
                cpu_session = create_inference_session(onnx_bytes)
                input_name = cpu_session.get_inputs()[0].name
                dummy_np = np.random.randn(1, 3, h, w).astype(np.float32)

                # Warmup
                for _ in range(10):
                    cpu_session.run(None, {input_name: dummy_np})

                # Benchmark
                times = []
                for _ in range(num_iterations):
                    start = time.perf_counter()
                    cpu_session.run(None, {input_name: dummy_np})
                    times.append((time.perf_counter() - start) * 1000)

                cpu_stats = {'mean': np.mean(times), 'fps': 1000 / np.mean(times)}
                print(f"{cpu_stats['mean']:.2f}ms ({cpu_stats['fps']:.1f} FPS)")
            except Exception as e:
                print(f"FAILED: {e}")
                cpu_stats = {'mean': -1, 'fps': 0}

            # GPU benchmark (DirectML)
            print("  GPU: ", end="", flush=True)
            try:
                gpu_session = ort.InferenceSession(
                    onnx_bytes,
                    providers=['DmlExecutionProvider', 'CPUExecutionProvider']
                )
                input_name = gpu_session.get_inputs()[0].name
                dummy_np = np.random.randn(1, 3, h, w).astype(np.float32)

                # Warmup
                for _ in range(10):
                    gpu_session.run(None, {input_name: dummy_np})

                # Benchmark
                times = []
                for _ in range(num_iterations):
                    start = time.perf_counter()
                    gpu_session.run(None, {input_name: dummy_np})
                    times.append((time.perf_counter() - start) * 1000)

                gpu_stats = {'mean': np.mean(times), 'fps': 1000 / np.mean(times)}
                print(f"{gpu_stats['mean']:.2f}ms ({gpu_stats['fps']:.1f} FPS)")
            except Exception as e:
                print(f"FAILED: {e}")
                gpu_stats = {'mean': -1, 'fps': 0}

            results.append({
                'name': name, 'model_name': model_name, 'size': f"{w}x{h}",
                'h': h, 'w': w, 'params': params,
                'cpu_ms': cpu_stats['mean'], 'cpu_fps': cpu_stats['fps'],
                'gpu_ms': gpu_stats['mean'], 'gpu_fps': gpu_stats['fps']
            })

        except Exception as e:
            print(f"  ERROR: {e}")

    # Rankings per resolution
    resolutions = [(128, 128), (180, 320), (720, 1280)]

    for h, w in resolutions:
        res_results = [r for r in results if r['h'] == h and r['w'] == w]
        if not res_results:
            continue

        print("\n" + "=" * 80)
        print(f"RANKING: {w}x{h}")
        print("=" * 80)

        # CPU ranking
        print(f"\n  CPU (fastest first):")
        print(f"  {'#':<3} {'Model':<28} {'Time':>10} {'FPS':>10} {'Params':>12}")
        print(f"  " + "-" * 70)
        for i, r in enumerate(sorted(res_results, key=lambda x: x['cpu_ms']), 1):
            print(f"  {i:<3} {r['model_name']:<28} {r['cpu_ms']:>8.2f}ms {r['cpu_fps']:>9.1f} {r['params']:>12,}")

        # GPU ranking
        gpu_results = [r for r in res_results if r['gpu_ms'] > 0]
        if gpu_results:
            print(f"\n  GPU (fastest first):")
            print(f"  {'#':<3} {'Model':<28} {'Time':>10} {'FPS':>10} {'Params':>12}")
            print(f"  " + "-" * 70)
            for i, r in enumerate(sorted(gpu_results, key=lambda x: x['gpu_ms']), 1):
                print(f"  {i:<3} {r['model_name']:<28} {r['gpu_ms']:>8.2f}ms {r['gpu_fps']:>9.1f} {r['params']:>12,}")

    print()


if __name__ == "__main__":
    try:
        print("\n" + "=" * 80)
        print("BALL DETECTION MODEL BENCHMARK")
        print("7 models x 3 resolutions = 21 tests")
        print("=" * 80 + "\n")

        run_fullframe_benchmark()

        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETE")
        print("=" * 80 + "\n")

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
