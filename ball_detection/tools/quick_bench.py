#!/usr/bin/env python3
"""
Quick benchmark for trained ONNX model.
"""
import numpy as np
import onnxruntime as ort
import time


def benchmark_onnx_model(model_path, input_size=128, num_iterations=100, stereo_mode=False):
    """Benchmark ONNX model on CPU and GPU."""

    print(f"Benchmarking: {model_path}")
    print(f"Input size: {input_size}x{input_size}")
    print(f"Mode: {'Stereo (2 frames)' if stereo_mode else 'Single frame'}")
    print(f"Iterations: {num_iterations}")
    print()

    # Test data (single frame, batch_size=1)
    dummy_input1 = np.random.rand(1, 3, input_size, input_size).astype(np.float32)
    dummy_input2 = np.random.rand(1, 3, input_size, input_size).astype(np.float32)

    # CPU session
    print("CPU Inference:")
    sess_cpu = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = sess_cpu.get_inputs()[0].name

    # Warmup
    for _ in range(10):
        sess_cpu.run(None, {input_name: dummy_input1})
        if stereo_mode:
            sess_cpu.run(None, {input_name: dummy_input2})

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        sess_cpu.run(None, {input_name: dummy_input1})
        if stereo_mode:
            sess_cpu.run(None, {input_name: dummy_input2})
        times.append((time.perf_counter() - start) * 1000)

    mean_time = np.mean(times)
    min_time = np.min(times)
    fps = 1000 / mean_time
    per_frame = mean_time / (2 if stereo_mode else 1)
    print(f"  Mean: {mean_time:.2f}ms ({fps:.1f} FPS)")
    if stereo_mode:
        print(f"  Per frame: {per_frame:.2f}ms")
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
            sess_gpu.run(None, {input_name: dummy_input1})
            if stereo_mode:
                sess_gpu.run(None, {input_name: dummy_input2})

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            sess_gpu.run(None, {input_name: dummy_input1})
            if stereo_mode:
                sess_gpu.run(None, {input_name: dummy_input2})
            times.append((time.perf_counter() - start) * 1000)

        mean_time = np.mean(times)
        min_time = np.min(times)
        fps = 1000 / mean_time
        per_frame = mean_time / (2 if stereo_mode else 1)
        print(f"  Mean: {mean_time:.2f}ms ({fps:.1f} FPS)")
        if stereo_mode:
            print(f"  Per frame: {per_frame:.2f}ms")
        print(f"  Min:  {min_time:.2f}ms ({1000/min_time:.1f} FPS)")

    except Exception as e:
        print(f"  GPU not available: {e}")

    print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "../models/best_pixel_error.onnx"

    # Benchmark single frame
    print("=" * 60)
    print("SINGLE FRAME (mono camera)")
    print("=" * 60)
    benchmark_onnx_model(model_path, input_size=128, num_iterations=1000, stereo_mode=False)

    # Benchmark stereo (2 sequential inferences)
    print("=" * 60)
    print("STEREO (ZED camera - left + right frames)")
    print("=" * 60)
    benchmark_onnx_model(model_path, input_size=128, num_iterations=1000, stereo_mode=True)

    print("=" * 60)
    print("RECOMMENDATION:")
    print("For ZED stereo: Use GPU if per-frame time < 10ms")
    print("Allows CPU to handle triangulation/control while GPU infers")
