"""
Export Trained PyTorch Model to ONNX Format

Converts the trained BallDetectorCNN or MobileNetV3 to ONNX for deployment with ONNX Runtime + DirectML.
"""

import argparse
import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

from model import BallDetectorCNN, BallDetectorMobileNetV3


def export_to_onnx(pytorch_model_path, output_path, crop_size=128, use_mobilenet=True, opset_version=14):
    """
    Export PyTorch model to ONNX format.

    Args:
        pytorch_model_path: Path to trained PyTorch model (.pth)
        output_path: Output path for ONNX model (.onnx)
        crop_size: Input crop size (128x128)
        use_mobilenet: Whether to use MobileNetV3 architecture
        opset_version: ONNX opset version (14 recommended for DirectML)
    """
    device = torch.device('cpu')  # Export on CPU for compatibility

    # Load PyTorch model
    print("=" * 60)
    print("PYTORCH TO ONNX EXPORT")
    print("=" * 60)
    print(f"Loading PyTorch model from: {pytorch_model_path}")

    if use_mobilenet:
        print("Architecture: MobileNetV3-Small")
        model = BallDetectorMobileNetV3(pretrained=False)
    else:
        print("Architecture: BallDetectorCNN")
        model = BallDetectorCNN()

    state_dict = torch.load(pytorch_model_path, map_location=device)

    # Handle checkpoint vs direct state_dict
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
        print(f"  Loaded from checkpoint (epoch {state_dict.get('epoch', 'unknown')})")

    # Remove _orig_mod. prefix from torch.compile
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("  Removing torch.compile wrapper (_orig_mod. prefix)...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    print("  ✓ Model weights loaded successfully")

    model.eval()
    model.to(device)

    param_count = model.count_parameters()
    print(f"  Parameters: {param_count:,} ({param_count * 4 / 1024:.1f} KB)")
    print()

    # Create dummy input (128x128 RGB crop)
    dummy_input = torch.randn(1, 3, crop_size, crop_size, device=device)

    # Export to ONNX
    print(f"\nExporting to ONNX (opset {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"ONNX model saved to: {output_path}")

    # Verify ONNX model
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX model is valid!")

    # Get model size
    model_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Model size: {model_size_mb:.2f} MB")

    return output_path


def test_onnx_inference(onnx_path, use_directml=True):
    """
    Test ONNX model inference.

    Args:
        onnx_path: Path to ONNX model
        use_directml: Whether to use DirectML execution provider (AMD GPU)
    """
    print("\n" + "=" * 60)
    print("Testing ONNX Inference")
    print("=" * 60)

    # Setup execution providers
    if use_directml:
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        print("Execution providers: DirectML (AMD GPU) + CPU fallback")
    else:
        providers = ['CPUExecutionProvider']
        print("Execution providers: CPU only")

    # Create inference session
    print(f"\nCreating inference session...")
    session = ort.InferenceSession(onnx_path, providers=providers)

    # Print model info
    print(f"\nModel inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: shape={inp.shape}, type={inp.type}")

    print(f"\nModel outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: shape={out.shape}, type={out.type}")

    # Test inference
    print(f"\nRunning test inference...")
    dummy_input = np.random.randn(1, 3, 128, 128).astype(np.float32)

    import time
    start = time.time()
    outputs = session.run(None, {'input': dummy_input})
    inference_time = (time.time() - start) * 1000  # Convert to ms

    print(f"\nInference time: {inference_time:.2f} ms")
    print(f"Output shape: {outputs[0].shape}")
    print(f"Output values: {outputs[0]}")

    x, y = outputs[0][0]
    print(f"\nParsed output:")
    print(f"  x (normalized): {x:.4f}")
    print(f"  y (normalized): {y:.4f}")

    # Benchmark with multiple runs
    print(f"\nBenchmarking (100 runs)...")
    times = []
    for _ in range(100):
        start = time.time()
        session.run(None, {'input': dummy_input})
        times.append((time.time() - start) * 1000)

    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"  Average: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")
    print(f"  Throughput: {1000/avg_time:.1f} FPS")

    print("\nONNX inference test successful!")


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX')

    parser.add_argument('--model', type=str,
                        default='./ball_detection/models/best_pixel_error.pth',
                        help='Path to PyTorch model (.pth)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for ONNX model (.onnx)')
    parser.add_argument('--crop-size', type=int, default=128,
                        help='Input crop size (default: 128)')
    parser.add_argument('--use-custom-cnn', action='store_true',
                        help='Use custom BallDetectorCNN instead of MobileNetV3')
    parser.add_argument('--opset', type=int, default=14,
                        help='ONNX opset version')
    parser.add_argument('--test', action='store_true',
                        help='Test ONNX inference after export')
    parser.add_argument('--no-directml', action='store_true',
                        help='Disable DirectML for testing (CPU only)')

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        model_path = Path(args.model)
        args.output = model_path.parent / f"{model_path.stem}.onnx"

    # Export to ONNX
    onnx_path = export_to_onnx(
        pytorch_model_path=args.model,
        output_path=args.output,
        crop_size=args.crop_size,
        use_mobilenet=not args.use_custom_cnn,
        opset_version=args.opset
    )

    # Test if requested
    if args.test:
        test_onnx_inference(
            onnx_path=onnx_path,
            use_directml=not args.no_directml
        )


if __name__ == "__main__":
    main()
