"""
Export Trained PyTorch Model to ONNX Format

Converts the trained BallDetectorCNN or MobileNetV3 to ONNX for deployment with ONNX Runtime + DirectML.

For STRUCTURED PRUNED models (torch_pruning), this script dynamically reconstructs
the pruned architecture from the checkpoint's layer shapes.
"""

import argparse
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

from ..core.model import BallDetectorCNN, BallDetectorMobileNetV3, BallDetectorShuffleNetV2


def count_onnx_nodes(model_path):
    """Count number of nodes in ONNX model."""
    model = onnx.load(model_path)
    return len(model.graph.node)


def count_onnx_params(model_path):
    """Count total parameters in ONNX model."""
    model = onnx.load(model_path)
    total = 0
    for init in model.graph.initializer:
        total += np.prod(init.dims) if init.dims else 1
    return total


def optimize_onnx_graph(model_path, output_path=None, for_quantization=False):
    """
    Optimize ONNX graph using ONNX Runtime's built-in optimizer.

    Removes identity ops, fuses layers, and eliminates dead code.
    Uses ONNX Runtime (already installed) instead of onnxoptimizer.

    Args:
        model_path: Path to input ONNX model
        output_path: Path to save optimized model (defaults to overwriting input)
        for_quantization: If True, use basic optimization to maintain
                         compatibility with INT8 quantization

    Returns:
        Path to optimized model
    """
    import shutil

    if output_path is None:
        output_path = model_path

    print(f"  Optimizing ONNX graph...")
    original_nodes = count_onnx_nodes(model_path)

    # Use ONNX Runtime session with optimization to create optimized model
    sess_options = ort.SessionOptions()

    # For INT8 quantization: use basic optimization only
    # ORT_ENABLE_ALL creates NCHWC-format nodes incompatible with ConvInteger
    if for_quantization:
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        print(f"  Using BASIC optimization (for INT8 quantization compatibility)")
    else:
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess_options.optimized_model_filepath = str(output_path) + ".tmp"

    try:
        # Create session which triggers optimization and saves to file
        _ = ort.InferenceSession(str(model_path), sess_options, providers=['CPUExecutionProvider'])

        # Move optimized model to output path
        if Path(sess_options.optimized_model_filepath).exists():
            shutil.move(sess_options.optimized_model_filepath, output_path)
            new_nodes = count_onnx_nodes(output_path)
            print(f"  Nodes: {original_nodes} -> {new_nodes} ({original_nodes - new_nodes} removed)")
        else:
            print(f"  [WARNING] Optimized model not created")

        return output_path
    except Exception as e:
        print(f"  [WARNING] Optimization failed: {e}")
        # Clean up temp file if exists
        tmp_path = Path(str(output_path) + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        return model_path


def quantize_to_int8(model_path, output_path=None):
    """
    Apply INT8 quantization using QDQ format for better compatibility.

    Uses static quantization with QDQ (QuantizeLinear/DequantizeLinear) format
    instead of fused integer operators (ConvInteger) which have limited support.

    Args:
        model_path: Path to input ONNX model (FP32)
        output_path: Path to save INT8 model (defaults to model_int8.onnx)

    Returns:
        Path to quantized model
    """
    import onnx
    import numpy as np
    from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader

    if output_path is None:
        output_path = str(model_path).replace('.onnx', '_int8.onnx')

    # Get input shape from model
    model = onnx.load(str(model_path))
    input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    if input_shape[0] == 0:  # Dynamic batch
        input_shape[0] = 1

    # Calibration data reader with random data (for weight quantization)
    class DummyCalibrationReader(CalibrationDataReader):
        def __init__(self, input_name, input_shape, num_samples=10):
            self.input_name = input_name
            self.input_shape = input_shape
            self.num_samples = num_samples
            self.current = 0

        def get_next(self):
            if self.current >= self.num_samples:
                return None
            self.current += 1
            # Random calibration data (normalized like ImageNet)
            data = np.random.randn(*self.input_shape).astype(np.float32)
            return {self.input_name: data}

    input_name = model.graph.input[0].name
    calibration_reader = DummyCalibrationReader(input_name, input_shape)

    print(f"  Applying INT8 quantization (QDQ format)...")
    quantize_static(
        model_input=str(model_path),
        model_output=str(output_path),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,  # Use QDQ format (universally supported)
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        extra_options={'DefaultTensorType': onnx.TensorProto.FLOAT}
    )

    print(f"  INT8 model saved: {output_path}")
    return output_path


def is_structurally_pruned(state_dict, model_type='mobilenet'):
    """Check if model has been structurally pruned by comparing first layer shape."""
    if model_type == 'mobilenet':
        expected_shape = (16, 3, 3, 3)  # MobileNetV3 first conv
        # Try different key patterns (with/without 'model.' prefix, etc.)
        possible_keys = [
            'features.0.0.weight',
            'model.features.0.0.weight',
            'backbone.features.0.0.weight',
        ]
        actual_shape = None
        for key in possible_keys:
            if key in state_dict:
                actual_shape = state_dict[key].shape
                break
        if actual_shape is None:
            # Key not found - assume not pruned (standard model)
            return False
    elif model_type == 'shufflenet':
        expected_shape = (24, 3, 3, 3)  # ShuffleNetV2 x0.5 first conv
        if 'conv1.0.weight' not in state_dict:
            return False
        actual_shape = state_dict['conv1.0.weight'].shape
    else:
        expected_shape = (16, 3, 3, 3)  # Custom CNN first conv
        if 'conv1.weight' not in state_dict:
            return False
        actual_shape = state_dict['conv1.weight'].shape

    return tuple(actual_shape) != expected_shape


def build_pruned_mobilenetv3_from_state_dict(state_dict):
    """
    Reconstruct a pruned MobileNetV3 model by using torch_pruning to re-prune.

    Since we can't infer the exact architecture, we'll:
    1. Load an unpruned model
    2. Progressively prune it until parameter count matches
    3. Load the state dict
    """
    print("  Reconstructing pruned model using iterative pruning...")

    # Count parameters in the pruned state dict (exclude buffers like running_mean/running_var)
    target_params = sum(v.numel() for k, v in state_dict.items()
                       if not any(x in k for x in ['running_mean', 'running_var', 'num_batches_tracked']))

    print(f"    Target parameters: {target_params:,}")

    # Load unpruned model
    from ..core.model import BallDetectorMobileNetV3
    model = BallDetectorMobileNetV3(pretrained=False)

    original_params = sum(p.numel() for p in model.parameters())
    print(f"    Original parameters: {original_params:,}")

    # Apply iterative pruning until we match target
    import torch_pruning as tp

    prune_ratio_needed = 1.0 - (target_params / original_params)
    print(f"    Need to prune: {prune_ratio_needed*100:.1f}%")

    # Prune in steps
    current_params = original_params
    step = 0

    while current_params > target_params * 1.01:  # Allow 1% tolerance
        step += 1

        # Calculate prune amount for this step
        remaining_to_prune = current_params - target_params
        prune_this_step = min(0.1, remaining_to_prune / current_params)  # Max 10% per step

        if prune_this_step < 0.01:  # Less than 1%, do final prune
            prune_this_step = remaining_to_prune / current_params

        print(f"    Step {step}: Pruning {prune_this_step*100:.1f}%...")

        # Apply pruning
        imp = tp.importance.MagnitudeImportance(p=1)
        ignored_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.out_features == 2:
                ignored_layers.append(module)

        try:
            pruner = tp.pruner.MagnitudePruner(
                model,
                example_inputs=torch.randn(1, 3, 128, 128),
                importance=imp,
                iterative_steps=1,
                pruning_ratio=prune_this_step,
                ignored_layers=ignored_layers,
            )
            pruner.step()
        except Exception as e:
            print(f"    [WARNING] Pruning failed: {e}")
            print(f"    Achieved {current_params:,} params (target: {target_params:,})")
            break

        # Reset BatchNorm stats
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.reset_running_stats()

        current_params = sum(p.numel() for p in model.parameters())
        print(f"      -> {current_params:,} params")

        if step > 50:  # Safety limit
            print(f"    [WARNING] Reached step limit")
            break

    print(f"    [OK] Pruned to {current_params:,} params (target: {target_params:,})")

    # Now load the state dict
    try:
        model.load_state_dict(state_dict, strict=False)
        print(f"    [OK] State dict loaded")
        return model
    except Exception as e:
        print(f"    [ERROR] Failed to load state dict: {e}")
        return None


def export_pruned_model_via_scripting(state_dict, output_path, crop_size, device):
    """
    Export a structurally pruned model by reconstructing from state dict.
    """
    print("\n" + "!" * 60)
    print("STRUCTURED PRUNING DETECTED")
    print("!" * 60)
    print("\nReconstructing pruned model architecture from state dict...")

    # Build a model that matches the pruned architecture
    model = build_pruned_mobilenetv3_from_state_dict(state_dict)

    if model is None:
        print("\n[WARNING] Automatic reconstruction not yet implemented.")
        print("The model architecture is too complex to infer from state dict alone.")
        print()
        print("SOLUTION: Check if ONNX was already exported during training.")
        print("The train.py script automatically exports ONNX at the end (line 1073).")
        print()
        print("If training completed, check the model directory for .onnx file.")
        print("If not exported, the training may have been interrupted.")
        print()
        raise RuntimeError("Cannot export structurally pruned models from checkpoint alone")

    # Load the state dict
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)

    print(f"  [OK] Pruned model reconstructed ({model.count_parameters():,} parameters)")

    # Export to ONNX
    dummy_input = torch.randn(1, 3, crop_size, crop_size, device=device)
    torch.onnx.export(
        model, dummy_input, output_path,
        opset_version=14,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print(f"  [OK] ONNX exported to: {output_path}")
    return output_path


def export_to_onnx(pytorch_model_path, output_path, crop_size=128, model_type='mobilenet', opset_version=14):
    """
    Export PyTorch model to ONNX format.

    Args:
        pytorch_model_path: Path to trained PyTorch model (.pth)
        output_path: Output path for ONNX model (.onnx)
        crop_size: Input crop size (128x128)
        model_type: Model architecture ('mobilenet', 'shufflenet', or 'custom')
        opset_version: ONNX opset version (14 recommended for DirectML)
    """
    device = torch.device('cpu')  # Export on CPU for compatibility

    # Load PyTorch model
    print("=" * 60)
    print("PYTORCH TO ONNX EXPORT")
    print("=" * 60)
    print(f"Loading PyTorch model from: {pytorch_model_path}")

    if model_type == 'shufflenet':
        print("Architecture: ShuffleNetV2 x0.5")
        model = BallDetectorShuffleNetV2(pretrained=False)
    elif model_type == 'mobilenet':
        print("Architecture: MobileNetV3-Small")
        model = BallDetectorMobileNetV3(pretrained=False)
    else:
        print("Architecture: BallDetectorCNN")
        model = BallDetectorCNN()

    checkpoint = torch.load(pytorch_model_path, map_location=device)

    # Handle checkpoint vs direct state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'unknown')
        val_error = checkpoint.get('val_pixel_error', 'unknown')
        print(f"  Loaded checkpoint from epoch {epoch}")
        print(f"  Validation error: {val_error}")
    else:
        state_dict = checkpoint
        epoch = 'unknown'
        val_error = 'unknown'

    # Check for STRUCTURED pruning (torch_pruning) - layer dimensions changed
    if is_structurally_pruned(state_dict, model_type):
        onnx_path = export_pruned_model_via_scripting(state_dict, output_path, crop_size, device)
        return onnx_path  # Return the exported path

    # Remove _orig_mod. prefix from torch.compile
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("  Removing torch.compile wrapper (_orig_mod. prefix)...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Handle UNSTRUCTURED pruned models (weight_orig + weight_mask → weight)
    # This is old-style pruning that doesn't change architecture
    if any(k.endswith('_orig') for k in state_dict.keys()):
        print("  Detected unstructured pruned model - making pruning permanent...")
        new_state_dict = {}

        # Find all pruned parameters
        pruned_params = set()
        for key in state_dict.keys():
            if key.endswith('_orig'):
                param_name = key[:-5]  # Remove '_orig'
                pruned_params.add(param_name)

        # Convert pruned parameters
        for param_name in pruned_params:
            orig_key = f"{param_name}_orig"
            mask_key = f"{param_name}_mask"

            if mask_key in state_dict:
                # Apply mask: weight = weight_orig * mask
                weight = state_dict[orig_key] * state_dict[mask_key]
                new_state_dict[param_name] = weight
                print(f"    {param_name}: {(state_dict[mask_key] == 0).sum().item()} / {state_dict[mask_key].numel()} weights pruned")
            else:
                # No mask, just copy original
                new_state_dict[param_name] = state_dict[orig_key]

        # Copy non-pruned parameters
        for key, value in state_dict.items():
            if not (key.endswith('_orig') or key.endswith('_mask')):
                new_state_dict[key] = value

        state_dict = new_state_dict
        print("  [OK] Unstructured pruning made permanent")

    model.load_state_dict(state_dict)
    print("  [OK] Model weights loaded successfully")

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
                        help='Path to PyTorch model (.pth) or existing ONNX model (.onnx)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for ONNX model (.onnx)')
    parser.add_argument('--crop-size', type=int, default=128,
                        help='Input crop size (default: 128)')
    parser.add_argument('--model-type', type=str, default='mobilenet',
                        choices=['mobilenet', 'shufflenet', 'custom'],
                        help='Model architecture (default: mobilenet)')
    parser.add_argument('--opset', type=int, default=14,
                        help='ONNX opset version')
    parser.add_argument('--test', action='store_true',
                        help='Test ONNX inference after export')
    parser.add_argument('--no-directml', action='store_true',
                        help='Disable DirectML for testing (CPU only)')
    parser.add_argument('--no-optimize', action='store_true',
                        help='Skip ONNX graph optimization')
    parser.add_argument('--quantize', type=str, choices=['int8'], default=None,
                        help='Quantization type (int8 for CPU inference)')

    args = parser.parse_args()

    model_path = Path(args.model)

    # Check if input is already ONNX (optimize/quantize only)
    if model_path.suffix.lower() == '.onnx':
        print("=" * 60)
        print("ONNX MODEL PROCESSING")
        print("=" * 60)
        print(f"Input: {model_path}")
        print(f"Nodes: {count_onnx_nodes(str(model_path))}")
        print(f"Params: {count_onnx_params(str(model_path)):,}")

        onnx_path = model_path
        if args.output:
            # Copy to output path first
            import shutil
            shutil.copy(model_path, args.output)
            onnx_path = Path(args.output)
    else:
        # Set default output path for PyTorch export
        if args.output is None:
            args.output = model_path.parent / f"{model_path.stem}.onnx"

        # Export to ONNX
        onnx_path = export_to_onnx(
            pytorch_model_path=args.model,
            output_path=args.output,
            crop_size=args.crop_size,
            model_type=args.model_type,
            opset_version=args.opset
        )

    # Optimize graph (remove empty ops from pruned models)
    if not args.no_optimize:
        print("\n" + "=" * 60)
        print("ONNX GRAPH OPTIMIZATION")
        print("=" * 60)
        # Use basic optimization if INT8 quantization is requested
        # (NCHWC transforms are incompatible with ConvInteger operators)
        optimize_onnx_graph(onnx_path, for_quantization=(args.quantize == 'int8'))

    # Quantize to INT8
    if args.quantize == 'int8':
        print("\n" + "=" * 60)
        print("INT8 QUANTIZATION")
        print("=" * 60)
        int8_path = quantize_to_int8(onnx_path)
        print(f"\nGenerated files:")
        print(f"  FP32: {onnx_path}")
        print(f"  INT8: {int8_path}")

    # Test if requested
    if args.test:
        test_onnx_inference(
            onnx_path=onnx_path,
            use_directml=not args.no_directml
        )


if __name__ == "__main__":
    main()
