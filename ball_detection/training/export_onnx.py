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

from ..core.model import BallDetectorCNN, BallDetectorMobileNetV3


def is_structurally_pruned(state_dict, use_mobilenet=True):
    """Check if model has been structurally pruned by comparing first layer shape."""
    if use_mobilenet:
        expected_shape = (16, 3, 3, 3)  # MobileNetV3 first conv
        actual_shape = state_dict['features.0.0.weight'].shape
    else:
        expected_shape = (16, 3, 3, 3)  # Custom CNN first conv
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
    if is_structurally_pruned(state_dict, use_mobilenet):
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
