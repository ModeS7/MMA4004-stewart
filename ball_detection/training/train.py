"""
Training Script for Ball Detection

Two modes:
- CROP: 128x128 crops, outputs (x, y) normalized coordinates
- FULLFRAME: 1280x720 or 320x180 input, outputs (x, y, confidence)

Edit settings below and run: python -m ball_detection.training.train
"""

import time
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..core.model import (
    BallDetectorCNN,
    BallDetectorMobileNetV3,
    BallDetectorShuffleNetV2,
    BallDetectorFullFrameTiny,
    BallDetectorFullFrameUltra,
    BallDetectorFullFrameMobileNet,
    BallDetectorFullFrameShuffleNet,
)
from ..core.dataset import create_dataloaders, create_fullframe_dataloaders, create_fullframe_memmap_dataloaders

# ============================================================
# SETTINGS
# ============================================================

# Mode: "crop" or "fullframe"
MODE = "fullframe"

# Model selection (depends on mode):
#
# CROP MODE (128x128 input → x, y output):
#   "cnn"        - Custom CNN with residual blocks
#   "mobilenet"  - MobileNetV3-Small backbone
#   "shufflenet" - ShuffleNetV2 x0.5 backbone
#
# FULLFRAME MODE (1280x720 or 320x180 input → x, y, confidence output):
#   "tiny"      - Custom lightweight backbone (~150K params)
#                 Aggressive downsampling, depthwise separable convs
#   "ultra"     - PixelUnshuffle + custom backbone (~138K params)
#                 Zero-compute 8x spatial reduction, very fast on GPU
#   "shufflenet"- PixelUnshuffle + ShuffleNetV2 x0.5 (~400K params)
#                 ImageNet pretrained, fast inference
#   "mobilenet" - PixelUnshuffle + MobileNetV3-Small (~1M params)
#                 ImageNet pretrained, best fullframe accuracy
#
MODEL = "shufflenet"

# Resolution for fullframe mode (width, height)
# Common: (320, 180) for speed, (1280, 720) for accuracy
RESOLUTION = (320, 180)

# Training parameters
EPOCHS = 2000
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 10
NUM_WORKERS = 8
TRAIN_SPLIT = 0.8

# Data
DATA_DIR = "./ball_detection/data/full_dataset/training_data_full"
OUTPUT_DIR = "./ball_detection/models"

# Memmap (fullframe mode only - faster loading)
USE_MEMMAP = True
MEMMAP_DIR = "./ball_detection/data/fullframe_memmap"

# Caching (fullframe + memmap only - pre-augmented images in RAM)
USE_CACHE = True
CACHE_MULTIPLIER = 3  # Cache size = dataset_size * multiplier

# Augmentation (crop mode only)
USE_SPATIAL_AUG = True
USE_APPEARANCE_AUG = True
USE_COLOR_INVARIANCE_AUG = True
USE_TEARING_AUG = True
TEARING_PROBABILITY = 0.01

# Checkpoints
SAVE_INTERVAL = 100
VALIDATION_INTERVAL = 2

# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(mode: str, model_name: str, pretrained: bool = True):
    """
    Create model based on mode and name.

    Args:
        mode: "crop" or "fullframe"
        model_name: Model architecture name
        pretrained: Use pretrained backbone (where applicable)

    Returns:
        PyTorch model
    """
    if mode == "crop":
        if model_name == "cnn":
            return BallDetectorCNN()
        elif model_name == "mobilenet":
            return BallDetectorMobileNetV3(pretrained=pretrained)
        elif model_name == "shufflenet":
            return BallDetectorShuffleNetV2(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown crop model: {model_name}. Use 'cnn', 'mobilenet', or 'shufflenet'")

    elif mode == "fullframe":
        if model_name == "tiny":
            return BallDetectorFullFrameTiny()
        elif model_name == "ultra":
            return BallDetectorFullFrameUltra()
        elif model_name == "shufflenet":
            return BallDetectorFullFrameShuffleNet(pretrained=pretrained)
        elif model_name == "mobilenet":
            return BallDetectorFullFrameMobileNet(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown fullframe model: {model_name}. Use 'tiny', 'ultra', 'shufflenet', or 'mobilenet'")

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'crop' or 'fullframe'")


# ============================================================
# LOSS FUNCTIONS
# ============================================================

class CropLoss(nn.Module):
    """MSE loss for (x, y) coordinate regression."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # pred: (batch, 2), target: (batch, 3) -> use only x, y
        return self.mse(pred, target[:, :2])


class FullframeLoss(nn.Module):
    """Combined MSE + BCE loss for (x, y, confidence)."""
    def __init__(self, coord_weight=1.0, conf_weight=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.coord_weight = coord_weight
        self.conf_weight = conf_weight

    def forward(self, pred, target):
        # pred: (batch, 3), target: (batch, 3)
        coord_loss = self.mse(pred[:, :2], target[:, :2])
        conf_loss = self.bce(torch.sigmoid(pred[:, 2]), target[:, 2])
        return self.coord_weight * coord_loss + self.conf_weight * conf_loss


# ============================================================
# METRICS
# ============================================================

def calculate_pixel_error(pred, target, size):
    """Calculate average pixel error."""
    # pred, target: (batch, 2+) normalized coordinates
    pred_pixels = pred[:, :2] * size
    target_pixels = target[:, :2] * size
    diff = pred_pixels - target_pixels
    distances = torch.sqrt((diff ** 2).sum(dim=1))
    return distances.mean().item()


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, pixel_size):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_pixel_error = 0
    num_batches = len(dataloader)
    use_amp = device.type == 'cuda'

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} Training')
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward pass with AMP
        if use_amp:
            with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                outputs = model(images)
        else:
            outputs = model(images)

        # Loss in fp32
        loss = criterion(outputs.float(), targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        pixel_error = calculate_pixel_error(outputs.float(), targets, pixel_size)
        total_loss += loss.detach()
        total_pixel_error += pixel_error

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'px_err': f'{pixel_error:.3f}'})

    avg_loss = (total_loss / num_batches).item()
    avg_pixel_error = total_pixel_error / num_batches
    return avg_loss, avg_pixel_error


def validate(model, dataloader, criterion, device, pixel_size):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_pixel_error = 0
    num_batches = len(dataloader)

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Validation'):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)
            pixel_error = calculate_pixel_error(outputs, targets, pixel_size)

            total_loss += loss.item()
            total_pixel_error += pixel_error

    return total_loss / num_batches, total_pixel_error / num_batches


# ============================================================
# MAIN
# ============================================================

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("=" * 60)
    print("BALL DETECTION TRAINING")
    print("=" * 60)
    print(f"Mode: {MODE}")
    print(f"Model: {MODEL}")
    if MODE == "fullframe":
        print(f"Resolution: {RESOLUTION[0]}x{RESOLUTION[1]}")
    print(f"Device: {device}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 60)

    # Create output directory
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"run_{timestamp}_{MODEL}"
    if MODE == "fullframe":
        run_name += f"_{RESOLUTION[0]}x{RESOLUTION[1]}"

    output_dir = Path(OUTPUT_DIR) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {output_dir}\n")

    # Save config
    config = {
        'mode': MODE,
        'model': MODEL,
        'resolution': RESOLUTION if MODE == "fullframe" else (128, 128),
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # TensorBoard
    writer = SummaryWriter(log_dir=output_dir / 'tensorboard')

    # Create dataloaders
    if MODE == "crop":
        print(f"Loading data from: {DATA_DIR}")
        train_loader, val_loader = create_dataloaders(
            data_dir=DATA_DIR,
            batch_size=BATCH_SIZE,
            crop_size=128,
            train_split=TRAIN_SPLIT,
            num_workers=NUM_WORKERS,
            use_spatial_augmentation=USE_SPATIAL_AUG,
            use_appearance_augmentation=USE_APPEARANCE_AUG,
            use_color_invariance_augmentation=USE_COLOR_INVARIANCE_AUG,
            use_tearing_augmentation=USE_TEARING_AUG,
            tearing_probability=TEARING_PROBABILITY,
        )
        pixel_size = 128
        criterion = CropLoss()
    else:  # fullframe
        if USE_MEMMAP:
            print(f"Loading memmap data from: {MEMMAP_DIR}")
            if USE_CACHE:
                print(f"Using cached augmentation (multiplier={CACHE_MULTIPLIER})")
            train_loader, val_loader = create_fullframe_memmap_dataloaders(
                data_dir=MEMMAP_DIR,
                batch_size=BATCH_SIZE,
                train_split=TRAIN_SPLIT,
                num_workers=NUM_WORKERS,
                use_cache=USE_CACHE,
                cache_multiplier=CACHE_MULTIPLIER,
            )
        else:
            print(f"Loading data from: {DATA_DIR}")
            train_loader, val_loader = create_fullframe_dataloaders(
                data_dir=DATA_DIR,
                batch_size=BATCH_SIZE,
                target_size=RESOLUTION,
                train_split=TRAIN_SPLIT,
                num_workers=NUM_WORKERS,
            )
        # Use diagonal for pixel error (accounts for both dimensions)
        pixel_size = (RESOLUTION[0]**2 + RESOLUTION[1]**2) ** 0.5
        criterion = FullframeLoss()

    # Create model
    print(f"\nCreating model: {MODEL}")
    model = create_model(MODE, MODEL, pretrained=True)
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Warmup + cosine annealing
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS]
    )

    # Training loop
    print(f"\nStarting training for {EPOCHS} epochs...")
    print("=" * 60)

    best_val_loss = float('inf')
    best_pixel_error = float('inf')
    training_history = []

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_px_err = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, pixel_size
        )

        # Validate (every N epochs)
        if epoch % VALIDATION_INTERVAL == 0 or epoch == 1:
            val_loss, val_px_err = validate(model, val_loader, criterion, device, pixel_size)

        scheduler.step()
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        # Print summary
        if epoch % VALIDATION_INTERVAL == 0 or epoch == 1:
            print(f"\nEpoch {epoch}/{EPOCHS}:")
            print(f"  Train: loss={train_loss:.4f}, px_err={train_px_err:.3f}")
            print(f"  Val:   loss={val_loss:.4f}, px_err={val_px_err:.3f}")
            print(f"  LR: {current_lr:.6f}, Time: {epoch_time:.1f}s")
        else:
            print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}, px_err={train_px_err:.3f}, time={epoch_time:.1f}s")

        # Save history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_pixel_error': train_px_err,
            'val_loss': val_loss if epoch % VALIDATION_INTERVAL == 0 else None,
            'val_pixel_error': val_px_err if epoch % VALIDATION_INTERVAL == 0 else None,
            'lr': current_lr,
        })

        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('PixelError/train', train_px_err, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        if epoch % VALIDATION_INTERVAL == 0 or epoch == 1:
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('PixelError/val', val_px_err, epoch)

        # Save best models (only on validation epochs)
        if epoch % VALIDATION_INTERVAL == 0 or epoch == 1:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), output_dir / 'best_model.pth')
                print(f"  * New best loss!")

            if val_px_err < best_pixel_error:
                best_pixel_error = val_px_err
                torch.save(model.state_dict(), output_dir / 'best_pixel_error.pth')
                print(f"  * New best pixel error!")

        # Periodic checkpoints
        if epoch % SAVE_INTERVAL == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, output_dir / f'checkpoint_epoch_{epoch}.pth')
            print(f"  Checkpoint saved")

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    # Export ONNX
    print("\n" + "=" * 60)
    print("EXPORTING TO ONNX")
    print("=" * 60)

    try:
        onnx_path = output_dir / f"{run_name}.onnx"

        if MODE == "crop":
            dummy_input = torch.randn(1, 3, 128, 128).to(device)
        else:
            dummy_input = torch.randn(1, 3, RESOLUTION[1], RESOLUTION[0]).to(device)

        # Load best model for export
        model.load_state_dict(torch.load(output_dir / 'best_pixel_error.pth'))
        model.eval()

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

        # Verify
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        model_size = onnx_path.stat().st_size / (1024 * 1024)
        print(f"Exported: {onnx_path}")
        print(f"Size: {model_size:.2f} MB")

    except Exception as e:
        print(f"ONNX export failed: {e}")

    # Close TensorBoard
    writer.close()

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Run: {run_name}")
    print(f"Best loss: {best_val_loss:.4f}")
    print(f"Best pixel error: {best_pixel_error:.3f}px")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
