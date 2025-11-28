"""
Training Script for Stereo Ball Detection (StereoTiny)

Trains the 6-channel stereo detector for simultaneous left+right ball detection.
Input: (B, 6, 720, 1280) - concatenated left+right RGB
Output: (B, 6) - (x_l, y_l, conf_l, x_r, y_r, conf_r)

Usage:
    python -m ball_detection.training.train_stereo
"""

import time
from pathlib import Path
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from ..core.model import StereoTiny
from ..core.stereo_dataset import create_stereo_dataloaders

# ============================================================
# SETTINGS - Edit these
# ============================================================
DATA_DIRS = [
    "./ball_detection/data/old_labels",
    "./ball_detection/data/new_labels",
]
OUTPUT_DIR = "./ball_detection/models"
EPOCHS = 500
BATCH_SIZE = 8  # 6-channel 1280x720 is large - adjust based on VRAM
LEARNING_RATE = 0.001
WARMUP_EPOCHS = 10
WEIGHT_DECAY = 1e-4
TRAIN_SPLIT = 0.8
NUM_WORKERS = 4
SAVE_INTERVAL = 50
VALIDATION_INTERVAL = 2

# Image dimensions (ZED camera)
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280

# ============================================================
# TENSORBOARD SETTINGS
# ============================================================
ENABLE_TENSORBOARD = True
TENSORBOARD_LOG_INTERVAL = 1
TENSORBOARD_IMAGE_INTERVAL = 50
# ============================================================


class StereoDetectionLoss(nn.Module):
    """
    Loss for stereo ball detection.

    Combines MSE for coordinates and BCE for confidence.
    Only penalizes coordinate error when confidence is high (valid detection).
    """
    def __init__(self, coord_weight=1.0, conf_weight=0.5):
        super().__init__()
        self.coord_weight = coord_weight
        self.conf_weight = conf_weight
        self.mse = nn.MSELoss(reduction='none')
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        """
        Args:
            pred: (batch, 6) - (x_l, y_l, conf_l, x_r, y_r, conf_r)
            target: (batch, 6) - same format

        Returns:
            loss: scalar tensor
        """
        # Split predictions
        pred_coords_left = pred[:, :2]  # x_l, y_l
        pred_conf_left = pred[:, 2]     # conf_l
        pred_coords_right = pred[:, 3:5]  # x_r, y_r
        pred_conf_right = pred[:, 5]    # conf_r

        # Split targets
        target_coords_left = target[:, :2]
        target_conf_left = target[:, 2]
        target_coords_right = target[:, 3:5]
        target_conf_right = target[:, 5]

        # Coordinate loss (weighted by target confidence)
        coord_loss_left = self.mse(pred_coords_left, target_coords_left)
        coord_loss_left = (coord_loss_left * target_conf_left.unsqueeze(1)).mean()

        coord_loss_right = self.mse(pred_coords_right, target_coords_right)
        coord_loss_right = (coord_loss_right * target_conf_right.unsqueeze(1)).mean()

        coord_loss = coord_loss_left + coord_loss_right

        # Confidence loss
        conf_loss_left = self.bce(pred_conf_left, target_conf_left)
        conf_loss_right = self.bce(pred_conf_right, target_conf_right)
        conf_loss = conf_loss_left + conf_loss_right

        # Combined loss
        total_loss = self.coord_weight * coord_loss + self.conf_weight * conf_loss

        return total_loss


def calculate_pixel_error(pred, target, image_width=1280, image_height=720):
    """
    Calculate average pixel error for valid detections.

    Args:
        pred: (batch, 6) predicted values
        target: (batch, 6) ground truth values
        image_width: Image width for denormalization
        image_height: Image height for denormalization

    Returns:
        avg_error_left, avg_error_right (in pixels)
    """
    # Split predictions
    pred_x_left = pred[:, 0] * image_width
    pred_y_left = pred[:, 1] * image_height
    pred_x_right = pred[:, 3] * image_width
    pred_y_right = pred[:, 4] * image_height

    # Split targets
    target_x_left = target[:, 0] * image_width
    target_y_left = target[:, 1] * image_height
    target_conf_left = target[:, 2]
    target_x_right = target[:, 3] * image_width
    target_y_right = target[:, 4] * image_height
    target_conf_right = target[:, 5]

    # Calculate Euclidean distance
    dist_left = torch.sqrt((pred_x_left - target_x_left)**2 + (pred_y_left - target_y_left)**2)
    dist_right = torch.sqrt((pred_x_right - target_x_right)**2 + (pred_y_right - target_y_right)**2)

    # Only count valid detections
    valid_left = target_conf_left > 0.5
    valid_right = target_conf_right > 0.5

    error_left = dist_left[valid_left].mean().item() if valid_left.any() else 0.0
    error_right = dist_right[valid_right].mean().item() if valid_right.any() else 0.0

    return error_left, error_right


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_error_left = 0
    total_error_right = 0
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

        # Loss computation in fp32
        loss = criterion(outputs.float(), targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        error_left, error_right = calculate_pixel_error(outputs.float(), targets)

        total_loss += loss.detach()
        total_error_left += error_left
        total_error_right += error_right

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'px_L': f'{error_left:.1f}',
            'px_R': f'{error_right:.1f}'
        })

    avg_loss = (total_loss / num_batches).item()
    avg_error_left = total_error_left / num_batches
    avg_error_right = total_error_right / num_batches

    return avg_loss, avg_error_left, avg_error_right


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()

    total_loss = 0
    total_error_left = 0
    total_error_right = 0
    num_batches = len(dataloader)

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Validation'):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            error_left, error_right = calculate_pixel_error(outputs, targets)

            total_loss += loss.item()
            total_error_left += error_left
            total_error_right += error_right

    avg_loss = total_loss / num_batches
    avg_error_left = total_error_left / num_batches
    avg_error_right = total_error_right / num_batches

    return avg_loss, avg_error_left, avg_error_right


def main():
    """Main training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # CUDA optimizations
    if device.type == 'cuda':
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    print("=" * 60)
    print("STEREO BALL DETECTION TRAINING (StereoTiny)")
    print("=" * 60)
    print(f"Data dirs: {DATA_DIRS}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Input: {IMAGE_WIDTH}x{IMAGE_HEIGHT} stereo (6 channels)")
    print("=" * 60)
    print()

    # Create output directory
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"run_{timestamp}_stereo_tiny"
    output_dir = Path(OUTPUT_DIR) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run Name: {run_name}")
    print(f"Output Directory: {output_dir}")
    print()

    # Save config
    config = {
        'run_name': run_name,
        'timestamp': timestamp,
        'data_dirs': DATA_DIRS,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'train_split': TRAIN_SPLIT,
        'model': 'StereoTiny',
        'input_shape': [6, IMAGE_HEIGHT, IMAGE_WIDTH],
        'output_shape': [6],  # x_l, y_l, conf_l, x_r, y_r, conf_r
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # TensorBoard
    if ENABLE_TENSORBOARD:
        tensorboard_base = Path(OUTPUT_DIR) / 'tensorboard_logs'
        tensorboard_dir = tensorboard_base / run_name
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
        print(f"TensorBoard: {tensorboard_dir}")
        print(f"  Command: tensorboard --logdir={tensorboard_base}")
        print()
    else:
        writer = None

    # Create dataloaders (includes ALL frames - model learns when ball is NOT present)
    print("Loading stereo data...")
    train_loader, val_loader = create_stereo_dataloaders(
        DATA_DIRS,
        batch_size=BATCH_SIZE,
        train_split=TRAIN_SPLIT,
        num_workers=NUM_WORKERS,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH
    )

    # Create model
    print("\nCreating StereoTiny model...")
    model = StereoTiny()
    model = model.to(device)

    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,} ({param_count * 4 / 1024:.1f} KB)")
    print()

    # Loss and optimizer
    criterion = StereoDetectionLoss(coord_weight=1.0, conf_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS]
    )

    print(f"LR schedule: warmup ({WARMUP_EPOCHS} epochs) + cosine annealing")
    print()

    # Training loop
    print(f"Starting training for {EPOCHS} epochs...")
    print("=" * 60)
    print()

    best_val_loss = float('inf')
    best_pixel_error = float('inf')
    training_history = []

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_err_left, train_err_right = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        if epoch % VALIDATION_INTERVAL == 0 or epoch == 1:
            val_loss, val_err_left, val_err_right = validate(
                model, val_loader, criterion, device
            )
        else:
            val_loss = val_loss if epoch > 1 else 0.0
            val_err_left = val_err_left if epoch > 1 else 0.0
            val_err_right = val_err_right if epoch > 1 else 0.0

        scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        # Print summary
        avg_train_err = (train_err_left + train_err_right) / 2
        avg_val_err = (val_err_left + val_err_right) / 2

        if epoch % VALIDATION_INTERVAL == 0 or epoch == 1:
            print(f"\nEpoch {epoch}/{EPOCHS} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Pixel Error: L={train_err_left:.1f}px R={train_err_right:.1f}px (avg={avg_train_err:.1f}px)")
            print(f"  Val Loss: {val_loss:.4f} | Pixel Error: L={val_err_left:.1f}px R={val_err_right:.1f}px (avg={avg_val_err:.1f}px)")
            print(f"  LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
        else:
            print(f"\nEpoch {epoch}/{EPOCHS}: Train Loss={train_loss:.4f} | Avg Pixel Error={avg_train_err:.1f}px | LR={current_lr:.6f}")

        # TensorBoard logging
        if writer is not None and epoch % TENSORBOARD_LOG_INTERVAL == 0:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('PixelError/train_left', train_err_left, epoch)
            writer.add_scalar('PixelError/train_right', train_err_right, epoch)
            writer.add_scalar('PixelError/train_avg', avg_train_err, epoch)
            writer.add_scalar('LearningRate/lr', current_lr, epoch)

            if epoch % VALIDATION_INTERVAL == 0 or epoch == 1:
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('PixelError/val_left', val_err_left, epoch)
                writer.add_scalar('PixelError/val_right', val_err_right, epoch)
                writer.add_scalar('PixelError/val_avg', avg_val_err, epoch)

        # Save history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_pixel_error_left': train_err_left,
            'train_pixel_error_right': train_err_right,
            'val_loss': val_loss if epoch % VALIDATION_INTERVAL == 0 else None,
            'val_pixel_error_left': val_err_left if epoch % VALIDATION_INTERVAL == 0 else None,
            'val_pixel_error_right': val_err_right if epoch % VALIDATION_INTERVAL == 0 else None,
            'learning_rate': current_lr,
            'time': epoch_time
        })

        # Save checkpoints
        if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

        # Save best model
        if epoch % VALIDATION_INTERVAL == 0 or epoch == 1:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = output_dir / 'best_model.pth'
                torch.save(model.state_dict(), best_path)
                print(f"  * New best validation loss! Saved to: {best_path}")

            if avg_val_err < best_pixel_error:
                best_pixel_error = avg_val_err
                best_pixel_path = output_dir / 'best_pixel_error.pth'
                torch.save(model.state_dict(), best_pixel_path)
                print(f"  * New best pixel error ({avg_val_err:.1f}px)! Saved to: {best_pixel_path}")

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    if writer is not None:
        writer.close()

    # Export to ONNX
    print("\n" + "=" * 60)
    print("EXPORTING TO ONNX")
    print("=" * 60)

    try:
        onnx_path = output_dir / f"{run_name}.onnx"
        dummy_input = torch.randn(1, 6, IMAGE_HEIGHT, IMAGE_WIDTH).to(device)

        print(f"Exporting model to: {onnx_path}")
        print(f"  Input shape: (1, 6, {IMAGE_HEIGHT}, {IMAGE_WIDTH})")
        print(f"  Output shape: (1, 6)")

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        model_size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"  ONNX model exported successfully!")
        print(f"  Model size: {model_size_mb:.2f} MB")

    except Exception as e:
        print(f"  [ERROR] ONNX export failed: {e}")

    # Training complete
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Run: {run_name}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best pixel error: {best_pixel_error:.1f}px")
    print()
    print(f"Run directory: {output_dir}")
    print(f"  ├── best_model.pth")
    print(f"  ├── best_pixel_error.pth")
    print(f"  ├── checkpoint_epoch_*.pth")
    print(f"  ├── {run_name}.onnx")
    print(f"  ├── config.json")
    print(f"  └── training_history.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
