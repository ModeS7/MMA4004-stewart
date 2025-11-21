"""
Training Script for Ball Detection CNN

Trains the lightweight CNN for sub-pixel ball center detection.
Edit settings below and run: python ball_detection/train.py
"""

import time
from pathlib import Path
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..core.model import BallDetectorCNN, BallDetectorMobileNetV3, create_model
from ..core.dataset import create_dataloaders

# ============================================================
# SETTINGS - Edit these
# ============================================================
DATA_DIR = "./ball_detection/data/final"
OUTPUT_DIR = "./ball_detection/models"
EPOCHS = 3200  # Overnight training
BATCH_SIZE = 512  # Increased for faster training on RTX 3090
CROP_SIZE = 128  # Crop size (scale variation handled by ShiftScaleRotate augmentation)
LEARNING_RATE = 0.001  # Lower LR for long training
WARMUP_EPOCHS = 10  # Linear warmup for stable start
WEIGHT_DECAY = 1e-5
TRAIN_SPLIT = 0.8
NUM_WORKERS = 24
SAVE_INTERVAL = 100  # Save less frequently (every 100 epochs for 3000 epoch run)
USE_MOBILENET = False  # Use MobileNetV3-Small with pretrained ImageNet weights
MOBILENET_PRETRAINED = True  # Load ImageNet pretrained weights for MobileNetV3
USE_SPATIAL_AUGMENTATION = True  # Offset, rotate, scale, shift (simulate CV detection error)
USE_APPEARANCE_AUGMENTATION = True  # Brightness, hue, blur, noise
# ============================================================


class DetectionLoss(nn.Module):
    """
    MSE loss for (x, y) coordinate regression.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        """
        Args:
            pred: (batch, 2) - predicted (x, y)
            target: (batch, 2) - ground truth (x, y)

        Returns:
            loss
        """
        # Coordinate loss (x, y)
        loss = self.mse(pred, target)

        return loss


def calculate_pixel_error(pred, target, crop_size=128):
    """
    Calculate average pixel error.

    Args:
        pred: (batch, 2) - predicted normalized coordinates
        target: (batch, 2) - ground truth normalized coordinates
        crop_size: Crop size in pixels

    Returns:
        Average pixel error
    """
    # Convert normalized coords to pixels
    pred_pixels = pred * crop_size
    target_pixels = target * crop_size

    # Calculate Euclidean distance
    diff = pred_pixels - target_pixels
    distances = torch.sqrt((diff ** 2).sum(dim=1))

    return distances.mean().item()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch with AMP bf16."""
    model.train()

    total_loss = 0
    total_pixel_error = 0
    num_batches = len(dataloader)
    use_amp = device.type == 'cuda'

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} Training')
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass with AMP bf16 (only on CUDA)
        if use_amp:
            with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                outputs = model(images)
        else:
            outputs = model(images)

        # Loss computation in fp32
        loss = criterion(outputs.float(), targets)

        # Backward pass (no GradScaler needed for bf16)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics (compute in fp32)
        pixel_error = calculate_pixel_error(outputs.float(), targets)

        total_loss += loss.item()
        total_pixel_error += pixel_error

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'px_err': f'{pixel_error:.3f}'
        })

    avg_loss = total_loss / num_batches
    avg_pixel_error = total_pixel_error / num_batches

    return avg_loss, avg_pixel_error


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()

    total_loss = 0
    total_pixel_error = 0
    num_batches = len(dataloader)

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass (no autocast needed for validation, keep in fp32)
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Metrics
            pixel_error = calculate_pixel_error(outputs, targets)

            total_loss += loss.item()
            total_pixel_error += pixel_error

    avg_loss = total_loss / num_batches
    avg_pixel_error = total_pixel_error / num_batches

    return avg_loss, avg_pixel_error


def main():
    """Main training loop."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # CUDA optimizations for maximum performance
    if device.type == 'cuda':
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = True

        # Scaled Dot Product Attention optimizations (PyTorch 2.0+)
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except AttributeError:
            pass  # Older PyTorch versions don't have these

        # Dynamo cache size for torch.compile
        torch._dynamo.config.cache_size_limit = 32

    print("=" * 60)
    print("BALL DETECTION CNN TRAINING")
    print("=" * 60)
    print(f"Data: {DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Device: {device}")

    # Print GPU info if available
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"AMP bf16: Enabled")
        print(f"TF32: Enabled (cuDNN & matmul)")
        print(f"cuDNN benchmark: Enabled")
    else:
        print("AMP: Disabled (CPU mode)")

    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Crop size: {CROP_SIZE}x{CROP_SIZE}")
    print("=" * 60)
    print()

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training config
    config = {
        'data_dir': DATA_DIR,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'crop_size': CROP_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'train_split': TRAIN_SPLIT,
        'use_mobilenet': USE_MOBILENET,
        'mobilenet_pretrained': MOBILENET_PRETRAINED,
        'use_spatial_augmentation': USE_SPATIAL_AUGMENTATION,
        'use_appearance_augmentation': USE_APPEARANCE_AUGMENTATION,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Create dataloaders
    print(f"Loading data from: {DATA_DIR}")
    print(f"Spatial augmentation: {'Enabled' if USE_SPATIAL_AUGMENTATION else 'Disabled'}")
    print(f"Appearance augmentation: {'Enabled' if USE_APPEARANCE_AUGMENTATION else 'Disabled'}")
    train_loader, val_loader = create_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        crop_size=CROP_SIZE,
        train_split=TRAIN_SPLIT,
        num_workers=NUM_WORKERS,
        use_spatial_augmentation=USE_SPATIAL_AUGMENTATION,
        use_appearance_augmentation=USE_APPEARANCE_AUGMENTATION
    )

    # Create model
    print("\nCreating model...")
    if USE_MOBILENET:
        print(f"Using MobileNetV3-Small (pretrained: {MOBILENET_PRETRAINED})")
        model = BallDetectorMobileNetV3(pretrained=MOBILENET_PRETRAINED)
    else:
        print("Using custom BallDetectorCNN")
        model = BallDetectorCNN()

    model = model.to(device)

    # Compile model for faster training (PyTorch 2.0+)
    model = torch.compile(model)

    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,} ({param_count * 4 / 1024:.1f} KB)")
    print(f"torch.compile: Enabled")
    print()

    # Loss and optimizer
    criterion = DetectionLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler: Warmup + Cosine Annealing with Warm Restarts
    # This periodically resets LR to help escape local minima during long training
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=WARMUP_EPOCHS
    )

    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,        # First restart after 50 epochs
        T_mult=2,      # Double the cycle length after each restart (50, 100, 200, ...)
        eta_min=1e-6   # Minimum LR
    )

    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS]
    )

    # Calculate restart epochs for checkpointing
    restart_epochs = [WARMUP_EPOCHS]  # After warmup
    cycle_length = 50
    next_restart = WARMUP_EPOCHS + cycle_length
    while next_restart <= EPOCHS:
        restart_epochs.append(next_restart)
        cycle_length *= 2  # T_mult = 2
        next_restart += cycle_length

    print(f"LR restart epochs: {restart_epochs[:10]}...")  # Show first 10

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
        train_loss, train_pixel_err = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_pixel_err = validate(
            model, val_loader, criterion, device
        )

        # Scheduler step (CosineAnnealingLR steps every epoch)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        # Print summary
        print(f"\nEpoch {epoch}/{EPOCHS} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Pixel Error: {train_pixel_err:.3f}px")
        print(f"  Val Loss: {val_loss:.4f} | Pixel Error: {val_pixel_err:.3f}px")
        print(f"  LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

        # Save history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_pixel_error': train_pixel_err,
            'val_loss': val_loss,
            'val_pixel_error': val_pixel_err,
            'learning_rate': current_lr,
            'time': epoch_time
        })

        # Save checkpoint at regular intervals OR before LR restarts
        is_restart_epoch = epoch in restart_epochs
        if (epoch % SAVE_INTERVAL == 0) or (epoch == EPOCHS) or is_restart_epoch:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'pixel_error': val_pixel_err,
            }, checkpoint_path)
            if is_restart_epoch:
                print(f"  🔄 Checkpoint saved before LR restart: {checkpoint_path}")
            else:
                print(f"  Checkpoint saved: {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / 'best_model.pth'
            torch.save(model.state_dict(), best_path)
            print(f"  ★ New best validation loss! Saved to: {best_path}")

        if val_pixel_err < best_pixel_error:
            best_pixel_error = val_pixel_err
            best_pixel_path = output_dir / 'best_pixel_error.pth'
            torch.save(model.state_dict(), best_pixel_path)
            print(f"  ★ New best pixel error! Saved to: {best_pixel_path}")

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    # Training complete
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best pixel error: {best_pixel_error:.3f}px")
    print(f"Models saved to: {output_dir}")
    print(f"History saved to: {output_dir}/training_history.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
