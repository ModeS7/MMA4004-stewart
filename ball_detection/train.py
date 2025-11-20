"""
Training Script for Ball Detection CNN

Trains the lightweight CNN for sub-pixel ball center detection.
Edit settings below and run: python ball_detection/train.py
"""

import time
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import BallDetectorCNN
from dataset import create_dataloaders

# ============================================================
# SETTINGS - Edit these
# ============================================================
DATA_DIR = "./ball_detection/data/final"
OUTPUT_DIR = "./ball_detection/models"
EPOCHS = 100
BATCH_SIZE = 512
IMAGE_SIZE = 64
CROP_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
TRAIN_SPLIT = 0.8
NUM_WORKERS = 0  # Windows compatibility
COORD_WEIGHT = 1.0
CONF_WEIGHT = 0.1
SAVE_INTERVAL = 10
# ============================================================


class DetectionLoss(nn.Module):
    """
    Combined loss for ball detection.

    Combines:
    - MSE loss for (x, y) coordinates
    - BCE loss for confidence (always 1.0 for valid detections)
    """
    def __init__(self, coord_weight=1.0, conf_weight=0.1):
        super().__init__()
        self.coord_weight = coord_weight
        self.conf_weight = conf_weight
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        """
        Args:
            pred: (batch, 3) - predicted (x, y, confidence)
            target: (batch, 3) - ground truth (x, y, confidence)

        Returns:
            total_loss, coord_loss, conf_loss
        """
        # Coordinate loss (x, y)
        coord_loss = self.mse(pred[:, :2], target[:, :2])

        # Confidence loss
        conf_loss = self.bce(pred[:, 2], target[:, 2])

        # Total weighted loss
        total_loss = self.coord_weight * coord_loss + self.conf_weight * conf_loss

        return total_loss, coord_loss, conf_loss


def calculate_pixel_error(pred, target, image_size=64):
    """
    Calculate average pixel error.

    Args:
        pred: (batch, 3) - predicted normalized coordinates
        target: (batch, 3) - ground truth normalized coordinates
        image_size: Image size in pixels

    Returns:
        Average pixel error
    """
    # Convert normalized coords to pixels
    pred_pixels = pred[:, :2] * image_size
    target_pixels = target[:, :2] * image_size

    # Calculate Euclidean distance
    diff = pred_pixels - target_pixels
    distances = torch.sqrt((diff ** 2).sum(dim=1))

    return distances.mean().item()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch with AMP bf16."""
    model.train()

    total_loss = 0
    total_coord_loss = 0
    total_conf_loss = 0
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

        # Loss computation (BCE is unsafe in autocast, compute in fp32)
        loss, coord_loss, conf_loss = criterion(outputs.float(), targets)

        # Backward pass (no GradScaler needed for bf16)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics (compute in fp32)
        pixel_error = calculate_pixel_error(outputs.float(), targets)

        total_loss += loss.item()
        total_coord_loss += coord_loss.item()
        total_conf_loss += conf_loss.item()
        total_pixel_error += pixel_error

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'px_err': f'{pixel_error:.3f}'
        })

    avg_loss = total_loss / num_batches
    avg_coord_loss = total_coord_loss / num_batches
    avg_conf_loss = total_conf_loss / num_batches
    avg_pixel_error = total_pixel_error / num_batches

    return avg_loss, avg_coord_loss, avg_conf_loss, avg_pixel_error


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()

    total_loss = 0
    total_coord_loss = 0
    total_conf_loss = 0
    total_pixel_error = 0
    num_batches = len(dataloader)

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass (no autocast needed for validation, keep in fp32)
            outputs = model(images)
            loss, coord_loss, conf_loss = criterion(outputs, targets)

            # Metrics
            pixel_error = calculate_pixel_error(outputs, targets)

            total_loss += loss.item()
            total_coord_loss += coord_loss.item()
            total_conf_loss += conf_loss.item()
            total_pixel_error += pixel_error

    avg_loss = total_loss / num_batches
    avg_coord_loss = total_coord_loss / num_batches
    avg_conf_loss = total_conf_loss / num_batches
    avg_pixel_error = total_pixel_error / num_batches

    return avg_loss, avg_coord_loss, avg_conf_loss, avg_pixel_error


def main():
    """Main training loop."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    else:
        print("AMP: Disabled (CPU mode)")

    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
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
        'image_size': IMAGE_SIZE,
        'crop_size': CROP_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'train_split': TRAIN_SPLIT,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Create dataloaders
    print(f"Loading data from: {DATA_DIR}")
    train_loader, val_loader = create_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        crop_size=CROP_SIZE,
        train_split=TRAIN_SPLIT,
        num_workers=NUM_WORKERS
    )

    # Create model
    print("\nCreating model...")
    model = BallDetectorCNN()
    model = model.to(device)
    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,} ({param_count * 4 / 1024:.1f} KB)")
    print()

    # Loss and optimizer
    criterion = DetectionLoss(coord_weight=COORD_WEIGHT, conf_weight=CONF_WEIGHT)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

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
        train_loss, train_coord, train_conf, train_pixel_err = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_coord, val_conf, val_pixel_err = validate(
            model, val_loader, criterion, device
        )

        # Scheduler step
        scheduler.step(val_loss)

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

        # Save checkpoint
        if (epoch % SAVE_INTERVAL == 0) or (epoch == EPOCHS):
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'pixel_error': val_pixel_err,
            }, checkpoint_path)
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
