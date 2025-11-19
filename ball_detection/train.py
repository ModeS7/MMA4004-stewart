"""
Training Script for Ball Detection CNN

Trains the lightweight CNN for sub-pixel ball center detection.
Includes validation, checkpointing, and tensorboard logging.
"""

import argparse
import os
import time
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import BallDetectorCNN
from dataset import create_dataloaders


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
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_coord_loss = 0
    total_conf_loss = 0
    total_pixel_error = 0
    num_batches = len(dataloader)

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(images)
        loss, coord_loss, conf_loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        pixel_error = calculate_pixel_error(outputs, targets)

        total_loss += loss.item()
        total_coord_loss += coord_loss.item()
        total_conf_loss += conf_loss.item()
        total_pixel_error += pixel_error

        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx+1}/{num_batches}] "
                  f"Loss: {loss.item():.4f}, "
                  f"Pixel Error: {pixel_error:.3f}px")

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
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
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


def main(args):
    """Main training loop."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup tensorboard
    writer = SummaryWriter(output_dir / 'runs')

    # Save training config
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Create dataloaders
    print(f"\nLoading data from: {args.data_dir}")
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        train_split=args.train_split,
        num_workers=args.num_workers
    )

    # Create model
    print("\nCreating model...")
    model = BallDetectorCNN()
    model = model.to(device)
    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,} ({param_count * 4 / 1024:.1f} KB)")

    # Loss and optimizer
    criterion = DetectionLoss(coord_weight=args.coord_weight, conf_weight=args.conf_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    best_pixel_error = float('inf')

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)

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

        # Print summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} (coord: {train_coord:.4f}, conf: {train_conf:.4f})")
        print(f"  Train Pixel Error: {train_pixel_err:.3f}px")
        print(f"  Val Loss: {val_loss:.4f} (coord: {val_coord:.4f}, conf: {val_conf:.4f})")
        print(f"  Val Pixel Error: {val_pixel_err:.3f}px")
        print(f"  Time: {epoch_time:.1f}s")

        # Tensorboard logging
        writer.add_scalars('Loss/total', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Loss/coord', {'train': train_coord, 'val': val_coord}, epoch)
        writer.add_scalars('Metric/pixel_error', {'train': train_pixel_err, 'val': val_pixel_err}, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # Save checkpoint
        if (epoch % args.save_interval == 0) or (epoch == args.epochs):
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'pixel_error': val_pixel_err,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / 'best_model.pth'
            torch.save(model.state_dict(), best_path)
            print(f"  New best validation loss! Saved to: {best_path}")

        if val_pixel_err < best_pixel_error:
            best_pixel_error = val_pixel_err
            best_pixel_path = output_dir / 'best_pixel_error.pth'
            torch.save(model.state_dict(), best_pixel_path)
            print(f"  New best pixel error! Saved to: {best_pixel_path}")

    # Training complete
    writer.close()
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best pixel error: {best_pixel_error:.3f}px")
    print(f"Models saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Ball Detection CNN')

    # Data
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='./ball_detection/models',
                        help='Output directory for models and logs')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Fraction of data for training')

    # Model
    parser.add_argument('--image-size', type=int, default=64,
                        help='Input image size')
    parser.add_argument('--coord-weight', type=float, default=1.0,
                        help='Weight for coordinate loss')
    parser.add_argument('--conf-weight', type=float, default=0.1,
                        help='Weight for confidence loss')

    # System
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()

    main(args)
