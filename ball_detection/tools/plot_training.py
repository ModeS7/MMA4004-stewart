"""
Plot Training History

Visualizes training metrics from training_history.json.
Shows loss, pixel error, and learning rate over epochs.

Usage: python ball_detection/plot_training.py [path_to_training_history.json]
"""

import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_history(history_path):
    """Plot training history from JSON file."""

    # Load training history
    with open(history_path, 'r') as f:
        history = json.load(f)

    if len(history) == 0:
        print("Error: Training history is empty!")
        return

    # Extract data
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    train_pixel_error = [h['train_pixel_error'] for h in history]
    val_pixel_error = [h['val_pixel_error'] for h in history]
    learning_rate = [h['learning_rate'] for h in history]

    # Calculate some statistics
    best_val_loss_epoch = np.argmin(val_loss) + 1
    best_val_loss = min(val_loss)
    best_pixel_error_epoch = np.argmin(val_pixel_error) + 1
    best_pixel_error = min(val_pixel_error)

    print("=" * 60)
    print("TRAINING HISTORY SUMMARY")
    print("=" * 60)
    print(f"Total epochs: {len(history)}")
    print(f"Best validation loss: {best_val_loss:.6f} (epoch {best_val_loss_epoch})")
    print(f"Best pixel error: {best_pixel_error:.4f}px (epoch {best_pixel_error_epoch})")
    print(f"Final validation loss: {val_loss[-1]:.6f}")
    print(f"Final pixel error: {val_pixel_error[-1]:.4f}px")
    print("=" * 60)
    print()

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')

    # Plot 1: Loss (log scale)
    ax1 = axes[0]
    ax1.plot(epochs, train_loss, label='Train Loss', linewidth=2, alpha=0.8)
    ax1.plot(epochs, val_loss, label='Val Loss', linewidth=2, alpha=0.8)
    ax1.axvline(best_val_loss_epoch, color='red', linestyle='--', alpha=0.5,
                label=f'Best Val Loss (epoch {best_val_loss_epoch})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training and Validation Loss (Log Scale)')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')

    # Plot 2: Pixel Error (log scale)
    ax2 = axes[1]
    ax2.plot(epochs, train_pixel_error, label='Train Pixel Error', linewidth=2, alpha=0.8)
    ax2.plot(epochs, val_pixel_error, label='Val Pixel Error', linewidth=2, alpha=0.8)
    ax2.axvline(best_pixel_error_epoch, color='red', linestyle='--', alpha=0.5,
                label=f'Best Pixel Error (epoch {best_pixel_error_epoch})')
    ax2.axhline(0.2, color='green', linestyle=':', alpha=0.5, label='Target: 0.2px')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Pixel Error (px)')
    ax2.set_title('Pixel Error (Log Scale)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    # Plot 3: Learning Rate
    ax3 = axes[2]
    ax3.plot(epochs, learning_rate, label='Learning Rate', linewidth=2, color='orange')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule (with Warm Restarts)')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = Path(history_path).parent / 'training_history.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Show plot
    plt.show()


def main():
    """Main entry point."""

    # Get history path from command line or use default
    if len(sys.argv) > 1:
        history_path = sys.argv[1]
    else:
        history_path = "./ball_detection/models/training_history.json"

    history_path = Path(history_path)

    # Check if file exists
    if not history_path.exists():
        print(f"Error: File not found: {history_path}")
        print()
        print("Usage: python ball_detection/plot_training.py [path_to_training_history.json]")
        print("Example: python ball_detection/plot_training.py ./ball_detection/models/training_history.json")
        sys.exit(1)

    print(f"Loading training history from: {history_path}")
    print()

    # Plot training history
    plot_training_history(history_path)


if __name__ == "__main__":
    main()
