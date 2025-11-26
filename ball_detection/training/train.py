"""
Training Script for Ball Detection CNN

Trains the lightweight CNN for sub-pixel ball center detection.
Edit settings below and run: python ball_detection/train.py
"""

import time
from pathlib import Path
import json
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from ..core.model import BallDetectorCNN, BallDetectorMobileNetV3, BallDetectorShuffleNetV2, create_model
from ..core.dataset import create_dataloaders
from ..core.gpu_augmentations import GPUAugmentationsWithTargets
from ..core.cached_dataset import CachedAugmentationDataset

# ============================================================
# SETTINGS - Edit these
# ============================================================
DATA_DIR = "./ball_detection/data/final"
OUTPUT_DIR = "./ball_detection/models"
EPOCHS = 4000  # Overnight training
BATCH_SIZE = 128 # Large batch for RTX 3090 (24GB VRAM) - max GPU utilization
CROP_SIZE = 128  # Crop size (scale variation handled by ShiftScaleRotate augmentation)
LEARNING_RATE = 0.001  # Max LR for warmup and cosine decay (should be > PRUNING_LR)
WARMUP_EPOCHS = 10  # Linear warmup for stable start
WEIGHT_DECAY = 1e-4  # Slightly higher weight decay for regularization
TRAIN_SPLIT = 0.8
NUM_WORKERS = 8  # Reduced workers for smaller batch size
SAVE_INTERVAL = 100  # Save less frequently (every 100 epochs for 3000 epoch run)
VALIDATION_INTERVAL = 2  # Validate every N epochs (reduce sync overhead)
USE_MOBILENET = False  # Use MobileNetV3-Small with pretrained ImageNet weights
USE_SHUFFLENET = True  # Use ShuffleNetV2 x0.5 (faster than MobileNetV3, ~350K params)
PRETRAINED_BACKBONE = True  # Load ImageNet pretrained weights for MobileNetV3/ShuffleNet
USE_SPATIAL_AUGMENTATION = True  # Offset, rotate, scale, shift (simulate CV detection error)
USE_APPEARANCE_AUGMENTATION = True  # Brightness, hue, blur, noise

# ============================================================
# GPU AUGMENTATION SETTINGS (Kornia - Much faster than CPU)
# ============================================================
USE_GPU_AUGMENTATIONS = False   # Use Kornia GPU augmentations instead of Albumentations
# When enabled, CPU augmentations (Albumentations) are disabled for max speed

# ============================================================
# CACHED AUGMENTATION SETTINGS (For Maximum GPU Utilization)
# ============================================================
USE_CACHED_AUGMENTATIONS = False  # Use pre-augmented cache in RAM
CACHE_SIZE_MULTIPLIER = 3        # Cache size = dataset_size x multiplier (3x = ~21K images)
CACHE_MAX_REUSE_COUNT = 2        # Replace cached items after N uses
CACHE_ENABLE_REFRESH = True      # Background thread continuously refreshes cache
# Memory usage: ~1.3GB for 7K images x 3 multiplier
# Expected GPU utilization: <20% -> 60-80%

# ============================================================
# PRUNING SETTINGS
# ============================================================
ENABLE_PRUNING = True
PRUNING_START_EPOCH = 150           # Start after initial convergence
PRUNING_CHECK_INTERVAL = 10         # Check every 10 epochs if ready to prune
INITIAL_TARGET_SPARSITY = 0.90      # 90% sparsity (~150K params)
SPARSITY_INCREMENT = 0.05           # After 90%, prune 2% more each time
PRUNE_AMOUNT_PER_STEP = 0.1        # Remove 20% of remaining params each step
VALIDATION_PIXEL_THRESHOLD = 1.0    # Max acceptable pixel error
PRUNING_PATIENCE = 10               # Must have px_error < 1.0 for 10 epochs

# ============================================================
# TENSORBOARD SETTINGS
# ============================================================
ENABLE_TENSORBOARD = True
TENSORBOARD_LOG_INTERVAL = 1        # Log scalars every N epochs
TENSORBOARD_IMAGE_INTERVAL = 50     # Log prediction images every N epochs
TENSORBOARD_HISTOGRAM_INTERVAL = 100 # Log weights/gradients every N epochs
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


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, gpu_aug=None):
    """Train for one epoch with AMP bf16."""
    model.train()
    if gpu_aug is not None:
        gpu_aug.train()

    total_loss = 0
    total_pixel_error = 0
    num_batches = len(dataloader)
    use_amp = device.type == 'cuda'

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} Training')
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply GPU augmentations (if enabled)
        if gpu_aug is not None:
            images, targets = gpu_aug(images, targets)

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

        # Metrics (compute in fp32) - accumulate tensors to reduce sync
        pixel_error = calculate_pixel_error(outputs.float(), targets)

        total_loss += loss.detach()
        total_pixel_error += pixel_error

        # Update progress bar (sync every batch for display)
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'px_err': f'{pixel_error:.3f}'
        })

    # Synchronize only at epoch end
    avg_loss = (total_loss / num_batches).item()
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
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

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


def apply_structured_pruning(model, amount):
    """
    Apply STRUCTURED pruning to convolutional layers - actually removes filters.

    This creates a permanently smaller model (not just zeroed weights):
    - Ranks filters by L1 norm (importance)
    - Removes least important filters
    - Reduces model dimensions (faster inference!)

    CRITICAL: Also resets BatchNorm statistics to prevent feature collapse.
    """
    import torch_pruning as tp

    # Use torch-pruning library for true structured pruning
    # This actually removes channels/filters, making the model smaller and faster

    # Define importance criterion (L1 norm of filters)
    imp = tp.importance.MagnitudeImportance(p=1)

    # Identify layers that can be pruned
    ignored_layers = []
    for name, module in model.named_modules():
        # Don't prune the final output layer
        if isinstance(module, nn.Linear) and module.out_features == 2:
            ignored_layers.append(module)

    # Create pruner
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=torch.randn(1, 3, 128, 128).to(next(model.parameters()).device),
        importance=imp,
        iterative_steps=1,
        pruning_ratio=amount,
        ignored_layers=ignored_layers,
    )

    # Apply pruning (actually removes filters)
    pruner.step()

    # CRITICAL: Reset BatchNorm statistics after pruning
    # Pruning changes activation distributions, making old running mean/var invalid
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.reset_running_stats()

    return model


def get_model_sparsity(model, original_params=None):
    """
    Calculate percentage parameter reduction from structured pruning.

    For structured pruning: compares current param count to original.
    For unstructured pruning: counts zero weights.
    """
    current_params = count_model_parameters(model)

    if original_params is not None:
        # Structured pruning: measure actual parameter reduction
        reduction_ratio = 1.0 - (current_params / original_params)
        return reduction_ratio * 100
    else:
        # Unstructured pruning: count zero weights
        total_params = 0
        zero_params = 0
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, 'weight_mask'):
                    mask = module.weight_mask
                    total_params += mask.numel()
                    zero_params += (mask == 0).sum().item()
                else:
                    total_params += module.weight.numel()

        return (zero_params / total_params * 100) if total_params > 0 else 0.0


def count_model_parameters(model, only_nonzero=False):
    """Count total parameters in the model."""
    total = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            total += module.weight.numel()
            if module.bias is not None:
                total += module.bias.numel()
    return total


def check_pruning_readiness(pixel_error_history, threshold=1.0):
    """Check if all values in deque are below threshold."""
    if len(pixel_error_history) < pixel_error_history.maxlen:
        return False
    return all(err < threshold for err in pixel_error_history)


def visualize_predictions(images, targets, preds, crop_size):
    """
    Create visualization grid with GT (green) and predictions (red).

    Args:
        images: (B, 3, H, W) tensor (ImageNet normalized)
        targets: (B, 2) normalized coordinates
        preds: (B, 2) normalized coordinates
        crop_size: Image size for denormalization

    Returns:
        (3, H, W*B) tensor grid
    """
    import cv2

    # ImageNet normalization stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    vis_images = []
    for i in range(len(images)):
        # Denormalize from ImageNet stats
        img = images[i].cpu()
        img = img * std + mean  # Denormalize to [0, 1]
        img = torch.clamp(img, 0, 1)  # Clamp to valid range

        img = img.numpy().transpose(1, 2, 0)  # CHW -> HWC
        img = (img * 255).astype(np.uint8).copy()

        # Convert normalized coords to pixels
        gt_x, gt_y = int(targets[i, 0] * crop_size), int(targets[i, 1] * crop_size)
        pred_x, pred_y = int(preds[i, 0] * crop_size), int(preds[i, 1] * crop_size)

        # Draw circles
        cv2.circle(img, (gt_x, gt_y), 3, (0, 255, 0), -1)  # Green = GT
        cv2.circle(img, (pred_x, pred_y), 3, (255, 0, 0), -1)  # Red = Pred

        # Draw line between them
        cv2.line(img, (gt_x, gt_y), (pred_x, pred_y), (255, 255, 0), 1)

        vis_images.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)

    return vutils.make_grid(vis_images, nrow=4, padding=2)


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

    # Create output directory and run name
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"run_{timestamp}"
    if USE_SHUFFLENET:
        run_name += "_shufflenetv2"
    elif USE_MOBILENET:
        run_name += "_mobilenetv3"
    else:
        run_name += "_customcnn"
    if ENABLE_PRUNING and not USE_SHUFFLENET:  # Pruning not supported for ShuffleNet
        run_name += "_pruning"

    # Each run gets its own directory under models/
    output_dir = Path(OUTPUT_DIR) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRun Name: {run_name}")
    print(f"Output Directory: {output_dir}")
    print()

    # Save training config
    config = {
        'run_name': run_name,
        'timestamp': timestamp,
        'data_dir': DATA_DIR,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'crop_size': CROP_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'train_split': TRAIN_SPLIT,
        'use_mobilenet': USE_MOBILENET,
        'use_shufflenet': USE_SHUFFLENET,
        'pretrained_backbone': PRETRAINED_BACKBONE,
        'use_spatial_augmentation': USE_SPATIAL_AUGMENTATION,
        'use_appearance_augmentation': USE_APPEARANCE_AUGMENTATION,
        'enable_pruning': ENABLE_PRUNING,
        'enable_tensorboard': ENABLE_TENSORBOARD,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Initialize TensorBoard - centralized in models/tensorboard_logs/
    if ENABLE_TENSORBOARD:
        tensorboard_base = Path(OUTPUT_DIR) / 'tensorboard_logs'
        tensorboard_dir = tensorboard_base / run_name
        writer = SummaryWriter(log_dir=tensorboard_dir)
        print(f"TensorBoard:")
        print(f"  Logs: {tensorboard_dir}")
        print(f"  Command: tensorboard --logdir={tensorboard_base}")
        print()
    else:
        writer = None

    # Create dataloaders
    print(f"Loading data from: {DATA_DIR}")

    # Disable CPU augmentations if using GPU augmentations or cached augmentations
    cpu_spatial_aug = USE_SPATIAL_AUGMENTATION and not USE_GPU_AUGMENTATIONS and not USE_CACHED_AUGMENTATIONS
    cpu_appearance_aug = USE_APPEARANCE_AUGMENTATION and not USE_GPU_AUGMENTATIONS and not USE_CACHED_AUGMENTATIONS

    if USE_CACHED_AUGMENTATIONS:
        print(f"Cached augmentations: Enabled")
        print(f"  Cache multiplier: {CACHE_SIZE_MULTIPLIER}x")
        print(f"  Max reuse: {CACHE_MAX_REUSE_COUNT}")
        print(f"  Background refresh: {'Enabled' if CACHE_ENABLE_REFRESH else 'Disabled'}")
        print()

        # Load labels and filter valid samples first
        labels_path = Path(DATA_DIR) / 'labels.json'
        with open(labels_path, 'r') as f:
            all_labels = json.load(f)

        # Filter valid samples (skip invalid ones)
        valid_samples = []
        for img_name, label in all_labels.items():
            if label.get('valid', True):
                valid_samples.append((img_name, label))

        # Calculate split indices based on valid samples
        dataset_size = len(valid_samples)
        train_size = int(TRAIN_SPLIT * dataset_size)
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, dataset_size))

        # Create cached datasets
        train_dataset = CachedAugmentationDataset(
            data_dir=DATA_DIR,
            crop_size=CROP_SIZE,
            cache_multiplier=CACHE_SIZE_MULTIPLIER,
            max_reuse_count=CACHE_MAX_REUSE_COUNT,
            use_spatial_aug=USE_SPATIAL_AUGMENTATION,
            use_appearance_aug=USE_APPEARANCE_AUGMENTATION,
            enable_refresh=CACHE_ENABLE_REFRESH,
            indices=train_indices
        )

        val_dataset = CachedAugmentationDataset(
            data_dir=DATA_DIR,
            crop_size=CROP_SIZE,
            cache_multiplier=1,  # No multiplier for validation
            max_reuse_count=999999,  # Never refresh validation
            use_spatial_aug=False,
            use_appearance_aug=False,
            enable_refresh=False,
            indices=val_indices
        )

        # Create dataloaders with minimal prefetching to save RAM
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,  # Fewer workers = less memory overhead
            pin_memory=True,  # Pin memory for faster GPU transfer
            prefetch_factor=2,  # Reduced prefetch to save RAM
            persistent_workers=True,  # Keep workers alive between epochs
            drop_last=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            drop_last=False
        )

    elif USE_GPU_AUGMENTATIONS:
        print(f"GPU augmentations (Kornia): Enabled")
        print(f"  CPU augmentations: Disabled (for max speed)")
        train_loader, val_loader = create_dataloaders(
            data_dir=DATA_DIR,
            batch_size=BATCH_SIZE,
            crop_size=CROP_SIZE,
            train_split=TRAIN_SPLIT,
            num_workers=NUM_WORKERS,
            use_spatial_augmentation=cpu_spatial_aug,
            use_appearance_augmentation=cpu_appearance_aug,
            disable_normalize=USE_GPU_AUGMENTATIONS
        )
    else:
        print(f"CPU Spatial augmentation: {'Enabled' if cpu_spatial_aug else 'Disabled'}")
        print(f"CPU Appearance augmentation: {'Enabled' if cpu_appearance_aug else 'Disabled'}")
        train_loader, val_loader = create_dataloaders(
            data_dir=DATA_DIR,
            batch_size=BATCH_SIZE,
            crop_size=CROP_SIZE,
            train_split=TRAIN_SPLIT,
            num_workers=NUM_WORKERS,
            use_spatial_augmentation=cpu_spatial_aug,
            use_appearance_augmentation=cpu_appearance_aug,
            disable_normalize=False
        )

    # Create model
    print("\nCreating model...")
    if USE_SHUFFLENET:
        print(f"Using ShuffleNetV2 x0.5 (pretrained: {PRETRAINED_BACKBONE})")
        model = BallDetectorShuffleNetV2(pretrained=PRETRAINED_BACKBONE)
    elif USE_MOBILENET:
        print(f"Using MobileNetV3-Small (pretrained: {PRETRAINED_BACKBONE})")
        model = BallDetectorMobileNetV3(pretrained=PRETRAINED_BACKBONE)
    else:
        print("Using custom BallDetectorCNN")
        model = BallDetectorCNN()

    model = model.to(device)

    # Create GPU augmentation module
    if USE_GPU_AUGMENTATIONS:
        gpu_aug = GPUAugmentationsWithTargets(crop_size=CROP_SIZE).to(device)
        print(f"GPU augmentations: Initialized on {device}")
    else:
        gpu_aug = None

    # Disable torch.compile for small models/datasets (overhead > benefit)
    # model = torch.compile(model)

    param_count = model.count_parameters()
    original_param_count = param_count  # Store for structured pruning tracking
    print(f"Model parameters: {param_count:,} ({param_count * 4 / 1024:.1f} KB)")
    print(f"torch.compile: Disabled (small model/dataset)")
    print()

    # Loss and optimizer
    criterion = DetectionLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Determine if pruning will be enabled (ShuffleNet doesn't support it)
    will_prune = ENABLE_PRUNING and not USE_SHUFFLENET

    # Learning rate scheduler - different strategies for pruning vs non-pruning
    PRUNING_LR = 2e-4  # Higher LR for structured pruning recovery
    MIN_LR = 1e-6  # Minimum LR for warm restarts

    if will_prune:
        # Pruning scheduler:
        # Phase 1: Warmup (0-10)
        # Phase 2: Cosine annealing until pruning starts (10-150)
        # Phase 3: Small constant LR during pruning (150+)
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=WARMUP_EPOCHS
        )

        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=PRUNING_START_EPOCH - WARMUP_EPOCHS,
            eta_min=PRUNING_LR
        )

        constant_scheduler = optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=EPOCHS - PRUNING_START_EPOCH
        )

        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler, constant_scheduler],
            milestones=[WARMUP_EPOCHS, PRUNING_START_EPOCH]
        )
        restart_epochs = []

        print(f"LR schedule (pruning mode):")
        print(f"  Warmup: epochs 1-{WARMUP_EPOCHS} ({LEARNING_RATE*0.1:.6f} -> {LEARNING_RATE:.6f})")
        print(f"  Cosine: epochs {WARMUP_EPOCHS+1}-{PRUNING_START_EPOCH} ({LEARNING_RATE:.6f} -> {PRUNING_LR:.6f})")
        print(f"  Pruning: epochs {PRUNING_START_EPOCH+1}-{EPOCHS} (constant {PRUNING_LR:.6f})")
    else:
        # Non-pruning scheduler: Warmup + CosineAnnealingWarmRestarts
        # T_0=400: cycle every 400 epochs
        # T_mult=1: fixed cycle length (no doubling)
        T_0 = 400
        T_mult = 1

        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=WARMUP_EPOCHS
        )

        cosine_restarts_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=MIN_LR
        )

        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_restarts_scheduler],
            milestones=[WARMUP_EPOCHS]
        )

        # Calculate restart epochs for logging (after warmup)
        restart_epochs = [WARMUP_EPOCHS + T_0 * i for i in range(1, (EPOCHS - WARMUP_EPOCHS) // T_0 + 1)]

        print(f"LR schedule (warm restarts):")
        print(f"  Warmup: epochs 1-{WARMUP_EPOCHS} ({LEARNING_RATE*0.1:.6f} -> {LEARNING_RATE:.6f})")
        print(f"  CosineAnnealingWarmRestarts: T_0={T_0}, T_mult={T_mult}")
        print(f"  LR range: {LEARNING_RATE:.6f} -> {MIN_LR:.6f}")
        print(f"  Restart epochs: {restart_epochs}")

    # Initialize pruning state (will_prune already accounts for ShuffleNet)
    enable_pruning = will_prune

    if enable_pruning:
        pixel_error_history = deque(maxlen=PRUNING_PATIENCE)
        current_sparsity_target = INITIAL_TARGET_SPARSITY
        pruning_active = True
        last_pruning_epoch = 0  # Track when we last pruned for LR decay
        last_successful_prune_epoch = 0  # Track when pruning actually happened (not just LR boost)
        lr_recovery_epochs = 50  # Decay boosted LR back over 50 epochs
        stall_reset_interval = 50  # Reset LR if stuck for this many epochs

        # Auto-finish when pruning hits limit
        pruning_limit_reached = False
        final_training_epochs_remaining = 200  # Train 200 more epochs after limit
        final_training_best_val_error = float('inf')
        final_training_best_model = None

        print(f"\nPruning Configuration:")
        print(f"  Enabled: True")
        print(f"  Start epoch: {PRUNING_START_EPOCH}")
        print(f"  Initial target: {INITIAL_TARGET_SPARSITY*100:.0f}% sparsity")
        print(f"  Check interval: every {PRUNING_CHECK_INTERVAL} epochs")
        print(f"  Patience: {PRUNING_PATIENCE} epochs < {VALIDATION_PIXEL_THRESHOLD}px")
        print(f"  Prune amount per step: {PRUNE_AMOUNT_PER_STEP*100:.0f}%")
        print(f"  LR after pruning: {PRUNING_LR * 0.5:.6f} (constant, no decay)")
        print(f"  Stall recovery: Reset to {PRUNING_LR:.6f} if no pruning for {stall_reset_interval} epochs")
        print(f"  Auto-finish: Train {final_training_epochs_remaining} epochs after pruning limit reached")
        print()

    # Training loop
    print(f"Starting training for {EPOCHS} epochs...")
    print("=" * 60)
    print()
    best_val_loss = float('inf')
    best_pixel_error = float('inf')

    # Track best model in each 100-epoch interval
    interval_best_error = float('inf')
    interval_best_model_state = None
    interval_best_epoch = 0
    current_interval_start = 1

    training_history = []

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_pixel_err = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, gpu_aug
        )

        # Validate only every N epochs (reduce sync overhead)
        if epoch % VALIDATION_INTERVAL == 0 or epoch == 1:
            val_loss, val_pixel_err = validate(
                model, val_loader, criterion, device
            )
        else:
            # Skip validation, reuse previous values
            val_loss = val_loss if epoch > 1 else 0.0
            val_pixel_err = val_pixel_err if epoch > 1 else 0.0

        # Scheduler step (CosineAnnealingLR steps every epoch)
        scheduler.step()

        # No LR decay after pruning - keep it constant at 1e-4
        # (Following structured pruning papers: stable low LR prevents overshoot)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        # Print summary
        if epoch % VALIDATION_INTERVAL == 0 or epoch == 1:
            # Add final training countdown to summary
            epoch_str = f"Epoch {epoch}/{EPOCHS}"
            if enable_pruning and pruning_limit_reached:
                epoch_str += f" [Final Training: {final_training_epochs_remaining} epochs remaining]"

            print(f"\n{epoch_str} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Pixel Error: {train_pixel_err:.3f}px")
            print(f"  Val Loss: {val_loss:.4f} | Pixel Error: {val_pixel_err:.3f}px")
            print(f"  LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
        else:
            epoch_str = f"Epoch {epoch}/{EPOCHS}"
            if enable_pruning and pruning_limit_reached:
                epoch_str += f" [Final: {final_training_epochs_remaining} left]"

            print(f"\n{epoch_str}: Train Loss: {train_loss:.4f} | Pixel Error: {train_pixel_err:.3f}px | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

        # TensorBoard logging
        if writer is not None:
            # Scalars (every epoch)
            if epoch % TENSORBOARD_LOG_INTERVAL == 0:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('PixelError/train', train_pixel_err, epoch)
                writer.add_scalar('LearningRate/lr', current_lr, epoch)
                writer.add_scalar('Performance/epoch_time', epoch_time, epoch)

                # Only log validation metrics when we actually validated
                if epoch % VALIDATION_INTERVAL == 0 or epoch == 1:
                    writer.add_scalar('Loss/val', val_loss, epoch)
                    writer.add_scalar('PixelError/val', val_pixel_err, epoch)

                if enable_pruning:
                    sparsity = get_model_sparsity(model, original_param_count)
                    writer.add_scalar('Sparsity/global', sparsity, epoch)

            # Images (every N epochs)
            if epoch % TENSORBOARD_IMAGE_INTERVAL == 0:
                sample_images, sample_targets = next(iter(val_loader))
                sample_images = sample_images[:8].to(device, non_blocking=True)
                sample_targets = sample_targets[:8].to(device, non_blocking=True)

                with torch.no_grad():
                    sample_preds = model(sample_images)

                vis_grid = visualize_predictions(
                    sample_images, sample_targets, sample_preds, CROP_SIZE
                )
                writer.add_image('Predictions/validation', vis_grid, epoch)

            # Histograms (every N epochs)
            if epoch % TENSORBOARD_HISTOGRAM_INTERVAL == 0:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        writer.add_histogram(f'Weights/{name}', param, epoch)
                        if param.grad is not None:
                            writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

        # Pruning logic
        if enable_pruning and pruning_active:
            # Track validation pixel error (only on validation epochs)
            if epoch % VALIDATION_INTERVAL == 0 or epoch == 1:
                pixel_error_history.append(val_pixel_err)

            # Stall detection: Boost LR if pruning hasn't happened for too long
            if epoch >= PRUNING_START_EPOCH and last_successful_prune_epoch > 0:
                epochs_since_last_prune = epoch - last_successful_prune_epoch

                if epochs_since_last_prune >= stall_reset_interval and epochs_since_last_prune % stall_reset_interval == 0:
                    # Model is stuck - try modest LR increase to help escape
                    # Not too high (avoids overshoot), just enough to explore
                    boost_lr = PRUNING_LR  # Reset to 2e-4 (from 1e-4)
                    current_lr = optimizer.param_groups[0]['lr']

                    if abs(current_lr - boost_lr) > 1e-7:  # Only if different
                        print(f"\n{'~'*60}")
                        print(f"PRUNING STALLED - LR RESET")
                        print(f"{'~'*60}")
                        print(f"  No pruning for {epochs_since_last_prune} epochs")
                        print(f"  Last prune: epoch {last_successful_prune_epoch}")
                        print(f"  Current LR: {current_lr:.6f}")
                        print(f"  Resetting LR to: {boost_lr:.6f}")
                        print(f"  (Modest increase to help escape, but not overshoot)")
                        print(f"{'~'*60}\n")

                        for param_group in optimizer.param_groups:
                            param_group['lr'] = boost_lr

                        # Track so we don't reset again immediately
                        last_pruning_epoch = epoch

            # Check if ready to prune (only on validation epochs)
            if (epoch >= PRUNING_START_EPOCH and
                epoch % PRUNING_CHECK_INTERVAL == 0 and
                (epoch % VALIDATION_INTERVAL == 0 or epoch == 1)):

                if check_pruning_readiness(pixel_error_history, VALIDATION_PIXEL_THRESHOLD):
                    current_sparsity = get_model_sparsity(model, original_param_count)

                    # Check if we need more pruning
                    if current_sparsity < current_sparsity_target * 100:
                        print(f"\n{'='*60}")
                        print(f"APPLYING STRUCTURED PRUNING at Epoch {epoch}")
                        print(f"  Current param reduction: {current_sparsity:.1f}%")
                        print(f"  Target reduction: {current_sparsity_target*100:.0f}%")

                        # Apply structured pruning (actually removes filters)
                        params_before_pruning = count_model_parameters(model)
                        model = apply_structured_pruning(model, PRUNE_AMOUNT_PER_STEP)

                        new_sparsity = get_model_sparsity(model, original_param_count)
                        remaining_params = count_model_parameters(model)
                        params_removed = params_before_pruning - remaining_params

                        print(f"  New param reduction: {new_sparsity:.1f}%")
                        print(f"  Remaining params: {remaining_params:,} (was {original_param_count:,})")
                        print(f"  Params removed this step: {params_removed:,}")
                        print(f"  Model is now {(1 - new_sparsity/100):.1%} of original size")

                        # Check if pruning actually removed parameters
                        # If no params removed, pruning can't go further (no more channels to remove)
                        if params_removed < 10:  # Essentially no change (< 10 params)
                            print(f"\n{'!'*60}")
                            print(f"PRUNING LIMIT REACHED!")
                            print(f"{'!'*60}")
                            print(f"  Cannot prune further (no more channels can be removed)")
                            print(f"  Current params: {remaining_params:,}")
                            print(f"  Param reduction: {new_sparsity:.2f}%")
                            print(f"\n  Switching to final training phase:")
                            print(f"  - Training for {final_training_epochs_remaining} more epochs")
                            print(f"  - Will save best model from this phase")
                            print(f"  - Then export ONNX and complete")
                            print(f"{'!'*60}\n")

                            pruning_limit_reached = True
                            pruning_active = False  # Disable further pruning attempts

                        else:
                            # CRITICAL: LOWER LR after pruning (following structured pruning papers)
                            # Papers (ThiNet, Network Slimming): Use low stable LR for fine-tuning
                            # This prevents overshoot: model briefly hits 0.5px, then overshoots to 1.5px
                            # Low LR = slower recovery but locks in the good solution
                            recovery_lr = PRUNING_LR * 0.5  # 1e-4 (half of 2e-4 min)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = recovery_lr
                            print(f"  LR lowered to {recovery_lr:.6f} for stable fine-tuning")
                            print(f"  (Prevents overshoot: keeps model at optimal 0.5-1.0px instead of settling at 1.5px)")
                            print(f"{'='*60}\n")

                            # Track pruning epoch (no decay needed with constant low LR)
                            last_pruning_epoch = epoch
                            last_successful_prune_epoch = epoch  # Track actual pruning (not just LR boost)

                            # Reset history after pruning to allow recovery
                            pixel_error_history.clear()

                    # If reached target and still capable, increase target
                    elif current_sparsity >= current_sparsity_target * 100:
                        print(f"\n  [OK] Reached {current_sparsity_target*100:.0f}% sparsity target!")
                        print(f"  Validation px_error has been < {VALIDATION_PIXEL_THRESHOLD} for {PRUNING_PATIENCE} epochs")
                        print(f"  Increasing target by {SPARSITY_INCREMENT*100:.0f}%")
                        current_sparsity_target += SPARSITY_INCREMENT
                        print(f"  New target: {current_sparsity_target*100:.0f}% sparsity\n")

                        # This is progress, update tracking
                        last_successful_prune_epoch = epoch

        # Final training phase tracking (after pruning limit reached)
        if enable_pruning and pruning_limit_reached:
            # Track best model during final training
            if epoch % VALIDATION_INTERVAL == 0 or epoch == 1:
                if val_pixel_err < final_training_best_val_error:
                    final_training_best_val_error = val_pixel_err
                    final_training_best_model = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': train_loss,
                        'train_pixel_error': train_pixel_err,
                        'val_loss': val_loss,
                        'val_pixel_error': val_pixel_err,
                    }
                    print(f"  [OK] New best final model: {val_pixel_err:.4f}px (epoch {epoch})")

            # Countdown
            final_training_epochs_remaining -= 1

            if final_training_epochs_remaining <= 0:
                print(f"\n{'='*60}")
                print(f"FINAL TRAINING COMPLETE!")
                print(f"{'='*60}")
                print(f"  Best final model: epoch {final_training_best_model['epoch']}")
                print(f"  Validation error: {final_training_best_val_error:.4f}px")
                print(f"  Saving best final model and exporting ONNX...")

                # Save best final model
                final_model_path = output_dir / 'final_pruned_model.pth'
                torch.save(final_training_best_model, final_model_path)
                print(f"  Saved to: {final_model_path}")

                # Load best model for ONNX export
                model.load_state_dict(final_training_best_model['model_state_dict'])
                print(f"  Loaded best model for export")
                print(f"{'='*60}\n")

                # Break out of training loop to export ONNX
                break

        # Save history
        history_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_pixel_error': train_pixel_err,
            'learning_rate': current_lr,
            'time': epoch_time,
            'sparsity': get_model_sparsity(model, original_param_count) if enable_pruning else 0.0
        }

        # Only include validation metrics when we actually validated
        if epoch % VALIDATION_INTERVAL == 0 or epoch == 1:
            history_entry['val_loss'] = val_loss
            history_entry['val_pixel_error'] = val_pixel_err

        training_history.append(history_entry)

        # Track best model within current 100-epoch interval
        if epoch % VALIDATION_INTERVAL == 0 or epoch == 1:
            if val_pixel_err < interval_best_error:
                interval_best_error = val_pixel_err
                interval_best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'train_pixel_error': train_pixel_err,
                    'val_loss': val_loss,
                    'val_pixel_error': val_pixel_err,
                }
                interval_best_epoch = epoch

        # Save best model from interval at interval boundaries
        is_restart_epoch = epoch in restart_epochs
        if (epoch % SAVE_INTERVAL == 0) or (epoch == EPOCHS) or is_restart_epoch:
            if interval_best_model_state is not None:
                # Save best from this interval
                checkpoint_path = output_dir / f'checkpoint_epoch_{interval_best_epoch}.pth'
                torch.save(interval_best_model_state, checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path}")
                print(f"    Epoch {interval_best_epoch} with {interval_best_error:.3f}px")

                # Reset interval tracking for next 100 epochs
                current_interval_start = epoch + 1
                interval_best_error = float('inf')
                interval_best_model_state = None
                interval_best_epoch = 0
            else:
                # Fallback: save current if no validation happened yet
                checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'train_pixel_error': train_pixel_err,
                }
                torch.save(checkpoint_data, checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path}")

        # Save best model (only on validation epochs)
        if epoch % VALIDATION_INTERVAL == 0 or epoch == 1:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = output_dir / 'best_model.pth'
                torch.save(model.state_dict(), best_path)
                print(f"  * New best validation loss! Saved to: {best_path}")

            if val_pixel_err < best_pixel_error:
                best_pixel_error = val_pixel_err
                best_pixel_path = output_dir / 'best_pixel_error.pth'
                torch.save(model.state_dict(), best_pixel_path)
                print(f"  * New best pixel error! Saved to: {best_pixel_path}")

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        print(f"\nTensorBoard logs saved to: {tensorboard_dir}")

    # Export to ONNX (for pruned models, export while model is in memory)
    print("\n" + "=" * 60)
    print("EXPORTING TO ONNX")
    print("=" * 60)
    try:
        onnx_path = output_dir / f"{run_name}.onnx"
        dummy_input = torch.randn(1, 3, CROP_SIZE, CROP_SIZE).to(device)

        print(f"Exporting model to: {onnx_path}")
        print(f"  Input shape: (1, 3, {CROP_SIZE}, {CROP_SIZE})")
        print(f"  Output shape: (1, 2)")

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,  # DirectML compatible
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        # Verify ONNX model
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        model_size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"  ONNX model exported successfully!")
        print(f"  Model size: {model_size_mb:.2f} MB")

        # Optimize ONNX graph when pruning is enabled (removes empty ops)
        if enable_pruning:
            try:
                from .export_onnx import optimize_onnx_graph, count_onnx_nodes
                print(f"\n  Optimizing pruned ONNX graph...")
                nodes_before = count_onnx_nodes(str(onnx_path))
                optimize_onnx_graph(str(onnx_path))
                nodes_after = count_onnx_nodes(str(onnx_path))
                print(f"  Nodes: {nodes_before} -> {nodes_after}")
            except Exception as opt_e:
                print(f"  [WARNING] ONNX optimization failed: {opt_e}")

    except Exception as e:
        print(f"  [ERROR] ONNX export failed: {e}")
        print(f"  (This is OK - you can export manually later)")

    # Training complete
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Run: {run_name}")

    # Show final pruned model info if applicable
    if enable_pruning and pruning_limit_reached and final_training_best_model is not None:
        print(f"\nFinal Pruned Model (auto-finished after pruning limit):")
        print(f"  Epoch: {final_training_best_model['epoch']}")
        print(f"  Validation error: {final_training_best_val_error:.4f}px")
        print(f"  Parameters: {count_model_parameters(model):,}")
        print(f"  Reduction: {get_model_sparsity(model, original_param_count):.2f}%")
        print(f"  Saved as: final_pruned_model.pth")
    else:
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best pixel error: {best_pixel_error:.3f}px")

    print()
    print(f"Run directory: {output_dir}")
    print(f"  ├── best_model.pth")
    print(f"  ├── best_pixel_error.pth")
    if enable_pruning and pruning_limit_reached:
        print(f"  ├── final_pruned_model.pth  ← Best from final training")
    print(f"  ├── checkpoint_epoch_*.pth")
    print(f"  ├── {run_name}.onnx  ← ONNX export")
    print(f"  ├── config.json")
    print(f"  └── training_history.json")
    print()
    print(f"TensorBoard logs: {tensorboard_dir}")
    print(f"TensorBoard: tensorboard --logdir={Path(OUTPUT_DIR) / 'tensorboard_logs'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
