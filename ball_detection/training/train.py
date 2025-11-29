"""
Training Script for Ball Detection

Two modes:
- CROP: 128x128 crops, outputs (x, y) normalized coordinates
- FULLFRAME: 1280x720 or 320x180 input, outputs (x, y, confidence)

Usage:
    python -m ball_detection.training.train --mode fullframe
    python -m ball_detection.training.train --mode crop
"""

import argparse
import time
from pathlib import Path
import numpy as np
from collections import deque
import json  # For config.json only

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from tqdm import tqdm

from ..core.model import (
    BallDetectorCNN,
    BallDetectorMobileNetV3,
    BallDetectorShuffleNetV2,
    BallDetectorFullFrameTiny,
    BallDetectorFullFrameTinyStereo,
    BallDetectorFullFrameUltra,
    BallDetectorFullFrameMobileNet,
    BallDetectorFullFrameShuffleNet,
)
from ..core.dataset import (
    create_dataloaders,
    create_fullframe_dataloaders,
    create_fullframe_memmap_dataloaders,
    create_crop_memmap_dataloaders,
    create_stereo_memmap_dataloaders,
)

# ============================================================
# SETTINGS
# ============================================================

# Model selection:
#
# CROP MODE (128x128 input → x, y output):
#   "cnn"        - Custom CNN with residual blocks
#   "mobilenet"  - MobileNetV3-Small backbone
#   "shufflenet" - ShuffleNetV2 x0.5 backbone
MODEL_CROP = "cnn"

# FULLFRAME MODE (1280x720 or 320x180 input → x, y, confidence output):
#   "tiny"      - Custom lightweight backbone (~150K params)
#   "ultra"     - PixelUnshuffle + custom backbone (~138K params)
#   "shufflenet"- ShuffleNetV2 x0.5 (~355K params)
#   "mobilenet" - PixelUnshuffle + MobileNetV3-Small (~1M params)
MODEL_FULLFRAME = "tiny"

# Resolution for fullframe mode (width, height)
RESOLUTION = (320, 180)

# Stereo mode (uses left+right pairs, 6-channel input)
USE_STEREO = True
STEREO_MEMMAP_DIR = "./ball_detection/data/stereo_memmap"

# Training parameters
EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 10
NUM_WORKERS = 8
TRAIN_SPLIT = 0.8

# Data
DATA_DIR = "./ball_detection/data/full_dataset/training_data_full"
OUTPUT_DIR = "./ball_detection/models"

# Memmap - faster loading from preprocessed data
USE_MEMMAP = True
MEMMAP_DIR = "./ball_detection/data/fullframe_memmap"  # Fullframe mode
CROP_MEMMAP_DIR = "./ball_detection/data/crop_memmap"  # Crop mode

# Augmentation
USE_SPATIAL_AUG = True
USE_APPEARANCE_AUG = True
USE_COLOR_INVARIANCE_AUG = True
USE_TEARING_AUG = True
TEARING_PROBABILITY = 0.01

# Checkpoints
SAVE_INTERVAL = 50
VALIDATION_INTERVAL = 2

# ============================================================
# PRUNING SETTINGS
# ============================================================
ENABLE_PRUNING = False
PRUNING_START_EPOCH = 150
PRUNING_CHECK_INTERVAL = 10
INITIAL_TARGET_SPARSITY = 0.90
SPARSITY_INCREMENT = 0.05
PRUNE_AMOUNT_PER_STEP = 0.1
VALIDATION_PIXEL_THRESHOLD = 1.0
PRUNING_PATIENCE = 10
PRUNING_LR = 2e-4

# ============================================================
# TENSORBOARD SETTINGS
# ============================================================
ENABLE_TENSORBOARD = True
TENSORBOARD_LOG_INTERVAL = 1
TENSORBOARD_IMAGE_INTERVAL = 50
TENSORBOARD_HISTOGRAM_INTERVAL = 100

# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(mode: str, model_name: str, pretrained: bool = True, stereo: bool = False):
    """Create model based on mode and name."""
    if mode == "crop":
        if model_name == "cnn":
            return BallDetectorCNN()
        elif model_name == "mobilenet":
            return BallDetectorMobileNetV3(pretrained=pretrained)
        elif model_name == "shufflenet":
            return BallDetectorShuffleNetV2(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown crop model: {model_name}")

    elif mode == "fullframe":
        # Stereo mode uses special model
        if stereo:
            if model_name == "tiny":
                return BallDetectorFullFrameTinyStereo()
            else:
                raise ValueError(f"Stereo mode only supports 'tiny' model, got: {model_name}")

        # Non-stereo models
        if model_name == "tiny":
            return BallDetectorFullFrameTiny()
        elif model_name == "ultra":
            return BallDetectorFullFrameUltra()
        elif model_name == "shufflenet":
            return BallDetectorFullFrameShuffleNet(pretrained=pretrained)
        elif model_name == "mobilenet":
            return BallDetectorFullFrameMobileNet(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown fullframe model: {model_name}")
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ============================================================
# LOSS FUNCTIONS
# ============================================================

class CropLoss(nn.Module):
    """MSE loss for (x, y) coordinate regression."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.mse(pred, target[:, :2])


class FullframeLoss(nn.Module):
    """
    Fullframe loss prioritizing:
    1. Confidence (ball detection) - most important
    2. Location with 30px tolerance - small errors OK
    3. Heavy outlier penalty - big mistakes punished hard
    """
    def __init__(self, resolution=(320, 180), tolerance_px=30, conf_weight=2.0, coord_weight=1.0, outlier_weight=5.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.conf_weight = conf_weight
        self.coord_weight = coord_weight
        self.outlier_weight = outlier_weight

        # Normalize tolerance to [0,1] range
        # Use average of width/height for threshold
        avg_size = (resolution[0] + resolution[1]) / 2
        self.tolerance = tolerance_px / avg_size  # ~0.12 for 30px on 320x180

        # Store for logging
        self.last_coord_loss = 0
        self.last_conf_loss = 0
        self.last_outlier_loss = 0

    def forward(self, pred, target):
        # Confidence loss (most important)
        conf_loss = self.bce(pred[:, 2], target[:, 2])

        # Only compute coord loss for positive samples (ball visible)
        positive_mask = target[:, 2] > 0.5

        if positive_mask.any():
            pred_coords = pred[positive_mask, :2]
            target_coords = target[positive_mask, :2]

            # Per-sample errors
            diff = pred_coords - target_coords
            distances = torch.sqrt((diff ** 2).sum(dim=1) + 1e-8)

            # Base coord loss (MSE-like but per sample)
            coord_loss = (diff ** 2).mean()

            # Outlier penalty: extra loss for samples > tolerance
            outlier_mask = distances > self.tolerance
            if outlier_mask.any():
                # Quadratic penalty for how much they exceed tolerance
                excess = distances[outlier_mask] - self.tolerance
                outlier_loss = (excess ** 2).mean()
            else:
                outlier_loss = torch.tensor(0.0, device=pred.device)
        else:
            coord_loss = torch.tensor(0.0, device=pred.device)
            outlier_loss = torch.tensor(0.0, device=pred.device)

        self.last_conf_loss = conf_loss.item()
        self.last_coord_loss = coord_loss.item() if torch.is_tensor(coord_loss) else coord_loss
        self.last_outlier_loss = outlier_loss.item() if torch.is_tensor(outlier_loss) else outlier_loss

        total = (self.conf_weight * conf_loss +
                 self.coord_weight * coord_loss +
                 self.outlier_weight * outlier_loss)

        return total


class StereoFullframeLoss(nn.Module):
    """
    Stereo fullframe loss for 5-output model.

    Inputs:
        pred: (batch, 5) - [x_left, y_left, x_right, y_right, confidence]
        target: (batch, 5) - [x_left, y_left, x_right, y_right, confidence]

    Loss:
        - BCE on confidence (index 4)
        - Coordinate loss for all 4 coords when confidence > 0.5
        - Outlier penalty for large errors
    """
    def __init__(self, resolution=(320, 180), tolerance_px=10, conf_weight=2.0, coord_weight=1.0, outlier_weight=5.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.conf_weight = conf_weight
        self.coord_weight = coord_weight
        self.outlier_weight = outlier_weight

        # Normalize tolerance to [0,1] range
        avg_size = (resolution[0] + resolution[1]) / 2
        self.tolerance = tolerance_px / avg_size

        # Store for logging
        self.last_coord_loss = 0
        self.last_conf_loss = 0
        self.last_outlier_loss = 0

    def forward(self, pred, target):
        # Confidence loss (index 4)
        conf_loss = self.bce(pred[:, 4], target[:, 4])

        # Only compute coord loss for positive samples (confidence > 0.5)
        positive_mask = target[:, 4] > 0.5

        if positive_mask.any():
            # Left coordinates (indices 0, 1)
            pred_left = pred[positive_mask, :2]
            target_left = target[positive_mask, :2]

            # Right coordinates (indices 2, 3)
            pred_right = pred[positive_mask, 2:4]
            target_right = target[positive_mask, 2:4]

            # Coordinate errors for both
            diff_left = pred_left - target_left
            diff_right = pred_right - target_right

            dist_left = torch.sqrt((diff_left ** 2).sum(dim=1) + 1e-8)
            dist_right = torch.sqrt((diff_right ** 2).sum(dim=1) + 1e-8)

            # Base coord loss (MSE-like)
            coord_loss = (diff_left ** 2).mean() + (diff_right ** 2).mean()

            # Outlier penalty for both views
            all_distances = torch.cat([dist_left, dist_right])
            outlier_mask = all_distances > self.tolerance
            if outlier_mask.any():
                excess = all_distances[outlier_mask] - self.tolerance
                outlier_loss = (excess ** 2).mean()
            else:
                outlier_loss = torch.tensor(0.0, device=pred.device)
        else:
            coord_loss = torch.tensor(0.0, device=pred.device)
            outlier_loss = torch.tensor(0.0, device=pred.device)

        self.last_conf_loss = conf_loss.item()
        self.last_coord_loss = coord_loss.item() if torch.is_tensor(coord_loss) else coord_loss
        self.last_outlier_loss = outlier_loss.item() if torch.is_tensor(outlier_loss) else outlier_loss

        total = (self.conf_weight * conf_loss +
                 self.coord_weight * coord_loss +
                 self.outlier_weight * outlier_loss)

        return total


# ============================================================
# METRICS
# ============================================================

def calculate_pixel_error(pred, target, size):
    """Calculate average pixel error.

    Args:
        size: Either int (for square) or tuple (width, height)
    """
    if isinstance(size, (tuple, list)):
        width, height = size
        scale = torch.tensor([width, height], device=pred.device)
    else:
        scale = size

    pred_pixels = pred[:, :2] * scale
    target_pixels = target[:, :2] * scale
    diff = pred_pixels - target_pixels
    distances = torch.sqrt((diff ** 2).sum(dim=1))
    return distances.mean().item()


# ============================================================
# PRUNING FUNCTIONS
# ============================================================

def apply_structured_pruning(model, amount, input_size):
    """Apply structured pruning using torch_pruning library."""
    import torch_pruning as tp

    imp = tp.importance.MagnitudeImportance(p=1)

    ignored_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features in [2, 3]:
            ignored_layers.append(module)

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=torch.randn(1, 3, input_size[1], input_size[0]).to(next(model.parameters()).device),
        importance=imp,
        iterative_steps=1,
        pruning_ratio=amount,
        ignored_layers=ignored_layers,
    )

    pruner.step()

    # Reset BatchNorm statistics
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.reset_running_stats()

    return model


def get_model_sparsity(model, original_params):
    """Calculate percentage parameter reduction."""
    current_params = count_model_parameters(model)
    if original_params is not None:
        reduction = 1.0 - (current_params / original_params)
        return reduction * 100
    return 0.0


def count_model_parameters(model):
    """Count total parameters."""
    return sum(p.numel() for p in model.parameters())


def check_pruning_readiness(history, threshold):
    """Check if all values in deque are below threshold."""
    if len(history) < history.maxlen:
        return False
    return all(err < threshold for err in history)


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_predictions(images, targets, preds, img_size, stereo=False):
    """Create visualization grid with GT (green) and predictions (red)."""
    import cv2

    # BGR order for denormalization (model trained on BGR)
    mean_bgr = torch.tensor([0.406, 0.456, 0.485]).view(3, 1, 1)
    std_bgr = torch.tensor([0.225, 0.224, 0.229]).view(3, 1, 1)

    vis_images = []
    for i in range(min(8, len(images))):
        img = images[i].cpu()

        if stereo:
            # Stereo: 6-channel image, show left (first 3 channels) and right (last 3)
            left_img = img[:3]
            right_img = img[3:]

            left_img = left_img * std_bgr + mean_bgr
            right_img = right_img * std_bgr + mean_bgr

            left_img = torch.clamp(left_img, 0, 1).numpy().transpose(1, 2, 0)
            right_img = torch.clamp(right_img, 0, 1).numpy().transpose(1, 2, 0)

            left_img = (left_img * 255).astype(np.uint8).copy()
            right_img = (right_img * 255).astype(np.uint8).copy()

            h, w = left_img.shape[:2]

            # Left image: left GT and pred (BGR colors for cv2)
            gt_x_l, gt_y_l = int(targets[i, 0] * w), int(targets[i, 1] * h)
            pred_x_l, pred_y_l = int(preds[i, 0] * w), int(preds[i, 1] * h)
            cv2.circle(left_img, (gt_x_l, gt_y_l), 3, (0, 255, 0), -1)
            cv2.circle(left_img, (pred_x_l, pred_y_l), 3, (0, 0, 255), -1)  # Red in BGR
            cv2.line(left_img, (gt_x_l, gt_y_l), (pred_x_l, pred_y_l), (0, 255, 255), 1)  # Yellow in BGR

            # Right image: right GT and pred (BGR colors for cv2)
            gt_x_r, gt_y_r = int(targets[i, 2] * w), int(targets[i, 3] * h)
            pred_x_r, pred_y_r = int(preds[i, 2] * w), int(preds[i, 3] * h)
            cv2.circle(right_img, (gt_x_r, gt_y_r), 3, (0, 255, 0), -1)
            cv2.circle(right_img, (pred_x_r, pred_y_r), 3, (0, 0, 255), -1)  # Red in BGR
            cv2.line(right_img, (gt_x_r, gt_y_r), (pred_x_r, pred_y_r), (0, 255, 255), 1)  # Yellow in BGR

            # Convert BGR->RGB for display and stack
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
            combined = np.concatenate([left_img, right_img], axis=1)
            vis_images.append(torch.from_numpy(combined).permute(2, 0, 1).float() / 255.0)
        else:
            # Non-stereo: 3-channel image
            img = img * std_bgr + mean_bgr
            img = torch.clamp(img, 0, 1)
            img = img.numpy().transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8).copy()

            h, w = img.shape[:2]
            gt_x, gt_y = int(targets[i, 0] * w), int(targets[i, 1] * h)
            pred_x, pred_y = int(preds[i, 0] * w), int(preds[i, 1] * h)

            # BGR colors for cv2
            cv2.circle(img, (gt_x, gt_y), 3, (0, 255, 0), -1)
            cv2.circle(img, (pred_x, pred_y), 3, (0, 0, 255), -1)  # Red in BGR
            cv2.line(img, (gt_x, gt_y), (pred_x, pred_y), (0, 255, 255), 1)  # Yellow in BGR

            # Convert BGR->RGB for display
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            vis_images.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)

    return vutils.make_grid(vis_images, nrow=4 if not stereo else 2, padding=2)


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

        if use_amp:
            with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                outputs = model(images)
        else:
            outputs = model(images)

        loss = criterion(outputs.float(), targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pixel_error = calculate_pixel_error(outputs.float(), targets, pixel_size)
        total_loss += loss.detach()
        total_pixel_error += pixel_error

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'px_err': f'{pixel_error:.3f}'})

    avg_loss = (total_loss / num_batches).item()
    avg_pixel_error = total_pixel_error / num_batches
    return avg_loss, avg_pixel_error


def validate(model, dataloader, criterion, device, pixel_size, stereo=False):
    """Validate model. Returns avg_loss, avg_pixel_error, worst_pixel_error, p98_pixel_error."""
    model.eval()
    total_loss = 0
    worst_pixel_error = 0
    all_errors_list = []
    num_batches = len(dataloader)

    # Handle tuple or scalar pixel_size
    if isinstance(pixel_size, (tuple, list)):
        scale = torch.tensor([pixel_size[0], pixel_size[1]], device=device)
    else:
        scale = pixel_size

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Validation'):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            if stereo:
                # Stereo mode: pool left and right errors together
                # targets: [x_left, y_left, x_right, y_right, confidence]
                positive_mask = targets[:, 4] > 0.5

                if positive_mask.any():
                    # Left errors
                    pred_left = outputs[positive_mask, :2] * scale
                    target_left = targets[positive_mask, :2] * scale
                    dist_left = torch.sqrt(((pred_left - target_left) ** 2).sum(dim=1))

                    # Right errors
                    pred_right = outputs[positive_mask, 2:4] * scale
                    target_right = targets[positive_mask, 2:4] * scale
                    dist_right = torch.sqrt(((pred_right - target_right) ** 2).sum(dim=1))

                    # Pool all errors together
                    valid_errors = torch.cat([dist_left, dist_right])
                else:
                    valid_errors = None
            else:
                # Non-stereo mode (original logic)
                pred_pixels = outputs[:, :2] * scale
                target_pixels = targets[:, :2] * scale
                diff = pred_pixels - target_pixels
                distances = torch.sqrt((diff ** 2).sum(dim=1))

                # Fullframe mode: only count positive samples (confidence > 0.5)
                # Crop mode: all samples are positive (ball always visible in crop)
                if targets.shape[1] > 2:
                    positive_mask = targets[:, 2] > 0.5
                    if positive_mask.any():
                        valid_errors = distances[positive_mask]
                    else:
                        valid_errors = None
                else:
                    valid_errors = distances

            if valid_errors is not None and len(valid_errors) > 0:
                max_error = valid_errors.max().item()
                worst_pixel_error = max(worst_pixel_error, max_error)
                all_errors_list.append(valid_errors.cpu())

            total_loss += loss.item()

    # Calculate avg and p98
    avg_pixel_error = 0
    p98_pixel_error = 0
    if all_errors_list:
        all_errors = torch.cat(all_errors_list)
        avg_pixel_error = all_errors.mean().item()
        p98_pixel_error = torch.quantile(all_errors, 0.98).item()

    return total_loss / num_batches, avg_pixel_error, worst_pixel_error, p98_pixel_error


# ============================================================
# MAIN
# ============================================================

def main(MODE):
    # Select model based on mode
    MODEL = MODEL_FULLFRAME if MODE == "fullframe" else MODEL_CROP

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
        if USE_STEREO:
            print(f"Stereo: Enabled (6-channel input)")
    print(f"Device: {device}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Pruning: {'Enabled' if ENABLE_PRUNING else 'Disabled'}")
    print("=" * 60)

    # Create output directory
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"run_{timestamp}_{MODEL}"
    if MODE == "fullframe":
        run_name += f"_{RESOLUTION[0]}x{RESOLUTION[1]}"
        if USE_STEREO:
            run_name += "_stereo"
    if ENABLE_PRUNING:
        run_name += "_pruning"

    output_dir = Path(OUTPUT_DIR) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {output_dir}\n")

    # Save config
    config = {
        'mode': MODE,
        'model': MODEL,
        'resolution': RESOLUTION if MODE == "fullframe" else (128, 128),
        'stereo': USE_STEREO if MODE == "fullframe" else False,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'enable_pruning': ENABLE_PRUNING,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # TensorBoard - centralized location
    writer = None
    if ENABLE_TENSORBOARD:
        tensorboard_base = Path(OUTPUT_DIR) / 'tensorboard_logs'
        tensorboard_dir = tensorboard_base / run_name
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tensorboard_dir), max_queue=10000, flush_secs=300)
        print(f"TensorBoard: {tensorboard_dir}")
        print(f"  Command: tensorboard --logdir={tensorboard_base}\n")

    # Create dataloaders
    if MODE == "crop":
        # Crop mode: only positive samples (ball always visible)
        if USE_MEMMAP:
            print(f"Loading crop memmap from: {CROP_MEMMAP_DIR}")
            train_loader, val_loader = create_crop_memmap_dataloaders(
                data_dir=CROP_MEMMAP_DIR,
                batch_size=BATCH_SIZE,
                crop_size=128,
                train_split=TRAIN_SPLIT,
                num_workers=NUM_WORKERS,
            )
        else:
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
                use_tearing_augmentation=False,
                tearing_probability=0,
            )
        pixel_size = 128
        input_size = (128, 128)
        criterion = CropLoss()
    else:
        # Fullframe mode - check for stereo
        if USE_STEREO:
            print(f"Loading stereo memmap data from: {STEREO_MEMMAP_DIR}")
            train_loader, val_loader = create_stereo_memmap_dataloaders(
                data_dir=STEREO_MEMMAP_DIR,
                batch_size=BATCH_SIZE,
                train_split=TRAIN_SPLIT,
                num_workers=NUM_WORKERS,
            )
            criterion = StereoFullframeLoss(resolution=RESOLUTION, tolerance_px=5)
        elif USE_MEMMAP:
            print(f"Loading memmap data from: {MEMMAP_DIR}")
            train_loader, val_loader = create_fullframe_memmap_dataloaders(
                data_dir=MEMMAP_DIR,
                batch_size=BATCH_SIZE,
                train_split=TRAIN_SPLIT,
                num_workers=NUM_WORKERS,
            )
            criterion = FullframeLoss(resolution=RESOLUTION, tolerance_px=5)
        else:
            print(f"Loading data from: {DATA_DIR}")
            train_loader, val_loader = create_fullframe_dataloaders(
                data_dir=DATA_DIR,
                batch_size=BATCH_SIZE,
                target_size=RESOLUTION,
                train_split=TRAIN_SPLIT,
                num_workers=NUM_WORKERS,
            )
            criterion = FullframeLoss(resolution=RESOLUTION, tolerance_px=5)
        pixel_size = RESOLUTION  # (width, height) tuple
        input_size = RESOLUTION

    # Create model
    print(f"\nCreating model: {MODEL}")
    model = create_model(MODE, MODEL, pretrained=True, stereo=USE_STEREO and MODE == "fullframe")
    model = model.to(device)

    original_param_count = count_model_parameters(model)
    print(f"Parameters: {original_param_count:,}")

    # Compile model for faster training (PyTorch 2.0+)
    if device.type == 'cuda':
        model = torch.compile(model, mode='reduce-overhead')
        print("Model compiled with torch.compile()")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-5
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS]
    )

    # Pruning state
    pruning_active = ENABLE_PRUNING
    pruning_limit_reached = False
    pixel_error_history = deque(maxlen=PRUNING_PATIENCE)
    current_sparsity_target = INITIAL_TARGET_SPARSITY
    last_pruning_epoch = 0

    # Training loop
    print(f"\nStarting training for {EPOCHS} epochs...")
    print("=" * 60)

    best_val_loss = float('inf')
    best_pixel_error = float('inf')
    val_loss, val_pixel_err, val_worst_px_err, val_p98_px_err = 0, 0, 0, 0

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_px_err = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, pixel_size
        )

        # Validate
        if epoch % VALIDATION_INTERVAL == 0 or epoch == 1:
            val_loss, val_pixel_err, val_worst_px_err, val_p98_px_err = validate(model, val_loader, criterion, device, pixel_size, stereo=USE_STEREO and MODE == "fullframe")

        scheduler.step()
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        # Print summary
        if epoch % VALIDATION_INTERVAL == 0 or epoch == 1:
            print(f"\nEpoch {epoch}/{EPOCHS}:")
            print(f"  Train: loss={train_loss:.4f}, px_err={train_px_err:.3f}")
            print(f"  Val:   loss={val_loss:.4f}, px_err={val_pixel_err:.3f}, p98={val_p98_px_err:.3f}, worst={val_worst_px_err:.3f}")
            print(f"  LR: {current_lr:.6f}, Time: {epoch_time:.1f}s")
            if ENABLE_PRUNING:
                sparsity = get_model_sparsity(model, original_param_count)
                print(f"  Sparsity: {sparsity:.1f}%")
        else:
            print(f"\nEpoch {epoch}: loss={train_loss:.4f}, px_err={train_px_err:.3f}, time={epoch_time:.1f}s")

        # TensorBoard logging
        if writer is not None:
            if epoch % TENSORBOARD_LOG_INTERVAL == 0:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('PixelError/train', train_px_err, epoch)
                writer.add_scalar('LearningRate/lr', current_lr, epoch)
                writer.add_scalar('Performance/epoch_time', epoch_time, epoch)

                # Fullframe mode: log coord, conf, and outlier losses separately
                if MODE == "fullframe" and hasattr(criterion, 'last_coord_loss'):
                    writer.add_scalar('Loss/coord', criterion.last_coord_loss, epoch)
                    writer.add_scalar('Loss/conf', criterion.last_conf_loss, epoch)
                    writer.add_scalar('Loss/outlier', criterion.last_outlier_loss, epoch)

                if epoch % VALIDATION_INTERVAL == 0 or epoch == 1:
                    writer.add_scalar('Loss/val', val_loss, epoch)
                    writer.add_scalar('PixelError/val', val_pixel_err, epoch)
                    writer.add_scalar('PixelError/val_p98', val_p98_px_err, epoch)
                    writer.add_scalar('PixelError/val_worst', val_worst_px_err, epoch)

                if ENABLE_PRUNING:
                    sparsity = get_model_sparsity(model, original_param_count)
                    writer.add_scalar('Sparsity/global', sparsity, epoch)

            # Images
            if epoch % TENSORBOARD_IMAGE_INTERVAL == 0:
                sample_images, sample_targets = next(iter(val_loader))
                sample_images = sample_images[:8].to(device)
                sample_targets = sample_targets[:8].to(device)
                with torch.no_grad():
                    sample_preds = model(sample_images)
                vis_grid = visualize_predictions(sample_images, sample_targets, sample_preds, pixel_size, stereo=USE_STEREO and MODE == "fullframe")
                writer.add_image('Predictions/validation', vis_grid, epoch)

            # Histograms
            if epoch % TENSORBOARD_HISTOGRAM_INTERVAL == 0:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        writer.add_histogram(f'Weights/{name}', param, epoch)
                        if param.grad is not None:
                            writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

            if epoch % 50 == 0:
                writer.flush()

        # Pruning logic
        if ENABLE_PRUNING and pruning_active and not pruning_limit_reached:
            if epoch % VALIDATION_INTERVAL == 0 or epoch == 1:
                pixel_error_history.append(val_pixel_err)

            if (epoch >= PRUNING_START_EPOCH and
                epoch % PRUNING_CHECK_INTERVAL == 0 and
                (epoch % VALIDATION_INTERVAL == 0 or epoch == 1)):

                if check_pruning_readiness(pixel_error_history, VALIDATION_PIXEL_THRESHOLD):
                    current_sparsity = get_model_sparsity(model, original_param_count)

                    if current_sparsity < current_sparsity_target * 100:
                        print(f"\n{'='*60}")
                        print(f"APPLYING STRUCTURED PRUNING at Epoch {epoch}")
                        print(f"  Current: {current_sparsity:.1f}%, Target: {current_sparsity_target*100:.0f}%")

                        params_before = count_model_parameters(model)
                        model = apply_structured_pruning(model, PRUNE_AMOUNT_PER_STEP, input_size)
                        params_after = count_model_parameters(model)

                        new_sparsity = get_model_sparsity(model, original_param_count)
                        print(f"  New: {new_sparsity:.1f}%, Params: {params_after:,}")

                        if params_before - params_after < 10:
                            print(f"  PRUNING LIMIT REACHED!")
                            pruning_limit_reached = True
                            pruning_active = False
                        else:
                            for pg in optimizer.param_groups:
                                pg['lr'] = PRUNING_LR * 0.5
                            pixel_error_history.clear()
                            last_pruning_epoch = epoch

                        print(f"{'='*60}\n")

                    elif current_sparsity >= current_sparsity_target * 100:
                        current_sparsity_target += SPARSITY_INCREMENT

        # Save best models
        if epoch % VALIDATION_INTERVAL == 0 or epoch == 1:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), output_dir / 'best_model.pth')
                print(f"  * New best loss!")

            if val_pixel_err < best_pixel_error:
                best_pixel_error = val_pixel_err
                torch.save(model.state_dict(), output_dir / 'best_pixel_error.pth')
                print(f"  * New best pixel error!")

        # Checkpoints
        if epoch % SAVE_INTERVAL == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, output_dir / f'checkpoint_epoch_{epoch}.pth')
            print(f"  Checkpoint saved")

    # Close TensorBoard
    if writer is not None:
        writer.close()

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Run: {run_name}")
    print(f"Best loss: {best_val_loss:.4f}")
    print(f"Best pixel error: {best_pixel_error:.3f}px")
    if ENABLE_PRUNING:
        print(f"Final params: {count_model_parameters(model):,} ({get_model_sparsity(model, original_param_count):.1f}% reduction)")
    print(f"\nOutput: {output_dir}")
    if ENABLE_TENSORBOARD:
        print(f"TensorBoard: tensorboard --logdir={Path(OUTPUT_DIR) / 'tensorboard_logs'}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ball detection model')
    parser.add_argument('--mode', type=str, choices=['crop', 'fullframe'],
                        default='fullframe', help='Training mode (default: fullframe)')
    args = parser.parse_args()

    main(args.mode)
