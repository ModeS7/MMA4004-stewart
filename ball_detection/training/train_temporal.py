"""
Training Script for EfficientTemporalTiny (Stereo + GRU)

Trains on sequences of consecutive stereo frames.
- Non-overlapping sequences with random offset per epoch
- Same augmentation across all frames in sequence
- Backprop through time (BPTT) for GRU learning

Usage:
    python -m ball_detection.training.train_temporal
"""

import time
from pathlib import Path
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from ..core.model import EfficientTemporalTiny
from ..core.stereo_dataset import MultiTemporalDataset

# ============================================================
# SETTINGS
# ============================================================
DATA_DIRS = [
    "./ball_detection/data/full_dataset",
]
OUTPUT_DIR = "./ball_detection/models"

# Training
EPOCHS = 100
SEQUENCE_LENGTH = 32        # Stereo frames per sequence
BATCH_SIZE = 4              # Number of sequences per batch (reduce if OOM)
LEARNING_RATE = 0.001
WARMUP_EPOCHS = 10
WEIGHT_DECAY = 1e-4
SAVE_INTERVAL = 10

# Model
HISTORY_LENGTH = 8          # GRU history (internal buffer for streaming)
HIDDEN_SIZE = 128           # GRU hidden size

# Robustness
FRAME_SKIP_PROB = 0.1       # 10% random frame skipping during training

# Image dimensions
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280

# Options (for debugging/speed)
ENABLE_AUGMENTATION = False  # Set False for faster training/debugging
ENABLE_TENSORBOARD = True
USE_TORCH_COMPILE = True    # Set False if experiencing crashes
# ============================================================


def temporal_collate_fn(batch):
    """Stack multiple sequences into a batch, padding to max length.

    Args:
        batch: List of (frames, targets) tuples
               Each frames: (T, 6, H, W), targets: (T, 6)

    Returns:
        frames: (B, T_max, 6, H, W)
        targets: (B, T_max, 6)
        lengths: (B,) actual lengths of each sequence
    """
    frames_list = [item[0] for item in batch]
    targets_list = [item[1] for item in batch]

    # Find max sequence length in batch
    lengths = [f.shape[0] for f in frames_list]
    max_len = max(lengths)

    # Pad sequences to max length
    B = len(batch)
    C, H, W = frames_list[0].shape[1:]

    frames_padded = torch.zeros(B, max_len, C, H, W)
    targets_padded = torch.zeros(B, max_len, 6)

    for i, (frames, targets) in enumerate(zip(frames_list, targets_list)):
        T = frames.shape[0]
        frames_padded[i, :T] = frames
        targets_padded[i, :T] = targets
        # Padded frames stay zero, padded targets have conf=0 (no ball)

    return frames_padded, targets_padded, torch.tensor(lengths)


class TemporalDetectionLoss(nn.Module):
    """Loss for temporal stereo detection with batch support."""

    def __init__(self, coord_weight=1.0, conf_weight=0.5):
        super().__init__()
        self.coord_weight = coord_weight
        self.conf_weight = conf_weight
        self.mse = nn.MSELoss(reduction='none')
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, pred, target, return_components=False):
        """
        Args:
            pred: (B, 6) batched predictions
            target: (B, 6) batched targets
            return_components: If True, return (total_loss, coord_loss, conf_loss)

        Returns:
            Scalar loss (mean over batch)
        """
        # Coordinates (weighted by target confidence)
        # pred[:, :2] = left xy, pred[:, 3:5] = right xy
        coord_loss_left = self.mse(pred[:, :2], target[:, :2]).mean(dim=1) * target[:, 2]  # (B,)
        coord_loss_right = self.mse(pred[:, 3:5], target[:, 3:5]).mean(dim=1) * target[:, 5]  # (B,)
        coord_loss = (coord_loss_left + coord_loss_right).mean()  # scalar

        # Confidence BCE
        conf_loss_left = self.bce(pred[:, 2], target[:, 2]).mean()  # scalar
        conf_loss_right = self.bce(pred[:, 5], target[:, 5]).mean()  # scalar
        conf_loss = conf_loss_left + conf_loss_right

        total_loss = self.coord_weight * coord_loss + self.conf_weight * conf_loss

        if return_components:
            return total_loss, coord_loss, conf_loss
        return total_loss


def calculate_classification_metrics_batch(pred, target, threshold=0.5):
    """
    Calculate classification metrics for batched confidence predictions.

    Args:
        pred: (B, 6) batched predictions
        target: (B, 6) batched targets

    Returns:
        dict with: tp, fp, tn, fn counts for both cameras (summed over batch)
    """
    # Binary predictions (B,)
    pred_left = (pred[:, 2] >= threshold).int()
    pred_right = (pred[:, 5] >= threshold).int()
    gt_left = (target[:, 2] >= threshold).int()
    gt_right = (target[:, 5] >= threshold).int()

    # Left camera counts
    tp_left = ((pred_left == 1) & (gt_left == 1)).sum().item()
    fp_left = ((pred_left == 1) & (gt_left == 0)).sum().item()
    tn_left = ((pred_left == 0) & (gt_left == 0)).sum().item()
    fn_left = ((pred_left == 0) & (gt_left == 1)).sum().item()

    # Right camera counts
    tp_right = ((pred_right == 1) & (gt_right == 1)).sum().item()
    fp_right = ((pred_right == 1) & (gt_right == 0)).sum().item()
    tn_right = ((pred_right == 0) & (gt_right == 0)).sum().item()
    fn_right = ((pred_right == 0) & (gt_right == 1)).sum().item()

    return {
        'tp_left': tp_left, 'fp_left': fp_left, 'tn_left': tn_left, 'fn_left': fn_left,
        'tp_right': tp_right, 'fp_right': fp_right, 'tn_right': tn_right, 'fn_right': fn_right,
    }


def calculate_pixel_error_batch(pred, target, image_width=1280, image_height=720):
    """
    Calculate pixel error for batched predictions.

    Args:
        pred: (B, 6) batched predictions
        target: (B, 6) batched targets

    Returns:
        (sum_error_left, sum_error_right, valid_left_count, valid_right_count)
    """
    # Scale to pixels
    pred_x_left = pred[:, 0] * image_width
    pred_y_left = pred[:, 1] * image_height
    pred_x_right = pred[:, 3] * image_width
    pred_y_right = pred[:, 4] * image_height

    target_x_left = target[:, 0] * image_width
    target_y_left = target[:, 1] * image_height
    target_x_right = target[:, 3] * image_width
    target_y_right = target[:, 4] * image_height

    # Euclidean error
    error_left = torch.sqrt((pred_x_left - target_x_left)**2 + (pred_y_left - target_y_left)**2)
    error_right = torch.sqrt((pred_x_right - target_x_right)**2 + (pred_y_right - target_y_right)**2)

    # Mask for valid detections
    valid_left = target[:, 2] >= 0.5
    valid_right = target[:, 5] >= 0.5

    # Sum errors only for valid detections
    sum_error_left = (error_left * valid_left.float()).sum().item()
    sum_error_right = (error_right * valid_right.float()).sum().item()
    valid_left_count = valid_left.sum().item()
    valid_right_count = valid_right.sum().item()

    return sum_error_left, sum_error_right, valid_left_count, valid_right_count


def train_epoch(model, data_loader, dataset, criterion, optimizer, device, epoch, use_amp=False, profile_batches=5):
    """Train for one epoch with sequence batching and AMP."""
    model.train()

    total_loss = 0
    total_coord_loss = 0
    total_conf_loss = 0
    total_error_left = 0
    total_error_right = 0
    total_valid_left = 0
    total_valid_right = 0
    total_frames = 0

    # Classification metrics
    cls_metrics = {
        'tp_left': 0, 'fp_left': 0, 'tn_left': 0, 'fn_left': 0,
        'tp_right': 0, 'fp_right': 0, 'tn_right': 0, 'fn_right': 0,
    }

    # Profiling timers (first epoch only)
    if epoch == 1 and profile_batches > 0:
        import time as _time
        profile_times = {'data': [], 'transfer': [], 'forward': [], 'backward': [], 'total': []}
    else:
        profile_times = None

    # Reshuffle sequences for this epoch
    dataset.reshuffle_epoch()

    # Process batches of sequences via DataLoader
    pbar = tqdm(data_loader, desc=f'Epoch {epoch} Training')
    batch_idx = 0
    t_batch_end = None  # For measuring data loading time

    for frames, targets, lengths in pbar:
        if profile_times is not None and batch_idx < profile_batches:
            torch.cuda.synchronize()
            t_after_data = _time.perf_counter()
            # Data loading time = time since last batch ended
            if t_batch_end is not None:
                profile_times['data'].append(t_after_data - t_batch_end)
        # frames: (B, T_max, 6, H, W), targets: (B, T_max, 6), lengths: (B,)
        # Non-blocking transfer to GPU
        frames = frames.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        if profile_times is not None and batch_idx < profile_batches:
            torch.cuda.synchronize()
            t_transfer_end = _time.perf_counter()
            profile_times['transfer'].append(t_transfer_end - t_after_data)

        B, T_max = frames.shape[:2]

        # Create mask for valid timesteps: (B, T_max)
        time_indices = torch.arange(T_max, device=device).unsqueeze(0)  # (1, T_max)
        mask = time_indices < lengths.unsqueeze(1)  # (B, T_max)

        # === BATCHED FEATURE EXTRACTION ===
        # Reshape: (B, T_max, 6, H, W) -> (B*T_max, 6, H, W)
        frames_flat = frames.view(B * T_max, *frames.shape[2:])

        # Process all B*T_max frames through CNN at once with AMP
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            all_feats = model.encoder.extract_features(frames_flat)  # (B*T_max, 256)
        all_feats = all_feats.float()  # Cast back to FP32 for GRU stability

        # Reshape: (B*T_max, 256) -> (B, T_max, 256)
        all_feats = all_feats.view(B, T_max, -1)

        # === PARALLEL GRU OVER B SEQUENCES ===
        # Initialize hidden state for B sequences
        hidden = torch.zeros(1, B, model.hidden_size, device=device)
        batch_loss = torch.tensor(0.0, device=device)
        batch_coord_loss = torch.tensor(0.0, device=device)
        batch_conf_loss = torch.tensor(0.0, device=device)

        # Store predictions for metrics: list of (B, 6)
        all_preds = []
        num_valid_frames = 0

        for t in range(T_max):
            feat = all_feats[:, t:t+1, :]  # (B, 1, 256)
            target_t = targets[:, t, :]    # (B, 6)
            mask_t = mask[:, t]            # (B,) - which sequences are valid at this timestep

            gru_out, hidden = model.gru(feat, hidden)  # (B, 1, hidden), (1, B, hidden)

            # Predict
            output = model.head(gru_out[:, -1])  # (B, hidden) -> (B, 6)
            pred = torch.sigmoid(output)  # (B, 6)
            all_preds.append(pred)

            # Only compute loss for valid (non-padded) frames
            if mask_t.any():
                valid_pred = pred[mask_t]      # (num_valid, 6)
                valid_target = target_t[mask_t]  # (num_valid, 6)
                frame_loss, coord_loss, conf_loss = criterion(valid_pred, valid_target, return_components=True)
                batch_loss = batch_loss + frame_loss
                batch_coord_loss = batch_coord_loss + coord_loss
                batch_conf_loss = batch_conf_loss + conf_loss
                num_valid_frames += mask_t.sum().item()

        if profile_times is not None and batch_idx < profile_batches:
            torch.cuda.synchronize()
            t_forward_end = _time.perf_counter()

        # Backprop through entire batch of sequences
        optimizer.zero_grad()
        batch_loss.backward()

        # Gradient clipping for RNN stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if profile_times is not None and batch_idx < profile_batches:
            torch.cuda.synchronize()
            t_backward_end = _time.perf_counter()
            profile_times['forward'].append(t_forward_end - t_transfer_end)
            profile_times['backward'].append(t_backward_end - t_forward_end)
            t_batch_end = t_backward_end  # For next iteration's data loading measurement

            # Print profiling summary immediately after collecting enough samples
            if batch_idx == profile_batches - 1:
                print("\n" + "="*60)
                print(f"PROFILING SUMMARY (first {profile_batches} batches)")
                print("="*60)
                avg_data = sum(profile_times['data']) / len(profile_times['data']) if profile_times['data'] else 0
                avg_transfer = sum(profile_times['transfer']) / len(profile_times['transfer']) if profile_times['transfer'] else 0
                avg_forward = sum(profile_times['forward']) / len(profile_times['forward'])
                avg_backward = sum(profile_times['backward']) / len(profile_times['backward'])
                avg_total = avg_data + avg_transfer + avg_forward + avg_backward
                print(f"Data Loading:  {avg_data:.3f}s ({100*avg_data/avg_total:.1f}%)")
                print(f"GPU Transfer:  {avg_transfer:.3f}s ({100*avg_transfer/avg_total:.1f}%)")
                print(f"Forward Pass:  {avg_forward:.3f}s ({100*avg_forward/avg_total:.1f}%)")
                print(f"Backward Pass: {avg_backward:.3f}s ({100*avg_backward/avg_total:.1f}%)")
                print(f"TOTAL:         {avg_total:.3f}s per batch")
                print("="*60 + "\n")

        batch_idx += 1

        # === SINGLE SYNC POINT: extract metrics ===
        with torch.no_grad():
            batch_loss_val = batch_loss.item()
            batch_coord_loss_val = batch_coord_loss.item()
            batch_conf_loss_val = batch_conf_loss.item()

            # Stack predictions: (T_max, B, 6)
            all_preds_stacked = torch.stack(all_preds, dim=0)  # (T_max, B, 6)
            targets_t = targets.transpose(0, 1)  # (B, T_max, 6) -> (T_max, B, 6)
            mask_t = mask.transpose(0, 1)  # (T_max, B)

            # Flatten and apply mask for metrics
            preds_flat = all_preds_stacked.view(-1, 6)  # (T_max*B, 6)
            targets_flat = targets_t.reshape(-1, 6)    # (T_max*B, 6)
            mask_flat = mask_t.reshape(-1)             # (T_max*B,)

            # Only compute metrics on valid frames
            valid_preds = preds_flat[mask_flat]
            valid_targets = targets_flat[mask_flat]

            # Pixel errors
            err_l, err_r, valid_l, valid_r = calculate_pixel_error_batch(valid_preds, valid_targets)

            # Classification metrics
            frame_cls = calculate_classification_metrics_batch(valid_preds, valid_targets)
            for k, v in frame_cls.items():
                cls_metrics[k] += v

        # Accumulate metrics
        total_loss += batch_loss_val
        total_coord_loss += batch_coord_loss_val
        total_conf_loss += batch_conf_loss_val
        total_error_left += err_l
        total_error_right += err_r
        total_valid_left += valid_l
        total_valid_right += valid_r
        total_frames += num_valid_frames

        pbar.set_postfix({
            'loss': f'{batch_loss_val/max(num_valid_frames,1):.4f}',
            'px_L': f'{err_l/max(valid_l,1):.1f}',
            'px_R': f'{err_r/max(valid_r,1):.1f}'
        })

    # Calculate final metrics
    avg_loss = total_loss / total_frames
    avg_coord_loss = total_coord_loss / total_frames
    avg_conf_loss = total_conf_loss / total_frames
    # Pixel error: divide by valid counts (frames where ball was visible)
    avg_error_left = total_error_left / max(total_valid_left, 1)
    avg_error_right = total_error_right / max(total_valid_right, 1)

    # Classification metrics
    tp = cls_metrics['tp_left'] + cls_metrics['tp_right']
    fp = cls_metrics['fp_left'] + cls_metrics['fp_right']
    tn = cls_metrics['tn_left'] + cls_metrics['tn_right']
    fn = cls_metrics['fn_left'] + cls_metrics['fn_right']

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        'loss': avg_loss,
        'coord_loss': avg_coord_loss,
        'conf_loss': avg_conf_loss,
        'error_left': avg_error_left,
        'error_right': avg_error_right,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

    # Print profiling summary
    if profile_times is not None and len(profile_times['forward']) > 0:
        print("\n" + "="*60)
        print("PROFILING SUMMARY (first {} batches)".format(len(profile_times['forward'])))
        print("="*60)
        avg_data = sum(profile_times['data']) / len(profile_times['data']) if profile_times['data'] else 0
        avg_transfer = sum(profile_times['transfer']) / len(profile_times['transfer']) if profile_times['transfer'] else 0
        avg_forward = sum(profile_times['forward']) / len(profile_times['forward'])
        avg_backward = sum(profile_times['backward']) / len(profile_times['backward'])
        avg_total = avg_data + avg_transfer + avg_forward + avg_backward
        print(f"  Data Loading: {avg_data*1000:7.1f}ms ({100*avg_data/avg_total:5.1f}%)")
        print(f"  GPU Transfer: {avg_transfer*1000:7.1f}ms ({100*avg_transfer/avg_total:5.1f}%)")
        print(f"  Forward:      {avg_forward*1000:7.1f}ms ({100*avg_forward/avg_total:5.1f}%)")
        print(f"  Backward:     {avg_backward*1000:7.1f}ms ({100*avg_backward/avg_total:5.1f}%)")
        print(f"  TOTAL:        {avg_total*1000:7.1f}ms per batch")
        print("="*60 + "\n")

    return metrics


def validate(model, data_loader, criterion, device, use_amp=False):
    """Validate on sequences with sequence batching."""
    model.eval()

    total_loss = 0
    total_coord_loss = 0
    total_conf_loss = 0
    total_error_left = 0
    total_error_right = 0
    total_valid_left = 0
    total_valid_right = 0
    total_frames = 0

    # Classification metrics
    cls_metrics = {
        'tp_left': 0, 'fp_left': 0, 'tn_left': 0, 'fn_left': 0,
        'tp_right': 0, 'fp_right': 0, 'tn_right': 0, 'fn_right': 0,
    }

    with torch.no_grad():
        for frames, targets, lengths in tqdm(data_loader, desc='Validation'):
            # frames: (B, T_max, 6, H, W), targets: (B, T_max, 6), lengths: (B,)
            frames = frames.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            B, T_max = frames.shape[:2]

            # Create mask for valid timesteps
            time_indices = torch.arange(T_max, device=device).unsqueeze(0)
            mask = time_indices < lengths.unsqueeze(1)  # (B, T_max)

            # === BATCHED FEATURE EXTRACTION ===
            frames_flat = frames.view(B * T_max, *frames.shape[2:])

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                all_feats = model.encoder.extract_features(frames_flat)  # (B*T_max, 256)
            all_feats = all_feats.float()
            all_feats = all_feats.view(B, T_max, -1)  # (B, T_max, 256)

            # === PARALLEL GRU OVER B SEQUENCES ===
            hidden = torch.zeros(1, B, model.hidden_size, device=device)
            batch_loss = 0
            batch_coord_loss = 0
            batch_conf_loss = 0
            all_preds = []
            num_valid_frames = 0

            for t in range(T_max):
                feat = all_feats[:, t:t+1, :]  # (B, 1, 256)
                target_t = targets[:, t, :]    # (B, 6)
                mask_t = mask[:, t]            # (B,)

                gru_out, hidden = model.gru(feat, hidden)
                output = model.head(gru_out[:, -1])
                pred = torch.sigmoid(output)  # (B, 6)
                all_preds.append(pred)

                # Only compute loss for valid frames
                if mask_t.any():
                    valid_pred = pred[mask_t]
                    valid_target = target_t[mask_t]
                    frame_loss, coord_loss, conf_loss = criterion(valid_pred, valid_target, return_components=True)
                    batch_loss += frame_loss.item()
                    batch_coord_loss += coord_loss.item()
                    batch_conf_loss += conf_loss.item()
                    num_valid_frames += mask_t.sum().item()

            # Stack predictions and flatten for metrics
            all_preds_stacked = torch.stack(all_preds, dim=0)  # (T_max, B, 6)
            targets_t = targets.transpose(0, 1)  # (T_max, B, 6)
            mask_t = mask.transpose(0, 1)  # (T_max, B)

            preds_flat = all_preds_stacked.view(-1, 6)
            targets_flat = targets_t.reshape(-1, 6)
            mask_flat = mask_t.reshape(-1)

            # Only compute metrics on valid frames
            valid_preds = preds_flat[mask_flat]
            valid_targets = targets_flat[mask_flat]

            # Pixel errors
            err_l, err_r, valid_l, valid_r = calculate_pixel_error_batch(valid_preds, valid_targets)

            # Classification metrics
            frame_cls = calculate_classification_metrics_batch(valid_preds, valid_targets)
            for k, v in frame_cls.items():
                cls_metrics[k] += v

            total_loss += batch_loss
            total_coord_loss += batch_coord_loss
            total_conf_loss += batch_conf_loss
            total_error_left += err_l
            total_error_right += err_r
            total_valid_left += valid_l
            total_valid_right += valid_r
            total_frames += num_valid_frames

    # Calculate final metrics
    avg_loss = total_loss / total_frames
    avg_coord_loss = total_coord_loss / total_frames
    avg_conf_loss = total_conf_loss / total_frames
    avg_error_left = total_error_left / max(total_valid_left, 1)
    avg_error_right = total_error_right / max(total_valid_right, 1)

    # Classification metrics
    tp = cls_metrics['tp_left'] + cls_metrics['tp_right']
    fp = cls_metrics['fp_left'] + cls_metrics['fp_right']
    tn = cls_metrics['tn_left'] + cls_metrics['tn_right']
    fn = cls_metrics['fn_left'] + cls_metrics['fn_right']

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        'loss': avg_loss,
        'coord_loss': avg_coord_loss,
        'conf_loss': avg_conf_loss,
        'error_left': avg_error_left,
        'error_right': avg_error_right,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

    return metrics


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # CUDA optimization flags
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("=" * 60)
    print("TEMPORAL STEREO TRAINING (EfficientTemporalTiny + GRU)")
    print("=" * 60)
    print(f"Data: {DATA_DIRS}")
    print(f"Device: {device}")
    print(f"Sequence length: {SEQUENCE_LENGTH} stereo frames")
    print(f"Batch size: {BATCH_SIZE} sequences")
    print(f"GRU hidden size: {HIDDEN_SIZE}")
    print(f"Frame skip prob: {FRAME_SKIP_PROB*100:.0f}%")
    print("=" * 60)

    # Output directory
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"run_{timestamp}_temporal_tiny_gru"
    output_dir = Path(OUTPUT_DIR) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRun: {run_name}")
    print(f"Output: {output_dir}\n")

    # Save config
    config = {
        'run_name': run_name,
        'model': 'EfficientTemporalTiny',
        'sequence_length': SEQUENCE_LENGTH,
        'batch_size': BATCH_SIZE,
        'history_length': HISTORY_LENGTH,
        'hidden_size': HIDDEN_SIZE,
        'frame_skip_prob': FRAME_SKIP_PROB,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # TensorBoard
    if ENABLE_TENSORBOARD:
        tb_dir = Path(OUTPUT_DIR) / 'tensorboard_logs' / run_name
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"TensorBoard: tensorboard --logdir={Path(OUTPUT_DIR) / 'tensorboard_logs'}\n")
    else:
        writer = None

    # Create datasets with proper train/val split by video segments
    print("Loading temporal datasets...")
    print(f"Augmentation: {'ON' if ENABLE_AUGMENTATION else 'OFF'}")
    train_dataset = MultiTemporalDataset(
        DATA_DIRS,
        sequence_length=SEQUENCE_LENGTH,
        use_augmentation=ENABLE_AUGMENTATION,
        frame_skip_prob=FRAME_SKIP_PROB if ENABLE_AUGMENTATION else 0.0,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        split='train',      # Use ~80% of video segments
        val_ratio=0.2,
        use_memmap=True     # Use preprocessed memmap (instant loading)
    )

    val_dataset = MultiTemporalDataset(
        DATA_DIRS,
        sequence_length=SEQUENCE_LENGTH,
        use_augmentation=False,
        frame_skip_prob=0.0,  # No skipping for validation
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        split='val',        # Use ~20% of video segments (different from train!)
        val_ratio=0.2,
        use_memmap=True     # Use preprocessed memmap (instant loading)
    )

    print(f"\nTrain sequences: {len(train_dataset)}")
    print(f"Val sequences: {len(val_dataset)}")

    # DataLoaders (num_workers=0 for WSL2 compatibility)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Dataset handles via reshuffle_epoch()
        num_workers=0,
        pin_memory=True,
        collate_fn=temporal_collate_fn,
        drop_last=True  # Drop incomplete batches for consistent tensor shapes
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=temporal_collate_fn
    )

    # AMP setup
    use_amp = device.type == 'cuda'

    # Create model
    print("\nCreating EfficientTemporalTiny...")
    model = EfficientTemporalTiny(history_length=HISTORY_LENGTH, hidden_size=HIDDEN_SIZE)
    model = model.to(device)

    # Compile model for faster training (PyTorch 2.0+)
    if USE_TORCH_COMPILE and hasattr(torch, 'compile') and device.type == 'cuda':
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')

    param_count = model.count_parameters()
    print(f"Parameters: {param_count:,}")

    # Loss and optimizer
    criterion = TemporalDetectionLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Scheduler
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[WARMUP_EPOCHS])

    # Training loop
    print(f"\nStarting training for {EPOCHS} epochs...")
    print("=" * 60)

    best_val_error = float('inf')
    history = []

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, train_dataset, criterion, optimizer, device, epoch, use_amp
        )

        # Validate every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            val_metrics = validate(model, val_loader, criterion, device, use_amp)
        else:
            val_metrics = None

        scheduler.step()

        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']

        avg_train_err = (train_metrics['error_left'] + train_metrics['error_right']) / 2

        print(f"\nEpoch {epoch}/{EPOCHS}:")
        print(f"  Train: loss={train_metrics['loss']:.4f} (coord={train_metrics['coord_loss']:.4f}, conf={train_metrics['conf_loss']:.4f})")
        print(f"         px_err={avg_train_err:.1f} (L={train_metrics['error_left']:.1f}, R={train_metrics['error_right']:.1f})")
        print(f"         acc={train_metrics['accuracy']*100:.1f}%, prec={train_metrics['precision']*100:.1f}%, rec={train_metrics['recall']*100:.1f}%, F1={train_metrics['f1']*100:.1f}%")

        if val_metrics:
            avg_val_err = (val_metrics['error_left'] + val_metrics['error_right']) / 2
            print(f"  Val:   loss={val_metrics['loss']:.4f} (coord={val_metrics['coord_loss']:.4f}, conf={val_metrics['conf_loss']:.4f})")
            print(f"         px_err={avg_val_err:.1f} (L={val_metrics['error_left']:.1f}, R={val_metrics['error_right']:.1f})")
            print(f"         acc={val_metrics['accuracy']*100:.1f}%, prec={val_metrics['precision']*100:.1f}%, rec={val_metrics['recall']*100:.1f}%, F1={val_metrics['f1']*100:.1f}%")

        print(f"  LR={lr:.6f}, Time={epoch_time:.1f}s")

        # TensorBoard
        if writer:
            # Training metrics
            writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            writer.add_scalar('Loss/train_coord', train_metrics['coord_loss'], epoch)
            writer.add_scalar('Loss/train_conf', train_metrics['conf_loss'], epoch)
            writer.add_scalar('PixelError/train', avg_train_err, epoch)
            writer.add_scalar('PixelError/train_left', train_metrics['error_left'], epoch)
            writer.add_scalar('PixelError/train_right', train_metrics['error_right'], epoch)
            writer.add_scalar('Classification/train_accuracy', train_metrics['accuracy'], epoch)
            writer.add_scalar('Classification/train_precision', train_metrics['precision'], epoch)
            writer.add_scalar('Classification/train_recall', train_metrics['recall'], epoch)
            writer.add_scalar('Classification/train_f1', train_metrics['f1'], epoch)
            writer.add_scalar('LR', lr, epoch)

            # Validation metrics
            if val_metrics:
                writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
                writer.add_scalar('Loss/val_coord', val_metrics['coord_loss'], epoch)
                writer.add_scalar('Loss/val_conf', val_metrics['conf_loss'], epoch)
                writer.add_scalar('PixelError/val', avg_val_err, epoch)
                writer.add_scalar('PixelError/val_left', val_metrics['error_left'], epoch)
                writer.add_scalar('PixelError/val_right', val_metrics['error_right'], epoch)
                writer.add_scalar('Classification/val_accuracy', val_metrics['accuracy'], epoch)
                writer.add_scalar('Classification/val_precision', val_metrics['precision'], epoch)
                writer.add_scalar('Classification/val_recall', val_metrics['recall'], epoch)
                writer.add_scalar('Classification/val_f1', val_metrics['f1'], epoch)

        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_coord_loss': train_metrics['coord_loss'],
            'train_conf_loss': train_metrics['conf_loss'],
            'train_error': avg_train_err,
            'train_accuracy': train_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'train_f1': train_metrics['f1'],
            'val_loss': val_metrics['loss'] if val_metrics else None,
            'val_error': avg_val_err if val_metrics else None,
            'val_accuracy': val_metrics['accuracy'] if val_metrics else None,
            'val_f1': val_metrics['f1'] if val_metrics else None,
            'lr': lr
        })

        # Save best model
        if val_metrics and avg_val_err < best_val_error:
            best_val_error = avg_val_err
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            print(f"  * New best model: {avg_val_err:.1f}px")

        # Checkpoint
        if epoch % SAVE_INTERVAL == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_dir / f'checkpoint_{epoch}.pth')

    # Save final
    torch.save(model.state_dict(), output_dir / 'final_model.pth')
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    if writer:
        writer.close()

    # Export ONNX
    print("\nExporting ONNX...")
    try:
        model.eval()
        # For ONNX, export the streaming version (single frame input)
        dummy = torch.randn(1, 6, IMAGE_HEIGHT, IMAGE_WIDTH).to(device)

        # Can't easily export stateful GRU to ONNX
        # Export just the encoder + head for now
        print("  Note: Exporting encoder only (GRU state handled in inference code)")

        torch.onnx.export(
            model.encoder,
            dummy,
            output_dir / f'{run_name}_encoder.onnx',
            opset_version=14,
            input_names=['stereo_frame'],
            output_names=['features'],
            dynamic_axes={'stereo_frame': {0: 'batch'}, 'features': {0: 'batch'}}
        )
        print(f"  Saved: {output_dir / f'{run_name}_encoder.onnx'}")
    except Exception as e:
        print(f"  ONNX export failed: {e}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best validation error: {best_val_error:.1f}px")
    print(f"Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
