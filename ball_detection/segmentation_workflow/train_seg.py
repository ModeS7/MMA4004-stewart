#!/usr/bin/env python3
"""
Segmentation Training - Optimized for Small Datasets

For ~500 images: heavy augmentation + pretrained encoder + regularization
Run: python ball_detection/segmentation_workflow/train_seg.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import segmentation_models_pytorch as smp
from pycocotools import mask as mask_utils
import random
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# ============================================================
# SETTINGS - Edit these
# ============================================================
DATA_DIR = "./ball_detection/data/sem_seg"
EPOCHS = 1000
BATCH_SIZE = 16
DEVICE = "cuda"
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
ENCODER = "efficientnet-b0"  # Smaller encoder for small dataset
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

# Fine-tuning settings
FINETUNE = True  # Set to False to train from scratch
PRETRAINED_WEIGHTS = "./ball_detection/models/segmentation/best_segmentation.pth"
FINETUNE_LR = 3e-4  # Lower LR for fine-tuning
# ============================================================


class SimpleSegDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir)
        self.split = split
        self.images_dir = self.data_dir / split

        # Load annotations
        ann_file = self.images_dir / "_annotations.coco.json"
        with open(ann_file, 'r') as f:
            coco = json.load(f)

        # Map images
        self.images = {img['id']: img for img in coco['images']}

        # Group annotations by image
        self.annotations = {}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        # Get valid image IDs
        self.image_ids = []
        for img_id, img_info in self.images.items():
            img_path = self.images_dir / img_info['file_name']
            if img_path.exists():
                self.image_ids.append(img_id)

        print(f"Loaded {len(self.image_ids)} images for {split}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = self.images_dir / img_info['file_name']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)

        if img_id in self.annotations:
            for ann in self.annotations[img_id]:
                if 'segmentation' in ann and ann['segmentation']:
                    seg = ann['segmentation']

                    # Check if RLE format (dict with 'counts')
                    if isinstance(seg, dict) and 'counts' in seg:
                        # Decode RLE
                        rle_mask = mask_utils.decode(seg)
                        mask = np.maximum(mask, rle_mask)
                    # Check if polygon format (list of lists)
                    elif isinstance(seg, list):
                        for poly in seg:
                            if len(poly) < 6:  # Need at least 3 points
                                continue
                            try:
                                pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                                cv2.fillPoly(mask, [pts], 1)
                            except:
                                continue

        # Resize to fixed size
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        mask = cv2.resize(mask, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)

        # Heavy augmentation for small dataset
        if self.split == 'train':
            image, mask = self._augment(image, mask)

        # Normalize image
        image = image.astype(np.float32) / 255.0
        image = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(torch.from_numpy(image).permute(2, 0, 1))

        mask = torch.from_numpy(mask).long()

        return image, mask

    def _augment(self, image, mask):
        """Moderate augmentation - allows some overfitting for representative data."""
        h, w = image.shape[:2]

        # Horizontal flip (50%)
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        # Random rotation (-10 to +10 degrees) - mild
        if random.random() > 0.6:
            angle = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

        # Random scale (0.95 to 1.05) - very mild
        if random.random() > 0.7:
            scale = random.uniform(0.95, 1.05)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            # Crop or pad back to original size
            if scale > 1:
                start_x = (new_w - w) // 2
                start_y = (new_h - h) // 2
                image = image[start_y:start_y+h, start_x:start_x+w]
                mask = mask[start_y:start_y+h, start_x:start_x+w]
            else:
                pad_x = (w - new_w) // 2
                pad_y = (h - new_h) // 2
                image = cv2.copyMakeBorder(image, pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x, cv2.BORDER_REFLECT)
                mask = cv2.copyMakeBorder(mask, pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x, cv2.BORDER_REFLECT)

        # Brightness/contrast adjustment - mild
        if random.random() > 0.6:
            alpha = random.uniform(0.9, 1.1)  # Contrast (mild)
            beta = random.randint(-15, 15)    # Brightness (mild)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # Gaussian blur (20%) - occasional
        if random.random() > 0.8:
            ksize = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)

        return image, mask


def dice_score(pred, target):
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)


def train():
    # Performance optimizations
    torch.backends.cudnn.allow_tf32 = True
    #torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch._dynamo.config.cache_size_limit = 32
    torch.set_float32_matmul_precision('high')

    print("=" * 60)
    print("SEGMENTATION TRAINING - Small Dataset Optimized")
    print("=" * 60)
    print(f"Data: {DATA_DIR}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Encoder: {ENCODER}")
    if FINETUNE and PRETRAINED_WEIGHTS and Path(PRETRAINED_WEIGHTS).exists():
        print(f"Mode: Fine-tuning from {PRETRAINED_WEIGHTS}")
        print(f"Learning rate: {FINETUNE_LR} (fine-tune)")
    else:
        print(f"Mode: Training from scratch")
        print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    print()

    # Load data
    train_dataset = SimpleSegDataset(DATA_DIR, 'train')
    valid_dataset = SimpleSegDataset(DATA_DIR, 'valid')

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

    # Create model - smaller encoder for small dataset
    print(f"Creating U-Net with {ENCODER} encoder...")
    model = smp.Unet(encoder_name=ENCODER, encoder_weights='imagenet', classes=2, activation=None)

    # Load pretrained weights if specified (for fine-tuning)
    if FINETUNE and PRETRAINED_WEIGHTS and Path(PRETRAINED_WEIGHTS).exists():
        print(f"Loading pretrained weights from: {PRETRAINED_WEIGHTS}")
        state_dict = torch.load(PRETRAINED_WEIGHTS, map_location=DEVICE)
        # Handle torch.compile saved models (remove _orig_mod prefix if present)
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print("Pretrained weights loaded - fine-tuning mode")
        is_finetuning = True
    else:
        print("Training from scratch with ImageNet encoder weights")
        is_finetuning = False

    model = model.to(DEVICE, memory_format=torch.channels_last)  # Faster for convs

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params*4/1024/1024:.1f} MB)")

    # # Compile model for faster training (disabled - can cause CUDA errors)
    # print("Compiling model with torch.compile...")
    model = torch.compile(model, mode="reduce-overhead")
    print()

    # Training setup with weight decay for regularization
    criterion = nn.CrossEntropyLoss()

    # Use lower LR for fine-tuning
    lr = FINETUNE_LR if (is_finetuning and FINETUNE_LR) else LEARNING_RATE
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # Warmup + Cosine annealing with warm restarts (better for long training)
    warmup_epochs = 2 if is_finetuning else 5
    min_lr = 1e-6
    T_0 = 100  # Restart every 100 epochs (10 restarts for 1000 epochs)

    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5 if is_finetuning else 0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=min_lr)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    print(f"LR schedule: Warmup {warmup_epochs} epochs, then CosineWarmRestarts (T_0={T_0})")

    # Output directory
    output_dir = Path("ball_detection/models/segmentation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard - centralized in models/tensorboard_logs/
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"seg_{timestamp}"
    if is_finetuning:
        run_name += "_finetune"
    tensorboard_base = Path("ball_detection/models/tensorboard_logs")
    tensorboard_dir = tensorboard_base / run_name
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir), max_queue=1000, flush_secs=120)
    print(f"TensorBoard: tensorboard --logdir {tensorboard_base}")

    best_dice = 0.0
    interval_best_dice = 0.0  # Best within current 100-epoch interval
    interval_size = 100

    print("Starting training...")
    print("=" * 60)

    for epoch in range(EPOCHS):
        # Reset interval best at start of each interval
        if epoch % interval_size == 0:
            interval_best_dice = 0.0
            current_interval = epoch // interval_size
            print(f"\n--- Starting interval {current_interval} (epochs {epoch+1}-{min(epoch+interval_size, EPOCHS)}) ---")
        lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{EPOCHS} (lr: {lr:.2e})")

        # Train
        model.train()
        train_loss = 0
        train_dice = 0

        pbar = tqdm(train_loader, desc='Training')
        for images, masks in pbar:
            images = images.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # Faster than zero

            # Mixed precision training with bfloat16 (no scaler needed)
            with autocast('cuda', dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred_masks = torch.argmax(outputs, dim=1)
                dice = dice_score(pred_masks, masks)

            train_loss += loss.item()
            train_dice += dice.item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice.item():.4f}'})

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        # Validate
        if len(valid_loader) > 0:
            model.eval()
            val_loss = 0
            val_dice = 0

            with torch.no_grad():
                for images, masks in tqdm(valid_loader, desc='Validating'):
                    images = images.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
                    masks = masks.to(DEVICE, non_blocking=True)

                    with autocast('cuda', dtype=torch.bfloat16):
                        outputs = model(images)
                        loss = criterion(outputs, masks)

                    pred_masks = torch.argmax(outputs, dim=1)
                    dice = dice_score(pred_masks, masks)

                    val_loss += loss.item()
                    val_dice += dice.item()

            val_loss /= len(valid_loader)
            val_dice /= len(valid_loader)

            print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

            # TensorBoard logging
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Dice/train', train_dice, epoch)
            writer.add_scalar('Dice/val', val_dice, epoch)
            writer.add_scalar('LR', lr, epoch)

            current_dice = val_dice
        else:
            print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Dice/train', train_dice, epoch)
            writer.add_scalar('LR', lr, epoch)
            current_dice = train_dice

        # Flush TensorBoard periodically (not every epoch to reduce overhead)
        if (epoch + 1) % 50 == 0:
            writer.flush()

        # Save best model (overall)
        if current_dice > best_dice:
            best_dice = current_dice
            torch.save(model.state_dict(), output_dir / "best_segmentation.pth")
            print(f"Saved best model! Dice: {best_dice:.4f}")

        # Save best model for current interval
        if current_dice > interval_best_dice:
            interval_best_dice = current_dice
            interval_start = (epoch // interval_size) * interval_size
            interval_end = interval_start + interval_size
            torch.save(model.state_dict(), output_dir / f"best_interval_{interval_start}-{interval_end}.pth")
            print(f"Saved interval best ({interval_start}-{interval_end})! Dice: {interval_best_dice:.4f}")

        # Step scheduler
        scheduler.step()

        # Save checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), output_dir / f"checkpoint_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved at epoch {epoch+1}")

    writer.close()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best Dice: {best_dice:.4f}")
    print(f"Model saved: {output_dir / 'best_segmentation.pth'}")
    print("\nNext step:")
    print("  python ball_detection/segmentation_workflow/auto_label_simple.py")
    print("=" * 60)


if __name__ == "__main__":
    train()
