#!/usr/bin/env python3
"""
Simple Segmentation Training - Just Run It!

Edit the settings below and run: python -m ball_detection.train_seg_simple
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

# ============================================================
# SETTINGS - Edit these
# ============================================================
DATA_DIR = "./ball_detection/training_data"
EPOCHS = 50
BATCH_SIZE = 8  # Reduced for 1280x720 images
DEVICE = "cuda"  # or "cpu"
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
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

        # Simple augmentation for training
        if self.split == 'train' and np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        # Normalize image
        image = image.astype(np.float32) / 255.0
        image = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(torch.from_numpy(image).permute(2, 0, 1))

        mask = torch.from_numpy(mask).long()

        return image, mask


def dice_score(pred, target):
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)


def train():
    print("=" * 60)
    print("SIMPLE SEGMENTATION TRAINING")
    print("=" * 60)
    print(f"Data: {DATA_DIR}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    print()

    # Load data
    train_dataset = SimpleSegDataset(DATA_DIR, 'train')
    valid_dataset = SimpleSegDataset(DATA_DIR, 'valid')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Create model
    print("Creating U-Net model...")
    model = smp.Unet(encoder_name='efficientnet-b4', encoder_weights='imagenet', classes=2, activation=None)
    model = model.to(DEVICE)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params*4/1024/1024:.1f} MB)")
    print()

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Output directory
    output_dir = Path("ball_detection/models/segmentation")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_dice = 0.0

    print("Starting training...")
    print("=" * 60)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # Train
        model.train()
        train_loss = 0
        train_dice = 0

        pbar = tqdm(train_loader, desc='Training')
        for images, masks in pbar:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

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
                    images = images.to(DEVICE)
                    masks = masks.to(DEVICE)

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

            current_dice = val_dice
        else:
            print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
            current_dice = train_dice

        # Save best
        if current_dice > best_dice:
            best_dice = current_dice
            torch.save(model.state_dict(), output_dir / "best_segmentation.pth")
            print(f"Saved best model! Dice: {best_dice:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), output_dir / f"checkpoint_epoch_{epoch+1}.pth")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best Dice: {best_dice:.4f}")
    print(f"Model saved: {output_dir / 'best_segmentation.pth'}")
    print("\nNext step:")
    print("  python -m ball_detection.auto_label_simple")
    print("=" * 60)


if __name__ == "__main__":
    train()
