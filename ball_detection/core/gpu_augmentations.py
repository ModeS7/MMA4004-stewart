"""
GPU-based data augmentations using Kornia.

These augmentations run on GPU during the forward pass, eliminating CPU bottleneck
from Albumentations. This dramatically improves GPU utilization.
"""

import torch
import torch.nn as nn
import kornia.augmentation as K


class GPUAugmentations(nn.Module):
    """
    GPU-based augmentations for ball detection training.

    Replicates the augmentations from dataset.py but runs on GPU.
    Applied during training only (controlled by model.train() / model.eval()).
    """

    def __init__(self, crop_size=128, spatial=True, appearance=True):
        super().__init__()

        self.crop_size = crop_size
        self.spatial = spatial
        self.appearance = appearance

        augmentations = []

        if spatial:
            # Spatial augmentations (similar to Albumentations ShiftScaleRotate)
            augmentations.extend([
                # Random rotation
                K.RandomRotation(degrees=15.0, p=0.5),

                # Random affine (handles shift and scale)
                K.RandomAffine(
                    degrees=0,  # Rotation handled above
                    translate=(0.1, 0.1),  # 10% shift
                    scale=(0.9, 1.1),  # 10% scale variation
                    p=0.7
                ),

                # Random perspective (slight distortion)
                K.RandomPerspective(distortion_scale=0.1, p=0.3),
            ])

        if appearance:
            # Appearance augmentations
            augmentations.extend([
                # Color jitter (brightness, contrast, saturation, hue)
                K.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.2,
                    hue=0.1,
                    p=0.7
                ),

                # Random gamma correction
                K.RandomGamma(gamma=(0.8, 1.2), p=0.3),

                # Random blur
                K.RandomGaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0), p=0.3),
                K.RandomMotionBlur(kernel_size=3, angle=35.0, direction=0.5, p=0.2),

                # Random noise
                K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.3),

                # Random sharpness
                K.RandomSharpness(sharpness=0.5, p=0.3),

                # Random posterize (reduce color depth)
                K.RandomPosterize(bits=4, p=0.2),

                # Random solarize
                K.RandomSolarize(thresholds=0.5, p=0.2),
            ])

        # Combine all augmentations
        self.aug = K.AugmentationSequential(
            *augmentations,
            data_keys=["input"],  # Only augment images, not targets
            same_on_batch=False,  # Different aug for each image in batch
        )

    def forward(self, images):
        """
        Apply augmentations to images on GPU.

        Args:
            images: (B, C, H, W) tensor on GPU

        Returns:
            Augmented images (B, C, H, W)
        """
        if self.training and (self.spatial or self.appearance):
            # Apply augmentations
            images = self.aug(images)

        return images


class GPUAugmentationsWithTargets(nn.Module):
    """
    GPU augmentations that also transform target coordinates.

    This is more complex because we need to apply the SAME geometric transforms
    to both images and coordinate targets.
    """

    def __init__(self, crop_size=128):
        super().__init__()

        self.crop_size = crop_size

        # Geometric augmentations (affect coordinates)
        # Match Albumentations: Rotate(15, p=0.5) + ShiftScaleRotate(shift=0.0625, scale=0.1, p=0.5)
        self.geometric_aug = K.AugmentationSequential(
            K.RandomRotation(degrees=15.0, p=0.5),
            K.RandomAffine(
                degrees=0,  # Rotation already done above
                translate=(0.0625, 0.0625),  # Match shift_limit=0.0625
                scale=(0.9, 1.1),  # Match scale_limit=0.1
                p=0.5  # Match p=0.5
            ),
            data_keys=["input", "keypoints"],
            same_on_batch=False,
        )

        # Appearance augmentations (don't affect coordinates)
        # Simplified version - Kornia doesn't have all Albumentations features
        # Missing: RandomShadow, RandomSunFlare, CLAHE, ISONoise, Downscale, MedianBlur, ImageCompression
        self.appearance_aug = K.AugmentationSequential(
            # Match RandomBrightnessContrast (brightness=0.3, contrast=0.3, p=0.7)
            K.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.0, hue=0.0, p=0.7),
            # Color variations (simplified - match OneOf[ColorJitter, HueSaturationValue, RGBShift], p=0.8)
            # Kornia doesn't have exact equivalents, use ColorJitter with all params
            K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
            # Blur - match OneOf[GaussianBlur, MotionBlur, MedianBlur], p=0.3
            K.RandomGaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0), p=0.15),  # p=0.3/2 since we have 2 blur types
            K.RandomMotionBlur(kernel_size=5, angle=35.0, direction=0.5, p=0.15),
            # Noise - match GaussNoise(var_limit=(10.0, 20.0), p=0.3)
            # Convert var_limit to std: std = sqrt(var/255^2) ≈ sqrt(15/255^2) ≈ 0.015
            K.RandomGaussianNoise(mean=0.0, std=0.015, p=0.3),
            # Additional augmentations (no exact Albumentations equivalent but useful)
            K.RandomPosterize(bits=4, p=0.1),  # Reduced from 0.2
            data_keys=["input"],
            same_on_batch=False,
        )

        # Normalization (match Albumentations normalize)
        self.normalize = K.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        )

    def forward(self, images, targets):
        """
        Apply augmentations to both images and targets.

        Args:
            images: (B, C, H, W) tensor in [0, 1] range (from ToTensor)
            targets: (B, 2) normalized coordinates [x, y] in [0, 1]

        Returns:
            augmented_images: (B, C, H, W) normalized with ImageNet stats
            augmented_targets: (B, 2)
        """
        # Normalize images first (they come in as [0, 1] from dataset ToTensor)
        images = self.normalize(images)

        if not self.training:
            return images, targets

        batch_size = images.shape[0]

        # Convert normalized targets to pixel coordinates for Kornia
        # Kornia expects keypoints in pixel space (x, y)
        targets_pixels = targets.clone()
        targets_pixels[:, 0] *= self.crop_size  # x
        targets_pixels[:, 1] *= self.crop_size  # y

        # Reshape to (B, 1, 2) for Kornia keypoints format
        keypoints = targets_pixels.unsqueeze(1)  # (B, 1, 2)

        # Apply geometric transformations
        images_aug, keypoints_aug = self.geometric_aug(images, keypoints)

        # Extract and normalize keypoints back to [0, 1]
        targets_aug = keypoints_aug.squeeze(1)  # (B, 2)
        targets_aug[:, 0] /= self.crop_size
        targets_aug[:, 1] /= self.crop_size

        # Clamp to valid range
        targets_aug = torch.clamp(targets_aug, 0.0, 1.0)

        # Apply appearance transformations (doesn't affect targets)
        images_aug = self.appearance_aug(images_aug)

        return images_aug, targets_aug
