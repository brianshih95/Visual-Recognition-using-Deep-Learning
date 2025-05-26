import glob
import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DegradedImageDataset(Dataset):
    """Dataset for degraded images with rain and snow corruptions."""
    
    def __init__(self, data_root, mode='train', transform=None, augment=False,
                 patch_size=256, strong_augment=True):
        """
        Initialize dataset.
        
        Args:
            data_root: Root directory containing train/test folders
            mode: 'train', 'val', or 'test'
            transform: Torchvision transforms to apply
            augment: Whether to apply data augmentation
            patch_size: Size for random cropping during training
            strong_augment: Whether to apply strong augmentation
        """
        self.transform = transform
        self.mode = mode
        self.augment = augment
        self.patch_size = patch_size
        self.strong_augment = strong_augment
        
        if mode in ['train', 'val']:
            # Get all degraded images
            self.degraded_paths = []
            rain_pattern = os.path.join(data_root, 'train/degraded/rain-*.png')
            snow_pattern = os.path.join(data_root, 'train/degraded/snow-*.png')
            self.degraded_paths.extend(glob.glob(rain_pattern))
            self.degraded_paths.extend(glob.glob(snow_pattern))
            self.degraded_paths.sort()
            
            # Create corresponding clean image paths
            self.clean_paths = []
            for path in self.degraded_paths:
                # Replace 'degraded' folder with 'clean' folder and remove
                # degradation type prefix
                filename = os.path.basename(path)
                degradation_type = 'rain' if 'rain' in filename else 'snow'
                clean_filename = filename.replace(
                    f'{degradation_type}', f'{degradation_type}_clean')
                clean_path = os.path.join(
                    data_root, 'train/clean', clean_filename)
                self.clean_paths.append(clean_path)
            
            # Split into train and validation sets (90/10 split)
            split_idx = int(len(self.degraded_paths) * 0.9)
            if mode == 'train':
                self.degraded_paths = self.degraded_paths[:split_idx]
                self.clean_paths = self.clean_paths[:split_idx]
            else:  # validation
                self.degraded_paths = self.degraded_paths[split_idx:]
                self.clean_paths = self.clean_paths[split_idx:]
        
        else:  # test mode
            test_pattern = os.path.join(data_root, 'test/degraded/*.png')
            self.degraded_paths = glob.glob(test_pattern)
            self.degraded_paths.sort()
            # No clean images for test set
            self.clean_paths = [None] * len(self.degraded_paths)
    
    def __len__(self):
        """Return dataset size."""
        return len(self.degraded_paths)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        img_idx = idx

        degraded_path = self.degraded_paths[img_idx]
        degraded_img = Image.open(degraded_path).convert('RGB')
        
        if self.mode == 'test':
            if self.transform:
                degraded_img = self.transform(degraded_img)
            filename = os.path.basename(degraded_path)
            return degraded_img, filename
        
        clean_path = self.clean_paths[img_idx]
        clean_img = Image.open(clean_path).convert('RGB')
        
        degradation_type = (
            1.0 if 'rain' in os.path.basename(degraded_path) else 0.0)
        
        if self.mode == 'train' and self.augment:
            degraded_img, clean_img = self._apply_augmentations(
                degraded_img, clean_img)
        
        if self.transform:
            degraded_img = self.transform(degraded_img)
            clean_img = self.transform(clean_img)
        
        return degraded_img, clean_img, degradation_type
    
    def _apply_augmentations(self, degraded_img, clean_img):
        """Apply data augmentations to image pairs."""
        # Random horizontal and vertical flips
        if random.random() > 0.5:
            degraded_img = transforms.functional.hflip(degraded_img)
            clean_img = transforms.functional.hflip(clean_img)
        
        if random.random() > 0.5:
            degraded_img = transforms.functional.vflip(degraded_img)
            clean_img = transforms.functional.vflip(clean_img)
            
        # Random 90-degree rotations
        rot_times = random.randint(0, 3)
        if rot_times > 0:
            angle = 90 * rot_times
            degraded_img = transforms.functional.rotate(degraded_img, angle)
            clean_img = transforms.functional.rotate(clean_img, angle)
            
        # Color jitter on degraded images to improve robustness
        if random.random() > 0.7:
            color_jitter = transforms.ColorJitter(
                brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)
            degraded_img = color_jitter(degraded_img)
        
        # Enhanced augmentation options
        if self.strong_augment:
            # Random adjustments to gamma
            if random.random() > 0.7:
                gamma = random.uniform(0.85, 1.15)
                degraded_img = transforms.functional.adjust_gamma(
                    degraded_img, gamma)
            
            # Add slight Gaussian noise to degraded images
            if random.random() > 0.8:
                degraded_np = np.array(degraded_img).astype(np.float32) / 255.0
                noise_std = random.uniform(0.01, 0.03)
                noise = np.random.normal(0, noise_std, degraded_np.shape)
                degraded_np = np.clip(degraded_np + noise, 0, 1) * 255.0
                degraded_img = Image.fromarray(degraded_np.astype(np.uint8))
        
        # Random cropping for patch-based training
        img_width, img_height = degraded_img.size
        if (img_width > self.patch_size and img_height > self.patch_size):
            i, j, h, w = transforms.RandomCrop.get_params(
                degraded_img, output_size=(self.patch_size, self.patch_size))
            degraded_img = transforms.functional.crop(degraded_img, i, j, h, w)
            clean_img = transforms.functional.crop(clean_img, i, j, h, w)
        
        return degraded_img, clean_img
