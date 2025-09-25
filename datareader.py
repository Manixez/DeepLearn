import numpy as np
import os
from torch.utils.data import Dataset
import torch
import random
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.transforms import (
    ToTensor, Normalize, Compose, RandomResizedCrop, 
    RandomHorizontalFlip, ColorJitter, RandomRotation,
    RandomAffine, Resize, CenterCrop
)

# Set random seed for reproducibility
RANDOM_SEED = 2025
random.seed(RANDOM_SEED) # random seed-nya Python
np.random.seed(RANDOM_SEED) # random seed-nya Numpy
torch.manual_seed(RANDOM_SEED) # random seed-nya PyTorch



class MakananIndo(Dataset):
    # ImageNet normalization values
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(self,
                 data_dir='train',
                 img_size=(224, 224),  # Update default size for ViT
                 transform=None,
                 split='train'
                 ):
        
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split
        
        # Default transforms with data augmentation for training
        if transform is None:
            if split == 'train':
                self.transform = Compose([
                    RandomResizedCrop(img_size, scale=(0.8, 1.0)),  # Crop with zoom
                    RandomHorizontalFlip(p=0.5),  # Horizontal flip
                    ColorJitter(
                        brightness=0.2,  # Brightness variation
                        contrast=0.2,    # Contrast variation
                        saturation=0.2,  # Saturation variation
                        hue=0.1         # Hue variation
                    ),
                    RandomRotation(15),  # Rotate ±15 degrees
                    RandomAffine(
                        degrees=0,
                        translate=(0.1, 0.1),  # Translation
                        scale=(0.9, 1.1),      # Scale variation
                    ),
                    ToTensor(),
                    Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
                ])
            else:  # validation/test transforms
                self.transform = Compose([
                    Resize(int(img_size[0] * 1.14)),  # Resize with aspect ratio
                    CenterCrop(img_size),             # Center crop to target size
                    ToTensor(),
                    Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
                ])
        else:
            self.transform = transform

        # List seluruh file gambar dalam direktori data (.jpg atau .png)
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')]
        # sort self.image_files to ensure consistent order
        self.image_files.sort()
        
        csv_path = os.path.join(os.path.dirname(data_dir), 'train.csv')
        df = pd.read_csv(csv_path)
        # Buat dictionary mapping filename ke label
        self.label_dict = dict(zip(df['filename'], df['label']))
        # Buat list label sesuai urutan image_files
        self.labels = [self.label_dict.get(f, None) for f in self.image_files]
        
        # Pasangkan setiap image file dengan labelnya dan simpan dalam list of tuples
        all_data = list(zip(self.image_files, self.labels))
        
        # Split data into train/val with 0.8/0.2 ratio
        total_len = len(all_data)
        train_len = int(0.8 * total_len)
        
        # Use random indices for splitting but maintain reproducibility
        indices = list(range(total_len))
        random.shuffle(indices)  # This uses the global random seed set above
        
        train_indices = indices[:train_len]
        val_indices = indices[train_len:]
        
        if split == 'train':
            self.data = [all_data[i] for i in train_indices]
        elif split == 'val':
            self.data = [all_data[i] for i in val_indices]
        else:
            raise ValueError("Split must be 'train' or 'val'")
        
        # Define default transforms
        self.default_transform = Compose([
            ToTensor(),  # Converts PIL/numpy to tensor and scales to [0,1]
            Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        # Load Data dan Label
        img_path = os.path.join(self.data_dir, self.data[idx][0])
        # Load image using cv2
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Convert numpy array to PIL Image
        image = Image.fromarray(image)
        
        # Apply transforms (including resize and normalization)
        if self.transform:
            image = self.transform(image)
        else:
            image = self.default_transform(image)
        
        label = self.data[idx][1]
        
        # return data gambar, label, dan file_path
        return_data = (image, label, img_path)
        
        return return_data

if __name__ == "__main__":
    # Test both splits
    train_dataset = MakananIndo(split='train')
    val_dataset = MakananIndo(split='val')
    
    print(f"Train data: {len(train_dataset)}")
    print(f"Val data: {len(val_dataset)}")
    print(f"Total: {len(train_dataset) + len(val_dataset)}")
    
    # Sample 5 random images from each dataset
    train_indices = random.sample(range(len(train_dataset)), min(5, len(train_dataset)))
    val_indices = random.sample(range(len(val_dataset)), min(5, len(val_dataset)))
    
    print("\nTrain Dataset Samples:")
    for i, idx in enumerate(train_indices):
        image, label, filepath = train_dataset[idx]
        print(f"Train data ke-{i} (index {idx})")
        print(f"Image shape: {image.shape}")
        print(f"Label: {label}")
        print(f"File path: {filepath}")
        print("-" * 40)
    
    print("\nValidation Dataset Samples:")
    for i, idx in enumerate(val_indices):
        image, label, filepath = val_dataset[idx]
        print(f"Val data ke-{i} (index {idx})")
        print(f"Image shape: {image.shape}")
        print(f"Label: {label}")
        print(f"File path: {filepath}")
        print("-" * 40)

    # Create a figure with subplots for both train and val
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    # Plot train images
    for i, idx in enumerate(train_indices):
        image, label, filepath = train_dataset[idx]
        
        # Convert tensor to displayable format
        img_display = image.clone()
        for j in range(3):
            img_display[j] = img_display[j] * train_dataset.IMAGENET_STD[j] + train_dataset.IMAGENET_MEAN[j]
        
        img_display = img_display.permute(1, 2, 0)
        img_display = torch.clamp(img_display, 0, 1)
        
        axes[0, i].imshow(img_display)
        axes[0, i].set_title(f"Train: {label}")
        axes[0, i].axis('off')
    
    # Plot val images
    for i, idx in enumerate(val_indices):
        image, label, filepath = val_dataset[idx]
        
        # Convert tensor to displayable format
        img_display = image.clone()
        for j in range(3):
            img_display[j] = img_display[j] * val_dataset.IMAGENET_STD[j] + val_dataset.IMAGENET_MEAN[j]
        
        img_display = img_display.permute(1, 2, 0)
        img_display = torch.clamp(img_display, 0, 1)
        
        axes[1, i].imshow(img_display)
        axes[1, i].set_title(f"Val: {label}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()