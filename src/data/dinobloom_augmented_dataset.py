import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from collections import Counter


class DinoBloomAugmentedDataset(Dataset):
    def __init__(self, root_dir, split, config, is_training=True):
        self.root_dir = root_dir
        self.split = split
        self.is_training = is_training

        csv_path = os.path.join(root_dir, f"{split}.csv")
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=['labels'])

        self.class_names = sorted(self.df['labels'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}

        label_counts = Counter(self.df['labels'])
        total_samples = len(self.df)

        self.class_to_aug_level = {}
        aggressive_thresh = config['aggressive_threshold']
        medium_thresh = config['medium_threshold']

        for cls in self.class_names:
            #freq = label_counts[cls] / total_samples
            self.class_to_aug_level[cls] = 'aggressive'
            #if freq < aggressive_thresh:
            #    self.class_to_aug_level[cls] = 'aggressive'
            #elif freq < medium_thresh:
            #    self.class_to_aug_level[cls] = 'medium'
            #else:
            #    self.class_to_aug_level[cls] = 'light'

        image_size = config.get('image_size', 368)
        target_size = config.get('target_size', 224)

        if is_training:
            self.transform_light = self._build_light_aug(image_size, target_size)
            self.transform_medium = self._build_medium_aug(image_size, target_size)
            self.transform_aggressive = self._build_aggressive_aug(image_size, target_size)
        else:
            self.transform_val = self._build_val_transform(image_size, target_size)

        self.image_dir = os.path.join(root_dir, split)

        if is_training:
            print(f"\nLoaded {len(self.df)} training samples")
            print("\nClass-to-Augmentation Mapping:")
            for cls in self.class_names:
                aug_level = self.class_to_aug_level[cls]
                count = label_counts[cls]
                print(f"  {cls:<5} ({count:>5} samples): {aug_level.upper()}")

    def _build_light_aug(self, image_size, target_size):
        return transforms.Compose([
            transforms.CenterCrop(target_size),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _build_medium_aug(self, image_size, target_size):
        return transforms.Compose([
            transforms.CenterCrop(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _build_aggressive_aug(self, image_size, target_size):
        return transforms.Compose([
            transforms.CenterCrop(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            #transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.05, scale=(0.02, 0.1))
        ])

    def _build_val_transform(self, image_size, target_size):
        return transforms.Compose([
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['ID']
        label_name = row['labels']

        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.is_training:
            aug_level = self.class_to_aug_level[label_name]
            if aug_level == 'aggressive':
                image = self.transform_aggressive(image)
            elif aug_level == 'medium':
                image = self.transform_medium(image)
            else:
                image = self.transform_light(image)
        else:
            image = self.transform_val(image)

        label_idx = self.class_to_idx[label_name]
        return image, label_idx
