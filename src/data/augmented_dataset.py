import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from collections import Counter


class AugmentedWBCDataset(Dataset):
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
            freq = label_counts[cls] / total_samples
            if freq < aggressive_thresh:
                self.class_to_aug_level[cls] = 'aggressive'
            elif freq < medium_thresh:
                self.class_to_aug_level[cls] = 'medium'
            else:
                self.class_to_aug_level[cls] = 'light'

        if is_training:
            self.transform_light = self._build_light_aug(config['light'])
            self.transform_medium = self._build_medium_aug(config['medium'])
            self.transform_aggressive = self._build_aggressive_aug(config['aggressive'])
        else:
            self.transform_val = self._build_val_transform()

        self.image_dir = os.path.join(root_dir, split)

    def _build_light_aug(self, cfg):
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=tuple(cfg['random_crop_scale'])),
            transforms.RandomHorizontalFlip(p=cfg['h_flip_prob']),
            transforms.ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation']
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _build_medium_aug(self, cfg):
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=tuple(cfg['random_crop_scale'])),
            transforms.RandomHorizontalFlip(p=cfg['h_flip_prob']),
            transforms.RandomVerticalFlip(p=cfg['v_flip_prob']),
            transforms.RandomRotation(degrees=cfg['rotation_degrees']),
            transforms.ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation'],
                hue=cfg['hue']
            ),
            transforms.RandomAffine(degrees=0, translate=tuple(cfg['translate'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _build_aggressive_aug(self, cfg):
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=tuple(cfg['random_crop_scale'])),
            transforms.RandomHorizontalFlip(p=cfg['h_flip_prob']),
            transforms.RandomVerticalFlip(p=cfg['v_flip_prob']),
            transforms.RandomRotation(degrees=cfg['rotation_degrees']),
            transforms.ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation'],
                hue=cfg['hue']
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=tuple(cfg['translate']),
                scale=tuple(cfg['scale'])
            ),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=cfg['gaussian_blur_prob']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=cfg['random_erasing_prob'], scale=(0.02, 0.1))
        ])

    def _build_val_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
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
