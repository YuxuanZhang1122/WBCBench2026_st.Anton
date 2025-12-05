import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Optional, Tuple


class WBCDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        transform: Optional[transforms.Compose] = None,
        return_path: bool = False
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.return_path = return_path

        csv_path = os.path.join(root_dir, f"{split}.csv")
        self.df = pd.read_csv(csv_path)

        # For test split, labels might be missing
        self.is_test = self.df['labels'].isna().all() if 'labels' in self.df.columns else True

        if not self.is_test:
            self.df = self.df.dropna(subset=['labels'])
            self.df['labels'] = self.df['labels'].astype(str)
            self.class_names = sorted(self.df['labels'].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
            self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        else:
            # Test set has no labels
            self.class_names = []
            self.class_to_idx = {}
            self.idx_to_class = {}

        self.image_dir = os.path.join(root_dir, split)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple:
        row = self.df.iloc[idx]
        img_name = row['ID']

        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            # Return dummy label for test set
            label_idx = -1
        else:
            label = row['labels']
            label_idx = self.class_to_idx[label]

        if self.return_path:
            return image, label_idx, img_name
        return image, label_idx

    def get_class_weights(self, method: str = "inverse") -> torch.Tensor:
        from collections import Counter
        import numpy as np

        label_counts = Counter(self.df['labels'])
        counts = np.array([label_counts[cls] for cls in self.class_names])

        if method == "inverse":
            weights = 1.0 / counts
        elif method == "sqrt_inverse":
            weights = 1.0 / np.sqrt(counts)
        else:
            raise ValueError(f"Unknown weighting method: {method}")

        weights = weights / weights.sum() * len(weights)

        return torch.FloatTensor(weights)


def get_transforms(image_size: int, split: str, use_center_crop: bool = False) -> transforms.Compose:
    if use_center_crop:
        return transforms.Compose([
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_dataloader(
    root_dir: str,
    split: str,
    batch_size: int,
    image_size: int,
    num_workers: int = 4,
    shuffle: bool = None,
    return_path: bool = False,
    use_center_crop: bool = False
):
    if shuffle is None:
        shuffle = (split == "train")

    transform = get_transforms(image_size, split, use_center_crop)
    dataset = WBCDataset(root_dir, split, transform, return_path)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False
    )

    return dataloader, dataset
