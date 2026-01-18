!pip install "protobuf<4"

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available, logging disabled")
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from collections import Counter
from tqdm import tqdm
import json
import argparse
import timm  # Required for ConvNeXt V2

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from skimage import color
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CurriculumScheduler:
    """Manages curriculum learning schedules for augmentation parameters."""

    def __init__(self, schedule_type='cosine', total_epochs=15):
        self.schedule_type = schedule_type
        self.total_epochs = total_epochs

    def get_progress(self, epoch):
        """Returns progress ratio [0, 1] based on schedule type"""
        if epoch >= self.total_epochs:
            return 1.0

        t = epoch / self.total_epochs

        if self.schedule_type == 'cosine':
            return 0.5 * (1 - np.cos(np.pi * t))
        elif self.schedule_type == 'linear':
            return t
        elif self.schedule_type == 'exponential':
            return t ** 2
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def interpolate(self, start_val, end_val, epoch):
        """Linearly interpolate between start and end values"""
        progress = self.get_progress(epoch)
        return start_val + (end_val - start_val) * progress

    def interpolate_range(self, start_range, end_range, epoch):
        """Interpolate tuple ranges"""
        progress = self.get_progress(epoch)
        return tuple(s + (e - s) * progress for s, e in zip(start_range, end_range))

    def get_curriculum_params(self, epoch):
        """Get all curriculum parameters for given epoch"""
        gauss_std_start = (0.0, 0.05)
        gauss_std_end = (0.2, 0.25)
        gauss_prob_start = 0.1
        gauss_prob_end = 0.4

        iso_color_shift = (0.1, 0.4)
        iso_intensity_start = (0.0, 0.05)
        iso_intensity_end = (0.4, 0.7)
        iso_prob_start = 0.1
        iso_prob_end = 0.4

        blur_sigma_start = (0.1, 1.0)
        blur_sigma_end = (0.1, 1.5)
        blur_prob_start = 0.0
        blur_prob_end = 0.4

        return {
            'gauss_noise_std_range': self.interpolate_range(gauss_std_start, gauss_std_end, epoch),
            'gauss_noise_prob': self.interpolate(gauss_prob_start, gauss_prob_end, epoch),
            'iso_noise_color_shift': iso_color_shift,
            'iso_noise_intensity': self.interpolate_range(iso_intensity_start, iso_intensity_end, epoch),
            'iso_noise_prob': self.interpolate(iso_prob_start, iso_prob_end, epoch),
            'gaussian_blur_sigma': self.interpolate_range(blur_sigma_start, blur_sigma_end, epoch),
            'gaussian_blur_prob': self.interpolate(blur_prob_start, blur_prob_end, epoch),
        }


class RandAdjustHE:
    """H&E stain augmentation using HED color space with gamma correction"""

    def __init__(self, prob=0.5, gamma_h=(0.8, 1.2), gamma_e=(0.8, 1.2), gamma_d=(0.8, 1.2)):
        self.prob = prob
        self.gamma_h_range = (np.log(min(gamma_h)), np.log(max(gamma_h)))
        self.gamma_e_range = (np.log(min(gamma_e)), np.log(max(gamma_e)))
        self.gamma_d_range = (np.log(min(gamma_d)), np.log(max(gamma_d)))

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img

        gamma_h = np.exp(np.random.uniform(*self.gamma_h_range))
        gamma_e = np.exp(np.random.uniform(*self.gamma_e_range))
        gamma_d = np.exp(np.random.uniform(*self.gamma_d_range))

        img_np = np.array(img).astype(np.float32) / 255.0
        hed = color.rgb2hed(img_np)

        h, e, d = hed[:, :, 0], hed[:, :, 1], hed[:, :, 2]
        h = h ** gamma_h
        e = e ** gamma_e
        d = d ** gamma_d

        hed = np.stack([h, e, d], axis=2)
        img_rgb = color.hed2rgb(hed)
        img_rgb = np.clip(img_rgb, 0.0, 1.0)

        img_uint8 = (img_rgb * 255.0).astype(np.uint8)
        return Image.fromarray(img_uint8)





class WBCDataset(Dataset):
    def __init__(self, root_dir, split, aug_config, is_training=True, img_size=224):
        self.root_dir = root_dir
        self.split = split
        self.is_training = is_training
        self.img_size = img_size
        self.aug_config = aug_config

        csv_path = os.path.join(root_dir, f"{split}.csv")
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=['labels'])

        self.class_names = sorted(self.df['labels'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}

        if is_training:
            self.transform = self._build_train_transform(aug_config)
        else:
            self.transform = self._build_val_transform()

        self.image_dir = os.path.join(root_dir, split)

    def _apply_noise(self, tensor, cfg):
        img_np = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        noise_transforms = []

        if hasattr(self, 'curr_gauss_prob') and hasattr(self, 'curr_gauss_std'):
            if self.curr_gauss_prob > 0:
                noise_transforms.append(A.GaussNoise(std_range=self.curr_gauss_std, p=self.curr_gauss_prob))
        elif cfg.get('gauss_noise_prob', 0) > 0:
            noise_transforms.append(A.GaussNoise(std_range=(0.2, 0.3), p=cfg['gauss_noise_prob']))

        if hasattr(self, 'curr_iso_prob') and hasattr(self, 'curr_iso_intensity'):
            if self.curr_iso_prob > 0:
                noise_transforms.append(A.ISONoise(color_shift=self.curr_iso_color, intensity=self.curr_iso_intensity, p=self.curr_iso_prob))
        elif cfg.get('iso_noise_prob', 0) > 0:
            noise_transforms.append(A.ISONoise(color_shift=(0.1, 0.5), intensity=(0.4, 0.7), p=cfg['iso_noise_prob']))

        if noise_transforms:
            transform = A.Compose(noise_transforms)
            img_np = transform(image=img_np)['image']

        return torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

    def _build_train_transform(self, cfg):
        transform_list = [
            transforms.RandomResizedCrop(self.img_size, scale=tuple(cfg['random_crop_scale'])),
            transforms.RandomHorizontalFlip(p=cfg['h_flip_prob']),
            transforms.RandomVerticalFlip(p=cfg['v_flip_prob']),
            transforms.RandomRotation(degrees=cfg['rotation_degrees']),
        ]

        if cfg.get('he_stain_prob', 0) > 0:
            transform_list.append(
                RandAdjustHE(prob=cfg['he_stain_prob'])
            )

        transform_list.extend([
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
            transforms.ToTensor(),
            transforms.Lambda(lambda x: self._apply_noise(x, cfg)),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=cfg['gaussian_blur_prob']),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=cfg['random_erasing_prob'], scale=(0.02, 0.15))
        ])

        return transforms.Compose(transform_list)

    def _build_val_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(self.img_size, scale=[0.8,1]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _build_train_transform_curriculum(self, cfg, curriculum_params):
        transform_list = [
            transforms.RandomResizedCrop(self.img_size, scale=tuple(cfg['random_crop_scale'])),
            transforms.RandomHorizontalFlip(p=cfg['h_flip_prob']),
            transforms.RandomVerticalFlip(p=cfg['v_flip_prob']),
            transforms.RandomRotation(degrees=cfg['rotation_degrees']),
        ]

        if cfg.get('he_stain_prob', 0) > 0:
            transform_list.append(
                RandAdjustHE(prob=cfg['he_stain_prob'])
            )

        transform_list.extend([
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
            transforms.ToTensor(),
            transforms.Lambda(lambda x: self._apply_noise(x, cfg)),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=curriculum_params['gaussian_blur_sigma'])], p=curriculum_params['gaussian_blur_prob']),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=cfg['random_erasing_prob'], scale=(0.02, 0.15))
        ])

        return transforms.Compose(transform_list)

    def update_curriculum_params(self, curriculum_params):
        if not self.is_training:
            return

        self.curr_gauss_std = curriculum_params['gauss_noise_std_range']
        self.curr_gauss_prob = curriculum_params['gauss_noise_prob']
        self.curr_iso_color = curriculum_params['iso_noise_color_shift']
        self.curr_iso_intensity = curriculum_params['iso_noise_intensity']
        self.curr_iso_prob = curriculum_params['iso_noise_prob']

        self.transform = self._build_train_transform_curriculum(self.aug_config, curriculum_params)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['ID']
        label_name = row['labels']

        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        label_idx = self.class_to_idx[label_name]
        return image, label_idx


class ConvNeXtV2Classifier(nn.Module):
    """
    ConvNeXt V2 (Base) Classifier using timm.
    Requires 'timm' library: pip install timm
    """

    def __init__(self, num_classes=13, pretrained=True, dropout=0.4):
        super().__init__()
        # Load ConvNeXt V2 Base from timm
        self.model = timm.create_model(
            'convnextv2_base',
            pretrained=pretrained
        )

        # In timm ConvNeXt models, the classifier head is usually 'model.head.fc'
        # We replace the final linear layer to include our custom dropout
        in_features = self.model.head.fc.in_features
        self.model.head.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def compute_class_weights(label_counts, method="sqrt_inverse"):
    counts = np.array([label_counts[i] for i in sorted(label_counts.keys())])

    if method == "inverse":
        weights = 1.0 / counts
    elif method == "sqrt_inverse":
        weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    weights = weights / weights.sum() * len(weights)
    return torch.FloatTensor(weights)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing"""
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        n_classes = pred.size(-1)

        if target.dim() == 1:
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), confidence)
        else:
            true_dist = target

        log_probs = F.log_softmax(pred, dim=-1)

        if self.weight is not None:
            if target.dim() == 1:
                weight = self.weight[target]
            else:
                weight = (true_dist * self.weight.unsqueeze(0)).sum(dim=1)
            loss = -(true_dist * log_probs).sum(dim=-1) * weight
        else:
            loss = -(true_dist * log_probs).sum(dim=-1)

        return loss.mean()


def get_loss_function(class_weights=None, device="cuda", label_smoothing=0):
    if label_smoothing > 0:
        return LabelSmoothingCrossEntropy(smoothing=label_smoothing, weight=class_weights).to(device)
    else:
        return nn.CrossEntropyLoss(weight=None)


def compute_metrics(y_true, y_pred, class_names):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    per_class_metrics = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    metrics = {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'per_class': {
            class_names[i]: {
                'precision': float(per_class_metrics[0][i]),
                'recall': float(per_class_metrics[1][i]),
                'f1': float(per_class_metrics[2][i]),
                'support': int(per_class_metrics[3][i])
            }
            for i in range(len(class_names))
        }
    }
    return metrics


def train_epoch(model, dataloader, criterion, optimizer, device, grad_clip_norm,
                accumulation_steps=1):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Training")):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss = loss / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    if (batch_idx + 1) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def main():
    # Detect if running in Jupyter/Kaggle environment
    try:
        get_ipython()
        in_notebook = True
    except NameError:
        in_notebook = False

    parser = argparse.ArgumentParser(description="Train ConvNeXt V2 Base for WBC classification")
    parser.add_argument("--data_dir", type=str, default="/kaggle/input/wbc-zyx/raw_data")
    parser.add_argument("--exp_dir", type=str, default="/kaggle/working/convnextv2_base_run")


    if in_notebook:
        # Running in notebook - use defaults or modify these values
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    config = {
        'data': {
            'root_dir': args.data_dir,
            'num_classes': 13,
            'img_size': 224,
        },
        'model': {
            'pretrained': True,
            'dropout': 0.4,
            'arch': 'convnextv2_base'
        },
        'augmentation': {
            'random_crop_scale': [0.8, 1.0],
            'h_flip_prob': 0.5,
            'v_flip_prob': 0.5,
            'rotation_degrees': 45,
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1,
            'translate': [0.125, 0.125],
            'scale': [0.9, 1.1],
            'gauss_noise_prob': 0.3,
            'iso_noise_prob': 0.3,
            'gaussian_blur_prob': 0.3,
            'random_erasing_prob': 0,
            'he_stain_prob': 0.3,
        },
        'curriculum': {
            'enabled': True,
            'schedule_type': 'linear',
            'total_epochs': 10,
        },
        'validation': {
            'use_noised_val': True,
            'noised_val_split': 'noised_val',
        },
        'training': {
            'seed': 42,
            'batch_size': 32,
            'num_epochs': 20,
            'learning_rate': 0.0001,
            'weight_decay': 0.0001,
            'early_stopping_patience': 5,
            'gradient_clip_norm': 1.0,
            'scheduler_factor': 0.5,
            'scheduler_patience': 3,
            'scheduler_min_lr': 1e-7,
            'label_smoothing': 0.1,
            'accumulation_steps': 2,
            'num_workers': 4,
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 3,
        },
        'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    }

    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])

    exp_dir = args.exp_dir
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)

    with open(os.path.join(exp_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Device: {config['device']}")
    print(f"Experiment directory: {exp_dir}\n")

    train_dataset = WBCDataset(
        config['data']['root_dir'],
        'train',
        config['augmentation'],
        is_training=True,
        img_size=config['data']['img_size']
    )

    val_dataset = WBCDataset(
        config['data']['root_dir'],
        'val',
        config['augmentation'],
        is_training=False,
        img_size=config['data']['img_size']
    )

    if config['validation']['use_noised_val']:
        noised_val_dataset = WBCDataset(
            config['data']['root_dir'],
            config['validation']['noised_val_split'],
            config['augmentation'],
            is_training=False,
            img_size=config['data']['img_size']
        )
    else:
        noised_val_dataset = None

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    if noised_val_dataset:
        print(f"Noised Val samples: {len(noised_val_dataset)}")
    print(f"Classes: {train_dataset.class_names}\n")

    label_counts = Counter(train_dataset.df['labels'])
    class_weights_dict = {train_dataset.class_to_idx[cls]: count
                          for cls, count in label_counts.items()}
    class_weights = compute_class_weights(class_weights_dict, method="sqrt_inverse")

    print("Class Distribution:")
    for cls in train_dataset.class_names:
        idx = train_dataset.class_to_idx[cls]
        count = label_counts[cls]
        weight = class_weights[idx].item()
        print(f"{cls:<5}: {count:>6} samples | weight: {weight:.4f}")
    print()

    sample_weights = []
    for idx in range(len(train_dataset)):
        label_name = train_dataset.df.iloc[idx]['labels']
        label_idx = train_dataset.class_to_idx[label_name]
        sample_weights.append(class_weights[label_idx].item())

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=sampler,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        persistent_workers=config['training']['persistent_workers'],
        prefetch_factor=config['training']['prefetch_factor']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        persistent_workers=config['training']['persistent_workers'],
        prefetch_factor=config['training']['prefetch_factor']
    )

    if noised_val_dataset:
        noised_val_loader = DataLoader(
            noised_val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory'],
            persistent_workers=config['training']['persistent_workers'],
            prefetch_factor=config['training']['prefetch_factor']
        )
    else:
        noised_val_loader = None

    model = ConvNeXtV2Classifier(
        num_classes=config['data']['num_classes'],
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout']
    ).to(config['device'])

    criterion = get_loss_function(
        class_weights=None,
        device=config['device'],
        label_smoothing=config['training']['label_smoothing']
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=(0.9, 0.999)
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config['training']['scheduler_factor'],
        patience=config['training']['scheduler_patience'],
        min_lr=config['training']['scheduler_min_lr']
    )


    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(os.path.join(exp_dir, "logs"))
    else:
        writer = None

    print(f"Model: ConvNeXt V2 Base ({sum(p.numel() for p in model.parameters()):,} parameters)")
    print(f"Loss: Label Smoothing CE (smoothing={config['training']['label_smoothing']}) with unweighted loss")
    print(f"Sampling: WeightedRandomSampler with sqrt_inverse class weights")
    print(f"Image Size: {config['data']['img_size']}x{config['data']['img_size']}")
    print(f"Effective Batch Size: {config['training']['batch_size'] * config['training']['accumulation_steps']}\n")

    if config['curriculum']['enabled']:
        curriculum_scheduler = CurriculumScheduler(
            schedule_type=config['curriculum']['schedule_type'],
            total_epochs=config['curriculum']['total_epochs']
        )
        print(f"Curriculum Learning: {config['curriculum']['schedule_type']} schedule over {config['curriculum']['total_epochs']} epochs")
    else:
        curriculum_scheduler = None

    best_f1_original = 0
    best_f1_noised = 0
    best_macro_f1 = 0
    best_val_acc = 0
    patience_counter = 0
    training_history = []

    for epoch in range(config['training']['num_epochs']):
        print(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")

        if curriculum_scheduler:
            curriculum_params = curriculum_scheduler.get_curriculum_params(epoch)
            train_dataset.update_curriculum_params(curriculum_params)

            print(f"Curriculum Params:")
            print(f"  GaussNoise: std={curriculum_params['gauss_noise_std_range']}, p={curriculum_params['gauss_noise_prob']:.3f}")
            print(f"  ISONoise: intensity={curriculum_params['iso_noise_intensity']}, p={curriculum_params['iso_noise_prob']:.3f}")
            print(f"  GaussianBlur: sigma={curriculum_params['gaussian_blur_sigma']}, p={curriculum_params['gaussian_blur_prob']:.3f}")

        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, config['device'],
            config['training']['gradient_clip_norm'],
            accumulation_steps=config['training']['accumulation_steps']
        )

        val_loss, val_preds, val_labels = evaluate(
            model, val_loader, criterion, config['device']
        )

        val_metrics = compute_metrics(val_labels, val_preds, train_dataset.class_names)

        if noised_val_loader:
            noised_val_loss, noised_val_preds, noised_val_labels = evaluate(
                model, noised_val_loader, criterion, config['device']
            )
            noised_val_metrics = compute_metrics(
                noised_val_labels, noised_val_preds, train_dataset.class_names
            )
        else:
            noised_val_metrics = None

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")
        print(f"Val (Original) - Accuracy: {val_metrics['accuracy']:.4f} | Macro F1: {val_metrics['macro_f1']:.4f}")
        if noised_val_metrics:
            print(f"Val (Noised)   - Accuracy: {noised_val_metrics['accuracy']:.4f} | Macro F1: {noised_val_metrics['macro_f1']:.4f}")
        print()

        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val_original', val_loss, epoch)
            writer.add_scalar('Metrics/val_original_accuracy', val_metrics['accuracy'], epoch)
            writer.add_scalar('Metrics/val_original_macro_f1', val_metrics['macro_f1'], epoch)

            if noised_val_metrics:
                writer.add_scalar('Loss/val_noised', noised_val_loss, epoch)
                writer.add_scalar('Metrics/val_noised_accuracy', noised_val_metrics['accuracy'], epoch)
                writer.add_scalar('Metrics/val_noised_macro_f1', noised_val_metrics['macro_f1'], epoch)

            writer.add_scalar('Learning_rate', current_lr, epoch)

            if curriculum_scheduler:
                writer.add_scalar('Curriculum/gauss_noise_prob', curriculum_params['gauss_noise_prob'], epoch)
                writer.add_scalar('Curriculum/iso_noise_prob', curriculum_params['iso_noise_prob'], epoch)
                writer.add_scalar('Curriculum/gaussian_blur_prob', curriculum_params['gaussian_blur_prob'], epoch)

        history_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_metrics['accuracy'],
            'val_macro_f1': val_metrics['macro_f1'],
            'learning_rate': current_lr
        }
        if noised_val_metrics:
            history_entry.update({
                'noised_val_loss': noised_val_loss,
                'noised_val_accuracy': noised_val_metrics['accuracy'],
                'noised_val_macro_f1': noised_val_metrics['macro_f1'],
            })
        training_history.append(history_entry)

        scheduler.step(val_metrics['macro_f1'])

        if val_metrics['macro_f1'] > best_f1_original:
            best_f1_original = val_metrics['macro_f1']
            best_macro_f1 = val_metrics['macro_f1']
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'macro_f1': best_f1_original,
                'accuracy': val_metrics['accuracy'],
                'val_type': 'original',
            }, os.path.join(exp_dir, "checkpoints", "best_f1_original.pth"))

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'macro_f1': best_macro_f1,
                'accuracy': val_metrics['accuracy'],
            }, os.path.join(exp_dir, "checkpoints", "best_macro_f1.pth"))

            print(f"✓ Saved best F1 (original val) model (Macro F1: {best_f1_original:.4f})\n")
        else:
            patience_counter += 1

        if noised_val_metrics and noised_val_metrics['macro_f1'] > best_f1_noised:
            best_f1_noised = noised_val_metrics['macro_f1']

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'macro_f1': best_f1_noised,
                'accuracy': noised_val_metrics['accuracy'],
                'val_type': 'noised',
            }, os.path.join(exp_dir, "checkpoints", "best_f1_noised.pth"))

            print(f"✓ Saved best F1 (noised val) model (Macro F1: {best_f1_noised:.4f})\n")

        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"Early stopping at epoch {epoch + 1}\n")
            break

    print("=" * 80)
    print("FINAL VALIDATION METRICS")
    print("=" * 80)

    checkpoint_f1_orig = torch.load(os.path.join(exp_dir, "checkpoints", "best_f1_original.pth"))
    model.load_state_dict(checkpoint_f1_orig['model_state_dict'])

    val_loss_f1, val_preds_f1, val_labels_f1 = evaluate(model, val_loader, criterion, config['device'])
    val_metrics_f1 = compute_metrics(val_labels_f1, val_preds_f1, train_dataset.class_names)

    print("\nBest F1 (Original Val) Model:")
    print(f"  Original Val - Accuracy: {val_metrics_f1['accuracy']:.4f} | Macro F1: {val_metrics_f1['macro_f1']:.4f}")

    if noised_val_loader:
        noised_val_loss_f1, noised_val_preds_f1, noised_val_labels_f1 = evaluate(
            model, noised_val_loader, criterion, config['device']
        )
        noised_val_metrics_f1 = compute_metrics(
            noised_val_labels_f1, noised_val_preds_f1, train_dataset.class_names
        )
        print(f"  Noised Val   - Accuracy: {noised_val_metrics_f1['accuracy']:.4f} | Macro F1: {noised_val_metrics_f1['macro_f1']:.4f}")

    if noised_val_loader:
        checkpoint_f1_noised = torch.load(os.path.join(exp_dir, "checkpoints", "best_f1_noised.pth"))
        model.load_state_dict(checkpoint_f1_noised['model_state_dict'])

        val_loss_f1n, val_preds_f1n, val_labels_f1n = evaluate(model, val_loader, criterion, config['device'])
        val_metrics_f1n = compute_metrics(val_labels_f1n, val_preds_f1n, train_dataset.class_names)

        noised_val_loss_f1n, noised_val_preds_f1n, noised_val_labels_f1n = evaluate(
            model, noised_val_loader, criterion, config['device']
        )
        noised_val_metrics_f1n = compute_metrics(
            noised_val_labels_f1n, noised_val_preds_f1n, train_dataset.class_names
        )

        print("\nBest F1 (Noised Val) Model:")
        print(f"  Original Val - Accuracy: {val_metrics_f1n['accuracy']:.4f} | Macro F1: {val_metrics_f1n['macro_f1']:.4f}")
        print(f"  Noised Val   - Accuracy: {noised_val_metrics_f1n['accuracy']:.4f} | Macro F1: {noised_val_metrics_f1n['macro_f1']:.4f}")

    with open(os.path.join(exp_dir, "metrics_best_f1_original.json"), 'w') as f:
        json.dump(val_metrics_f1, f, indent=2)

    if noised_val_loader:
        with open(os.path.join(exp_dir, "metrics_best_f1_original_on_noised.json"), 'w') as f:
            json.dump(noised_val_metrics_f1, f, indent=2)

        with open(os.path.join(exp_dir, "metrics_best_f1_noised.json"), 'w') as f:
            json.dump(noised_val_metrics_f1n, f, indent=2)

        with open(os.path.join(exp_dir, "metrics_best_f1_noised_on_original.json"), 'w') as f:
            json.dump(val_metrics_f1n, f, indent=2)

    with open(os.path.join(exp_dir, "training_history.json"), 'w') as f:
        json.dump(training_history, f, indent=2)

    report_f1_orig = classification_report(
        val_labels_f1,
        val_preds_f1,
        target_names=train_dataset.class_names,
        digits=4,
        zero_division=0
    )

    with open(os.path.join(exp_dir, "classification_report_best_f1_original.txt"), 'w') as f:
        f.write("Best F1 (Original Val) Model on Original Validation\n")
        f.write("=" * 80 + "\n\n")
        f.write(report_f1_orig)

    if noised_val_loader:
        report_f1_noised = classification_report(
            noised_val_labels_f1n,
            noised_val_preds_f1n,
            target_names=train_dataset.class_names,
            digits=4,
            zero_division=0
        )

        with open(os.path.join(exp_dir, "classification_report_best_f1_noised.txt"), 'w') as f:
            f.write("Best F1 (Noised Val) Model on Noised Validation\n")
            f.write("=" * 80 + "\n\n")
            f.write(report_f1_noised)

    print("\nClassification Report (Best F1 Original Val Model):")
    print(report_f1_orig)

    if writer is not None:
        writer.close()
    print(f"\nResults saved to {exp_dir}")


if __name__ == "__main__":
    main()