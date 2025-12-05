import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
from collections import Counter

from src.models.resnet50_classifier import ResNet50Classifier
from src.data.augmented_dataset import AugmentedWBCDataset
from src.utils.losses import get_loss_function, compute_class_weights
from src.utils.metrics import compute_metrics, print_metrics, plot_confusion_matrix, save_classification_report
from src.utils.config_loader import load_config, save_config


def train_epoch(model, dataloader, criterion, optimizer, device, grad_clip_norm):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

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
    parser = argparse.ArgumentParser(description="Train ResNet50 for WBC classification")
    parser.add_argument("--config", type=str, default="configs/resnet50_fast.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)

    exp_dir = "experiments/resnet50_end2end"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)

    save_config(config, os.path.join(exp_dir, "config.yaml"))
    writer = SummaryWriter(os.path.join(exp_dir, "logs"))

    train_dataset = AugmentedWBCDataset(
        config.data.root_dir,
        "train",
        config.augmentation,
        is_training=True
    )

    val_dataset = AugmentedWBCDataset(
        config.data.root_dir,
        "val",
        config.augmentation,
        is_training=False
    )

    label_counts = Counter(train_dataset.df['labels'])
    class_weights_dict = {train_dataset.class_to_idx[cls]: count
                          for cls, count in label_counts.items()}
    class_weights = compute_class_weights(class_weights_dict, method="sqrt_inverse")

    # Create WeightedRandomSampler to oversample rare classes
    sample_weights = []
    for idx in range(len(train_dataset)):
        label_name = train_dataset.df.iloc[idx]['labels']
        label_idx = train_dataset.class_to_idx[label_name]
        # Use sqrt inverse weighting for sampling (same as loss)
        sample_weights.append(class_weights[label_idx].item())

    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=sampler,
        num_workers=getattr(config.training, 'num_workers', 4),
        pin_memory=getattr(config.training, 'pin_memory', False),
        persistent_workers=getattr(config.training, 'persistent_workers', False) if getattr(config.training, 'num_workers', 4) > 0 else False,
        prefetch_factor=getattr(config.training, 'prefetch_factor', 2) if getattr(config.training, 'num_workers', 4) > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=getattr(config.training, 'num_workers', 4),
        pin_memory=getattr(config.training, 'pin_memory', False),
        persistent_workers=getattr(config.training, 'persistent_workers', False) if getattr(config.training, 'num_workers', 4) > 0 else False,
        prefetch_factor=getattr(config.training, 'prefetch_factor', 2) if getattr(config.training, 'num_workers', 4) > 0 else None
    )

    model = ResNet50Classifier(
        num_classes=config.data.num_classes,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout
    ).to(config.device)

    criterion = get_loss_function(
        config.training.loss_function,
        class_weights,
        device=config.device
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
        min_lr=float(config.training.scheduler_min_lr)
    )

    best_macro_f1 = 0
    patience_counter = 0

    print("\nClass-to-Augmentation Mapping:")
    for cls in train_dataset.class_names:
        aug_level = train_dataset.class_to_aug_level[cls]
        count = label_counts[cls]
        print(f"{cls:<5} ({count:>5} samples): {aug_level.upper()}")

    for epoch in range(config.training.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.num_epochs}")

        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, config.device,
            config.training.gradient_clip_norm
        )

        val_loss, val_preds, val_labels = evaluate(
            model, val_loader, criterion, config.device
        )

        val_metrics = compute_metrics(val_labels, val_preds, train_dataset.class_names)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f} | Val Macro F1: {val_metrics['macro_f1']:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/accuracy', val_metrics['accuracy'], epoch)
        writer.add_scalar('Metrics/macro_f1', val_metrics['macro_f1'], epoch)

        scheduler.step(val_metrics['macro_f1'])

        if val_metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = val_metrics['macro_f1']
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'macro_f1': best_macro_f1,
            }, os.path.join(exp_dir, "checkpoints", "best_model.pth"))

            print(f"Saved best model (Macro F1: {best_macro_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.training.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    checkpoint = torch.load(os.path.join(exp_dir, "checkpoints", "best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])

    val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, config.device)
    val_metrics = compute_metrics(val_labels, val_preds, train_dataset.class_names)

    print_metrics(val_metrics, "Final Validation Metrics")

    with open(os.path.join(exp_dir, "metrics.json"), 'w') as f:
        json.dump(val_metrics, f, indent=2)

    plot_confusion_matrix(
        val_labels, val_preds, train_dataset.class_names,
        os.path.join(exp_dir, "confusion_matrix.png"),
        normalize=True
    )

    save_classification_report(
        val_labels, val_preds, train_dataset.class_names,
        os.path.join(exp_dir, "classification_report.txt")
    )

    writer.close()
    print(f"\nResults saved to {exp_dir}")


if __name__ == "__main__":
    main()
