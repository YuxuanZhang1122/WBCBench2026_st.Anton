import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import json
from collections import Counter

from src.data.dinobloom_augmented_dataset import DinoBloomAugmentedDataset
from src.models.dinobloom_classifier import DinoBloomClassifier
from src.utils.losses import get_loss_function, compute_class_weights
from src.utils.metrics import compute_metrics, print_metrics, plot_confusion_matrix, save_classification_report
from src.utils.config_loader import load_config, save_config


def train_epoch(model, dataloader, criterion, optimizer, device, use_amp=False, scaler=None):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Mixed precision backward pass
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
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
        for features, labels in tqdm(dataloader, desc="Evaluating"):
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description="Train WBC classifier on DinoBloom features")
    parser.add_argument("--config", type=str, default="configs/dinobloom_config.yaml", help="Path to config file")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use automatic mixed precision (default: True)")
    parser.add_argument("--no_amp", action="store_false", dest="use_amp", help="Disable automatic mixed precision")
    parser.add_argument("--data_dir", type=str, default="raw_data", help="Root data directory")

    args = parser.parse_args()

    config = load_config(args.config)

    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)

    feature_type = config.feature_extraction.feature_type
    resolution_strategy = config.feature_extraction.resolution_strategy
    classifier_head = config.training.classifier_head
    loss_function = config.training.loss_function

    exp_name = args.exp_name or f"dinobloom_{feature_type}_{classifier_head}_{loss_function}_{resolution_strategy}"

    exp_dir = os.path.join("experiments/", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)

    save_config(config, os.path.join(exp_dir, "config.yaml"))

    writer = SummaryWriter(os.path.join(exp_dir, "logs")) if config.logging.use_tensorboard else None

    print("=" * 80)
    print(f"Experiment: {exp_name}")
    print("=" * 80)
    print(f"Feature type: {feature_type}")
    print(f"Resolution strategy: {resolution_strategy}")
    print(f"Classifier: {classifier_head}")
    print(f"Loss function: {loss_function}")
    print(f"Device: {config.device}")
    print("=" * 80)

    print("\nLoading training data...")
    aug_config = {
        'aggressive_threshold': config.augmentation['aggressive_threshold'],
        'medium_threshold': config.augmentation['medium_threshold'],
        'image_size': config.data.image_size,
        'target_size': config.data.target_size
    }

    train_dataset = DinoBloomAugmentedDataset(
        args.data_dir,
        "train",
        aug_config,
        is_training=True
    )

    print("\nLoading validation data...")
    val_dataset = DinoBloomAugmentedDataset(
        args.data_dir,
        "val",
        aug_config,
        is_training=False
    )

    label_counts = Counter(train_dataset.df['labels'])
    class_weights_dict = {train_dataset.class_to_idx[cls]: count
                          for cls, count in label_counts.items()}
    class_weights = compute_class_weights(class_weights_dict, method="sqrt_inverse")

    # Create WeightedRandomSampler for oversampling rare classes
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
        batch_size=config.training.batch_size,
        sampler=sampler,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=config.training.persistent_workers if config.training.num_workers > 0 else False,
        prefetch_factor=config.training.prefetch_factor if config.training.num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=config.training.persistent_workers if config.training.num_workers > 0 else False,
        prefetch_factor=config.training.prefetch_factor if config.training.num_workers > 0 else None
    )

    print(f"\nClass weights: {class_weights}")

    model = DinoBloomClassifier(
        checkpoint_path=config.model.checkpoint_path,
        num_classes=config.data.num_classes,
        classifier_head=config.training.classifier_head,
        mlp_hidden_dims=config.training.mlp_hidden_dims,
        dropout=config.training.dropout,
        feature_type=feature_type,
        resolution_strategy=resolution_strategy,
        freeze_backbone=config.model.freeze_backbone
    )
    model = model.to(config.device)
    model.freeze_backbone()

    criterion = get_loss_function(
        config.training.loss_function,
        class_weights,
        config.training.focal_gamma,
        config.training.label_smoothing,
        config.device
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler() if args.use_amp else None

    if args.use_amp:
        print("\n✓ Automatic Mixed Precision (AMP) enabled")
    else:
        print("\n✗ AMP disabled (FP32 training)")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )

    best_macro_f1 = 0
    patience_counter = 0

    for epoch in range(config.training.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.num_epochs}")
        print("-" * 80)

        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, config.device,
            use_amp=args.use_amp, scaler=scaler
        )

        val_loss, val_preds, val_labels = evaluate(
            model, val_loader, criterion, config.device
        )

        val_metrics = compute_metrics(val_labels, val_preds, train_dataset.class_names)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f} | Val Macro F1: {val_metrics['macro_f1']:.4f}")

        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            writer.add_scalar('F1/macro', val_metrics['macro_f1'], epoch)

        scheduler.step(val_metrics['macro_f1'])

        if val_metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = val_metrics['macro_f1']
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'macro_f1': best_macro_f1,
                'config': config.__dict__
            }
            torch.save(checkpoint, os.path.join(exp_dir, "checkpoints", "best_model.pth"))
            print(f"✓ Saved best model (Macro F1: {best_macro_f1:.4f})")

        else:
            patience_counter += 1
            if patience_counter >= config.training.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(os.path.join(exp_dir, "checkpoints", "best_model.pth"), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, config.device)

    val_metrics = compute_metrics(val_labels, val_preds, train_dataset.class_names)

    print_metrics(val_metrics, "Final Validation Metrics")

    with open(os.path.join(exp_dir, "metrics.json"), 'w') as f:
        json.dump(val_metrics, f, indent=2)

    if config.evaluation.save_confusion_matrix:
        plot_confusion_matrix(
            val_labels,
            val_preds,
            train_dataset.class_names,
            os.path.join(exp_dir, "confusion_matrix.png"),
            normalize=True
        )

    if config.evaluation.save_per_class_metrics:
        save_classification_report(
            val_labels,
            val_preds,
            train_dataset.class_names,
            os.path.join(exp_dir, "classification_report.txt")
        )

    if writer:
        writer.close()

    print(f"\nResults saved to {exp_dir}")


if __name__ == "__main__":
    main()
