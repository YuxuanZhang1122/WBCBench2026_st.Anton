import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from collections import Counter
from tqdm import tqdm
import argparse
import timm
from skimage import color


class RandAdjustHE:
    def __init__(self, prob=0.5, gamma_h=(0.7, 1.3), gamma_e=(0.7, 1.3), gamma_d=(0.7, 1.3)):
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


class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=13, pretrained=True, dropout=0.4):
        super().__init__()
        self.resnet = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


class ConvNeXtV2Classifier(nn.Module):
    def __init__(self, num_classes=13, pretrained=True, dropout=0.4):
        super().__init__()
        self.model = timm.create_model('convnextv2_base', pretrained=pretrained)
        in_features = self.model.head.fc.in_features
        self.model.head.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class SwinTransformerClassifier(nn.Module):
    def __init__(self, num_classes=13, pretrained=True, dropout=0.4):
        super().__init__()
        self.model = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout
        )

    def forward(self, x):
        return self.model(x)


class WBCTestDataset(Dataset):
    def __init__(self, csv_file, root_dir, class_names, transform=None, n_tta=1):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.n_tta = n_tta
        self.class_names = sorted(class_names)
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.class_names)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['ID'])
        image = Image.open(img_path).convert('RGB')

        if self.n_tta > 1 and self.transform:
            augmented_images = []
            for _ in range(self.n_tta):
                aug_img = self.transform(image)
                augmented_images.append(aug_img)
            augmented_images = torch.stack(augmented_images)
            return augmented_images, row['ID']
        else:
            if self.transform:
                image = self.transform(image)
            return image, row['ID']


def get_tta_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        RandAdjustHE(prob=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.125, 0.125), scale=(0.9, 1.1)),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_model(model_type, weight_path, num_classes, device):
    if model_type == 'resnet50':
        model = ResNet50Classifier(num_classes=num_classes, pretrained=False)
    elif model_type == 'convnext':
        model = ConvNeXtV2Classifier(num_classes=num_classes, pretrained=False)
    elif model_type == 'swin':
        model = SwinTransformerClassifier(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint = torch.load(weight_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def predict_single_model(model, dataloader, n_tta, device):
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            if n_tta > 1:
                aug_imgs, ids = batch
                batch_size, n_tta_actual, c, h, w = aug_imgs.shape
                aug_imgs = aug_imgs.view(batch_size * n_tta_actual, c, h, w).to(device)

                outputs = model(aug_imgs)
                outputs = outputs.view(batch_size, n_tta_actual, -1)

                for i in range(batch_size):
                    tta_preds = outputs[i].argmax(dim=1).cpu().numpy()
                    majority_vote = Counter(tta_preds).most_common(1)[0][0]
                    predictions.append((ids[i], majority_vote))
            else:
                imgs, ids = batch
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1).cpu().numpy()
                predictions.extend(zip(ids, preds))

    return predictions


def ensemble_predictions(all_model_predictions, idx_to_class, use_resnet_fallback=True):
    id_to_preds = {}
    resnet_preds = {}

    for model_idx, model_preds in enumerate(all_model_predictions):
        for img_id, pred_idx in model_preds:
            if img_id not in id_to_preds:
                id_to_preds[img_id] = []
            id_to_preds[img_id].append(pred_idx)

            if model_idx == 0 and use_resnet_fallback:
                resnet_preds[img_id] = pred_idx

    final_predictions = []
    for img_id, pred_indices in id_to_preds.items():
        counts = Counter(pred_indices)
        most_common = counts.most_common(1)[0]

        if most_common[1] >= 2:
            final_pred_idx = most_common[0]
        elif use_resnet_fallback and img_id in resnet_preds:
            final_pred_idx = resnet_preds[img_id]
        else:
            final_pred_idx = most_common[0]

        label = idx_to_class[final_pred_idx]
        final_predictions.append((img_id, label))

    return final_predictions


def main():
    parser = argparse.ArgumentParser(description="WBC Test Prediction with Ensemble & TTA")
    parser.add_argument("--data_dir", type=str,
                        default="/kaggle/input/wbc2026bench/raw_data/raw_data", # "raw_data/"
                        help="Root directory containing test images and CSV")
    parser.add_argument("--weight_dir", type=str,
                        default="/kaggle/input/wbc2026bench", # ""
                        help="Directory containing model weights")
    parser.add_argument("--output", type=str,
                        default="predictions.csv",
                        help="Output CSV file path")
    parser.add_argument("--models", type=str, nargs='+',
                        default=['ensemble'],
                        choices=['resnet50', 'convnext', 'swin', 'ensemble'],
                        help="Models to use (default: ensemble)")
    parser.add_argument("--resnet_weight", type=str,
                        default="resnet50_best.pth",
                        help="ResNet50 weight filename")
    parser.add_argument("--convnext_weight", type=str,
                        default="convnext_best.pth",
                        help="ConvNeXt weight filename")
    parser.add_argument("--swin_weight", type=str,
                        default="swin_best.pth",
                        help="Swin Transformer weight filename")
    parser.add_argument("--n_tta", type=int, default=10,
                        help="Number of test-time augmentations (1=disabled)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--num_classes", type=int, default=13,
                        help="Number of classes")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')

    test_csv = os.path.join(args.data_dir, 'test.csv')
    test_img_dir = os.path.join(args.data_dir, 'test')

    train_csv = os.path.join(args.data_dir, 'train.csv')
    if not os.path.exists(train_csv):
        print(f"Error: Training CSV not found at {train_csv}")
        print("Using default class names (may cause incorrect predictions)")
        class_names = ['BA', 'BL', 'BNE', 'EO', 'LY', 'MMY', 'MO',
                       'MY', 'PC', 'PLY', 'PMY', 'SNE', 'VLY']
    else:
        train_df = pd.read_csv(train_csv)
        class_names = sorted(train_df['labels'].dropna().unique())
        print(f"Loaded {len(class_names)} classes from training data: {class_names}")

    idx_to_class = {idx: cls for idx, cls in enumerate(class_names)}

    transform = get_tta_transform() if args.n_tta > 1 else get_val_transform()
    test_dataset = WBCTestDataset(test_csv, test_img_dir, class_names,
                                   transform=transform, n_tta=args.n_tta)

    batch_size = args.batch_size if args.n_tta == 1 else max(1, args.batch_size // args.n_tta)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    models_to_use = args.models
    if 'ensemble' in models_to_use:
        models_to_use = ['resnet50', 'convnext', 'swin']

    all_model_predictions = []

    for model_type in models_to_use:
        if model_type == 'resnet50':
            weight_path = os.path.join(args.weight_dir, args.resnet_weight)
        elif model_type == 'convnext':
            weight_path = os.path.join(args.weight_dir, args.convnext_weight)
        elif model_type == 'swin':
            weight_path = os.path.join(args.weight_dir, args.swin_weight)

        if not os.path.exists(weight_path):
            print(f"Warning: Weight file {weight_path} not found, skipping {model_type}")
            continue

        print(f"\n{'='*60}")
        print(f"Loading {model_type.upper()} model from {weight_path}")
        print(f"{'='*60}")

        model = load_model(model_type, weight_path, args.num_classes, device)
        predictions = predict_single_model(model, test_loader, args.n_tta, device)
        all_model_predictions.append(predictions)

        individual_preds = [(img_id, idx_to_class[pred_idx]) for img_id, pred_idx in predictions]
        individual_output = args.output.replace('.csv', f'_{model_type}.csv')
        individual_df = pd.DataFrame(individual_preds, columns=['ID', 'labels'])
        individual_df.to_csv(individual_output, index=False)
        print(f"Saved {model_type} predictions to {individual_output}")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if len(all_model_predictions) == 0:
        print("Error: No models were loaded successfully!")
        return

    if len(all_model_predictions) == 1:
        final_predictions = [(img_id, idx_to_class[pred_idx])
                           for img_id, pred_idx in all_model_predictions[0]]
    else:
        print(f"\n{'='*60}")
        print(f"Ensembling predictions from {len(all_model_predictions)} models")
        print(f"{'='*60}")
        final_predictions = ensemble_predictions(all_model_predictions, idx_to_class)

    output_df = pd.DataFrame(final_predictions, columns=['ID', 'labels'])
    output_df.to_csv(args.output, index=False)

    print(f"\n{'='*60}")
    print(f"Predictions saved to {args.output}")
    print(f"Total predictions: {len(final_predictions)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
