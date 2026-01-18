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
    def __init__(self, prob=0.3, gamma_h=(0.8, 1.2), gamma_e=(0.8, 1.2), gamma_d=(0.8, 1.2)):
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


class RandomScaleCenterCrop:
    def __init__(self, size, scale=(0.8, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, img):
        w, h = img.size
        scale_factor = np.random.uniform(self.scale[0], self.scale[1])
        crop_size = int(min(w, h) * scale_factor)

        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size

        img_cropped = img.crop((left, top, right, bottom))
        return img_cropped.resize((self.size, self.size), Image.Resampling.BILINEAR)


# def get_tta_transform():
#     return transforms.Compose([
#         RandomScaleCenterCrop(224, scale=(0.85, 1.0)),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomVerticalFlip(p=0.5),
#         transforms.RandomChoice([
#             transforms.RandomRotation(degrees=(0, 0)),
#             transforms.RandomRotation(degrees=(90, 90)),
#             transforms.RandomRotation(degrees=(180, 180)),
#             transforms.RandomRotation(degrees=(270, 270))
#         ]),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

def get_tta_transform(model_type='resnet50'):
    he_prob = 0.0 if model_type == 'convnext' else 0.2
    return transforms.Compose([
        RandomScaleCenterCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        RandAdjustHE(prob=he_prob),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        # transforms.RandomAffine(degrees=0, translate=(0.125, 0.125), scale=(0.9, 1.1)),
        # transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    all_predictions = {}
    all_probabilities = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            if n_tta > 1:
                aug_imgs, ids = batch
                batch_size, n_tta_actual, c, h, w = aug_imgs.shape
                aug_imgs = aug_imgs.view(batch_size * n_tta_actual, c, h, w).to(device)

                outputs = model(aug_imgs)
                outputs = outputs.view(batch_size, n_tta_actual, -1)

                for i in range(batch_size):
                    img_id = ids[i]
                    probs = torch.nn.functional.softmax(outputs[i], dim=-1).cpu().numpy()
                    preds = outputs[i].argmax(dim=-1).cpu().numpy()

                    all_predictions[img_id] = preds
                    all_probabilities[img_id] = probs
            else:
                imgs, ids = batch
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.nn.functional.softmax(outputs, dim=-1).cpu().numpy()
                preds = outputs.argmax(dim=-1).cpu().numpy()

                for i, img_id in enumerate(ids):
                    all_predictions[img_id] = np.array([preds[i]])
                    all_probabilities[img_id] = np.array([probs[i]])

    return all_predictions, all_probabilities




def save_individual_predictions(predictions_dict, idx_to_class, output_path):
    results = []

    for img_id, preds in predictions_dict.items():
        vote_counts = Counter(preds)
        most_common = vote_counts.most_common(2)

        predicted_class = idx_to_class[most_common[0][0]]
        top_votes = most_common[0][1]

        if len(most_common) > 1:
            second_class = idx_to_class[most_common[1][0]]
            second_votes = most_common[1][1]
        else:
            second_class = ''
            second_votes = 0

        results.append({
            'ID': img_id,
            'labels': predicted_class,
            'top_votes': top_votes,
            'second_class': second_class,
            'second_votes': second_votes
        })

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Saved individual predictions to {output_path}")


def vote_strategy_mean_probs(all_probabilities_list, id_list, idx_to_class):
    results = []

    for img_id in id_list:
        all_probs = []
        for model_probs in all_probabilities_list:
            all_probs.append(model_probs[img_id])

        all_probs_array = np.concatenate(all_probs, axis=0)
        mean_probs = np.mean(all_probs_array, axis=0)
        predicted_idx = np.argmax(mean_probs)
        predicted_class = idx_to_class[predicted_idx]

        results.append({
            'ID': img_id,
            'labels': predicted_class
        })

    return results


def vote_strategy_consensus(all_predictions_list, id_list, idx_to_class):
    results = []

    for img_id in id_list:
        model_votes = []

        for model_preds in all_predictions_list:
            preds = model_preds[img_id]
            vote_counts = Counter(preds)
            majority_pred = vote_counts.most_common(1)[0][0]
            model_votes.append(majority_pred)

        vote_counts = Counter(model_votes)
        most_common = vote_counts.most_common(1)[0]

        if most_common[1] >= 2:
            final_pred_idx = most_common[0]
        else:
            final_pred_idx = model_votes[1] if len(model_votes) > 1 else model_votes[0]

        predicted_class = idx_to_class[final_pred_idx]
        results.append({
            'ID': img_id,
            'labels': predicted_class
        })

    return results


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
                        default="/kaggle/input/wbc-zyx/test_data/test_data", # "raw_data/"
                        help="Root directory containing test images and CSV")
    parser.add_argument("--weight_dir", type=str,
                        default="/kaggle/input/wbc-zyx", # ""
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
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--num_classes", type=int, default=13,
                        help="Number of classes")

    args, unknown = parser.parse_known_args()

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

    models_to_use = args.models
    if 'ensemble' in models_to_use:
        models_to_use = ['resnet50', 'convnext', 'swin']

    all_predictions_list = []
    all_probabilities_list = []

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

        transform = get_tta_transform(model_type) if args.n_tta > 1 else get_val_transform()
        test_dataset = WBCTestDataset(test_csv, test_img_dir, class_names,
                                       transform=transform, n_tta=args.n_tta)
        batch_size = args.batch_size if args.n_tta == 1 else max(1, args.batch_size // args.n_tta)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)

        model = load_model(model_type, weight_path, args.num_classes, device)
        predictions, probabilities = predict_single_model(model, test_loader, args.n_tta, device)

        all_predictions_list.append(predictions)
        all_probabilities_list.append(probabilities)

        individual_output = args.output.replace('.csv', f'_{model_type}.csv')
        save_individual_predictions(predictions, idx_to_class, individual_output)

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    id_list = list(all_predictions_list[0].keys())

    if len(all_predictions_list) == 0:
        print("Error: No models were loaded successfully!")
        return

    print(f"\n{'='*60}")
    print(f"Generating ensemble predictions with dual voting strategies")
    print(f"{'='*60}")

    vote30_results = vote_strategy_mean_probs(all_probabilities_list, id_list, idx_to_class)
    vote30_df = pd.DataFrame(vote30_results)
    vote30_output = args.output.replace('.csv', '_vote30.csv')
    vote30_df.to_csv(vote30_output, index=False)
    print(f"Saved 30-vote mean probability predictions to {vote30_output}")

    consensus_results = vote_strategy_consensus(all_predictions_list, id_list, idx_to_class)
    consensus_df = pd.DataFrame(consensus_results)
    consensus_output = args.output.replace('.csv', '_vote_consensus.csv')
    consensus_df.to_csv(consensus_output, index=False)
    print(f"Saved consensus voting predictions to {consensus_output}")

    print(f"\n{'='*60}")
    print(f"All predictions completed!")
    print(f"Output files:")
    print(f"  - Individual models: {len(models_to_use)} files")
    print(f"  - Ensemble (30-vote mean probs): {vote30_output}")
    print(f"  - Ensemble (consensus): {consensus_output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
