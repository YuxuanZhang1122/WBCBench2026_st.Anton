import os
import argparse
import torch
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

from src.models.classifiers import get_classifier
from src.utils.config_loader import load_config


def load_model_from_checkpoint(checkpoint_path, feature_type="cls", device="mps"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config_dict = checkpoint['config']

    # Infer input_dim from feature_type
    if feature_type == "cls" or feature_type == "avg_patch":
        input_dim = 768
    elif feature_type == "concat":
        input_dim = 1536
    else:
        input_dim = 768  # default

    try:
        classifier_head = config_dict['training'].classifier_head
        mlp_hidden_dims = config_dict['training'].mlp_hidden_dims
        dropout = config_dict['training'].dropout
        num_classes = config_dict['data'].num_classes
    except:
        classifier_head = "linear"
        mlp_hidden_dims = [256]
        dropout = 0.3
        num_classes = 13

    model = get_classifier(
        classifier_head,
        input_dim,
        num_classes,
        mlp_hidden_dims,
        dropout
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, config_dict


def extract_test_features(
    test_dir,
    model_path,
    dinobloom_checkpoint,
    resolution_strategy="resize",
    device="mps"
):
    from src.models.dinobloom import load_dinobloom
    from src.data.dataset import get_dataloader

    print("Loading DinoBloom for test feature extraction...")
    dinobloom = load_dinobloom(
        dinobloom_checkpoint,
        resolution_strategy,
        freeze=True,
        device=device
    )

    print("Loading test images...")
    test_loader, test_dataset = get_dataloader(
        root_dir=test_dir,
        split="test",
        batch_size=32,
        image_size=368,
        num_workers=4,
        shuffle=False,
        return_path=True
    )

    print(f"Extracting features from {len(test_dataset)} test images...")
    all_cls = []
    all_avg_patch = []
    all_concat = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc="Extracting test features"):
            images = images.to(device)

            cls_token, avg_patch, concat = dinobloom(images)

            all_cls.append(cls_token.cpu().numpy())
            all_avg_patch.append(avg_patch.cpu().numpy())
            all_concat.append(concat.cpu().numpy())
            all_paths.extend(paths)

    features = {
        'cls': np.concatenate(all_cls, axis=0),
        'avg_patch': np.concatenate(all_avg_patch, axis=0),
        'concat': np.concatenate(all_concat, axis=0),
        'paths': np.array(all_paths)
    }

    return features


def predict(model, features, batch_size=256, device="mps"):
    model.eval()
    all_preds = []
    all_probs = []

    num_samples = features.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Generating predictions"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            batch = torch.FloatTensor(features[start_idx:end_idx]).to(device)
            outputs = model(batch)

            probs = torch.softmax(outputs, dim=1)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_probs)


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for test set")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--feature_type", type=str, choices=["cls", "avg_patch", "concat"], required=True)
    parser.add_argument("--test_features", type=str, default=None, help="Path to pre-extracted test features (HDF5)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")

    args = parser.parse_args()

    exp_dir = os.path.join("experiments", args.exp_name)
    checkpoint_path = os.path.join(exp_dir, "checkpoints", "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading model from {checkpoint_path}...")
    model, config_dict = load_model_from_checkpoint(checkpoint_path, feature_type=args.feature_type)

    if args.test_features and os.path.exists(args.test_features):
        print(f"Loading pre-extracted test features from {args.test_features}...")
        with h5py.File(args.test_features, 'r') as f:
            features = f[args.feature_type][:]
            paths = [p.decode('utf-8') if isinstance(p, bytes) else p for p in f['paths'][:]]
    else:
        print("Test features not provided. You need to extract them first.")
        print("Run: python3 extract_features.py --splits test --resolution_strategy resize")
        return

    print(f"\nGenerating predictions for {len(features)} test samples...")
    predictions, probabilities = predict(model, features, device=config_dict.get('device', 'mps'))

    class_names = ['BA', 'BL', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PC', 'PLY', 'PMY', 'SNE', 'VLY']
    pred_labels = [class_names[p] for p in predictions]

    df = pd.DataFrame({
        'ID': paths,
        'label': pred_labels
    })

    output_path = args.output or os.path.join(exp_dir, "test_predictions.csv")
    df.to_csv(output_path, index=False)

    print(f"\nPredictions saved to {output_path}")
    print(f"Total predictions: {len(df)}")
    print(f"\nPrediction distribution:")
    print(df['label'].value_counts().sort_index())


if __name__ == "__main__":
    main()
