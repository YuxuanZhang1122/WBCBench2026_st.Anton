import os
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
from tqdm import tqdm
import argparse


def generate_noised_validation(data_dir, output_dir, num_variations=1, seed=42):
    np.random.seed(seed)

    val_csv = os.path.join(data_dir, 'val.csv')
    df = pd.read_csv(val_csv)
    df = df.dropna(subset=['labels'])

    noised_val_dir = os.path.join(output_dir, 'noised_val')
    os.makedirs(noised_val_dir, exist_ok=True)

    albu_transform = A.Compose([
        A.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.0), p=1.0),
        A.GaussNoise(std_range=(0.15, 0.25), p=1.0),
        A.ISONoise(color_shift=(0.1, 0.5), intensity=(0.3, 0.6), p=1.0),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 2.0), p=1.0),
    ])

    val_dir = os.path.join(data_dir, 'val')
    new_rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating noised validation"):
        img_name = row['ID']
        label = row['labels']
        img_path = os.path.join(val_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        img_np = np.array(image)

        for var_idx in range(num_variations):
            augmented = albu_transform(image=img_np)
            noised_img_np = augmented['image']

            noised_img = Image.fromarray(noised_img_np)

            if num_variations == 1:
                output_name = img_name
            else:
                base_name, ext = os.path.splitext(img_name)
                output_name = f"{base_name}_noise{var_idx}{ext}"

            output_path = os.path.join(noised_val_dir, output_name)
            noised_img.save(output_path, quality=95)

            new_rows.append({'ID': output_name, 'labels': label})

    noised_df = pd.DataFrame(new_rows)
    noised_csv_path = os.path.join(output_dir, 'noised_val.csv')
    noised_df.to_csv(noised_csv_path, index=False)

    print(f"\nGenerated {len(noised_df)} noised validation images")
    print(f"Saved to: {noised_val_dir}")
    print(f"CSV saved to: {noised_csv_path}")
    print(f"\nClass distribution:")
    print(noised_df['labels'].value_counts().sort_index())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate noised validation dataset for robust evaluation")
    parser.add_argument("--data_dir", type=str, default="raw_data", help="Directory containing val.csv and val/ folder")
    parser.add_argument("--output_dir", type=str, default="raw_data", help="Output directory for noised_val/ and noised_val.csv")
    parser.add_argument("--num_variations", type=int, default=1, help="Number of noised variations per image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    generate_noised_validation(args.data_dir, args.output_dir, args.num_variations, args.seed)
