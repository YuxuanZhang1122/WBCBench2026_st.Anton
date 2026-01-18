#!/usr/bin/env python3
"""
Script to set up the custom WBC dataset for CytoDiff training.
This script creates the required metadata files and organizes the dataset structure.
"""

import os
import shutil
from pathlib import Path

# Define the class mapping from abbreviations to full names
CLASS_MAPPING = {
    'BA': 'basophil',
    'BL': 'blast', 
    'BNE': 'neutrophil_band',
    'EO': 'eosinophil',
    'LY': 'lymphocyte',
    'MMY': 'metamyelocyte',
    'MO': 'monocyte',
    'MY': 'myelocyte',
    'PC': 'plasma',
    'PLY': 'prolymphocyte',
    'PMY': 'promyelocyte',
    'SNE': 'neutrophil_segmented',
    'VLY': 'lymphocyte_variant'
}

# Reverse mapping for generating prompts
REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

# Define paths
SOURCE_DIR = "/home/xiaos7/data_areas/lmr-ihb-imaging/xiaos7/public_datasets/wbc_class/class_samples_for_cytodiff"
PROJECT_DIR = "/home/xiaos7/projects/CytoDiff"
DATASET_DIR = os.path.join(PROJECT_DIR, "datasets", "custom_wbc")
METADATA_DIR = os.path.join(DATASET_DIR, "metadata")

def create_directory_structure():
    """Create the required directory structure."""
    print("Creating directory structure...")
    
    # Create main dataset directory
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)
    
    # Create subdirectories for each class
    for class_name in CLASS_MAPPING.values():
        class_dir = os.path.join(DATASET_DIR, "images", class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    print(f"Created directory structure at {DATASET_DIR}")

def organize_images():
    """Copy and organize images by class."""
    print("Organizing images by class...")
    
    source_path = Path(SOURCE_DIR)
    
    for image_file in source_path.glob("*.png"):
        # Extract class abbreviation from filename
        class_abbr = image_file.name.split('_')[0]
        
        if class_abbr in CLASS_MAPPING:
            class_name = CLASS_MAPPING[class_abbr]
            dest_dir = os.path.join(DATASET_DIR, "images", class_name)
            dest_path = os.path.join(dest_dir, image_file.name)
            
            # Copy the image
            shutil.copy2(image_file, dest_path)
            print(f"Copied {image_file.name} to {class_name}/")
        else:
            print(f"Warning: Unknown class abbreviation '{class_abbr}' for file {image_file.name}")

def generate_metadata_files():
    """Generate image_ids.txt and class_labels.txt files."""
    print("Generating metadata files...")
    
    image_ids = []
    class_labels = []
    
    # Process each class directory
    for class_name in sorted(CLASS_MAPPING.values()):
        class_dir = os.path.join(DATASET_DIR, "images", class_name)
        
        if os.path.exists(class_dir):
            image_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.png')])
            
            for image_file in image_files:
                # Remove extension for image ID
                image_id = os.path.splitext(image_file)[0]
                image_ids.append(f"{class_name}/{image_id}")
                class_labels.append(class_name)
    
    # Write image_ids.txt
    image_ids_file = os.path.join(METADATA_DIR, "image_ids.txt")
    with open(image_ids_file, 'w') as f:
        f.write('\n'.join(image_ids))
    print(f"Created {image_ids_file} with {len(image_ids)} entries")
    
    # Write class_labels.txt
    class_labels_file = os.path.join(METADATA_DIR, "class_labels.txt")
    with open(class_labels_file, 'w') as f:
        f.write('\n'.join(class_labels))
    print(f"Created {class_labels_file} with {len(class_labels)} entries")
    
    # Create class_names.txt (unique class names)
    unique_classes = sorted(list(set(class_labels)))
    class_names_file = os.path.join(METADATA_DIR, "class_names.txt")
    with open(class_names_file, 'w') as f:
        f.write('\n'.join(unique_classes))
    print(f"Created {class_names_file} with {len(unique_classes)} unique classes")

def create_dataset_config():
    """Create a dataset configuration file."""
    config_content = f"""# Custom WBC Dataset Configuration
dataset_name: custom_wbc
num_classes: {len(CLASS_MAPPING)}
dataset_path: {DATASET_DIR}
metadata_path: {METADATA_DIR}

# Class mapping
classes:
"""
    
    for i, (abbr, class_name) in enumerate(CLASS_MAPPING.items()):
        config_content += f"  {i}: {class_name}  # {abbr}\n"
    
    config_file = os.path.join(METADATA_DIR, "dataset_config.yaml")
    with open(config_file, 'w') as f:
        f.write(config_content)
    print(f"Created {config_file}")

def generate_statistics():
    """Generate and display dataset statistics."""
    print("\nDataset Statistics:")
    print("=" * 50)
    
    total_images = 0
    for class_name in CLASS_MAPPING.values():
        class_dir = os.path.join(DATASET_DIR, "images", class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) if f.endswith('.png')])
            print(f"{class_name:20}: {count:3d} images")
            total_images += count
    
    print("-" * 30)
    print(f"{'Total':20}: {total_images:3d} images")
    print(f"Number of classes: {len(CLASS_MAPPING)}")

def main():
    """Main function to set up the dataset."""
    print("Setting up Custom WBC Dataset for CytoDiff...")
    print("=" * 60)
    
    # Check if source directory exists
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory {SOURCE_DIR} does not exist!")
        return
    
    # Create directory structure
    create_directory_structure()
    
    # Organize images
    organize_images()
    
    # Generate metadata files
    generate_metadata_files()
    
    # Create dataset config
    create_dataset_config()
    
    # Display statistics
    generate_statistics()
    
    print("\nDataset setup completed successfully!")
    print(f"Dataset location: {DATASET_DIR}")
    print(f"Metadata location: {METADATA_DIR}")
    
    print(f"\nNext steps:")
    print(f"1. Update training/util_data.py to include your custom dataset")
    print(f"2. Modify local.yaml configuration files")
    print(f"3. Update training scripts to use your dataset")

if __name__ == "__main__":
    main()