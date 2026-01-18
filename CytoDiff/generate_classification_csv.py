#!/usr/bin/env python3
"""
Generate proper CSV metadata for custom WBC dataset compatible with classification scripts.
This creates k-fold cross-validation splits matching the expected format.
"""

import os
import pandas as pd
import random
from pathlib import Path

# Define the correct class mapping to match the classification script expectations
CLASS_MAPPING = {
    'BA': 'Basophil',           # Instead of 'basophil' 
    'BL': 'Blast',              # Instead of 'blast' (this might need to map to existing class)
    'BNE': 'Band Neutrophil',   # Instead of 'neutrophil_band'
    'EO': 'Eosinophil',         # Instead of 'eosinophil'
    'LY': 'Typical Lymphocyte', # Instead of 'lymphocyte' 
    'MMY': 'Metamyelocyte',     # Instead of 'metamyelocyte'
    'MO': 'Monocyte',           # Instead of 'monocyte'
    'MY': 'Myelocyte',          # Instead of 'myelocyte'
    'PC': 'Plasma',             # This class doesn't exist in original - might need mapping
    'PLY': 'Prolymphocyte',     # This class doesn't exist in original - might need mapping  
    'PMY': 'Promyelocyte',      # Instead of 'promyelocyte'
    'SNE': 'Segmented Neutrophil', # Instead of 'neutrophil_segmented'
    'VLY': 'Atypical Lymphocyte',   # Instead of 'lymphocyte_variant'
}

# Original Matek classes from the script
ORIGINAL_LABELS = [
    'Basophil', 'Erythroblast', 'Eosinophil', 'Smudge cell', 
    'Atypical Lymphocyte', 'Typical Lymphocyte', 'Metamyelocyte', 
    'Monoblast', 'Monocyte', 'Myelocyte', 'Myeloblast', 
    'Band Neutrophil', 'Segmented Neutrophil', 'Promyelocyte Bilobed', 
    'Promyelocyte'
]

def map_custom_to_matek_classes():
    """Map custom WBC classes to existing Matek classes."""
    # Classes that need special handling
    class_mapping_decisions = {
        'PC': 'Monoblast',  # Map plasma cells to monoblasts (similar precursor cells)
        'PLY': 'Typical Lymphocyte',  # Map prolymphocyte to lymphocyte
        'BL': 'Myeloblast',  # Map blast to myeloblast
    }
    
    # Update the mapping
    updated_mapping = CLASS_MAPPING.copy()
    updated_mapping.update(class_mapping_decisions)
    
    return updated_mapping

def create_kfold_metadata_csv():
    """Create CSV metadata file with k-fold splits compatible with classification script."""
    
    # Source dataset directory
    source_dir = "/home/xiaos7/projects/CytoDiff/datasets/custom_wbc/images"
    
    # Get the proper class mapping
    final_class_mapping = map_custom_to_matek_classes()
    
    data_list = []
    
    # Process each class directory
    for class_dir in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_dir)
        
        if os.path.isdir(class_path):
            # Get the corresponding abbreviation from directory name
            abbr = None
            for k, v in CLASS_MAPPING.items():
                if v.lower().replace(' ', '_') == class_dir.replace('_', ' ').lower() or \
                   v.lower().replace(' ', '') == class_dir.replace('_', '').lower():
                    abbr = k
                    break
            
            if not abbr:
                # Try to infer from common patterns
                abbr_map = {
                    'basophil': 'BA', 'blast': 'BL', 'neutrophil_band': 'BNE',
                    'eosinophil': 'EO', 'lymphocyte': 'LY', 'metamyelocyte': 'MMY',
                    'monocyte': 'MO', 'myelocyte': 'MY', 'plasma': 'PC',
                    'prolymphocyte': 'PLY', 'promyelocyte': 'PMY',
                    'neutrophil_segmented': 'SNE', 'lymphocyte_variant': 'VLY'
                }
                abbr = abbr_map.get(class_dir, 'UNKNOWN')
            
            if abbr in final_class_mapping:
                matek_label = final_class_mapping[abbr]
                
                # Get all image files in this class
                image_files = [f for f in os.listdir(class_path) if f.endswith('.png')]
                
                # Create k-fold splits (5 folds)
                random.shuffle(image_files)
                n_folds = 5
                fold_size = len(image_files) // n_folds
                
                for i, image_file in enumerate(image_files):
                    image_path = os.path.join(class_path, image_file)
                    
                    # Determine which fold this image belongs to for testing
                    test_fold = i // max(1, fold_size) if fold_size > 0 else i % n_folds
                    test_fold = min(test_fold, n_folds - 1)  # Ensure it doesn't exceed fold count
                    
                    # Create fold assignments
                    fold_assignments = []
                    for fold_idx in range(n_folds):
                        if fold_idx == test_fold:
                            fold_assignments.append('test')
                        elif fold_idx == (test_fold + 1) % n_folds:
                            fold_assignments.append('val')
                        else:
                            fold_assignments.append('train')
                    
                    # Add to data
                    row = [image_path, matek_label, 'matek'] + fold_assignments
                    data_list.append(row)
    
    # Create DataFrame
    columns = ['image', 'label', 'dataset', 'kfold0', 'kfold1', 'kfold2', 'kfold3', 'kfold4']
    df = pd.DataFrame(data_list, columns=columns)
    
    # Save CSV files
    output_dir = "/home/xiaos7/projects/CytoDiff/classification/csv_files/custom_wbc"
    os.makedirs(output_dir, exist_ok=True)
    
    # Main metadata file
    metadata_file = os.path.join(output_dir, "matek_metadata.csv")
    df.to_csv(metadata_file, index=False)
    
    # Base metadata file (without k-folds)
    base_df = df[['image', 'label', 'dataset']].copy()
    base_file = os.path.join(output_dir, "matek_metadata_base.csv")
    base_df.to_csv(base_file, index=False)
    
    print(f"Created metadata files in {output_dir}")
    print(f"Total samples: {len(df)}")
    print("\nClass distribution:")
    print(df['label'].value_counts().sort_index())
    
    # Also create a symbolic link or copy to expected location for classification script
    classification_dir = "/home/xiaos7/projects/CytoDiff/datasets/custom_wbc_classification"
    os.makedirs(classification_dir, exist_ok=True)
    
    import shutil
    shutil.copy2(metadata_file, os.path.join(classification_dir, "matek_metadata.csv"))
    print(f"\nCopied metadata to classification directory: {classification_dir}")
    
    return df

def update_dataset_image_size():
    """Update the dataset_image_size in dataset_wbc.py to include custom_wbc"""
    print("\nNote: You'll need to update dataset_wbc.py to include:")
    print('dataset_image_size = {')
    print('    "Ace_20":250,')
    print('    "matek":345,') 
    print('    "custom_wbc":345,  # Add this line')
    print('    "MLL_20":288,')
    print('    "BMC_22":250,')
    print('}')

if __name__ == "__main__":
    print("Creating proper CSV metadata for custom WBC dataset...")
    print("=" * 60)
    
    # Set random seed for reproducible splits
    random.seed(42)
    
    df = create_kfold_metadata_csv()
    update_dataset_image_size()
    
    print("\n" + "=" * 60)
    print("CSV generation completed!")
    print("\nNext steps:")
    print("1. Update dataset_wbc.py with custom_wbc image size")
    print("2. Update dataroot path in classification scripts to point to custom_wbc_classification")
    print("3. Use dataset_selection='matek' in classification scripts")