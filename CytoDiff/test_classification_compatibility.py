#!/usr/bin/env python3
"""
Test script to verify that the custom WBC dataset works with the classification pipeline.
"""

import sys
import os
sys.path.append('/home/xiaos7/projects/CytoDiff/classification')

import pandas as pd
import torch
from dataset_wbc import DatasetMarr, labels_map

def test_custom_dataset():
    """Test loading the custom WBC dataset."""
    print("Testing Custom WBC Dataset Loading")
    print("=" * 50)
    
    # Dataset parameters
    dataroot = '/home/xiaos7/projects/CytoDiff/datasets/custom_wbc_classification'
    dataset_selection = 'matek'  # Use 'matek' as expected by the script
    fold = 0
    
    # Check if CSV exists
    csv_path = os.path.join(dataroot, 'matek_metadata.csv')
    print(f"Checking CSV file: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found at {csv_path}")
        return False
    
    # Read and inspect CSV
    df = pd.read_csv(csv_path)
    print(f"✅ CSV loaded successfully")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Shape: {df.shape}")
    print(f"   Datasets: {df['dataset'].unique()}")
    print(f"   Classes: {df['label'].unique()}")
    print("\nClass distribution:")
    print(df['label'].value_counts())
    
    # Test dataset loading
    try:
        print(f"\n🧪 Testing DatasetMarr loading...")
        dataset = DatasetMarr(
            dataroot=dataroot,
            dataset_selection=dataset_selection,
            labels_map=labels_map,
            fold=fold,
            transform=None,
            state='train',
            is_hsv=False,
            is_hed=False
        )
        
        print(f"✅ Dataset loaded successfully")
        print(f"   Number of samples: {len(dataset)}")
        
        # Test loading a sample
        if len(dataset) > 0:
            print(f"\n🔍 Testing sample loading...")
            image, label = dataset[0]
            print(f"✅ Sample loaded successfully")
            print(f"   Image shape: {image.shape if hasattr(image, 'shape') else 'PIL Image'}")
            print(f"   Label: {label}")
            print(f"   Label type: {type(label)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def test_image_paths():
    """Test if image paths in CSV are valid."""
    print(f"\n🔍 Testing image path validity...")
    
    csv_path = '/home/xiaos7/projects/CytoDiff/datasets/custom_wbc_classification/matek_metadata.csv'
    df = pd.read_csv(csv_path)
    
    # Check first few image paths
    valid_paths = 0
    invalid_paths = 0
    
    for i, row in df.head(10).iterrows():
        image_path = row['image']
        if os.path.exists(image_path):
            valid_paths += 1
        else:
            invalid_paths += 1
            print(f"   ⚠️  Invalid path: {image_path}")
    
    print(f"   Valid paths: {valid_paths}/{valid_paths + invalid_paths}")
    
    if valid_paths > 0:
        print("✅ Image paths are accessible")
        return True
    else:
        print("❌ No valid image paths found")
        return False

if __name__ == "__main__":
    print("Custom WBC Dataset Compatibility Test")
    print("=" * 60)
    
    success = True
    
    # Test 1: CSV and dataset loading
    success &= test_custom_dataset()
    
    # Test 2: Image path validity
    success &= test_image_paths()
    
    print(f"\n{'=' * 60}")
    if success:
        print("🎉 ALL TESTS PASSED! Your dataset is compatible with the classification pipeline.")
        print("\nYou can now run classification experiments with:")
        print(f"   dataroot = '/home/xiaos7/projects/CytoDiff/datasets/custom_wbc_classification'")
        print(f"   dataset_selection = 'matek'")
    else:
        print("❌ Some tests failed. Please check the issues above.")