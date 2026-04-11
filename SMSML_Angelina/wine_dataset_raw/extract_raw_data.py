"""
Wine Dataset Raw Data Extractor
==============================

This script extracts the raw wine dataset from sklearn and saves it as CSV files.
Use this to create actual raw data files for the project structure.

Author: Angelina
Date: April 2026
"""

import pandas as pd
from sklearn.datasets import load_wine
import os

def extract_raw_dataset(output_dir="wine_dataset_raw"):
    """
    Extract and save the raw wine dataset as CSV files.
    
    Args:
        output_dir (str): Directory to save raw data files
    """
    print("="*50)
    print("WINE DATASET RAW DATA EXTRACTOR")
    print("="*50)
    
    # Load the wine dataset
    print("Loading wine dataset from sklearn...")
    wine = load_wine()
    
    # Create DataFrame with features
    wine_data = pd.DataFrame(wine.data, columns=wine.feature_names)
    wine_data['target'] = wine.target
    
    # Create target mapping DataFrame
    target_mapping = pd.DataFrame({
        'class_id': range(len(wine.target_names)),
        'class_name': wine.target_names
    })
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save complete raw dataset
    wine_data.to_csv(f"{output_dir}/wine_raw_complete.csv", index=False)
    print(f"✓ Saved: {output_dir}/wine_raw_complete.csv ({wine_data.shape})")
    
    # Save features only (without target)
    features_only = wine_data.drop('target', axis=1)
    features_only.to_csv(f"{output_dir}/wine_features_raw.csv", index=False)
    print(f"✓ Saved: {output_dir}/wine_features_raw.csv ({features_only.shape})")
    
    # Save target only
    target_only = wine_data[['target']]
    target_only.to_csv(f"{output_dir}/wine_target_raw.csv", index=False)
    print(f"✓ Saved: {output_dir}/wine_target_raw.csv ({target_only.shape})")
    
    # Save target mapping
    target_mapping.to_csv(f"{output_dir}/wine_classes.csv", index=False)
    print(f"✓ Saved: {output_dir}/wine_classes.csv ({target_mapping.shape})")
    
    # Save dataset info as text
    with open(f"{output_dir}/dataset_info.txt", 'w') as f:
        f.write("Wine Dataset Information\n")
        f.write("========================\n\n")
        f.write(f"Total samples: {len(wine_data)}\n")
        f.write(f"Total features: {len(wine.feature_names)}\n")
        f.write(f"Total classes: {len(wine.target_names)}\n\n")
        
        f.write("Feature Names:\n")
        for i, feature in enumerate(wine.feature_names, 1):
            f.write(f"{i:2d}. {feature}\n")
        
        f.write(f"\nClass Names:\n")
        for i, class_name in enumerate(wine.target_names):
            f.write(f"Class {i}: {class_name}\n")
        
        f.write(f"\nClass Distribution:\n")
        class_counts = wine_data['target'].value_counts().sort_index()
        for class_id, count in class_counts.items():
            f.write(f"Class {class_id}: {count} samples\n")
    
    print(f"✓ Saved: {output_dir}/dataset_info.txt")
    
    print("="*50)
    print("RAW DATA EXTRACTION COMPLETED!")
    print("="*50)
    print(f"Files saved in: {os.path.abspath(output_dir)}")
    print("\nFiles created:")
    print("- wine_raw_complete.csv (complete dataset)")
    print("- wine_features_raw.csv (features only)")  
    print("- wine_target_raw.csv (target only)")
    print("- wine_classes.csv (class mapping)")
    print("- dataset_info.txt (dataset information)")

if __name__ == "__main__":
    # Extract raw dataset
    extract_raw_dataset()