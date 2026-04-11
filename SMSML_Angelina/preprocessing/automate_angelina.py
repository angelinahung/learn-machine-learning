"""
Automated Data Preprocessing Pipeline for Wine Dataset
=====================================================

This script provides automated functions to preprocess the wine dataset
following the same steps as the manual experimentation notebook.

Author: Angelina
Date: April 2026
"""

import pandas as pd
import numpy as np
import os
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class WineDataPreprocessor:
    """
    Automated preprocessing pipeline for wine classification dataset.
    """
    
    def __init__(self):
        """Initialize the preprocessor with default parameters."""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
    def load_data(self):
        """
        Load the wine dataset from sklearn.
        
        Returns:
            pd.DataFrame: Raw wine dataset with features and target
        """
        print("=== LOADING WINE DATASET ===")
        wine = load_wine()
        data = pd.DataFrame(wine.data, columns=wine.feature_names)
        data['target'] = wine.target
        
        print(f"Dataset loaded successfully!")
        print(f"Dataset shape: {data.shape}")
        print(f"Target classes: {wine.target_names}")
        
        return data, wine.target_names
    
    def handle_missing_values(self, data):
        """
        Handle missing values in the dataset using median imputation.
        
        Args:
            data (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        print("=== HANDLING MISSING VALUES ===")
        data_processed = data.copy()
        
        if data_processed.isnull().sum().sum() > 0:
            # Fill numeric columns with median
            numeric_columns = data_processed.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if data_processed[col].isnull().sum() > 0:
                    data_processed[col].fillna(data_processed[col].median(), inplace=True)
            print("Missing values handled using median imputation")
        else:
            print("No missing values found - no imputation needed")
        
        print(f"Final missing values: {data_processed.isnull().sum().sum()}")
        return data_processed
    
    def encode_features(self, data):
        """
        Encode categorical features (demonstration with wine dataset).
        
        Args:
            data (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with encoded features
        """
        print("=== ENCODING FEATURES ===")
        
        # For wine dataset, target is already numeric, but we demonstrate encoding
        print("Target values before encoding:", data['target'].unique())
        
        # Keep original target as it's already properly encoded
        print("Target is already numeric - no encoding needed")
        print("Feature columns (all numeric, no categorical encoding needed)")
        
        feature_columns = [col for col in data.columns if col != 'target']
        print(f"Number of features: {len(feature_columns)}")
        
        return data
    
    def scale_features(self, X_train, X_test=None):
        """
        Scale features using StandardScaler.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame, optional): Test features
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled) or just X_train_scaled
        """
        print("=== FEATURE SCALING ===")
        
        # Fit scaler on training data
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        print("Feature scaling completed using StandardScaler")
        print(f"Training features shape: {X_train_scaled.shape}")
        
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            print(f"Test features shape: {X_test_scaled.shape}")
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and test sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            test_size (float): Proportion of test set
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("=== DATASET SPLITTING ===")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        print(f"Training set shape: X_train {X_train.shape}, y_train {y_train.shape}")
        print(f"Test set shape: X_test {X_test.shape}, y_test {y_test.shape}")
        print(f"Training set ratio: {len(X_train) / len(X):.2f}")
        print(f"Test set ratio: {len(X_test) / len(X):.2f}")
        
        # Display target distribution
        print("\nTarget distribution in training set:")
        print(y_train.value_counts().sort_index())
        print("\nTarget distribution in test set:")
        print(y_test.value_counts().sort_index())
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train, X_test, y_train, y_test, 
                           processed_data=None, output_dir="wine_dataset_preprocessing"):
        """
        Save processed datasets to files.
        
        Args:
            X_train, X_test, y_train, y_test: Split datasets
            processed_data (pd.DataFrame, optional): Complete processed dataset
            output_dir (str): Output directory path
        """
        print("=== SAVING PROCESSED DATA ===")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training and test sets
        X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
        X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv", index=False, header=['target'])
        y_test.to_csv(f"{output_dir}/y_test.csv", index=False, header=['target'])
        
        # Save complete processed dataset if provided
        if processed_data is not None:
            processed_data.to_csv(f"{output_dir}/wine_processed_complete.csv", index=False)
        
        # Save scaler for future use
        import joblib
        joblib.dump(self.scaler, f"{output_dir}/scaler.joblib")
        
        print(f"All processed datasets saved to {output_dir}/")
        print("Files created:")
        print("- X_train.csv, X_test.csv")
        print("- y_train.csv, y_test.csv") 
        print("- wine_processed_complete.csv")
        print("- scaler.joblib")
    
    def preprocess_pipeline(self, output_dir="wine_dataset_preprocessing"):
        """
        Complete preprocessing pipeline that returns ready-to-train data.
        
        Args:
            output_dir (str): Directory to save processed data
            
        Returns:
            dict: Dictionary containing all processed datasets and metadata
        """
        print("="*60)
        print("AUTOMATED WINE DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # Step 1: Load data
        data, target_names = self.load_data()
        
        # Step 2: Handle missing values
        data_processed = self.handle_missing_values(data)
        
        # Step 3: Encode features
        data_encoded = self.encode_features(data_processed)
        
        # Step 4: Prepare features and target
        X = data_encoded.drop('target', axis=1)
        y = data_encoded['target']
        
        # Step 5: Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Step 6: Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Step 7: Save processed data
        self.save_processed_data(
            X_train_scaled, X_test_scaled, y_train, y_test,
            processed_data=data_encoded, output_dir=output_dir
        )
        
        # Mark as fitted
        self.is_fitted = True
        
        print("="*60)
        print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'target_names': target_names,
            'scaler': self.scaler,
            'processed_data': data_encoded
        }


def main():
    """
    Main function to run the automated preprocessing pipeline.
    """
    # Initialize preprocessor
    preprocessor = WineDataPreprocessor()
    
    # Run complete pipeline
    results = preprocessor.preprocess_pipeline(
        output_dir="wine_dataset_preprocessing"
    )
    
    # Display final results summary
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print(f"Training samples: {len(results['X_train'])}")
    print(f"Test samples: {len(results['X_test'])}")
    print(f"Number of features: {results['X_train'].shape[1]}")
    print(f"Number of classes: {len(results['target_names'])}")
    print(f"Classes: {results['target_names']}")
    print(f"Data ready for model training: {'YES' if preprocessor.is_fitted else 'NO'}")
    
    return results


if __name__ == "__main__":
    # Run the automated preprocessing
    processed_data = main()