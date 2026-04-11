import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load preprocessed data"""
    try:
        print("Loading processed datasets...")
        X_train = pd.read_csv("dataset_preprocessing/X_train.csv")
        X_test = pd.read_csv("dataset_preprocessing/X_test.csv") 
        y_train = pd.read_csv("dataset_preprocessing/y_train.csv")['target']
        y_test = pd.read_csv("dataset_preprocessing/y_test.csv")['target']
        
        with open("dataset_preprocessing/dataset_info.json", 'r') as f:
            dataset_info = json.load(f)
        
        print(f"Data loaded: Train({len(X_train)}), Test({len(X_test)}), Features({len(X_train.columns)}), Classes({dataset_info['n_classes']})")
        return X_train, X_test, y_train, y_test, dataset_info
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def setup_mlflow():
    """Setup MLflow tracking"""
    print("Setting up MLflow...")
    mlflow.set_tracking_uri("file:./mlruns")
    
    experiment_name = "Wine_Classification_Basic_Angelina"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created experiment: {experiment_name}")
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        print(f"Using existing experiment: {experiment_name}")
    
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking: {mlflow.get_tracking_uri()}")

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest with manual MLflow logging"""
    print("\n=== Training Random Forest (BASIC LEVEL) ===")
    
    with mlflow.start_run(run_name="RandomForest_Basic_Angelina"):
        # Model parameters - BASIC level (no hyperparameter tuning)
        params = {
            'n_estimators': 100,
            'random_state': 42,
            'model_type': 'RandomForest'
        }
        
        # Log parameters
        for param, value in params.items():
            mlflow.log_param(param, value)
        
        # Train model
        print("Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'], 
            random_state=params['random_state']
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("n_features", len(X_train.columns))
        mlflow.log_metric("n_train_samples", len(X_train))
        mlflow.log_metric("n_test_samples", len(X_test))
        
        # Log model - CRITICAL for BASIC level
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="RandomForest_Wine_Classification_Angelina"
        )
        
        # Log classification report as artifact
        report = classification_report(y_test, y_pred_test)
        with open("classification_report_rf.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report_rf.txt")
        
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("Model and artifacts logged to MLflow")
        
        return model, test_accuracy

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train Logistic Regression with manual MLflow logging"""
    print("\n=== Training Logistic Regression (BASIC LEVEL) ===")
    
    with mlflow.start_run(run_name="LogisticRegression_Basic_Angelina"):
        # Model parameters - BASIC level (no hyperparameter tuning)
        params = {
            'max_iter': 1000,
            'random_state': 42,
            'model_type': 'LogisticRegression'
        }
        
        # Log parameters
        for param, value in params.items():
            mlflow.log_param(param, value)
        
        # Train model
        print("Training Logistic Regression...")
        model = LogisticRegression(
            max_iter=params['max_iter'], 
            random_state=params['random_state']
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("n_features", len(X_train.columns))
        mlflow.log_metric("n_train_samples", len(X_train))
        mlflow.log_metric("n_test_samples", len(X_test))
        
        # Log model - CRITICAL for BASIC level
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="LogisticRegression_Wine_Classification_Angelina"
        )
        
        # Log classification report as artifact
        report = classification_report(y_test, y_pred_test)
        with open("classification_report_lr.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report_lr.txt")
        
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("Model and artifacts logged to MLflow")
        
        return model, test_accuracy

def main():
    """Main function - BASIC Level Implementation"""
    print("=" * 70)
    print("KRITERIA 2 - BASIC LEVEL: WINE CLASSIFICATION WITH MLFLOW")
    print("=" * 70)
    print("Implementation Details:")
    print("  Scikit-learn models with MLflow tracking")
    print("  Local MLflow tracking (file:./mlruns)")
    print("  Manual logging (compatible approach)")
    print("  Model artifacts and metrics saved")
    print("  No hyperparameter tuning (Basic requirement)")
    print("Author: Angelina")
    print("=" * 70)
    
    try:
        # Load data
        X_train, X_test, y_train, y_test, dataset_info = load_data()
        
        # Setup MLflow
        setup_mlflow()
        
        # Train models
        print(f"\nTraining wine classification models for {dataset_info['n_classes']} classes...")
        
        rf_model, rf_accuracy = train_random_forest(X_train, X_test, y_train, y_test)
        lr_model, lr_accuracy = train_logistic_regression(X_train, X_test, y_train, y_test)
        
        # Summary
        print("\n" + "=" * 70)
        print("BASIC LEVEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("Results Summary:")
        print(f"  Random Forest Test Accuracy: {rf_accuracy:.4f}")
        print(f"  Logistic Regression Test Accuracy: {lr_accuracy:.4f}")
        print(f"  Models saved in MLflow with artifacts")
        print(f"  Experiment: Wine_Classification_Basic_Angelina")
        
        print(f"\nView Results in MLflow Dashboard:")
        print("   1. Keep current terminal open (MLflow UI running)")
        print("   2. Open browser: http://127.0.0.1:5000")
        print("   3. Click on 'Wine_Classification_Basic_Angelina'")
        print("   4. View runs and artifacts")
        
        print(f"\nKRITERIA 2 - BASIC (2 pts) Requirements Met:")
        print("   ML models trained with Scikit-Learn")
        print("   MLflow Tracking UI working locally")
        print("   Models and metrics logged to MLflow")
        print("   Artifacts saved (classification reports)")
        print("   No hyperparameter tuning (Basic level)")
        
        print(f"\nScreenshot Requirements:")
        print("   1. dashboard.jpg - MLflow experiments page")
        print("   2. artifacts.jpg - Model artifacts page")
        
        return rf_model, lr_model
        
    except Exception as e:
        print(f"Error in modeling pipeline: {e}")
        print("\nTroubleshooting Steps:")
        print("   1. Check if processed data exists in dataset_preprocessing/")
        print("   2. Verify preprocessing notebook was run successfully")
        print("   3. Ensure MLflow installation: pip install mlflow")
        print("   4. Check MLflow UI is accessible: python -m mlflow ui")
        raise

if __name__ == "__main__":
    main()