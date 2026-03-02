"""
Email Spam Classifier - Main Pipeline
=====================================

A complete machine learning pipeline for email spam detection using
Naive Bayes and Logistic Regression classifiers.

Author: ML Engineer
Version: 1.0
"""

import sys
import os

# Add current directory and src directory to Python path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.loader import load_data
from src.data.preprocessor import preprocess
from src.features.vectorizer import vectorize
from src.models.trainer import train_models
from src.evaluation.evaluator import evaluate
from src.visualization.plotter import plot_all


def interactive_classification():
    """Interactive email classification using text_classifier module."""
    try:
        from text_classifier import interactive_text_classification as text_classify
        return text_classify()
    except ImportError:
        print("Error: text_classifier.py not found. Please ensure the file exists.")
        print("You can still train models using option 1.")
        return True


def main():
    """
    Main pipeline function that runs the complete email spam classification process.
    
    Pipeline Steps:
    1. Load dataset
    2. Preprocess data
    3. Vectorize text
    4. Train models
    5. Evaluate models
    6. Generate visualizations
    """
    print("=" * 80)
    print("EMAIL SPAM CLASSIFIER - MACHINE LEARNING PIPELINE")
    print("=" * 80)
    
    # Check if running in interactive mode
    try:
        choice = input("Choose an option:\n1. Train Models (Full Pipeline)\n2. Classify Email (Interactive Mode)\n3. Exit\n\nEnter your choice (1, 2, or 3): ").strip()
    except (EOFError, KeyboardInterrupt):
        # Non-interactive mode, run full pipeline
        print("Non-interactive mode detected. Running full pipeline...")
        choice = "1"
    
    if choice == "2":
        # Interactive classification mode - use separate module
        return interactive_classification()
    elif choice == "3":
        print("Goodbye!")
        return
    elif choice != "1":
        print("Invalid choice. Running full pipeline...")
    
    print("Starting complete pipeline execution...")
    print()
    
    try:
        # Step 1: Load dataset
        print("STEP 1: LOADING DATASET")
        print("-" * 40)
        dataset_path = "data/dataset.csv"
        
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset file not found at {dataset_path}")
            print("Please place your dataset.csv file in the data/ directory.")
            print("  - One text column (email content)")
            print("  - One label column (spam/ham or 1/0)")
            return
        
        df = load_data(dataset_path)
        print("Dataset loaded successfully!")
        print()
        
        # Step 2: Preprocess data
        print("STEP 2: DATA PREPROCESSING")
        print("-" * 40)
        X_train, X_test, y_train, y_test = preprocess(df)
        print("Data preprocessing completed!")
        print()
        
        # Step 3: Vectorize (feature selection)
        print("STEP 3: FEATURE PROCESSING")
        print("-" * 40)
        X_train_vec, X_test_vec = vectorize(X_train, X_test, y_train)
        print("Feature processing completed!")
        print()
        
        # Step 4: Train models
        print("STEP 4: MODEL TRAINING")
        print("-" * 40)
        nb_model, lr_model, nb_preds, lr_preds = train_models(
            X_train_vec, X_test_vec, y_train
        )
        print("Model training completed!")
        print()
        
        # Step 5: Evaluate models
        print("STEP 5: MODEL EVALUATION")
        print("-" * 40)
        evaluation_results = evaluate(
            y_test, nb_preds, lr_preds, nb_model, lr_model, X_test_vec
        )
        print("Model evaluation completed!")
        print()
        
        # Step 6: Generate visualizations
        print("STEP 6: VISUALIZATION GENERATION")
        print("-" * 40)
        plot_all(df, y_test, nb_preds, lr_preds, nb_model, lr_model, X_test_vec)
        print("Visualizations generated!")
        print()
        
        # Final summary
        print("=" * 80)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("Generated Outputs:")
        print("  Models saved to: outputs/models/")
        print("    - naive_bayes_model.pkl")
        print("    - logistic_regression_model.pkl")
        print("    - scaler.pkl")
        print("    - feature_selector.pkl")
        print("    - variance_selector.pkl")
        print()
        print("  Plots saved to: outputs/plots/")
        print("    - class_distribution.png")
        print("    - confusion_matrix_nb.png")
        print("    - confusion_matrix_lr.png")
        print("    - roc_curve.png")
        print("    - feature_importance.png")
        print()
        print("Your Email Spam Classifier is ready!")
        print("You can now use the saved models to classify new emails.")
        
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        print("Please check that all required files are in place.")
        
    except ValueError as e:
        print(f"Data processing error: {e}")
        print("Please check your dataset format and column names.")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install required packages using: pip install -r requirements.txt")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Please check the error message and try again.")
        
    finally:
        print()
        print("=" * 80)
        print("PIPELINE FINISHED")
        print("=" * 80)


if __name__ == "__main__":
    # Run main and handle interactive mode loop
    result = main()
    
    # If interactive classification returned True (back to menu), restart
    if result is True:
        main()
