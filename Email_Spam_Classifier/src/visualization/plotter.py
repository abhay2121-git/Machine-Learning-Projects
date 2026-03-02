"""
Visualization module for email spam classifier.
Generates all required plots for model analysis and comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve
import joblib
import os
from typing import Optional


def plot_class_distribution(df: pd.DataFrame) -> None:
    """
    Plot the distribution of spam vs ham classes.
    
    Args:
        df (pd.DataFrame): Original dataframe with Prediction column
        
    Returns:
        None (saves plot to outputs/plots/class_distribution.png)
    """
    print("Generating class distribution plot...")
    
    # Check for Prediction column
    if 'Prediction' not in df.columns:
        print("Warning: Prediction column not found. Skipping class distribution plot.")
        return
    
    # Convert labels to readable format
    df['label_binary'] = df['Prediction'].apply(lambda x: 'Spam' if x == 1 else 'Ham')
    
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x='label_binary', palette={'Ham': 'green', 'Spam': 'red'})
    
    # Add count labels on bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.title('Spam vs Ham Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Email Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Save plot
    output_path = "outputs/plots/class_distribution.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Class distribution plot saved to: {output_path}")


def plot_confusion_matrix_nb(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Plot confusion matrix for Naive Bayes classifier.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        
    Returns:
        None (saves plot to outputs/plots/confusion_matrix_nb.png)
    """
    print("Generating Naive Bayes confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    
    plt.title('Confusion Matrix — Naive Bayes', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j + 0.5, i + 0.5, cm[i, j], 
                    ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Save plot
    output_path = "outputs/plots/confusion_matrix_nb.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Naive Bayes confusion matrix saved to: {output_path}")


def plot_confusion_matrix_lr(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Plot confusion matrix for Logistic Regression classifier.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        
    Returns:
        None (saves plot to outputs/plots/confusion_matrix_lr.png)
    """
    print("Generating Logistic Regression confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    
    plt.title('Confusion Matrix — Logistic Regression', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j + 0.5, i + 0.5, cm[i, j], 
                    ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Save plot
    output_path = "outputs/plots/confusion_matrix_lr.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Logistic Regression confusion matrix saved to: {output_path}")


def plot_roc_curve(y_true: np.ndarray, nb_proba: np.ndarray, lr_proba: np.ndarray) -> None:
    """
    Plot ROC curves for both models on the same graph.
    
    Args:
        y_true (np.ndarray): True labels
        nb_proba (np.ndarray): Naive Bayes prediction probabilities
        lr_proba (np.ndarray): Logistic Regression prediction probabilities
        
    Returns:
        None (saves plot to outputs/plots/roc_curve.png)
    """
    print("Generating ROC curve comparison...")
    
    # Calculate ROC curves
    nb_fpr, nb_tpr, _ = roc_curve(y_true, nb_proba)
    lr_fpr, lr_tpr, _ = roc_curve(y_true, lr_proba)
    
    # Calculate AUC scores
    from sklearn.metrics import auc
    nb_auc = auc(nb_fpr, nb_tpr)
    lr_auc = auc(lr_fpr, lr_tpr)
    
    plt.figure(figsize=(10, 6))
    
    # Plot ROC curves
    plt.plot(nb_fpr, nb_tpr, color='blue', lw=2, 
             label=f'Naive Bayes (AUC = {nb_auc:.3f})')
    plt.plot(lr_fpr, lr_tpr, color='red', lw=2, 
             label=f'Logistic Regression (AUC = {lr_auc:.3f})')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
             label='Random Classifier (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.3)
    
    # Save plot
    output_path = "outputs/plots/roc_curve.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved to: {output_path}")


def plot_feature_importance(lr_model: object, feature_names: list = None) -> None:
    """
    Plot top 20 most important features for spam detection.
    
    Args:
        lr_model (object): Trained Logistic Regression model
        feature_names (list): Names of features (optional)
        
    Returns:
        None (saves plot to outputs/plots/feature_importance.png)
    """
    print("Generating feature importance plot...")
    
    try:
        coefficients = lr_model.coef_[0]
        
        # Load feature names if not provided
        if feature_names is None:
            try:
                # Try to load the original dataset to get feature names
                import joblib
                selector = joblib.load("outputs/models/feature_selector.pkl")
                
                # Load original dataset to get column names
                df = pd.read_csv("data/dataset.csv")
                if 'Email No.' in df.columns:
                    df = df.drop('Email No.', axis=1)
                if 'Prediction' in df.columns:
                    df = df.drop('Prediction', axis=1)
                
                # Get the names of selected features
                selected_mask = selector.get_support()
                feature_names = df.columns[selected_mask].tolist()
                
            except Exception as e:
                print(f"Could not load feature names: {e}")
                feature_names = [f"Feature_{i}" for i in range(len(coefficients))]
        
        # Ensure we have the right number of feature names
        if len(feature_names) != len(coefficients):
            feature_names = [f"Feature_{i}" for i in range(len(coefficients))]
        
        # Create dataframe of features and coefficients
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })
        
        # Get top features by absolute coefficient value
        top_features = feature_importance.nlargest(20, 'abs_coefficient').sort_values('coefficient', ascending=True)
        
        # Create horizontal bar plot
        plt.figure(figsize=(10, 8))
        colors = ['red' if x > 0 else 'green' for x in top_features['coefficient']]
        
        bars = plt.barh(range(len(top_features)), top_features['coefficient'], color=colors, alpha=0.7)
        
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Coefficient Value', fontsize=12)
        plt.title('Top 20 Most Predictive Features for Spam', fontsize=16, fontweight='bold')
        
        # Add legend
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='red', alpha=0.7, label='Indicates Spam')
        green_patch = mpatches.Patch(color='green', alpha=0.7, label='Indicates Ham')
        plt.legend(handles=[red_patch, green_patch], loc='lower right')
        
        plt.grid(axis='x', alpha=0.3)
        
        # Save plot
        output_path = "outputs/plots/feature_importance.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance plot saved to: {output_path}")
        
        # Print top features for reference
        print("\nTop 10 Spam Indicators (positive coefficients):")
        spam_features = feature_importance.nlargest(10, 'coefficient')
        for _, row in spam_features.iterrows():
            print(f"  {row['feature']}: {row['coefficient']:.4f}")
        
        print("\nTop 10 Ham Indicators (negative coefficients):")
        ham_features = feature_importance.nsmallest(10, 'coefficient')
        for _, row in ham_features.iterrows():
            print(f"  {row['feature']}: {row['coefficient']:.4f}")
        
    except Exception as e:
        print(f"Warning: Could not generate feature importance plot - {e}")


def plot_all(df: pd.DataFrame, y_test: np.ndarray, nb_preds: np.ndarray, 
             lr_preds: np.ndarray, nb_model: object, lr_model: object, 
             X_test: np.ndarray) -> None:
    """
    Generate all visualization plots.
    
    Args:
        df (pd.DataFrame): Original dataframe
        y_test (np.ndarray): True test labels
        nb_preds (np.ndarray): Naive Bayes predictions
        lr_preds (np.ndarray): Logistic Regression predictions
        nb_model (object): Trained Naive Bayes model
        lr_model (object): Trained Logistic Regression model
        X_test (np.ndarray): Vectorized test features
        
    Returns:
        None (saves all plots to outputs/plots/)
    """
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Get prediction probabilities
    nb_proba = nb_model.predict_proba(X_test)[:, 1]
    lr_proba = lr_model.predict_proba(X_test)[:, 1]
    
    # Generate all plots
    plot_class_distribution(df)
    plot_confusion_matrix_nb(y_test, nb_preds)
    plot_confusion_matrix_lr(y_test, lr_preds)
    plot_roc_curve(y_test, nb_proba, lr_proba)
    plot_feature_importance(lr_model)
    
    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS COMPLETED")
    print("=" * 60)
    print("Plots saved to outputs/plots/ directory:")
    print("  - class_distribution.png")
    print("  - confusion_matrix_nb.png")
    print("  - confusion_matrix_lr.png")
    print("  - roc_curve.png")
    print("  - feature_importance.png")
