"""
Model evaluator module for email spam classifier.
Computes comprehensive metrics and compares model performance.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import pandas as pd
from typing import Tuple


def evaluate(y_test: np.ndarray, nb_preds: np.ndarray, lr_preds: np.ndarray,
            nb_model: object, lr_model: object, X_test: np.ndarray) -> None:
    """
    Evaluate both models and display comprehensive comparison.
    
    Args:
        y_test (np.ndarray): True test labels
        nb_preds (np.ndarray): Naive Bayes predictions
        lr_preds (np.ndarray): Logistic Regression predictions
        nb_model (object): Trained Naive Bayes model
        lr_model (object): Trained Logistic Regression model
        X_test (np.ndarray): Vectorized test features
        
    Returns:
        None (prints evaluation results)
    """
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    
    # Calculate metrics for Naive Bayes
    nb_accuracy = accuracy_score(y_test, nb_preds)
    nb_precision = precision_score(y_test, nb_preds)
    nb_recall = recall_score(y_test, nb_preds)
    nb_f1 = f1_score(y_test, nb_preds)
    
    # Calculate metrics for Logistic Regression
    lr_accuracy = accuracy_score(y_test, lr_preds)
    lr_precision = precision_score(y_test, lr_preds)
    lr_recall = recall_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds)
    
    # Calculate ROC AUC scores
    nb_proba = nb_model.predict_proba(X_test)[:, 1]
    lr_proba = lr_model.predict_proba(X_test)[:, 1]
    nb_roc_auc = roc_auc_score(y_test, nb_proba)
    lr_roc_auc = roc_auc_score(y_test, lr_proba)
    
    # Create comparison table
    print("\nPERFORMANCE COMPARISON TABLE")
    print("-" * 80)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC AUC':<10}")
    print("-" * 80)
    print(f"{'Naive Bayes':<20} {nb_accuracy:<10.2%} {nb_precision:<10.2%} {nb_recall:<10.2%} {nb_f1:<10.2%} {nb_roc_auc:<10.2%}")
    print(f"{'Logistic Regression':<20} {lr_accuracy:<10.2%} {lr_precision:<10.2%} {lr_recall:<10.2%} {lr_f1:<10.2%} {lr_roc_auc:<10.2%}")
    print("-" * 80)
    
    # Determine which model performed better
    print(f"\nWINNER ANALYSIS:")
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
    nb_scores = [nb_accuracy, nb_precision, nb_recall, nb_f1, nb_roc_auc]
    lr_scores = [lr_accuracy, lr_precision, lr_recall, lr_f1, lr_roc_auc]
    
    nb_wins = 0
    lr_wins = 0
    
    for i, metric in enumerate(metrics):
        if nb_scores[i] > lr_scores[i]:
            nb_wins += 1
            print(f"  {metric}: Naive Bayes ({nb_scores[i]:.2%}) > Logistic Regression ({lr_scores[i]:.2%})")
        elif lr_scores[i] > nb_scores[i]:
            lr_wins += 1
            print(f"  {metric}: Logistic Regression ({lr_scores[i]:.2%}) > Naive Bayes ({nb_scores[i]:.2%})")
        else:
            print(f"  {metric}: Tie ({nb_scores[i]:.2%})")
    
    if nb_wins > lr_wins:
        print(f"\nOverall Winner: Naive Bayes ({nb_wins}/{len(metrics)} metrics)")
    elif lr_wins > nb_wins:
        print(f"\nOverall Winner: Logistic Regression ({lr_wins}/{len(metrics)} metrics)")
    else:
        print(f"\nOverall Result: Tie ({nb_wins}/{len(metrics)} metrics each)")
    
    # Detailed classification reports
    print("\n" + "=" * 80)
    print("DETAILED CLASSIFICATION REPORTS")
    print("=" * 80)
    
    print("\nNAIVE BAYES CLASSIFICATION REPORT:")
    print("-" * 40)
    print(classification_report(y_test, nb_preds, target_names=['Ham', 'Spam']))
    
    print("\nLOGISTIC REGRESSION CLASSIFICATION REPORT:")
    print("-" * 40)
    print(classification_report(y_test, lr_preds, target_names=['Ham', 'Spam']))
    
    # Confusion matrices
    print("\n" + "=" * 80)
    print("CONFUSION MATRICES")
    print("=" * 80)
    
    nb_cm = confusion_matrix(y_test, nb_preds)
    lr_cm = confusion_matrix(y_test, lr_preds)
    
    print("\nNaive Bayes Confusion Matrix:")
    print("Predicted ->  Ham    Spam")
    print("Actual |")
    print(f"Ham        {nb_cm[0,0]:<6} {nb_cm[0,1]:<6}")
    print(f"Spam       {nb_cm[1,0]:<6} {nb_cm[1,1]:<6}")
    
    print("\nLogistic Regression Confusion Matrix:")
    print("Predicted ->  Ham    Spam")
    print("Actual |")
    print(f"Ham        {lr_cm[0,0]:<6} {lr_cm[0,1]:<6}")
    print(f"Spam       {lr_cm[1,0]:<6} {lr_cm[1,1]:<6}")
    
    # Error analysis
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)
    
    # False positives (ham classified as spam)
    nb_fp = nb_cm[0,1]
    lr_fp = lr_cm[0,1]
    
    # False negatives (spam classified as ham)
    nb_fn = nb_cm[1,0]
    lr_fn = lr_cm[1,0]
    
    print(f"\nFalse Positives (Ham to Spam):")
    print(f"  Naive Bayes: {nb_fp} cases")
    print(f"  Logistic Regression: {lr_fp} cases")
    
    print(f"\nFalse Negatives (Spam to Ham):")
    print(f"  Naive Bayes: {nb_fn} cases")
    print(f"  Logistic Regression: {lr_fn} cases")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED")
    print("=" * 80)
    
    # Store results for plotting (return as dictionary for potential use)
    evaluation_results = {
        'naive_bayes': {
            'accuracy': nb_accuracy,
            'precision': nb_precision,
            'recall': nb_recall,
            'f1': nb_f1,
            'roc_auc': nb_roc_auc,
            'confusion_matrix': nb_cm,
            'predictions': nb_preds,
            'probabilities': nb_proba
        },
        'logistic_regression': {
            'accuracy': lr_accuracy,
            'precision': lr_precision,
            'recall': lr_recall,
            'f1': lr_f1,
            'roc_auc': lr_roc_auc,
            'confusion_matrix': lr_cm,
            'predictions': lr_preds,
            'probabilities': lr_proba
        }
    }
    
    return evaluation_results
