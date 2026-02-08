"""
Model evaluation and visualization utilities for house price prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Optional


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, 
                             figsize: tuple = (10, 6)) -> None:
    """
    Plot predictions vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, label='Predictions')
    
    # Perfect prediction line
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actual Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, 
                  figsize: tuple = (15, 5)) -> None:
    """
    Plot residual analysis.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        figsize: Figure size
    """
    residuals = y_true - y_pred
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted')
    ax1.grid(True, alpha=0.3)
    
    # Residuals histogram
    ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residuals Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15, 
                          figsize: tuple = (10, 8)) -> None:
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with feature importance
        top_n: Number of top features to show
        figsize: Figure size
    """
    if importance_df.empty:
        print("No feature importance data available!")
        return
    
    # Get top features
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=figsize)
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title(f'Top {top_n} Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()


def plot_learning_curves(train_scores: List[float], val_scores: List[float], 
                        epochs: Optional[List[int]] = None, 
                        figsize: tuple = (10, 6)) -> None:
    """
    Plot learning curves.
    
    Args:
        train_scores: Training scores
        val_scores: Validation scores
        epochs: Epoch numbers (optional)
        figsize: Figure size
    """
    if epochs is None:
        epochs = range(1, len(train_scores) + 1)
    
    plt.figure(figsize=figsize)
    plt.plot(epochs, train_scores, 'b-', label='Training Score')
    plt.plot(epochs, val_scores, 'r-', label='Validation Score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_model_comparison(models_metrics: dict, metric: str = 'r2', 
                         figsize: tuple = (10, 6)) -> None:
    """
    Plot comparison of multiple models.
    
    Args:
        models_metrics: Dictionary of model metrics
        metric: Metric to compare
        figsize: Figure size
    """
    model_names = list(models_metrics.keys())
    metric_values = [models_metrics[model][metric] for model in model_names]
    
    plt.figure(figsize=figsize)
    bars = plt.bar(model_names, metric_values, alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.xlabel('Models')
    plt.ylabel(metric.upper())
    plt.title(f'Model Comparison - {metric.upper()}')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_prediction_intervals(y_true: np.ndarray, y_pred: np.ndarray, 
                            y_pred_lower: np.ndarray, y_pred_upper: np.ndarray,
                            figsize: tuple = (12, 6)) -> None:
    """
    Plot predictions with confidence intervals.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_pred_lower: Lower bound of predictions
        y_pred_upper: Upper bound of predictions
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Sort by actual values for better visualization
    sort_idx = np.argsort(y_true)
    
    x = range(len(y_true))
    plt.plot(x, y_true[sort_idx], 'b-', label='Actual', alpha=0.7)
    plt.plot(x, y_pred[sort_idx], 'r-', label='Predicted', alpha=0.7)
    plt.fill_between(x, y_pred_lower[sort_idx], y_pred_upper[sort_idx], 
                     alpha=0.3, color='red', label='Confidence Interval')
    
    plt.xlabel('Sample Index (sorted by actual value)')
    plt.ylabel('Values')
    plt.title('Predictions with Confidence Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Model plotting utilities loaded successfully!")
