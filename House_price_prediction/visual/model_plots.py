"""
Model Performance Visualization module for house price prediction.
Contains functions for creating comprehensive model evaluation plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Any, List, Optional

from evaluate import evaluate_model


def plot_learning_curves(model: Any, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_val: pd.DataFrame, y_val: pd.Series) -> None:
    """
    Plot learning curves to analyze model performance.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
    """
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, 
        scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    
    # Calculate mean and std
    train_mean = -np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = -np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    
    plt.fill_between(train_sizes, train_mean - train_std, 
                     train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, val_mean - val_std, 
                     val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Negative Mean Squared Error')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_residual_analysis(y_true: pd.Series, y_pred: pd.Series) -> None:
    """
    Analyze model residuals for assumptions checking.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Residual Analysis', fontsize=16, fontweight='bold')
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.6)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Price ($100k)')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted')
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model: Any, feature_names: List[str]) -> None:
    """
    Create feature importance visualization.
    
    Args:
        model: Trained model
        feature_names: List of feature names
    """
    # Get feature importance
    importances = np.abs(model.coef_)
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], 
                color='steelblue', edgecolor='black')
    
    plt.xlabel('Absolute Coefficient Value')
    plt.title('Feature Importance')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add importance values as text
    for i, (idx, row) in enumerate(feature_importance_df.itertuples()):
        plt.text(row['Importance'] * 1.05, i, row['Feature'], 
                f'{row["Importance"]:.3f}', fontsize=9, va='center')
    
    plt.tight_layout()
    plt.show()


def plot_model_comparison(models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Compare multiple models side by side.
    
    Args:
        models: Dictionary of model names and trained models
        X_test: Test features
        y_test: Test targets
    """
    fig, axes = plt.subplots(1, len(models), figsize=(15, 6))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    
    metrics = {}
    
    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        metrics[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}
        
        # Create subplot
        ax = axes[i]
        
        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.6, s=20, edgecolors='k', linewidths=0.5)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Price ($100k)')
        ax.set_ylabel('Predicted Price ($100k)')
        ax.set_title(f'{name} (R²: {r2:.3f})')
        ax.grid(True, alpha=0.3)
        
        # Add metrics text
        metrics_text = f"R²: {r2:.3f}\\nRMSE: {rmse:.2f}\\nMAE: {mae:.2f}"
        ax.text(0.05, 0.95, metrics_text, fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return metrics
