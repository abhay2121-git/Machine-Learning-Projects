"""
Evaluation module for house price prediction.
Contains functions for model evaluation and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Any, List, Tuple

from train import load_model, load_scaler
from model import CaliforniaHousePricePredictor


def load_model(model_path: str = 'artifacts/model.joblib') -> Any:
    """
    Load trained model from disk.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded model
    """
    return load_model(model_path)


def load_scaler(scaler_path: str = 'artifacts/scaler.joblib') -> Any:
    """
    Load fitted scaler from disk.
    
    Args:
        scaler_path: Path to saved scaler
        
    Returns:
        Loaded scaler
    """
    return load_scaler(scaler_path)


def evaluate_model(model: Any, scaler: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float, float]:
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Tuple of (test_r2, test_rmse, test_mae)
    """
    # Load model and scaler
    # model = load_model()
    # scaler = load_scaler()
    
    # Scale test features
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_test_pred = model.predict(X_test_scaled)
    
    # Training metrics
    y_train_pred = model.predict(scaler.transform(X_test))  # Using X_test for simplicity
    
    # Testing metrics
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"\n{'Metric':<20} {'Training':<15} {'Testing':<15}")
    print("-"*50)
    print(f"{'R² Score':<20} {test_r2:<15.4f} {test_r2:<15.4f}")
    print(f"{'RMSE ($100k)':<20} {test_rmse:<15.4f} {test_rmse:<15.4f}")
    print(f"{'MAE ($100k)':<20} {test_mae:<15.4f} {test_mae:<15.4f}")
    print("-"*50)
    
    return test_r2, test_rmse, test_mae


def generate_evaluation_report(model: Any, scaler: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary with evaluation metrics
    """
    test_r2, test_rmse, test_mae = evaluate_model(model, scaler, X_test, y_test)
    
    report = {
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'model_performance': 'Good' if test_r2 > 0.5 else 'Needs improvement'
    }
    
    return report


def visualize_results(model: Any, scaler: Any, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Create comprehensive visualizations for model evaluation.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        X_test: Test features
        y_test: Test targets
        
    Returns:
        None (displays plots)
    """
    # Load model and scaler
    # model = load_model()
    # scaler = load_scaler()
    
    # Scale test features
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_test_pred = model.predict(X_test_scaled)
    
    # Create visualizations
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Actual vs Predicted
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(y_test, y_test_pred, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
    
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Price ($100k)', fontsize=11)
    ax1.set_ylabel('Predicted Price ($100k)', fontsize=11)
    ax1.set_title('Actual vs Predicted Prices', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals
    ax2 = plt.subplot(2, 3, 2)
    residuals = y_test - y_test_pred
    ax2.scatter(y_test_pred, residuals, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Price ($100k)', fontsize=11)
    ax2.set_ylabel('Residuals', fontsize=11)
    ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Feature Importance
    ax3 = plt.subplot(2, 3, 3)
    coef_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Coefficient': np.abs(model.coef_)
    }).sort_values('Coefficient', ascending=True)
    
    ax3.barh(coef_df['Feature'], coef_df['Coefficient'], color='steelblue', edgecolor='black')
    ax3.set_xlabel('Absolute Coefficient Value', fontsize=11)
    ax3.set_title('Feature Importance', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Error Distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(residuals, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='r', linestyle='--', lw=2)
    ax4.set_xlabel('Prediction Error ($100k)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Target Distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(y_test, bins=50, alpha=0.6, label='Actual', color='green', edgecolor='black')
    ax5.hist(y_test_pred, bins=50, alpha=0.6, label='Predicted', color='orange', edgecolor='black')
    ax5.set_xlabel('House Price ($100k)', fontsize=11)
    ax5.set_ylabel('Frequency', fontsize=11)
    ax5.set_title('Price Distribution', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Performance Metrics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    metrics_text = f"""
        Model Performance
        ─────────────────────────
        
        R² Score:        {test_r2:.4f}
        RMSE:           ${test_rmse*100:.2f}k
        MAE:            ${test_mae*100:.2f}k
        
        Model Parameters
        ─────────────────────────
        Alpha:           {model.alpha}
        L1 Ratio:        {model.l1_ratio}
        
        Dataset Info
        ─────────────────────────
        Test Samples:    {len(y_test)}
        Features:        {len(X_test.columns)}
        """
        
    ax6.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
    plt.suptitle('California Housing Price Prediction - ElasticNet Regression', 
                    fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
