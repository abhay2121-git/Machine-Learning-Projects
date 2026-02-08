"""
Model definition module for house price prediction.
Contains ML model classes and training utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, Any, Tuple, Optional


class CaliforniaHousePricePredictor:
    """
    California Housing Price Prediction using ElasticNet Regression
    Dataset: 20,640 samples with 8 features (median income, house age, rooms, etc.)
    """
    
    def __init__(self, alpha=0.1, l1_ratio=0.5, test_size=0.2, random_state=42):
        """
        Initialize ElasticNet model with California Housing dataset
        
        Parameters:
        -----------
        alpha : float - Regularization strength (default: 0.1)
        l1_ratio : float - Mix between L1 (Lasso) and L2 (Ridge)
            - 0.0: Pure Ridge regression
            - 1.0: Pure Lasso regression
            - 0.5: Equal mix (default)
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.test_size = test_size
        self.random_state = random_state
        
        # Load dataset
        print("Loading California Housing Dataset...")
        california = fetch_california_housing()
        
        self.X = pd.DataFrame(california.data, columns=california.feature_names)
        self.y = pd.Series(california.target, name='Price')
        
        # Feature names
        self.feature_names = california.feature_names
        
        # Initialize scaler and model
        self.scaler = StandardScaler()
        self.model = ElasticNet(alpha=self.alpha, 
                               l1_ratio=self.l1_ratio, 
                               max_iter=10000,
                               random_state=self.random_state)
        
        print(f"Dataset loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features\n")
    
    def explore_data(self):
        """Display dataset information and statistics"""
        print("="*70)
        print("DATASET OVERVIEW")
        print("="*70)
        print(f"\nShape: {self.X.shape}")
        print(f"\nFeatures:")
        for i, feature in enumerate(self.feature_names, 1):
            print(f"  {i}. {feature}")
        
        print(f"\nTarget: Median House Value (in $100,000s)")
        print(f"\nFirst 5 samples:")
        print(pd.concat([self.X.head(), self.y.head()], axis=1))
        
        print(f"\n\nDataset Statistics:")
        print(self.X.describe())
        
        print(f"\n\nTarget Statistics:")
        print(self.y.describe())
        print()
    
    def split_and_preprocess(self):
        """Split data and apply feature scaling"""
        print("="*70)
        print("DATA PREPROCESSING")
        print("="*70)
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        
        print(f"\nTrain-Test Split ({int((1-self.test_size)*100)}-{int(self.test_size*100)}):")
        print(f"  Training samples: {len(self.X_train)}")
        print(f"  Testing samples: {len(self.X_test)}")
        
        # Feature scaling
        print(f"\nApplying StandardScaler (mean=0, std=1)...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("  Features scaled successfully")
        print()
    
    def train_model(self):
        """Train ElasticNet regression model"""
        print("="*70)
        print("MODEL TRAINING")
        print("="*70)
        
        print(f"\nTraining ElasticNet Regression...")
        print(f"  Hyperparameters:")
        print(f"    - Alpha (regularization): {self.alpha}")
        print(f"    - L1 ratio: {self.l1_ratio}")
        print(f"    - L1 (Lasso) weight: {self.l1_ratio * 100:.0f}%")
        print(f"    - L2 (Ridge) weight: {(1 - self.l1_ratio) * 100:.0f}%")
        
        self.model.fit(self.X_train_scaled, self.y_train)
        
        print("  Model trained successfully")
        print(f"\n  Model Coefficients:")
        
        # Display feature importance
        coef_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        print(coef_df.to_string(index=False))
        print(f"\n  Intercept: {self.model.intercept_:.4f}")
        print()
    
    def evaluate_model(self):
        """Evaluate model performance on test set"""
        print("="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        # Predictions
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)
        
        # Training metrics
        train_r2 = r2_score(self.y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        
        # Testing metrics
        test_r2 = r2_score(self.y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        print(f"\n{'Metric':<20} {'Training':<15} {'Testing':<15}")
        print("-"*50)
        print(f"{'R² Score':<20} {train_r2:<15.4f} {test_r2:<15.4f}")
        print(f"{'RMSE ($100k)':<20} {train_rmse:<15.4f} {test_rmse:<15.4f}")
        print(f"{'MAE ($100k)':<20} {train_mae:<15.4f} {test_mae:<15.4f}")
        
        # Sample predictions
        print(f"\n\nSample Predictions (first 10 test samples):")
        print("-"*70)
        results_df = pd.DataFrame({
            'Actual': self.y_test.values[:10],
            'Predicted': y_test_pred[:10],
            'Error': self.y_test.values[:10] - y_test_pred[:10]
        })
        print(results_df.to_string(index=False))
        print()
        
        return test_r2, test_rmse, test_mae
    
    def predict_custom(self):
        """Interactive price prediction"""
        print("="*70)
        print("CUSTOM HOUSE PRICE PREDICTION")
        print("="*70)
        
        print("\nEnter house features:")
        
        features = {}
        feature_descriptions = {
            'MedInc': 'Median Income (in $10k)',
            'HouseAge': 'House Age (years)',
            'AveRooms': 'Average Rooms per household',
            'AveBedrms': 'Average Bedrooms per household',
            'Population': 'Block Population',
            'AveOccup': 'Average Occupancy per household',
            'Latitude': 'Latitude',
            'Longitude': 'Longitude'
        }
        
        for feature in self.feature_names:
            value = float(input(f"  {feature_descriptions[feature]}: "))
            features[feature] = value
        
        # Create input array
        input_df = pd.DataFrame([features])
        input_scaled = self.scaler.transform(input_df)
        
        # Predict
        prediction = self.model.predict(input_scaled)[0]
        
        print(f"\n{'='*70}")
        print(f"PREDICTED HOUSE PRICE: ${prediction * 100000:,.2f}")
        print(f"                       (${prediction:.2f} in $100k)")
        print(f"{'='*70}\n")
    
    def visualize_results(self):
        """Create comprehensive visualizations"""
        y_pred = self.model.predict(self.X_test_scaled)
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Actual vs Predicted
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter(self.y_test, y_pred, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
        
        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax1.set_xlabel('Actual Price ($100k)', fontsize=11)
        ax1.set_ylabel('Predicted Price ($100k)', fontsize=11)
        ax1.set_title('Actual vs Predicted Prices', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals
        ax2 = plt.subplot(2, 3, 2)
        residuals = self.y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted Price ($100k)', fontsize=11)
        ax2.set_ylabel('Residuals', fontsize=11)
        ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Feature Importance
        ax3 = plt.subplot(2, 3, 3)
        coef_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': np.abs(self.model.coef_)
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
        ax5.hist(self.y_test, bins=50, alpha=0.6, label='Actual', color='green', edgecolor='black')
        ax5.hist(y_pred, bins=50, alpha=0.6, label='Predicted', color='orange', edgecolor='black')
        ax5.set_xlabel('House Price ($100k)', fontsize=11)
        ax5.set_ylabel('Frequency', fontsize=11)
        ax5.set_title('Price Distribution', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Performance Metrics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        r2 = r2_score(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        
        metrics_text = f"""
        Model Performance
        ─────────────────────────
        
        R² Score:        {r2:.4f}
        RMSE:           ${rmse*100:.2f}k
        MAE:            ${mae*100:.2f}k
        
        Model Parameters
        ─────────────────────────
        Alpha:           {self.alpha}
        L1 Ratio:        {self.l1_ratio}
        
        Dataset Info
        ─────────────────────────
        Test Samples:    {len(self.y_test)}
        Features:        {len(self.feature_names)}
        """
        
        ax6.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.suptitle('California Housing Price Prediction - ElasticNet Regression', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()


def main():
    """Main execution pipeline"""
    
    print("\n" + "="*70)
    print("CALIFORNIA HOUSING PRICE PREDICTION")
    print("ElasticNet Regression Model")
    print("="*70 + "\n")
    
    # Initialize predictor
    predictor = CaliforniaHousePricePredictor(alpha=0.1, l1_ratio=0.5)
    
    # Explore dataset
    predictor.explore_data()
    
    # Preprocess
    predictor.split_and_preprocess()
    
    # Train
    predictor.train_model()
    
    # Evaluate
    predictor.evaluate_model()
    
    # Visualize
    predictor.visualize_results()
    
    # Interactive prediction
    while True:
        choice = input("\nWould you like to predict a custom house price? (y/n): ").lower()
        if choice == 'y':
            predictor.predict_custom()
        else:
            break
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()