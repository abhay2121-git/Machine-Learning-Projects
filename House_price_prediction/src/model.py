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
        print(f"    - L2 (Ridge) weight: {(1-self.l1_ratio) * 100:.0f}%")
        
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
        print("-"*50)
        
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
    
    def predict_sample_house(self, medinc=5.0, houseage=25.0, avelrooms=4.5, 
                          avebedrms=1.2, population=500, aveoccup=2.0,
                          latitude=34.05, longitude=-118.24):
        """
        Predict house price for specific sample values.
        
        Args:
            medinc: Median income in $10k (default: 5.0)
            houseage: House age in years (default: 25.0)
            avelrooms: Average rooms per household (default: 4.5)
            avebedrms: Average bedrooms per household (default: 1.2)
            population: Block population (default: 500)
            aveoccup: Average occupancy per household (default: 2.0)
            latitude: Latitude coordinate (default: 34.05)
            longitude: Longitude coordinate (default: -118.24)
        
        Returns:
            Predicted house price in dollars
        """
        # Create input array with your values
        features = {
            'MedInc': medinc,
            'HouseAge': houseage,
            'AveRooms': avelrooms,
            'AveBedrms': avebedrms,
            'Population': population,
            'AveOccup': aveoccup,
            'Latitude': latitude,
            'Longitude': longitude
        }
        
        input_df = pd.DataFrame([features])
        input_scaled = self.scaler.transform(input_df)
        
        # Predict
        prediction = self.model.predict(input_scaled)[0]
        
        print(f"\n{'='*70}")
        print(f"SAMPLE HOUSE PREDICTION")
        print(f"{'='*70}")
        print(f"Input Features:")
        print(f"  Median Income: ${medinc*10:,.2f}k")
        print(f"  House Age: {houseage} years")
        print(f"  Average Rooms: {avelrooms}")
        print(f"  Average Bedrooms: {avebedrms}")
        print(f"  Population: {population}")
        print(f"  Average Occupancy: {aveoccup}")
        print(f"  Location: ({latitude}, {longitude})")
        print(f"\nPredicted Price: ${prediction * 100000:,.2f}")
        print(f"                 (${prediction:.2f} in $100k)")
        print(f"{'='*70}\n")
        
        return prediction
    
    def load_user_config(self, config_file='user_config.json'):
        """
        Load user configuration from JSON file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Dictionary with user configuration
        """
        import json
        import os
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                'medinc': 8.5,
                'houseage': 35.0,
                'avelrooms': 4.0,
                'avebedrms': 1.5,
                'population': 600,
                'aveoccup': 2.2,
                'latitude': 38.5,
                'longitude': -121.0
            }
    
    def save_user_config(self, config_data, config_file='user_config.json'):
        """
        Save user configuration to JSON file.
        
        Args:
            config_data: Dictionary with user configuration
            config_file: Path to save configuration
        
        """
        import json
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        print(f"Configuration saved to {config_file}")
    
    def predict_with_user_config(self, config_file='user_config.json'):
        """
        Predict using user configuration file.
        
        Args:
            config_file: Path to configuration file
        """
        config = self.load_user_config(config_file)
        
        print(f"\n{'='*70}")
        print(f"USER CONFIGURATION PREDICTION")
        print(f"{'='*70}")
        print(f"Using configuration from: {config_file}")
        print(f"Input Features:")
        print(f"  Median Income: ${config['medinc']:.2f}k")
        print(f"  House Age: {config['houseage']} years")
        print(f"  Average Rooms: {config['avelrooms']}")
        print(f"  Average Bedrooms: {config['avebedrms']}")
        print(f"  Population: {config['population']}")
        print(f"  Average Occupancy: {config['aveoccup']}")
        print(f"  Location: ({config['latitude']}, {config['longitude']})")
        
        # Create input array
        features = {
            'MedInc': config['medinc'],
            'HouseAge': config['houseage'],
            'AveRooms': config['avelrooms'],
            'AveBedrms': config['avebedrms'],
            'Population': config['population'],
            'AveOccup': config['aveoccup'],
            'Latitude': config['latitude'],
            'Longitude': config['longitude']
        }
        
        input_df = pd.DataFrame([features])
        input_scaled = self.scaler.transform(input_df)
        
        # Predict
        prediction = self.model.predict(input_scaled)[0]
        
        print(f"\nPredicted Price: ${prediction * 100000:,.2f}")
        print(f"                 (${prediction:.2f} in $100k)")
        print(f"{'='*70}\n")
        
        return prediction
    
    def predict_custom(self):
        """Interactive price prediction with error handling"""
        print("="*70)
        print("CUSTOM HOUSE PRICE PREDICTION")
        print("="*70)
        
        print("\nEnter house features (or 'quit' to exit):")
        
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
        
        try:
            for feature in self.feature_names:
                while True:
                    user_input = input(f"  {feature_descriptions[feature]}: ").strip()
                    
                    if user_input.lower() == 'quit':
                        print("Exiting prediction mode...")
                        return
                    
                    try:
                        value = float(user_input)
                        if value >= 0:  # Basic validation
                            features[feature] = value
                            break
                        else:
                            print("    Please enter a positive number.")
                    except ValueError:
                        print("    Please enter a valid number.")
            
            # Create input array
            input_df = pd.DataFrame([features])
            input_scaled = self.scaler.transform(input_df)
            
            # Predict
            prediction = self.model.predict(input_scaled)[0]
            
            print(f"\n{'='*70}")
            print(f"PREDICTED HOUSE PRICE: ${prediction * 100000:,.2f}")
            print(f"                       (${prediction:.2f} in $100k)")
            print(f"{'='*70}\n")
            
        except KeyboardInterrupt:
            print("\n\nPrediction cancelled by user.")
        except Exception as e:
            print(f"\nError during prediction: {str(e)}")
    
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
    
    try:
        # Initialize predictor
        predictor = CaliforniaHousePricePredictor(alpha=0.1, l1_ratio=0.5)
        
        # Display dataset information
        predictor.explore_data()
        
        # Preprocess data
        predictor.split_and_preprocess()
        
        # Train model
        predictor.train_model()
        
        # Evaluate model
        predictor.evaluate_model()
        
        # Visualize results
        predictor.visualize_results()
        
        # Interactive user input prediction
        print("\n" + "="*70)
        print("Would you like to enter your own house values? (y/n): ")
        
        try:
            choice = input().strip().lower()
            
            if choice == 'y':
                print("\n" + "="*70)
                print("ENTER YOUR HOUSE VALUES")
                print("="*70)
                
                # Get user input for each feature
                medinc = float(input("Median Income (in $10k, e.g., 8.5 for $85k): "))
                houseage = float(input("House Age (in years): "))
                avelrooms = float(input("Average Rooms per household: "))
                avebedrms = float(input("Average Bedrooms per household: "))
                population = float(input("Block Population: "))
                aveoccup = float(input("Average Occupancy per household: "))
                latitude = float(input("Latitude (e.g., 37.8 for SF): "))
                longitude = float(input("Longitude (e.g., -122.4 for SF): "))
                
                # Predict with user values
                predictor.predict_sample_house(
                    medinc=medinc, houseage=houseage, avelrooms=avelrooms,
                    avebedrms=avebedrms, population=population, aveoccup=aveoccup,
                    latitude=latitude, longitude=longitude
                )
            else:
                print("Using default sample values above.")
                
        except (EOFError, KeyboardInterrupt):
            print("\nUsing default sample values above.")
        except ValueError:
            print("\nInvalid input! Please enter valid numbers. Using default sample values above.")
        except Exception as e:
            print(f"Input error: {str(e)}. Using default sample values above.")
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please check your environment and dependencies.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
