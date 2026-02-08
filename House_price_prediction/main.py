"""
Main entry point for House Price Prediction Project.
Uses modular structure with separate files for different functionalities.
"""

import sys
import os

# Add src directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import CaliforniaHousePricePredictor


def main():
    """Main execution function"""
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