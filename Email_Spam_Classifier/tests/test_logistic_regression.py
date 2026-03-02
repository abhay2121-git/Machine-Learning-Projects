"""
Test suite for Logistic Regression module.
"""

import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.models.logistic_regression import train_logistic_regression


class TestLogisticRegression(unittest.TestCase):
    """Test cases for the train_logistic_regression function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample training data
        self.X_train = np.array([
            [1, 0, 1, 0],  # Sample 1
            [0, 1, 0, 1],  # Sample 2
            [1, 1, 0, 0],  # Sample 3
            [0, 0, 1, 1],  # Sample 4
            [1, 0, 0, 1]   # Sample 5
        ])
        
        self.y_train = np.array([0, 1, 0, 1, 0])
        
        self.X_test = np.array([
            [1, 0, 1, 0],  # Test sample 1
            [0, 1, 0, 1]   # Test sample 2
        ])
    
    def test_train_logistic_regression_returns_model_and_predictions(self):
        """Test that train_logistic_regression returns model and predictions."""
        model, predictions = train_logistic_regression(
            self.X_train, self.y_train, self.X_test
        )
        
        # Check return types
        self.assertIsInstance(model, LogisticRegression)
        self.assertIsInstance(predictions, np.ndarray)
        
        # Check predictions shape
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Check predictions are binary (0 or 1)
        unique_preds = set(predictions)
        self.assertTrue(unique_preds.issubset({0, 1}))
    
    def test_model_attributes(self):
        """Test that trained model has expected attributes."""
        model, _ = train_logistic_regression(
            self.X_train, self.y_train, self.X_test
        )
        
        # Check that model is fitted
        self.assertTrue(hasattr(model, 'coef_'))
        self.assertTrue(hasattr(model, 'intercept_'))
        self.assertTrue(hasattr(model, 'n_iter_'))
        
        # Check coefficient shape
        self.assertEqual(model.coef_.shape[0], 1)  # Binary classification
        self.assertEqual(model.coef_.shape[1], self.X_train.shape[1])
    
    def test_predictions_reasonable(self):
        """Test that predictions are reasonable."""
        model, predictions = train_logistic_regression(
            self.X_train, self.y_train, self.X_test
        )
        
        # Predictions should be either 0 or 1
        for pred in predictions:
            self.assertIn(pred, [0, 1])
        
        # Should have valid predictions for all test samples
        self.assertEqual(len(predictions), len(self.X_test))
    
    def test_model_training(self):
        """Test that model is actually trained."""
        model, _ = train_logistic_regression(
            self.X_train, self.y_train, self.X_test
        )
        
        # Model should be fitted (have learned parameters)
        self.assertIsNotNone(model.coef_)
        self.assertIsNotNone(model.intercept_)
        self.assertIsNotNone(model.n_iter_)
        
        # Should be able to predict
        test_predictions = model.predict(self.X_test)
        self.assertIsInstance(test_predictions, np.ndarray)
    
    def test_probability_predictions(self):
        """Test that model can predict probabilities."""
        model, _ = train_logistic_regression(
            self.X_train, self.y_train, self.X_test
        )
        
        # Should be able to predict probabilities
        probabilities = model.predict_proba(self.X_test)
        
        # Check shape
        self.assertEqual(probabilities.shape[0], len(self.X_test))
        self.assertEqual(probabilities.shape[1], 2)  # Binary classification
        
        # Check that probabilities sum to 1
        for prob_row in probabilities:
            self.assertAlmostEqual(prob_row.sum(), 1.0, places=6)
        
        # Check that probabilities are between 0 and 1
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))


if __name__ == '__main__':
    unittest.main()
