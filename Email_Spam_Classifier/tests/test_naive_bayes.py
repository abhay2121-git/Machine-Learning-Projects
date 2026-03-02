"""
Test suite for Naive Bayes module.
"""

import unittest
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from src.models.naive_bayes import train_naive_bayes


class TestNaiveBayes(unittest.TestCase):
    """Test cases for the train_naive_bayes function."""
    
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
    
    def test_train_naive_bayes_returns_model_and_predictions(self):
        """Test that train_naive_bayes returns model and predictions."""
        model, predictions = train_naive_bayes(self.X_train, self.y_train, self.X_test)
        
        # Check return types
        self.assertIsInstance(model, MultinomialNB)
        self.assertIsInstance(predictions, np.ndarray)
        
        # Check predictions shape
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Check predictions are binary (0 or 1)
        unique_preds = set(predictions)
        self.assertTrue(unique_preds.issubset({0, 1}))
    
    def test_model_attributes(self):
        """Test that trained model has expected attributes."""
        model, _ = train_naive_bayes(self.X_train, self.y_train, self.X_test)
        
        # Check that model is fitted
        self.assertTrue(hasattr(model, 'class_count_'))
        self.assertTrue(hasattr(model, 'class_prior_'))
        
        # Check class counts
        self.assertEqual(len(model.class_count_), 2)  # Binary classification
        self.assertEqual(len(model.class_prior_), 2)
    
    def test_predictions_reasonable(self):
        """Test that predictions are reasonable."""
        model, predictions = train_naive_bayes(self.X_train, self.y_train, self.X_test)
        
        # Predictions should be either 0 or 1
        for pred in predictions:
            self.assertIn(pred, [0, 1])
        
        # Should have valid predictions for all test samples
        self.assertEqual(len(predictions), len(self.X_test))
    
    def test_model_training(self):
        """Test that model is actually trained."""
        model, _ = train_naive_bayes(self.X_train, self.y_train, self.X_test)
        
        # Model should be fitted (have learned parameters)
        self.assertIsNotNone(model.class_count_)
        self.assertIsNotNone(model.class_prior_)
        
        # Should be able to predict
        test_predictions = model.predict(self.X_test)
        self.assertIsInstance(test_predictions, np.ndarray)


if __name__ == '__main__':
    unittest.main()
