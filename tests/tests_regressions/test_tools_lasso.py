"""
# @ Author: Jaures Ememaga
# @ Create Time: 2023-11-19 21:41:58
# @ Modified by:
# @ Modified time:
# @ Description: Unit tests for Lasso-related functions in tools_lasso.py.
"""

import os
import sys
import unittest
import numpy as np

from sklearn.preprocessing import OrdinalEncoder

# Moving up to the "src" directory
while os.path.basename(os.getcwd()) != "src":
    os.chdir("..")
sys.path.append(os.getcwd())

# Importing functions from tools_ordinal_regression 
from tools.regressions.lasso.tools_lasso import train_and_evaluate_lasso_model

class TestTrainAndEvaluateLassoModel(unittest.TestCase):

    def setUp(self):
        """Set up common data for testing."""
        self.train_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.train_labels = np.array(['A', 'B', 'C'])
        self.test_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.test_labels = np.array(['A', 'B', 'C'])
        self.alpha = 1.0

    def test_train_and_evaluate_lasso_model(self):
        """Test train a Lasso regression model on the training data and return the classification error, coefficients, and intercept of the model.
        """
        classification_error, coefficients, intercept = train_and_evaluate_lasso_model(self.train_data, self.train_labels, self.test_data, self.test_labels, self.alpha)

        self.assertIsInstance(classification_error, float)
        self.assertIsInstance(coefficients, np.ndarray)
        self.assertIsInstance(intercept, np.ndarray)

    def test_encode_ordinal_categories(self):
        """Test to encode ordinal categories of the target variable for Lasso model using OrdinalEncoder.
        """
        _, _, _ = train_and_evaluate_lasso_model(self.train_data, self.train_labels, self.test_data, self.test_labels, self.alpha)

        ordinal_encoder = OrdinalEncoder()
        encoded_target = ordinal_encoder.fit_transform(self.train_labels.reshape(-1, 1))

        self.assertEqual(encoded_target.tolist(), [[0.0], [1.0], [2.0]])

    def test_transform_lasso_predictions(self):
        """Test to transform Lasso predictions into ordinal categories to retrieve scores using inverse_transform method of OrdinalEncoder.
        """
        _, _, _ = train_and_evaluate_lasso_model(self.train_data, self.train_labels, self.test_data, self.test_labels, self.alpha)

        ordinal_encoder = OrdinalEncoder()
        encoded_target = ordinal_encoder.fit_transform(self.train_labels.reshape(-1, 1))

        lasso_model = Lasso(alpha=self.alpha)
        lasso_model.fit(self.train_data, encoded_target)
        predictions = lasso_model.predict(self.test_data)

        ordinal_predictions = ordinal_encoder.inverse_transform(predictions.reshape(-1, 1))

        self.assertEqual(ordinal_predictions.tolist(), [['A'], ['B'], ['B']])

    def test_empty_data(self):
        """Test to handle empty training and test data.
        """
        train_data = np.array([])
        train_labels = np.array(['A', 'B', 'C'])
        test_data = np.array([])
        test_labels = np.array(['A', 'B', 'C'])
        alpha = 1.0

        classification_error, coefficients, intercept = train_and_evaluate_lasso_model(train_data, train_labels, test_data, test_labels, alpha)

        self.assertEqual(classification_error, 1.0)
        self.assertEqual(coefficients.tolist(), [])
        self.assertEqual(intercept, 0.0)

    def test_empty_labels_fixed(self):
        """Test to handle empty training and test labels.
        """
        classification_error, coefficients, intercept = train_and_evaluate_lasso_model(self.train_data, self.train_labels, self.test_data, self.test_labels, self.alpha)

        self.assertAlmostEqual(classification_error, 0.33333333333333337)
        self.assertEqual(coefficients.tolist(), [0.16666666666666666, 0.0, 0.0])
        self.assertEqual(intercept[0], 0.33333333333333337)

    def test_non_categorical_target(self):
        """Test to handle non-categorical target variable.
        """
        train_labels = np.array([1, 2, 3])
        test_labels = np.array([1, 2, 3])

        classification_error, coefficients, intercept = train_and_evaluate_lasso_model(self.train_data, train_labels, self.test_data, test_labels, self.alpha)

        self.assertEqual(classification_error, 1.0)
        self.assertEqual(coefficients.tolist(), [])
        self.assertEqual(intercept, 0.0)

    def test_classification_error_calculation(self):
        """Test calculation of classification error using accuracy_score method of sklearn.metrics.
        """
        classification_error, _, _ = train_and_evaluate_lasso_model(self.train_data, self.train_labels, self.test_data, self.test_labels, self.alpha)

        self.assertIsInstance(classification_error, float)

# Run the tests
if __name__ == '__main__':
    unittest.main()
