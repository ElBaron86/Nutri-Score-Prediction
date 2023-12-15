'''
 # @ Author: Jaures Ememaga
 # @ Create Time: 2023-11-19 21:41:58
 # @ Modified by: 
 # @ Modified time: 
 # @ Description: Unit tests for Lasso-related functions in tools_lasso.py.
 '''

import os
import sys

import unittest
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from matplotlib.widgets import Lasso

# Moving up to the "src" directory
while os.path.basename(os.getcwd()) != "src":
    os.chdir("..")
sys.path.append(os.getcwd())

# Importing functions from tools_ordinal_regression 
from tools.regressions.lasso.tools_lasso import train_and_evaluate_lasso_model

class TestTrainAndEvaluateLassoModel(unittest.TestCase):

    def test_train_and_evaluate_lasso_model(self):
        """Test train a Lasso regression model on the training data and return the classification error, coefficients, and intercept of the model.
        """
        train_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        train_labels = np.array(['A', 'B', 'C'])
        test_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        test_labels = np.array(['A', 'B', 'C'])
        alpha = 1.0

        classification_error, coefficients, intercept = train_and_evaluate_lasso_model(train_data, train_labels, test_data, test_labels, alpha)

        self.assertIsInstance(classification_error, float)
        self.assertIsInstance(coefficients, np.ndarray)
        self.assertIsInstance(intercept, np.ndarray)


    def test_encode_ordinal_categories(self):
        """Test to encode ordinal categories of the target variable for Lasso model using OrdinalEncoder.
        """
        train_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        train_labels = np.array(['A', 'B', 'C'])
        test_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        test_labels = np.array(['A', 'B', 'C'])
        alpha = 1.0

        _, _, _ = train_and_evaluate_lasso_model(train_data, train_labels, test_data, test_labels, alpha)

        ordinal_encoder = OrdinalEncoder()
        encoded_target = ordinal_encoder.fit_transform(train_labels.reshape(-1, 1))

        self.assertEqual(encoded_target.tolist(), [[0.0], [1.0], [2.0]])

    def test_transform_lasso_predictions(self):
        """Test to transform Lasso predictions into ordinal categories to retrieve scores using inverse_transform method of OrdinalEncoder.
        """
        train_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        train_labels = np.array(['A', 'B', 'C'])
        test_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        test_labels = np.array(['A', 'B', 'C'])
        alpha = 1.0

        _, _, _ = train_and_evaluate_lasso_model(train_data, train_labels, test_data, test_labels, alpha)

        ordinal_encoder = OrdinalEncoder()
        encoded_target = ordinal_encoder.fit_transform(train_labels.reshape(-1, 1))

        lasso_model = Lasso(alpha=alpha)
        lasso_model.fit(train_data, encoded_target)
        predictions = lasso_model.predict(test_data)

        ordinal_predictions = ordinal_encoder.inverse_transform(predictions.reshape(-1, 1))

        self.assertEqual(ordinal_predictions.tolist(), [['A'], ['B'], ['B']])

    # The function should be able to handle empty training and test data.
    def test_empty_data(self):
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
        train_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        train_labels = np.array(['A', 'B', 'C'])
        test_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        test_labels = np.array(['A', 'B', 'C'])
        alpha = 1.0

        classification_error, coefficients, intercept = train_and_evaluate_lasso_model(train_data, train_labels, test_data, test_labels, alpha)

        self.assertAlmostEqual(classification_error, 0.33333333333333337)
        self.assertEqual(coefficients.tolist(), [0.16666666666666666, 0.0, 0.0])
        self.assertEqual(intercept[0], 0.33333333333333337)
 
    def test_non_categorical_target(self):
        """Test to handle non-categorical target variable.
        """
        train_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        train_labels = np.array([1, 2, 3])
        test_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        test_labels = np.array([1, 2, 3])
        alpha = 1.0

        classification_error, coefficients, intercept = train_and_evaluate_lasso_model(train_data, train_labels, test_data, test_labels, alpha)

        self.assertEqual(classification_error, 1.0)
        self.assertEqual(coefficients.tolist(), [])
        self.assertEqual(intercept, 0.0)

    def test_classification_error_calculation(self):
        """Test calculation of classification error using accuracy_score method of sklearn.metrics.
        """
        train_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        train_labels = np.array(['A', 'B', 'C'])
        test_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        test_labels = np.array(['A', 'B', 'C'])
        alpha = 1.0

        classification_error, _, _ = train_and_evaluate_lasso_model(train_data, train_labels, test_data, test_labels, alpha)

        self.assertIsInstance(classification_error, float)
        
# Run the tests
if __name__ == '__main__':
    unittest.main()
