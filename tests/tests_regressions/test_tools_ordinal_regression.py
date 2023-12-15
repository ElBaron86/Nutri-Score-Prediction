'''
 # @ Author: Jaures Ememaga
 # @ Create Time: 2023-11-15 21:28:00
 # @ Modified by: 
 # @ Modified time:
 # @ Description: Unit tests for OrderedModel-related functions in tools_ordinal_regression.py.
 '''



import os
import sys

import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.miscmodels.ordinal_model import OrderedModel


# Moving up to the "src" directory
while os.path.basename(os.getcwd()) != "src":
    os.chdir("..")
sys.path.append(os.getcwd())

# Importing functions from tools_ordinal_regression 
from tools.regressions.ordinal.tools_ordinal_regression import calculate_aic_bic, fit_model, backward_variable_selection, cross_validation

class TestOrdinalRegressionFunctions(unittest.TestCase):

    def setUp(self):
        # Creating sample data for testing
        np.random.seed(42)
        self.X_sample = pd.DataFrame(np.random.rand(100, 5), columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
        self.y_sample = pd.Series(np.random.choice(['A', 'B', 'C', 'D', 'E'], size=100), name='score')

    def test_calculate_aic_bic(self):
        aic, bic = calculate_aic_bic(self.X_sample, self.y_sample)
        self.assertIsInstance(aic, float)
        self.assertIsInstance(bic, float)

    def test_fit_model(self):
        features = ['feature1', 'feature2']
        model = fit_model(self.X_sample, self.y_sample, features)
        self.assertIsInstance(model, OrderedModel)

    def test_backward_variable_selection(self):
        selected_features, criterion_values, unused_features = backward_variable_selection(self.X_sample, self.y_sample, criterion='aic')
        self.assertIsInstance(selected_features, list)
        self.assertIsInstance(criterion_values, list)
        self.assertIsInstance(unused_features, list)

    def test_cross_validation(self):
        features = ['feature1', 'feature2']
        mse = cross_validation(k=5, x=self.X_sample, y=self.y_sample, features=features)
        self.assertIsInstance(mse, float)

# Run the tests
if __name__ == '__main__':
    unittest.main()
