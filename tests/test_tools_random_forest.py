'''
 # @ Author: Jaures Ememaga
 # @ Create Time: 2023-11-23 20:11:31
 # @ Modified by: 
 # @ Modified time:
 # @ Description: Unit tests for RandomForestClassifier-related functions in tools_random_forest.py.
 '''

#### Imports ####

import sys
import os
import unittest
from unittest.mock import patch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Recovery of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Back to the project directory 
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from src.tools.tools_random_forest import (
    hyper_params_search,
    make_prediction,
    hyper_params_search_with_feature_elimination
)


class TestFunctions(unittest.TestCase):

    def setUp(self):
        # Generate a database for tests
        x, y = make_classification(n_samples=100, n_features=10, n_classes=3, n_informative=5, n_clusters_per_class=2, random_state=42)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)


    def test_hyper_params_search(self):
        """Test hyper_params_search function."""
        model = hyper_params_search(pd.DataFrame(self.x_train), pd.Series(self.y_train), n_iter=2)
        self.assertIsInstance(model, RandomForestClassifier)

    def test_make_prediction(self):
        """Test make_prediction function."""
        model = hyper_params_search(pd.DataFrame(self.x_train), pd.Series(self.y_train), n_iter=2)
        y_pred = make_prediction(model, pd.DataFrame(self.x_test), pd.Series(self.y_test))
        self.assertEqual(len(y_pred), len(self.y_test))

    def test_hyper_params_search_with_feature_elimination(self):
        """Test hyper_params_search_with_feature_elimination function."""
        actual_model = hyper_params_search(pd.DataFrame(self.x_train), pd.Series(self.y_train))
        best_model, best_features = hyper_params_search_with_feature_elimination(
            pd.DataFrame(self.x_train), pd.Series(self.y_train),
            pd.DataFrame(self.x_test), pd.Series(self.y_test),
            actual_model, n_iter=2
        )

        self.assertIsInstance(best_model, RandomForestClassifier)
        self.assertIsInstance(best_features, list)

    def test_hyper_params_search_empty_input(self):
        """Test hyper_params_search function with empty input."""
        with self.assertRaises(ValueError):
            hyper_params_search(pd.DataFrame(), pd.Series())


    def test_hyper_params_search_with_feature_elimination_empty_input(self):
        """Test hyper_params_search_with_feature_elimination function with empty input."""
        actual_model = hyper_params_search(pd.DataFrame(self.x_train), pd.Series(self.y_train), n_iter=2)
        with self.assertRaises(ValueError):
            hyper_params_search_with_feature_elimination(pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series(), actual_model, n_iter=2)
            
    @patch('matplotlib.pyplot.show')  # Mock the show function
    def test_make_prediction_plot_conf_mat(self, mock_show):
        """Test make_prediction function with plot_conf_mat."""
        model = hyper_params_search(pd.DataFrame(self.x_train), pd.Series(self.y_train))
        with self.assertLogs() as log:
            make_prediction(model, pd.DataFrame(self.x_test), pd.Series(self.y_test), plot_conf_mat=True)
        self.assertTrue("Accuracy:" in log.output[0])
        mock_show.assert_called_once()  # Assert that show function is called

    @patch('matplotlib.pyplot.show')  # Mock the show function
    def test_hyper_params_search_with_feature_elimination_plot_graph(self, mock_show):
        """Test hyper_params_search_with_feature_elimination function with plot_graph."""
        actual_model = hyper_params_search(pd.DataFrame(self.x_train), pd.Series(self.y_train), n_iter=2)
        best_model, best_features = hyper_params_search_with_feature_elimination(
            pd.DataFrame(self.x_train), pd.Series(self.y_train),
            pd.DataFrame(self.x_test), pd.Series(self.y_test), 
            actual_model, plot_graph=True, n_iter=2
        )

        self.assertIsInstance(best_model, RandomForestClassifier)
        self.assertIsInstance(best_features, list)
        mock_show.assert_called_once()  # Assert that show function is called

    def tearDown(self):
        # Clean resources after tests if necessary
        pass

if __name__ == '__main__':
    unittest.main()
