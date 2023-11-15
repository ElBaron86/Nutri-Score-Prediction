'''
 # @ Author: Jaures Ememaga
 # @ Create Time: 2023-11-15 14:30:20
 # @ Modified by: jaures Ememaga
 # @ Modified time: 2023-11-15 14:32:40
 # @ Description: Test des fonctions du script cross_validation.py dans src/tools
 '''


import sys
import os
# récupération du répertoire actuel
current_dir = os.path.dirname(os.path.abspath(__file__))

# retour au répertoire du projet 
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)

# importation des modules essentiels
import pickle
import unittest
from pandas.api.types import CategoricalDtype
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel, OrderedResultsWrapper
from src.tools.cross_validation import scale_data, cross_validation_selection

class TestScaleData(unittest.TestCase):
    def setUp(self):
        # Recupération des données pour les tests
        self.data_fake = pd.read_csv("src/data/train.csv")
        self.data_fake = self.data_fake.drop(labels=['score'], axis=1)

    def tearDown(self):
        pass

    def test_scale_data(self):
        # chemin du pickle contenant le scaler
        scaler_path = 'src/tools/scaler.pkl'

        # Entraîner le scaler_fake sur les données d'entraînement factices
        scaler_fake = StandardScaler()
        scaler_fake.fit(self.data_fake)

        # Utiliser la fonction scale_data pour normaliser les données
        scaled_data_function = scale_data(self.data_fake, scaler_path)

        # Utiliser le scaler directement pour normaliser les données
        scaled_data_direct = pd.DataFrame(scaler_fake.transform(self.data_fake), columns=self.data_fake.columns)

        # Vérifier que les deux résultats sont équivalents
        pd.testing.assert_frame_equal(scaled_data_function, scaled_data_direct)

if __name__ == '__main__':
    unittest.main()

# Test de la fonction cross_validation_selection

class TestCrossValidationSelection(unittest.TestCase):
    def setUp(self):
        
        self.data_fake = pd.read_csv("src/data/train.csv")
        score_type = CategoricalDtype(categories=['E', 'D', 'C', 'B', 'A'], ordered=True)
        self.data_fake['score'] = self.data_fake['score'].astype(score_type)
        
        # Recupération des données pour les tests

        X_train = self.data_fake.drop(labels = ['score'], axis=1)
        y_train = self.data_fake['score']

        self.X_train = scale_data(X_train, 'src/tools/scaler.pkl')
        self.y_train = y_train

    def tearDown(self):
        pass

    def test_cross_validation_selection(self):
        k = 5
        plot_results = False
        save_path = ""

        best_model, performance_per_split = cross_validation_selection(self.X_train, self.y_train, k, plot_results, save_path)

        # Vérifier que le modèle retourné est une instance de OrderedModel
        self.assertTrue(isinstance(best_model, (OrderedModel, OrderedResultsWrapper)))

        # Vérifier que la liste des performances par split a la longueur attendue (k)
        self.assertEqual(len(performance_per_split), k)

if __name__ == '__main__':
    unittest.main(verbosity=2)

