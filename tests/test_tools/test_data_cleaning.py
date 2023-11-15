'''
 # @ Author: Jaures Ememaga
 # @ Create Time: 2023-11-12 17:42:06
 # @ Modified by: Jaures Ememaga
 # @ Modified time: 2023-11-15 15:41:52
 # @ Description: Test ds fonctions du script data_cleaning das src/tools
 '''

# importation des modules
import sys
import os
import pandas as pd
import unittest

# récupération du répertoire actuel
current_dir = os.path.dirname(os.path.abspath(__file__))

# retour au répertoire du projet 
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)

# Importation des fonctions de data_cleaning.py
from src.tools.data_cleaning import select_columns, split_train_test
    
# test de la fonction select_columns 

class TestSelectColumns(unittest.TestCase):
    def test_select_columns(self):
        # Creer un dataframe de test
        data = {'column1': [1, 2, 3], 'column2': [4, 5, 6], 'column3': [7, 8, 9]}
        df_test = pd.DataFrame(data)

        # Definir quelques colonnes a selectionner dans df_test
        list_columns = ['column1', 'column3']
        df_result = select_columns(df_test, list_columns)

        # On verifie que le nouveau DataFrame a les bonnes colonnes
        self.assertListEqual(list(df_result.columns), list_columns, "La sélection des colonnes a échoué.")

    def test_missing_columns(self):
        # Creer un DataFrame test
        data = {'column1': [1, 2, 3], 'column2': [4, 5, 6]}
        df_test = pd.DataFrame(data)

        # Tenter de sélectionner les colonnes manquantes
        list_columns = ['column1', 'column3']

        # Verifier que ValueError est affiche avec un message correct
        with self.assertRaises(ValueError) as context:
            select_columns(df_test, list_columns)

        self.assertIn("Le DataFrame ne contient pas les colonnes attendues", str(context.exception))

# Effectuer le test
if __name__ == '__main__':
    unittest.main(verbosity=2)
    

# test de la fonction split_train_test 
class TestSplitTrainTest(unittest.TestCase):
    def test_split_train_test(self):
        # Creates a test dataframe
        data = {'column1': [1, 2, 3, 4, 5], 'column2': [10, 20, 30, 40, 50]}
        df_test = pd.DataFrame(data)

        # Diviser les données en lots d'entrainement et de test
        train_set, test_set = split_train_test(df_test, test_ratio=0.2)

        # Verifier que les tailles des lots sont comme prévu
        self.assertEqual(len(train_set) + len(test_set), len(df_test), "Les tailles des ensembles ne correspondent pas")

if __name__ == '__main__':
    unittest.main(verbosity=2)
