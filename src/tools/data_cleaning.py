"""Ce script rassemble l'ensemble des fonctions utilisees pour le traitement des données. 
 Ces fonctions serviront a lire, nettoyer et selectionner les variables dont nous aurons besoin pour mener a
 bien le projet. 
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import unittest
import os
from typing import List, Tuple


# Fonction pour charger un fichier csv

def read_file(path : str) -> pd.DataFrame:
    """
    This function takes the way to a CSV or Excel file as a starter and returns a pandas dataframe.
    
    Parameters:
        path (str): The path to the CSV or Excel file.
        
    Returns:
        pd.DataFrame: A dataframe pandas containing the data from the file.
    """
    # Check the file extension to determine the format
    if path.endswith('.csv'):
        # Read a CSV file
        df = pd.read_csv(path)
    else:
        # If the extension is not CSV, displays an error message
        raise ValueError("File format not supported. Use a CSV file.")
    
    return df


# Selection des colonnes utiles pour le calcul du Nutri-Score #

list_all_columns = ["energy_100g", "fat_100g", "saturated-fat_100g", "trans-fat_100g", "cholesterol_100g",
                    "carbohydrates_100g", "sugars_100g", "fiber_100g", "proteins_100g", "salt_100g", "sodium_100g",
                    "vitamin-a_100g", "vitamin-c_100g", "calcium_100g", "iron_100g", "score"]  # `score` is the nutrition_grade_fr transformed to int

def select_columns(df: pd.DataFrame, list_columns: List[str] = list_all_columns) -> pd.DataFrame:
    """
    Cette fonction prend un dataframe et une liste de noms de colonnes,
et renvoie un nouveau DataFrame contenant uniquement les colonnes specifiees.

    Parameters:
        df (pd.DataFrame): Le DataFrame d'origine.
        list_columns (List[str]): La liste des noms de colonne a selectionner.

    Returns:
        pd.DataFrame: Un nouveau DataFrame avec uniquement les colonnes specifiees.
        
    Raises:
        ValueError: Si une colonne dans list_columns n'est pas présente dans df.
    """
    # Vérifiez que toutes les colonnes de list_columns existent dans df
    missing_columns = [col for col in list_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Le DataFrame ne contient pas les colonnes attendues: {missing_columns}")

    # Selection des colonnes spécifiées
    df_selected_columns = df[list_columns]
    # Suppression des valeurs manquantes
    df_selected_columns.dropna(inplace=True)

    return df_selected_columns


    
# Fonction pour diviser les données en train et ensembles de test

def split_train_test(df: pd.DataFrame, test_ratio: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cette fonction prend un dataframe et un rapport de test,
    et renvoie deux dataframes: un pour la formation et un pour les tests.

    Parameters:
        df (pd.DataFrame): Le DataFrame d'origine.
        test_ratio (float): Le ratio de la dataframe à utiliser pour les tests.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Un tuple contenant la formation et le test de données de données.
    """
    # Split the dataframe using train_test_split
    train_set, test_set = train_test_split(df, test_size=test_ratio, random_state=42)

    return train_set, test_set

