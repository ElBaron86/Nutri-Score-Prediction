# Importation des modules
import argparse
import numpy as np
import pickle
from mord import LogisticAT

def load_model(model_path):
    """Fonction pour charger le modèle ordinal à partir du fichier pickle

    Args:
        model_path (str): Chemin du fichier pickle contenant le modèle

    Returns:
        model : Modèle ordinal chargé
    """
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

def score_probas(model, input_values):
    """Fonction pour obtenir les probabilités de classe pour une liste de valeurs d'entrée

    Args:
        model : Modèle utilisé
        input_values (list): Liste des valeurs d'entrée

    Returns:
        probas : Vecteur des 5 probabilités de classes rangées dans cet ordre : [E, D, C, B, A] 
    """
    input_array = np.array(input_values).reshape(1, -1)
    probas = model.predict_proba(input_array)
    probas = np.round(probas, decimals=2)  # J'arrondis à 2 décimales
    print(probas)
    return probas

if __name__ == "__main__":
    # Paramètres en ligne de commande
    parser = argparse.ArgumentParser(description='Faire des prédictions avec le modèle ordinal.')
    parser.add_argument('model_path', type=str, help='Chemin du fichier pickle du modèle ordinal')
    parser.add_argument('input_values', type=float, nargs='+', help='Liste des valeurs d\'entrée pour les prédictions')
    args = parser.parse_args()

    # Chargement du modèle
    model_path = args.model_path
    model_ordinal = load_model(model_path)

    # Prédictions
    input_values = args.input_values
    probas = score_probas(model_ordinal, input_values)

