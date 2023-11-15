'''
 # @ Author: KPADONOU Carlos
 # @ Create Time: 2023-11-15 18:14:14
 # @ Modified by: 
 # @ Modified time: 
 # @ Description: Ce code regroupe des fonctions pour effectuer la sélection de variable dans le cadre de
 la regression logistique ordinale : Backward
 '''

import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel


# Identifier la variable à retirer dans le cadre d'un Backward-critère p-value

def select_pvalue_high(ordinal_results_train):
    """ Cette fonction nous permet d'identifier la variable avec la plus grande p_value
        afin de le retirer pour un backward.

    Args:
        ordinal_results_train (statsmodels.miscmodels.ordinal_model.OrderedResultsWrapper): 
        le modèle sur lequel on prévoit effectue un feature séléction

    Returns:
        str : le nom de la variable avec la plus grande valeur de p_value la plus élevée
    """
    # Le tableau des coefficients est recupéré dans une variable 
    a = ordinal_results_train.summary().tables[1].data
    # Mettre le tableau sous forme dataframe est prenant la première ligne comme nom de variable
    df = pd.DataFrame(a[1:-4], columns=a[0])
    # définir la première colonne comme index du dataframe
    df.set_index(df.columns[0], inplace=True)
    # identifier la variable avec la plus grande p_value
    feature_move=df['P>|z|'].astype(float).idxmax()
    return feature_move
