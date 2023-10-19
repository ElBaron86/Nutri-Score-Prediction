'''-----------------------------------------------
Importation des outils nécessaires 
-----------------------------------------------'''
import pandas as pd

'''-----------------------------------------------
Chargement et nettoyage de la base de données 
-----------------------------------------------'''

food= pd.read_csv("data_clean_principal.csv", sep=',')

#Vérification que la base est bien importée
print(food.head())

#Suppression des doublons
food.drop_duplicates(inplace=True) 

#Déplacement de "nutrition_grade_fr" après "iron_100g"
#Puis on enlève les colonnes qui ne nous seront pas utiles/intraitables, etc.

colonnes = food.columns.tolist()
indice_iron = colonnes.index("iron_100g")
indice_nutrition_grade = colonnes.index("nutrition_grade_fr")

# Réorganisation la dataframe en incluant les colonnes nécessaires
food = food[colonnes[:indice_iron + 1] + colonnes[indice_nutrition_grade:]]
