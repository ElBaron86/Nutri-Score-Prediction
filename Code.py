'''-----------------------------------------------
Importation des outils nécessaires 
-----------------------------------------------'''
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


'''-----------------------------------------------
Chargement et nettoyage de la base de données 
-----------------------------------------------'''

food= pd.read_csv("data_clean_principal.csv", sep=',', header=0, encoding='utf-8', low_memory=False)

food.drop_duplicates(inplace=True) #Enlève les doublons
print(food.head()) #Vérifie que la base est bien importée

# Nombre de colonnes et de lignes avant "nettoyage"
lignes, colonnes = food.shape
print("Avant nettoyage on compte : {} lignes et {} colonnes".format(lignes,colonnes))

# On enlève les colonnes qui ne nous seront pas utiles (exemple date de la dernière mise à jour des infos) ATTENTION: voir pour en enlever d'autres, juste un exemple
columns_to_drop = ['code', 'creator','countries','countries_fr','brands','created_t','last_modified_t']
food = food.drop(columns=columns_to_drop)

columns_to_check = [0, 1, 2, 3, 4, 5,6, 25]

rows_to_drop = []

# Parcoure les lignes du DataFrame et repérez les lignes avec des NA dans les colonnes spécifiées
for index, row in food.iterrows():
    if row[columns_to_check].isna().any():
        rows_to_drop.append(index)

# Supprimez les lignes avec des NA en utilisant la méthode drop()
food = food.drop(index=rows_to_drop)


# Nombre de colonnes et de lignes "nettoyage" + vérification que les deux drop ont fonctionné
lignes, colonnes = food.shape
print("Après nettoyage on compte : {} lignes et {} colonnes".format(lignes,colonnes))

# Pour faire la régression il faut remplacer les lettres du nutriscire par des nombres
nutriscore_mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}

# Application de la transformation à la colonne de nutriscores
food['nutrition_grade_fr'] = food['nutrition_grade_fr'].map(nutriscore_mapping)



'''-----------------------------------------------
Régression linéaire multiple
-----------------------------------------------'''

X = food.iloc[:,0:7] #### ATTENTION : j'ai choisi ces colonnes un peu au hasard, juste pour voir si ça fonctionnait
Y = food.iloc[:,25]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, Y_train)

coefficients = model.coef_
interception = model.intercept_

print("Coefficients : ", coefficients)
print("Interception : ", interception)

Y_pred = model.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)

print("R² :", r2)
print("MSE :", mse)

'''-----------------------------------------------
Test
-----------------------------------------------'''

X_new = [1133,13.46,7.46,0.11,0.06,33.7,25.68]

# Convertissez la liste en un tableau NumPy
X_new = np.array(X_new)

# Remodelez en un tableau 2D
X_new= X_new.reshape(1, -1)


# Utilisez le modèle pour faire des prédictions sur de nouvelles données
predictions = model.predict(X_new)

# Affichez le DataFrame avec les prédictions
print("Le nutriscore vaut environ {}, soit la lettre {}".format(predictions, 'd')) # Voir pour arrondir "prédictions" et mettre la lettre correspondante
