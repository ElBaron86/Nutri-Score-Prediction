'''
 # @ Author: KPADONOU Carlos
 # @ Create Time: 2023-11-19 17:50:36
 # @ Modified by: KPADONOU Carlos
 # @ Modified time: 
 # @ Description: Ce code effectue la sélection de variable dans le cadre de
 la regression logistique ordinale : Backward
 '''


from feature_selection import cross_validation_,variable_selection_ordered_model_aic,variable_selection_ordered_model_bic
import os
import pandas as pd

script_dir=os.path.dirname(os.path.realpath(__file__))
train_file_path=os.path.join(script_directory, '..', 'data', 'train.csv')

# Chargement des données depuis un fichier CSV
df_train = pd.read_csv(train_file_path)

# Séparation des caractéristiques (X_train) et de la variable cible (Y_train)
X_train = df_train.drop(labels=['score'], axis=1)
Y_train = df_train['score']

# Normalisation des caractéristiques avec StandardScaler
scale = StandardScaler()
X_train_scale = scale.fit_transform(X_train)

# Définition du type de score comme catégorique ordonné
score_type = CategoricalDtype(categories=['E', 'D', 'C', 'B', 'A'], ordered=True)
Y_train = Y_train.astype(score_type)

# Conversion des caractéristiques normalisées en DataFrame
X_train_scale = pd.DataFrame(X_train_scale)
X_train_scale.columns = X_train.columns.tolist()

# rendre numérique les lables pour le calcul de l'accuracy_score

Y_train_predict=Y_train.map({'E': 0, 'D': 1, 'C': 2, 'B': 3, 'A': 4})

# Sélection de modèle par AIC
best_variables_aic, aic_values, unused_variables_aic = variable_selection_ordered_model_aic(X_train_scale, Y_train)

#calculer de l'erreur théorique pour le modèle sélectionné selon aic
X_train_aic = X_train_scale.drop(labels=['sodium_100g', 'vitamin-a_100g', 'proteins_100g'], axis=1)
mse_aic = cross_validation_(k=10, X=X_train_aic, y=Y_train_predict)
print(f"L'erreur de classification du modèle sélectionné par AIC est : {mse_aic}")

# Sélection de modèle par BIC
best_variables_bic, bic_values, unused_variables_bic = variable_selection_ordered_model_bic(X_train_scale, Y_train)

#calculer de l'erreur théorique pour le modèle sélectionné selon bic
X_train_bic = X_train_scale.drop(labels=['sodium_100g', 'vitamin-a_100g', 'proteins_100g', 'energy_100g'], axis=1)
mse_bic = cross_validation_(k=10, X=X_train_bic, y=Y_train_predict)
print(f"L'erreur de classification du modèle sélectionné par BIC est : {mse_bic}")