import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report
import mord as md

# Charger les données
df_aliments = pd.read_csv("data_clean_principal.csv")

# Remplacez toutes les valeurs manquantes par des zéros
df_aliments = df_aliments.fillna(0)

# Sélectionner les variables explicatives (15 premières variables) et la variable cible (33e variable)
X = df_aliments.iloc[:, :15]  # Variables explicatives
y = df_aliments.iloc[:, 32]  # Variable cible

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

###############################################################################

#Modèle de régression logistique normal

# Créer et entraîner le modèle

model1 = LogisticRegression(penalty="none",multi_class='multinomial',solver='saga')
model1.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred1 = model1.predict(X_test_scaled)

#Coefficients et erreur du modèle 
coefficients1= model1.coef_,2
print("Erreur de classification :", round(1 - accuracy_score(y_test, y_pred1),2))

###############################################################################

#Modèle de régression logistique Ridge

# Créer et entraîner le modèle
model2 = LogisticRegression(penalty="l2",multi_class='multinomial',solver='saga')
model2.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred2 = model2.predict(X_test)

#Coefficients et l'erreur du modèle
coefficients2= model2.coef_
print("Erreur de classification :", round(1 - accuracy_score(y_test, y_pred2),2))

###############################################################################

#Modèle de régression logistique Lasso

model3 = LogisticRegression(penalty="l1",multi_class='multinomial',
                            solver='saga',max_iter = 100)
model3.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred3 = model3.predict(X_test)

#Coefficients et erreur du modèle
coefficients3= model3.coef_
print("Erreur de classification :", round(1 - accuracy_score(y_test, y_pred3),2))













import pandas as pd
from pandas.api.types import CategoricalDtype
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.metrics import accuracy_score
from statsmodels.stats.outliers_influence import somersd
import mord as md
from sklearn.model_selection import train_test_split


# Charger les données
df_aliments = pd.read_csv("data_clean_principal.csv")

# Remplacez toutes les valeurs manquantes par des zéros
df_aliments = df_aliments.fillna(0)
df_aliments.dtypes

#Comme notre variable cible est du type "objet", on la convertit
cat_type = CategoricalDtype(categories=['e', 'd', 'c', 'b', 'a'], ordered=True)
df_aliments["nutrition_grade_fr"] = df_aliments["nutrition_grade_fr"].astype(cat_type)

#On revérifie son type
df_aliments["nutrition_grade_fr"].dtype
#Ici, nous pouvons voir que les valeurs sous la variable cible sont sous une forme ordonnée par catégorie.

# Sélectionner les variables explicatives (15 premières variables) et la variable cible (33e variable)
X = df_aliments.iloc[:, :15]  # Variables explicatives
y = df_aliments.iloc[:, 32]  # Variable cible

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

model = md.LogisticAT()

model.fit(X_train, y_train)