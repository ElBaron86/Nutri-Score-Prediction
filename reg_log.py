#Packages
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

#Importation de la base de données
data = pd.read_csv("general_data.csv", sep=';')


# Cas de la régression logistique simple
X = data[['DistanceFromHome']]
y = data['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle de régression logistique
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# Évaluation des performances
accuracy = accuracy_score(y_test, y_pred)
print("Précision du modèle : {:.2f}".format(accuracy))

# Résumé de la régression logistique
summary = pd.DataFrame({'Variable': X.columns, 'Coefficient': logreg.coef_[0]})
print(summary)

# Représentation visuelle
plt.figure(figsize=(8, 6))
sns.scatterplot(x='DistanceFromHome', y='Attrition', data=data)
sns.lineplot(x=X_test['DistanceFromHome'], y=y_pred, color='red', linewidth=2)
plt.xlabel('DistanceFromHome', fontsize=12)
plt.ylabel('Attrition', fontsize=12)
plt.title('Régression logistique - Prédiction de Médaille en fonction de Marathon', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(['Prédiction'], loc='upper right', fontsize=10)
plt.tight_layout()
plt.grid(True)
plt.show()

# Cas de la régression logistique multiple
X2 = data[['Marathon', '10K', '5K', '1500m']]
y2 = data['Medaille']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Création et ajustement du modèle de régression logistique
logreg2 = LogisticRegression()
logreg2.fit(X2_train, y2_train)

# Prédiction sur l'ensemble de test
y2_pred = logreg2.predict(X2_test)

# Évaluation des performances
accuracy2 = accuracy_score(y2_test, y2_pred)
print("Précision du modèle : {:.2f}".format(accuracy2))

# Résumé de la régression logistique
summary2 = pd.DataFrame({'Variable': X2.columns, 'Coefficient': logreg2.coef_[0]})
print(summary2)