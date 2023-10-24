# Modèle de classement NutriScore pour les aliments (Projet Académique)

![Logo](https://github.com/Alfex-1/Projet_digital/blob/main/data/nutri.jpg)

Pour rappel, le NutriScore est un système d'étiquetage nutritionnel largement adopté à l'échelle nationale, destiné à éclairer les consommateurs sur la qualité nutritionnelle des produits alimentaires transformés. Forts de cet outil, nous avons entrepris un projet académique visant à prédire, au moyen d'un modèle statistique, le NutriScore d'un ou de plusieurs produits en fonction de leurs caractéristiques nutritionnelles. Cette initiative a pour vocation d'assister les consommateurs dans leurs choix alimentaires en favorisant des options plus saines et mieux éclairées.

L'équipe derrière ce projet est constituée d'étudiants du programme de Master 2 en Sciences de l'Économie et de la Prévention (SEP) de l'Université de Reims Champagne Ardenne, promotion 2023/2024. Notre démarche s'inscrit dans le souhait de développer une application à l'attention de toute personne désireuse de bénéficier d'une information claire et rapide sur la qualité nutritionnelle d'un produit alimentaire.


# Contributeurs

- Brunet Alexandre (Scrum Master)
- Ertas Elif (Product Owner)
- Kpadondou Carlos (Data scientist)
- Jupin Manon (Data governance)
- Gabet Léo (Front/User Interface)
- Otogondoua Ememag Jordhy Jean Jaurès (Data Engineer)




## Prérequis

Avant de commencer, assurez-vous d'avoir Python installé sur votre système. Vous pouvez télécharger Python depuis [python.org](https://www.python.org/).

## Installation

1. **Clonez le dépôt GitHub sur votre machine locale:**

git clone https://github.com/Alfex-1/Projet_digital.git


3. **Installez les dépendances requises:**

pip install -r requirements.txt


## Utilisation

Pour utiliser le modèle de classement NutriScore, suivez ces étapes :

1. **Préparez vos données:** Assurez-vous d'avoir un ensemble de données contenant les informations nutritionnelles des aliments, y compris les valeurs nécessaires pour calculer le NutriScore (ici on met la liste des variables qu'on aura séléctionnées au final).

4. **Faites des prédictions:** Utilisez le modèle entraîné pour faire des prédictions sur de nouveaux aliments en utilisant le script `predict.py`.
(une fois à cette étape on ajustera l'utilisation du modèle en fonction du formulaire etc...)
