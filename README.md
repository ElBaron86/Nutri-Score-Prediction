# Modèle de classement Nutri-Score pour les aliments (Projet Académique)

![Logo](https://github.com/Alfex-1/Projet_digital/blob/main/docs/nutri.jpg)

Pour rappel, le Nutri-Score est un système d'étiquetage nutritionnel largement adopté à l'échelle nationale, destiné à éclairer les consommateurs sur la qualité nutritionnelle des produits alimentaires transformés. Forts de cet outil, nous avons entrepris un projet académique visant à prédire, au moyen d'un modèle statistique, le Nutri-Score d'un ou de plusieurs produits en fonction de leurs caractéristiques nutritionnelles. Cette initiative a pour vocation d'assister les consommateurs dans leurs choix alimentaires en favorisant des options plus saines et mieux éclairées.

L'équipe derrière ce projet est constituée d'étudiants de Master 2 de Statistique pour l'évaluation et la Prévention (SEP) de l'Université de Reims Champagne Ardenne, promotion 2023/2024. Notre démarche s'inscrit dans le souhait de développer une application à l'attention de toute personne désireuse de bénéficier d'une information claire et rapide sur la qualité nutritionnelle d'un produit alimentaire.

# Structure et méthode choisie

Notre application fonctionne grâce aux logiciels VBA et Python :
- VBA sert à afficher une formulaire qui sera affiché pour que l'utilisateur puisse entrer les informations nutritionnelles d'un produit de son choix
- Python sert à entraîner un modèle de régression logistique ordinale à partir d'une base de données recensant de très nombreux produits (avec leurs informations nutritionnelles et leur nutri-score).
- VBA enverra les informations sur Python qui retournera les résultats sur un tableau de bord Excel qui montera à la fois les différentes probabilités associées à chaque nutri-score, et de plus une jauge qui montre clairement quel nutri-score a le plus de chance de correspondre aux informations rentrées par l'utilisateur.
- L'opération peut se dérouler autant de fois que l'utilisateur le souhaite. Excel gardera une trace de toutes ses saisies dans une feuille (historique).

# Contributeurs

- Brunet Alexandre (Scrum Master)
- Ertas Elif (Product Owner)
- Kpadondou Carlos (Data scientist)
- Jupin Manon (Data governance)
- Gabet Léo (Front/User Interface)
- Jaurès Ememaga (Data Engineer)




## Prérequis

Avant de commencer, assurez-vous d'avoir Python installé sur votre système. Vous pouvez télécharger Python depuis [python.org](https://www.python.org/).


## Installation

1. **Clonez le dépôt GitHub sur votre machine locale:**

git clone https://github.com/Alfex-1/Projet_digital.git


3. **Installez les dépendances requises:**

pip install -r requirements.txt


## Utilisation

Pour utiliser le modèle de classement Nutri-Score, suivez ces étapes :

1. **Préparez vos données:** Assurez-vous d'avoir un ensemble de données contenant les informations nutritionnelles des aliments, y compris les valeurs nécessaires pour calculer le Nutri-Score (ici on met la liste des variables qu'on aura séléctionnées au final).

4. **Faites des prédictions:** Utilisez le modèle entraîné pour faire des prédictions sur de nouveaux aliments en utilisant le script `predict.py`.
(une fois à cette étape on ajustera l'utilisation du modèle en fonction du formulaire etc...)
