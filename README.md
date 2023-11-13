# Estimation du nutri-score des aliments (Projet Académique)

![Logo](https://github.com/Alfex-1/Projet_digital/blob/main/docs/nutri.jpg)

Pour rappel, le Nutri-Score est un système d'étiquetage nutritionnel largement adopté à l'échelle nationale, destiné à éclairer les consommateurs sur la qualité nutritionnelle des produits alimentaires transformés. Forts de cet outil, nous avons entrepris un projet académique visant à prédire, au moyen d'un modèle statistique, le nutri-score d'un produits en fonction de ses caractéristiques nutritionnelles. Cette initiative a pour vocation d'assister les consommateurs dans leurs choix alimentaires en favorisant des options plus saines et mieux éclairées.

L'équipe derrière ce projet est constituée d'étudiants de Master 2 de Statistique pour l'évaluation et la Prévention (SEP) de l'Université de Reims Champagne Ardenne, promotion 2023/2024. Notre démarche s'inscrit dans le souhait de développer une application à l'attention de toute personne désireuse de bénéficier d'une information claire et rapide sur la qualité nutritionnelle d'un produit alimentaire.

# Contributeurs

- Brunet Alexandre (Scrum Master) ;
- Ertas Elif (Product Owner) ;
- Kpadondou Carlos (Data Scientist) ;
- Jupin Manon (Data Governance) ;
- Gabet Léo (Front/User Interface) ;
- Jaurès Ememaga (Data Engineer).

## Prérequis

1. Assurez-vous d'avoir Python installé sur votre système. Vous pouvez télécharger Python depuis [python.org](https://www.python.org/).
2. Disposer d'une connexion internet (afin de faire appel à l'API)

## Structure du dépôt 

- __docs__ : les supports business présentation ainsi que le rapport écrit de notre projet.
    - \demos : les vidéos de démonstration à chaque étape de notre projet.      
- __src__         
    - \api : application qui permet l'envoie des informations sur le formulaire d'entrée, l'application du modèle puis le renvoie de la réponse au dashboard.     
    - \data : dossier où on retrouve tous les fichiers .csv, en particulier la base de donnée nettoyée (data_clean.csv).        
    - \interfaces : on retrouve le fichier au format .xlsm de l'application, regroupant le formulaire d'entrée ainsi que le tableau de bord de réponse.        
    - \tools : on y retrouve tous les codes Python, en particulier les fonctions de sélections de variables et les modèles de prédictions.       
- __tests__ : dans ce dossier vous y retrouverez tous les tests unitaires effectués sur les fonctions présentent dans le fichier tools.       
- __README.md__ : le présent message que vous lisez actuellement         
- __requirement.txt__ : dans ce fichier on y retrouve la liste de tous les packages/modules nécessaires à l'éxecution des codes Python du projet.        

## Installation

1. **Clonez le dépôt GitHub sur votre machine locale:** git clone https://github.com/Alfex-1/Projet_digital.git
2. **Installez les dépendances requises:** pip install -r requirements.txt

## Utilisation

Pour utiliser l'application afin d'estimer le nutri-score d'un aliment, il vous suffit d'ouvrir le fichier ~\src\interfaces\V3_formulaire.xlsm 
Afin de n'avoir aucun conflit, il faut que vous autorisiez les fichiers munis de macros, pour cela pour faite clique droit sur le fichier concerné, vous allez dans propriétés, puis tout en bas de l'onglet Général vous cochez la case "Débloquer".
