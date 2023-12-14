# Estimation du nutri-score des aliments (Projet Académique)

![Logo](https://github.com/Alfex-1/Projet_digital/blob/main/docs/nutri.jpg)

Le Nutri-Score est un système d'étiquetage nutritionnel largement adopté à l'échelle nationale, destiné à éclairer les consommateurs sur la qualité nutritionnelle des produits alimentaires transformés. Forts de cet outil, nous avons entrepris un projet académique visant à prédire, au moyen d'un modèle statistique, le nutri-score d'un produits en fonction de ses caractéristiques nutritionnelles. Cette initiative a pour vocation d'assister les consommateurs dans leurs choix alimentaires en favorisant des options plus saines et mieux éclairées.

L'équipe derrière ce projet est constituée d'étudiants de Master 2 de Statistique pour l'Evaluation et la Prévision (SEP) de l'Université de Reims Champagne Ardenne, promotion 2023/2024. Notre démarche s'inscrit dans le souhait de développer une application à l'attention de toute personne désireuse de bénéficier d'une information claire et rapide sur la qualité nutritionnelle d'un produit alimentaire.


## Prérequis
Les conditions préalables pour exploiter efficacement ce projet varient selon l'utilisation que vous comptez en faire. Voici les recommandations spécifiques :

### Utilisation de l'Interface de Calcul du Nutri-Score :

1. **Disposer d'un environnement Windows**
2. **Connexion Internet :** Une connexion internet est indispensable pour accéder à l'API utilisée par notre interface.
3. **Microsoft Excel :** Assurez-vous d'avoir Microsoft Excel installé sur votre machine pour pouvoir utilisé notre  interface.

### Utilisation de l'Algorithme de Calcul du Nutri-Score :

1. **Installation de Python :** Pour utiliser notre algorithme, veuillez installer Python dans sa version 3.9. Vous pouvez la télécharger  sur [python.org](https://www.python.org/).

   
## Structure du dépôt 

- __docs__ : Les supports business présentation ainsi que le rapport écrit de notre projet.
    - **`\demos`** : Les vidéos de démonstration à chaque étape de notre projet.      
- __src__         
    - **`\api`** : Application qui permet l'envoi des informations sur le formulaire d'entrée, les prédictions par le modèle puis le renvoi de la réponse au dashboard.     
    - **`\data`** : Dossier où on retrouve tous les fichiers .csv, en particulier la base de donnée nettoyée (data_clean.csv).        
    - **`\interfaces`** : On retrouve le fichier au format .xlsm de l'application, regroupant le formulaire d'entrée ainsi que le tableau de bord de réponse.        
    - **`\tools`** : On y retrouve tous les codes Python, en particulier les fonctions de sélections de variables et les modèles de prédictions.       
- __tests__ : Dans ce dossier vous retrouverez tous les tests unitaires effectués sur les fonctions présentent dans le dossier tools.       
- __README.md__ : Le présent message que vous lisez actuellement         
- __requirement.txt__ : Fichier contenant la liste de tous les modules nécessaires à l'éxecution des codes Python du projet.        

## Installation

1. **Clonez le dépôt GitHub sur votre machine locale:** git clone https://github.com/Alfex-1/Projet_digital.git
2. **Installez les dépendances requises:**
```bash
pip install -r requirements.txt
```

## Utilisation

Deux modèles RandomForest sont déjà entrainés et à disposition au format pickle (**`random_forest_prod.pickle`** et **`random_forest_conso.pickle`**) dans le répertoire **`src\tools`**. L'un est à disposition des **consommateurs** et l'autre à l'attention des **producteurs**.

Si vous voulez Générer un nouveau modèle, il vous suffit de lancer dans un terminal le script `make_random_forest.py` disponible dans ce même répertoire. Il vous permettra de créer un nouveau modèle RandomForest cosommateur ou producteur. Avant de l'exécuter, assurez vous d'être dans le répertoire `src\tools`.  

**Exécutez le script:** 
```bash
python main_random_forest.py  
```
Faites un choix entre **P** pour construire le modèle producteur et **C** pour le modèle consommateur.  
Une fois le modèle construit, il vous sera demandé si vous souhaitez enregistrer le modèle produit (**O**/**N**).

Pour utiliser l'application afin d'estimer le nutri-score d'un aliment, il vous suffit d'ouvrir le fichier **`~\src\interface\dashboard_nutri_score.xlsm`**.  
Afin de n'avoir aucun conflit, il faut que vous autorisiez les fichiers munis de macros, pour cela, faites **click droit** sur le fichier concerné, vous allez dans **propriétés**, puis tout en bas de l'onglet **Général** vous cochez la case **Débloquer**.

# Contributeurs

- Alexandre Brunet **`@Alfex-1`** (Scrum Master) ;
- Elif Ertas **`@Elifets`** (Product Owner) ;
- Carlos Kpadondou **`@Data1User`** (Data Scientist) ;
- Manon Jupin **`@ManJup1`** (Data Governance) ;
- Léo Gabet **`@Meetic08`** (Front/User Interface) ;
- Jaurès Ememaga **`@ElBaron86`** (Data Engineer).
