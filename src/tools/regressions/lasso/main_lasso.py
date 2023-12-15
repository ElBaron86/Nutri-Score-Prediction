'''
 # @ Author: Jaures Ememaga
 # @ Create Time: 2023-11-10 16:23:51
 # @ Modified by: 
 # @ Modified time: 
 # @ Description: This script makes it possible to build the penalized model Lasso
 '''


# Basic modules
import sys
import os
import logging

# Data manipulation
import pandas as pd
from sklearn.model_selection import train_test_split

# Move up to the "Projet_digital" directory
while os.path.basename(os.getcwd()) != "src":
    os.chdir("..")
sys.path.append(os.getcwd())

# Importing our function
from tools.regressions.lasso.tools_lasso import train_and_evaluate_lasso_model  # Import the Lasso function

# Configure logging
logging.basicConfig(filename='tools/regressions/lasso/lasso_model.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Loading dataset
df = pd.read_csv(r'data/data_clean.csv')

# We recover the explanatory variables and the labels
X = df.drop(labels=['score', 'product_name'], axis=1).values
y = df['score'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regularization parameter (ALPHA) for Lasso regression
ALPHA = 1.0

# Train and evaluate the Lasso model
classification_error, coefficients, intercept = train_and_evaluate_lasso_model(
    X_train, y_train, X_test, y_test, ALPHA
)

# Log the results
logging.info("Lasso Model Results:")
logging.info("Classification Error: {classification_error}")
logging.info("Coefficients: {coefficients}")
logging.info("Intercept: {intercept}")
