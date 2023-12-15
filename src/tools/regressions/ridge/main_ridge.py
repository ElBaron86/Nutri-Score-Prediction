'''
 # @ Author: Jaures Ememaga
 # @ Create Time: 2023-12-15 16:54:48
 # @ Modified by: 
 # @ Modified time: 
 # @ Description:
 '''
 
# Basic modules
import sys
import os

# Data manipulation
import pandas as pd
from sklearn.model_selection import train_test_split

# Logging
import logging

# Move up to the "Projet_digital" directory
while os.path.basename(os.getcwd()) != "src":
    os.chdir("..")
sys.path.append(os.getcwd())


# Importing our function
from tools.regressions.ridge.tools_ridge import train_and_evaluate_ridge_model

# Configure logging
logging.basicConfig(filename='tools/regressions/ridge/ridge_model.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Loading dataset
df = pd.read_csv(r'data/data_clean.csv')

# We recover the explanatory variables and the labels
X = df.drop(labels=['score','product_name'], axis=1).values
y = df['score'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regularization parameter (alpha) for Ridge regression
alpha = 1.0

# Train and evaluate the Ridge model
classification_error, coefficients, intercept = train_and_evaluate_ridge_model(
    X_train, y_train, X_test, y_test, alpha
)

# Log the results
logging.info(f"Ridge Model Results:")
logging.info(f"Classification Error: {classification_error}")
logging.info(f"Coefficients: {coefficients}")
logging.info(f"Intercept: {intercept}")

print("Ridge Model Results:")
print(f"Classification Error: {classification_error}")
print(f"Coefficients: {coefficients}")
print(f"Intercept: {intercept}")
