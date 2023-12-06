'''
 # @ Author: Jaures Ememaga
 # @ Create Time: 2023-11-05 22:00:10
 # @ Modified by: 
 # @ Modified time: 
 # @ Description: Apply tools_data_cleaning.py on our data base
 '''

#### Imports ####
import sys
import os
import pandas as pd

# Importing the function
from tools_data_cleaning import clean_data

# Recovery of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Back to the project directory
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)

# data base #
data_base = pd.read_csv(r"..\data\data.csv")

# clean the data and get train ant test data
data_clean, train, test = clean_data(data=data_base)

data_clean.to_csv(r"..\data\data_clean.csv")
train.to_csv(r"..\data\train.csv")
test.to_csv(r"..\data\test.csv")