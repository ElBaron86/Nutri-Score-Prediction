'''
 # @ Author: Jaures Ememaga
 # @ Create Time: 2023-11-05 21:50:13
 # @ Modified by: 
 # @ Modified time: 
 # @ Description: Main data cleaning function.
 '''



import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean the data, including variable selection and data balancing based on the occurrence count per score.

    Args:
        data (pd.DataFrame): The base DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with selected variables and balanced scores.
    """

    # Select relevant features
    features = pd.concat([data['product_name'], data.filter(regex='_100g')], axis=1)
    features.drop(labels='nutrition-score-fr_100g', axis=1, inplace=True)

    # Extract labels (scores) and create a new DataFrame
    labels = data['nutrition_grade_fr'].str.upper()
    filtered_data = pd.concat([features, labels], axis=1)
    filtered_data.dropna(inplace=True)
    filtered_data.rename(columns={"nutrition_grade_fr": "score"}, inplace=True)

        # Find the number of samples in the least represented class
    min_samples = filtered_data['score'].value_counts().min()

    # Random undersampling for each class
    balanced_data = filtered_data.groupby('score', group_keys=False).apply(lambda x: resample(x, n_samples=min_samples, random_state=42))

    # dropping duplicated 
    balanced_data.drop_duplicates(inplace=True)

    # separate data in train and test, we drop the product names
    train_data, test_data = train_test_split(balanced_data.drop(labels='product_name', axis=1), test_size=0.3, random_state=42)

    return balanced_data, train_data, test_data
