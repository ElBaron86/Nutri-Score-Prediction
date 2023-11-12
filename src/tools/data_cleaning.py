import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import unittest
import os
from typing import List, Tuple

# Checking directory #
current_directory = os.getcwd()
print("Current directory :", current_directory)
if current_directory.endswith('tools'):
    print("You are in the right directory")
else :
    print("Please place yourself in the 'src/tools' folder")


# Function to read file #

def read_file(path : str) -> pd.DataFrame:
    """
    This function takes the way to a CSV or Excel file as a starter and returns a pandas dataframe.
    
    Parameters:
        path (str): The path to the CSV or Excel file.
        
    Returns:
        pd.DataFrame: A dataframe pandas containing the data from the file.
    """
    # Check the file extension to determine the format
    if path.endswith('.csv'):
        # Read a CSV file
        df = pd.read_csv(path)
    else:
        # If the extension is not CSV, displays an error message
        raise ValueError("File format not supported. Use a CSV file.")
    
    return df

# Unit test on the read_file function

class TestReadFile(unittest.TestCase):
    def test_read_csv_file(self):
        path_file = "../data/data_clean.csv"
        df_csv = read_file(path_file)
        self.assertIsInstance(df_csv, pd.DataFrame, "Reading the CSV file has failed.")

# Perform the test
if __name__ == '__main__':
    unittest.main(verbosity=2) # `verbosity` to display a detailed error message

# Selection of columns useful for Nutri-Score #

list_all_columns = ["energy_100g", "fat_100g", "saturated-fat_100g", "trans-fat_100g", "cholesterol_100g",
                    "carbohydrates_100g", "sugars_100g", "fiber_100g", "proteins_100g", "salt_100g", "sodium_100g",
                    "vitamin-a_100g", "vitamin-c_100g", "calcium_100g", "iron_100g", "score"]  # `score` is the nutrition_grade_fr transformed to int

def select_columns(df: pd.DataFrame, list_columns: List[str] = list_all_columns) -> pd.DataFrame:
    """
    This function takes a dataframe and a list of column names,
    and returns a new dataframe containing only the specified columns.

    Parameters:
        df (pd.DataFrame): The original dataframe.
        list_columns (List[str]): The list of column names to be selected.

    Returns:
        pd.DataFrame: A new dataframe with only the specified columns.
        
    Raises:
        ValueError: If any column in list_columns is not present in df.
    """
    # Check that all columns in list_columns exist in df
    missing_columns = [col for col in list_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"The dataframe does not contain the expected columns: {missing_columns}")

    # Select the specified columns
    df_selected_columns = df[list_columns]
    # drop missing values
    df_selected_columns.dropna(inplace=True)

    return df_selected_columns

# Unit test for the select_columns function

class TestSelectColumns(unittest.TestCase):
    def test_select_columns(self):
        # Creates a test dataframe
        data = {'column1': [1, 2, 3], 'column2': [4, 5, 6], 'column3': [7, 8, 9]}
        df_test = pd.DataFrame(data)

        # Select some columns
        list_columns = ['column1', 'column3']
        df_result = select_columns(df_test, list_columns)

        # Check that the new dataframe has the correct columns
        self.assertListEqual(list(df_result.columns), list_columns, "The selection of columns failed.")

    def test_missing_columns(self):
        # Creates a test dataframe
        data = {'column1': [1, 2, 3], 'column2': [4, 5, 6]}
        df_test = pd.DataFrame(data)

        # Attempt to select missing columns
        list_columns = ['column1', 'column3']

        # Check that a ValueError is raised with the correct message
        with self.assertRaises(ValueError) as context:
            select_columns(df_test, list_columns)

        self.assertIn("The dataframe does not contain the expected columns", str(context.exception))

# Perform the test for select_columns function
if __name__ == '__main__':
    unittest.main(verbosity=2)
    
# Function to split data into train and test sets

def split_train_test(df: pd.DataFrame, test_ratio: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function takes a dataframe and a test ratio,
    and returns two dataframes: one for training and one for testing.

    Parameters:
        df (pd.DataFrame): The original dataframe.
        test_ratio (float): The ratio of the dataframe to be used for testing.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and testing dataframes.
    """
    # Split the dataframe using train_test_split
    train_set, test_set = train_test_split(df, test_size=test_ratio, random_state=42)

    return train_set, test_set

# Unit test for the split_train_test function
class TestSplitTrainTest(unittest.TestCase):
    def test_split_train_test(self):
        # Creates a test dataframe
        data = {'column1': [1, 2, 3, 4, 5], 'column2': [10, 20, 30, 40, 50]}
        df_test = pd.DataFrame(data)

        # Split the dataframe
        train_set, test_set = split_train_test(df_test, test_ratio=0.2)

        # Check that the sizes of the sets are as expected
        self.assertEqual(len(train_set) + len(test_set), len(df_test), "The sizes of the sets do not match.")

if __name__ == '__main__':
    unittest.main(verbosity=2)

    
    
