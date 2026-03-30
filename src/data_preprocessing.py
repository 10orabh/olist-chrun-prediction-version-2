import pandas as pd 
import numpy as np 
from utils.logger import Logger
from typing import Union
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.data_ingestion import save_data_to_csv,load_data_from_csv
import os



logger = Logger('data_preprocessing', level="DEBUG").get_logger()

def data_clean(data: pd.DataFrame) -> pd.DataFrame:
    """Cleans the data by handling missing values, encoding categorical variables, and performing feature engineering.
    
    Args:
        data (pd.DataFrame): The raw data to be cleaned.
    
    Returns:
        pd.DataFrame: The cleaned data.
    """
    try:
        logger.debug("Starting data cleaning process.")
        # Example cleaning steps
        data = data.drop_duplicates()    
        logger.debug("Data cleaning completed.")
        return data
    except Exception as e:
        logger.error(f"Error occurred while cleaning data: {e}")
        raise



def data_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data by performing scaling, encoding, and feature selection.
    
    Args:
        data (pd.DataFrame): The cleaned data to be preprocessed.
        
    Returns:
        pd.DataFrame: The preprocessed data ready for modeling.
    """
    
    try:
        logger.debug("Starting data preprocessing.")
        # feature selection 
        data = data.drop(columns=['customer_unique_id','recency']) 
        logger.debug("Dropped unnecessary columns: 'customer_unique_id', 'recency'")
        
        # feature encoding stage 
        categorical_cols = data.select_dtypes(include=['object']).columns   
        logger.debug(f"Categorical columns identified for encoding: {categorical_cols.tolist()}")
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_features = encoder.fit_transform(data[categorical_cols])
        logger.debug("Categorical variables encoded successfully.")
        logger.debug("Data preprocessing completed.")
        return data
    except Exception as e:
        logger.error(f"Error occurred while preprocessing data: {e}")
        raise

def save_preprocessed_data(data: pd.DataFrame, file_name: str, file_path: str) -> None:
    """Saves the preprocessed data to a CSV file.
    
    Args:
        data (pd.DataFrame): The preprocessed data to be saved.
        file_name (str): The name of the CSV file to save the data to.
        file_path (str): The directory path where the CSV file will be saved.
    
    Returns:
        None
    """
    try:
        logger.debug("Saving preprocessed data to CSV.")
        save_data_to_csv(data, file_name, file_path)
        logger.debug("Preprocessed data saved successfully.")
    except Exception as e:
        logger.error(f"Error occurred while saving preprocessed data: {e}")

def main():
    try:
        uncleaned_data_path = './data/processed'
        preprocessed_data_path = './data/processed'
        
        logger.info("Starting data preprocessing process")
       
        
        train_data = load_data_from_csv(os.path.join(uncleaned_data_path, 'train_data.csv'))
        logger.debug(f"Train data loaded with shape: {train_data.shape}")
        test_data = load_data_from_csv(os.path.join(uncleaned_data_path, 'test_data.csv'))
        logger.debug(f"Test data loaded with shape: {test_data.shape}")
        
        # Clean data
        cleaned_train_data = data_clean(train_data)
        logger.debug(f"Cleaned train data shape: {cleaned_train_data.shape}")
        cleaned_test_data = data_clean(test_data)
        logger.debug(f"Cleaned test data shape: {cleaned_test_data.shape}")

        # Preprocess data
        preprocessed_train_data = data_preprocessing(cleaned_train_data)
        logger.debug(f"Preprocessed train data shape: {preprocessed_train_data.shape}")
        preprocessed_test_data = data_preprocessing(cleaned_test_data)
        logger.debug(f"Preprocessed test data shape: {preprocessed_test_data.shape}")

        # Save preprocessed data
        save_preprocessed_data(preprocessed_train_data, 'preprocessed_train_data', './data/processed')
        save_preprocessed_data(preprocessed_test_data, 'preprocessed_test_data', './data/processed')
        logger.info("Data preprocessing process completed successfully")
    except Exception as e:
        logger.error(f"Error occurred in the main function of data preprocessing: {e}")
        raise

if __name__ == "__main__":
    main()