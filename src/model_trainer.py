from utils.logger import Logger
from src.data_ingestion import load_data_from_csv
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin
import pickle
from typing import Union
import os

logger = Logger('model_training', level="DEBUG").get_logger()

def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, pd.DataFrame],
    **kwargs
) -> ClassifierMixin:
    """Trains a machine learning model using the provided training data.
    
    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The target variable for training.
    
    Returns:
        ClassifierMixin: The trained machine learning model.
    """
    try:
        logger.debug("Starting model training.")

        if model_name == 'logistic regression':
            model = LogisticRegression(**kwargs)
        else:
            raise ValueError(f"Model {model_name} not supported")

        model.fit(X_train, y_train)

        logger.debug("Model training completed successfully.")
        return model

    except Exception as e:
        logger.error(f"Error occurred during model training: {e}")
        raise

def save_model(model: ClassifierMixin, file_name: str, file_path: str) -> None:
    """Saves the trained model to a file using pickle.
    
    Args:
        model (ClassifierMixin): The trained machine learning model to be saved.
        file_name (str): The name of the file to save the model to.
        file_path (str): The directory path where the model file will be saved.
    
    Returns:
        None
    """
    try:
        logger.debug("Saving trained model to file.")
        os.makedirs(file_path, exist_ok=True)
        with open(f"{file_path}/{file_name}.pkl", 'wb') as f:
            pickle.dump(model, f)
        logger.debug("Model saved successfully.")
    except Exception as e:
        logger.error(f"Error occurred while saving the model: {e}")
        raise

def main():
    try:
        logger.info("Starting model training process")
        # Load preprocessed data
        X_train = load_data_from_csv('./data/processed/transformed_data/transformed_X_train.csv')
        y_train = load_data_from_csv('./data/processed/split/y_train.csv')

        # Train model
        model = train_model('logistic regression', X_train, y_train)

        # Save model
        save_model(model, 'logistic_regression_model', './artifacts/models')
        logger.debug("Trained model saved successfully.")
       
    except Exception as e:
        logger.error(f"Error occurred in the main function of model training: {e}")
        raise

if __name__ == "__main__":
    main()