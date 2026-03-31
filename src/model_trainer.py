import http

from matplotlib.pyplot import grid
import mlflow
import mlflow.sklearn
import dagshub
import json

from sklearn.model_selection import GridSearchCV

from utils.logger import Logger
from utils.yaml_loader import load_yaml
from src.data_ingestion import load_data_from_csv

import pandas as pd 

from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin

import pickle
from typing import Union
import os

dagshub.init(repo_owner='10orabh', repo_name='olist-chrun-prediction-version-2', mlflow=True)

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

def best_hyperparameters(param_grid: dict,modelname: str, X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """Performs hyperparameter tuning using GridSearchCV to find the best parameters for the model.
    
    Args:
        param_grid (dict): A dictionary containing the hyperparameters to be tuned and their respective values.
        modelname (str): The name of the model for which hyperparameters are to be tuned.
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The target variable for training.
        
        Returns:
            tuple: A tuple containing the best hyperparameters, best model, and best score for the model.
        """
    try:
        logger.debug("Starting hyperparameter tuning using GridSearchCV.")
        if modelname == 'logistic regression':
            model = LogisticRegression()
        else:
            raise ValueError(f"Model {modelname} not supported for hyperparameter tuning")

        grid_search = GridSearchCV(
            estimator=model, 
            param_grid=param_grid, 
            cv=5, 
            n_jobs=-1,
            scoring='precision')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        logger.debug(f"Best hyperparameters found: {best_params}")
        return best_params,best_model
    except Exception as e:
        logger.error(f"Error occurred during hyperparameter tuning: {e}")
        raise
def main():
    try:
        logger.info("Starting model training process")
        # Load preprocessed data
        X_train = load_data_from_csv('./data/processed/transformed_data/transformed_X_train.csv')
        y_train = load_data_from_csv('./data/processed/split/y_train.csv')

        # Train model
        params = load_yaml('./params.yaml')
        model_name = params['model_trainer']['model_name']
        params_grid = params['model_trainer']['param_grid']
        mlflow.set_tracking_uri("https://dagshub.com/10orabh/olist-chrun-prediction-version-2.mlflow")
        mlflow.set_experiment("Churn Prediction Experiment")
        with mlflow.start_run() as run:
            
            best_params, best_model = best_hyperparameters(params_grid, model_name, X_train, y_train.iloc[:, 0])
            
            for param_name, param_value in best_params.items():
                mlflow.log_param(param_name, param_value)
            mlflow.sklearn.log_model(best_model, "model") #type: ignore
            runid = run.info.run_id

            logger.debug(f"Model training run logged with run ID: {runid}")
            with open('./run_info.json', 'w') as f:
                json.dump({"run_id": runid}, f, indent=4)
        #Save model
        save_model(best_model, 'logistic_regression_model', './artifacts/models')
        logger.debug("Trained model saved successfully.")
       
    except Exception as e:
        logger.error(f"Error occurred in the main function of model training: {e}")
        raise

if __name__ == "__main__":
    
    main()