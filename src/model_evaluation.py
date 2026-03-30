import json
import os
import pickle
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.logger import Logger
from typing import Union, Dict, Any

# Initialize logger
logger = Logger('model_evaluation', level="DEBUG").get_logger()

def load_model(model_path: str) -> Any:
    """Loads a trained model from the specified path.
    
    Args:
        model_path (str): The file path to the saved model.
    
    Returns:
        ClassifierMixin: The loaded model.
    """
    try:
        logger.debug(f"Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error occurred while loading model: {e}")
        raise

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluates the model on the test data and returns performance metrics.
    
    Args:
        model (Any): The trained model to be evaluated.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The true labels for the test data.
    
    Returns:
        Dict[str, float]: A dictionary containing accuracy, precision, recall, and F1-score.
    """
    try:
        logger.debug("Starting model evaluation.")
        y_pred = model.predict(X_test) 
        logger.debug("Predictions made successfully.")
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        logger.debug(f"Evaluation metrics calculated: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error occurred during model evaluation: {e}")
        raise

def save_metrics(metrics: Dict[str, float], file_path: str) -> None:
    """Saves the evaluation metrics to a JSON file.
    
    Args:
        metrics (Dict[str, float]): The evaluation metrics to be saved.
        file_path (str): The file path where the metrics will be saved.
    
    Returns:
        None
    """
    try:
        logger.debug(f"Saving evaluation metrics to: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.debug("Evaluation metrics saved successfully.")
    except Exception as e:
        logger.error(f"Error occurred while saving evaluation metrics: {e}")
        raise

def main():
    try:
        model_path = './artifacts/models/logistic_regression_model.pkl'
        test_data_path = './data/processed/transformed_data/transformed_X_test.csv'
        test_labels_path = './data/processed/split/y_test.csv'
        metrics_output_path = './artifacts/reports/metrics.json'
        
        logger.info("Starting model evaluation process.")
        model = load_model(model_path)
        
        X_test = pd.read_csv(test_data_path)
        logger.debug(f"Test features loaded with shape: {X_test.shape}")
        
        y_test = pd.read_csv(test_labels_path)
        logger.debug(f"Test labels loaded with shape: {y_test.shape}")
        
        metrics = evaluate_model(model, X_test, pd.Series(y_test.iloc[:, 0]))
        save_metrics(metrics, metrics_output_path)
        logger.info("Model evaluation process completed successfully.")
    except Exception as e:
        logger.error(f"Error occurred in the main function of model evaluation: {e}")
        raise

if __name__ == "__main__":
    main()