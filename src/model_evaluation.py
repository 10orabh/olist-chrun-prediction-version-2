import mlflow
from mlflow.tracking import MlflowClient
import json
import os
import pickle
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.logger import Logger
import dagshub
import warnings
logger = Logger('model_evaluation', level="DEBUG").get_logger()
dagshub.init(repo_owner='10orabh', repo_name='olist-chrun-prediction-version-2', mlflow=True)

warnings.filterwarnings("ignore")
def load_model(model_path):
    try:
        logger.info(f"Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    try:
        logger.info("Generating predictions for evaluation.")
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        logger.info(f"Evaluation metrics calculated: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

def save_metrics(metrics, file_path):
    try:
        logger.info(f"Saving metrics to: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info("Metrics saved successfully.")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

def main():
    try:
        logger.info("Starting model evaluation stage.")
        
        model_path = './artifacts/models/best_model.pkl'
        test_data_path = './data/processed/test.npy'
        metrics_output_path = './artifacts/reports/metrics.json'

        model = load_model(model_path)
        
        logger.info(f"Loading test data from: {test_data_path}")
        test_data = np.load(test_data_path)
        X_arr = test_data[:, :-1]
        y_arr = test_data[:, -1]
        logger.info(f"Test data loaded. Shape: {X_arr.shape}")

        logger.info("Setting MLflow tracking URI.")
        mlflow.set_tracking_uri("https://dagshub.com/10orabh/olist-chrun-prediction-version-2.mlflow")

        logger.info("Reading run information from run_info.json")
        if os.path.exists('./run_info.json'):
            with open('./run_info.json', 'r') as f:
                run_info = json.load(f)
            run_id = run_info.get("run_id")
            logger.info(f"Run ID found: {run_id}")
        else:
            raise FileNotFoundError("run_info.json not found.")

        metrics = evaluate_model(model, X_arr, y_arr)
        
        logger.info(f"Logging metrics to MLflow for run_id: {run_id}")
        client = MlflowClient()
        for metric_name, metric_value in metrics.items():
            client.log_metric(run_id, metric_name, metric_value)
        logger.info("Metrics logged to MLflow successfully.")

        save_metrics(metrics, metrics_output_path)
        
        logger.info("Model evaluation stage completed successfully.")
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise

if __name__ == "__main__":
    main()