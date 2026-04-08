import mlflow
import mlflow.sklearn
import dagshub
import json
import os
import pickle
import importlib
import pandas as pd 
import numpy as np 

from utils.logger import Logger
from utils.yaml_loader import load_yaml
from sklearn.base import ClassifierMixin

dagshub.init(repo_owner='10orabh', repo_name='olist-chrun-prediction-version-2', mlflow=True)
logger = Logger('model_training', level="DEBUG").get_logger()

def save_model(model, file_name, file_path):
    try:
        os.makedirs(file_path, exist_ok=True)
        full_path = os.path.join(file_path, f"{file_name}.pkl")
        with open(full_path, 'wb') as f:
            pickle.dump(model, f)
        return full_path
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

def main():
    try:
        train_arr = np.load('./data/processed/train.npy') 
        logger.info("Train data loaded")
        X_train = train_arr[:, :-1]
        y_train = train_arr[:, -1]
        logger.info(f"Train data splitted into X_train with shape {X_train.shape} and y_train with shape {y_train.shape}")

        logger.info("Models and paramters are laoding from yaml file")
        config = load_yaml('./params.yaml')
        model_info = config['model_trainer']['models']['LogisticRegression']
        
        module = importlib.import_module(model_info['module'])
        model_class = getattr(module, model_info['class_name'])
        model = model_class(**model_info['best_params'])

        logger.info("Parameter and Model are loaded.")
        logger.info("Start Experimenting ...")
        mlflow.set_tracking_uri("https://dagshub.com/10orabh/olist-chrun-prediction-version-2.mlflow")
        mlflow.set_experiment("Churn Prediction Experiment")

        with mlflow.start_run() as run:
            logger.info("Model training Started...")
            model.fit(X_train, y_train)
            logger.info("Model Training complete!")
            mlflow.log_params(model_info['best_params'])
            mlflow.sklearn.log_model(model, "model")
            
            run_id = run.info.run_id
            with open('./run_info.json', 'w') as f:
                json.dump({"run_id": run_id}, f, indent=4)
            logger.info(f"Experiment run_id saved")
        save_model(model, 'best_model', './artifacts/models')
        logger.info("Model saved")
        logger.info("Success.")

    except Exception as e:
        
        logger.error(f"Failed: {e}")
        raise

if __name__ == "__main__":
    main()