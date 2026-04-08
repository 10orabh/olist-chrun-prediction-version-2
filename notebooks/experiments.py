import os
import json
import yaml
import numpy as np
import warnings
import mlflow
import mlflow.sklearn
import dagshub
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from utils.logger import Logger
from utils.yaml_loader import load_yaml

warnings.filterwarnings("ignore")
logger = Logger('experiments', level="DEBUG").get_logger()

dagshub.init(repo_owner='10orabh', repo_name='olist-chrun-prediction-version-2', mlflow=True)

def run_experiments():
    try:
        logger.info("Loading transformed data for hyperparameter tuning.")
        train_data = np.load('./data/processed/train.npy')
        test_data = np.load('./data/processed/test.npy')
        
        X_train, y_train = train_data[:, :-1], train_data[:, -1]
        X_test, y_test = test_data[:, :-1], test_data[:, -1]
        logger.info(f"Data loaded successfully. Train shape: {X_train.shape}")

        config = load_yaml('experiments.yaml')
        
        # Models and their parameter grids
        models = {
            "RandomForest": {
                "class": RandomForestClassifier(),
                "params": config['experiments']['random_forest_grid']
            },
            "XGBoost": {
                "class": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                "params": config['experiments']['xgboost_grid']
            }
        }

        mlflow.set_tracking_uri("https://dagshub.com/10orabh/olist-chrun-prediction-version-2.mlflow")

        mlflow.set_experiment("Hyperparameter_Tuning")

        for model_name, setup in models.items():
            logger.info(f"Starting GridSearchCV for {model_name}")
            
            with mlflow.start_run(run_name=f"Tuning_{model_name}", nested=True):
                grid_search = GridSearchCV(
                    estimator=setup['class'],
                    param_grid=setup['params'],
                    cv=3,
                    scoring='f1',
                    n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                
                best_params = grid_search.best_params_
                best_score = grid_search.best_score_
                
                logger.info(f"Best parameters for {model_name}: {best_params}")

                mlflow.log_params(best_params)
                mlflow.log_metric("best_cv_f1", best_score)
                
                # Evaluate on test data
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)
                
                metrics = {
                    "test_f1": f1_score(y_test, y_pred),
                    "test_accuracy": accuracy_score(y_test, y_pred),
                    "test_precision": precision_score(y_test, y_pred),
                    "test_recall": recall_score(y_test, y_pred)
                }
                
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(best_model, f"best_{model_name}_model") 
                
                logger.info(f"Logged metrics for {model_name}: {metrics}")

        logger.info("✅ Hyperparameter tuning experiments completed.")

    except Exception as e:
        logger.error(f"Error in experiments: {e}")
        raise

if __name__ == "__main__":
    run_experiments()