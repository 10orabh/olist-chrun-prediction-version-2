import os
import json
import pickle
import numpy as np
from sklearn.metrics import precision_score
from utils.logger import Logger
from utils.yaml_loader import load_yaml
from connectors.s3_connector import S3Connector

logger = Logger('model_pusher', level="DEBUG").get_logger()

def evaluate_s3_model(model_path, X_test, y_test):
    """S3 se download kiye gaye champion model ko test karne ke liye"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        y_pred = model.predict(X_test)
        return precision_score(y_test, y_pred)
    except Exception as e:
        logger.error(f"S3 model evaluation error: {e}")
        return 0.0

def main():
    try:
        logger.info("--- Model Pusher: Smart Logic Started ---")
        params = load_yaml('params.yaml')
        config = params['model_pusher']
        
    
        local_metrics_path = './artifacts/reports/metrics.json'
        local_model_path = './artifacts/models/best_model.pkl'
        
        with open(local_metrics_path, 'r') as f:
            new_model_report = json.load(f)
        
        new_precision = new_model_report.get('precision', 0.0)
        logger.info(f"New Model (Challenger) Precision: {new_precision}")

        # 2. S3 (Champion) Model ke saath comparison ki taiyari
        s3 = S3Connector()
        bucket = config['s3_bucket']
        s3_model_key = config['s3_model_key']
        
        
        temp_s3_model = "champion_model_from_s3.pkl"
        is_model_accepted = False

        logger.info("Checking for existing model in S3...")
        
        # Check: Kya S3 mein model hai?
        if s3.download_file(bucket, s3_model_key, temp_s3_model):
            # Agar model mil gaya, toh current test data par evaluate karo
            test_data = np.load('./data/processed/test.npy')
            X_test, y_test = test_data[:, :-1], test_data[:, -1]
            
            old_precision = evaluate_s3_model(temp_s3_model, X_test, y_test)
            logger.info(f"S3 Model (Champion) Precision: {old_precision}")
            
            # Comparison Logic
            if new_precision > old_precision:
                logger.info(f"Challenger is better: {new_precision} > {old_precision}")
                is_model_accepted = True
            else:
                logger.info("Champion is still better. No push needed.")
                is_model_accepted = False
            
            # Temp file delete karein
            if os.path.exists(temp_s3_model):
                os.remove(temp_s3_model)
        else:
            # AGAR MODEL NAHI MILA TOH:
            logger.warning("S3 bucket is empty or model not found. This is the FIRST push.")
            is_model_accepted = True

        # 3. Push Logic
        if is_model_accepted:
            logger.info("Pushing model and metrics to S3...")
            s3.upload_file(local_model_path, bucket, s3_model_key)
            logger.info("✅ Successfully pushed to S3.")
        else:
            logger.info("❌ Push rejected. Production model remains unchanged.")

    except Exception as e:
        logger.error(f"Model Pusher failed: {e}")
        raise

if __name__ == "__main__":
    main()