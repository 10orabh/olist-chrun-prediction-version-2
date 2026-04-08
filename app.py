from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from utils.logger import Logger

logger = Logger('api_stage', level="DEBUG").get_logger()
app = FastAPI(title="Olist Churn Prediction API")

MODEL_PATH = './artifacts/models/best_model.pkl'
PREPROCESSOR_PATH = './artifacts/preprocessor.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
    logger.info("Artifacts loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load artifacts: {e}")
    raise e

class ChurnInput(BaseModel):
    customer_id: str
    frequency: int        # Integer input
    monetary: float
    review_score: int     # Integer input
    installments: int     # Integer input
    total_payment: float

@app.post("/predict")
def predict(input_data: ChurnInput):
    try:
        logger.info(f"Processing request for: {input_data.customer_id}")
        
        data_dict = input_data.model_dump()
        df = pd.DataFrame([data_dict])
        
        # Internal Renaming: Taaki preprocessor ko purane naam mil sakein
        df = df.rename(columns={
            'review_score': 'avg_review',
            'installments': 'avg_installments'
        })
        
        df = df.drop(columns=['customer_id'], errors='ignore')
        
        transformed_data = preprocessor.transform(df)
        prediction = model.predict(transformed_data)
        probability = model.predict_proba(transformed_data)[:, 1]
        
        result = {
            "customer_id": input_data.customer_id,
            "churn_prediction": int(prediction[0]),
            "churn_probability": float(probability[0])
        }
        
        logger.info(f"Prediction success.")
        return result

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)