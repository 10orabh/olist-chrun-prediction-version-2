from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import os
import logging
from contextlib import asynccontextmanager
from connectors.s3_connector import S3Connector 
from utils.yaml_loader import load_yaml

# 1. Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 2. Configuration & Paths
params = load_yaml('params.yaml')
BUCKET = params['model_pusher']['s3_bucket']
MODEL_KEY = params['model_pusher']['s3_model_key']
PREPROCESSOR_KEY = "preprocessor/preprocessor.pkl"

LOCAL_MODEL = "./artifacts/models/best_model.pkl"
LOCAL_PRE = "./artifacts/preprocessor.pkl"

# Global objects
model = None
preprocessor = None

# 3. Lifespan Manager (Startup aur Shutdown logic yahan rahega)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, preprocessor
    try:
        # Folder create karein agar nahi hai
        os.makedirs("./artifacts/models", exist_ok=True)
        
        s3 = S3Connector()
        logger.info(" Downloading artifacts from S3...")
        
        # S3 se files uthana
        s3.download_file(BUCKET, MODEL_KEY, LOCAL_MODEL)
        s3.download_file(BUCKET, PREPROCESSOR_KEY, LOCAL_PRE)

        # Files ko load karna
        with open(LOCAL_MODEL, 'rb') as f:
            model = pickle.load(f)
        with open(LOCAL_PRE, 'rb') as f:
            preprocessor = pickle.load(f)
            
        logger.info("✅ Artifacts loaded successfully! Ready for predictions.")
    except Exception as e:
        logger.error(f"❌ Startup Error: {e}")
        # Demo ke liye: Agar startup fail hua toh app band ho jaye taki debug kar sako
        raise SystemExit(1) 
    
    yield
    # Shutdown logic (agar kuch ho toh yahan likhein)
    logger.info("Shutting down API...")

# 4. FastAPI App Initialization
app = FastAPI(title="Olist Churn Prediction API", lifespan=lifespan)

# 5. Input Schema
class ChurnInput(BaseModel):
    customer_id: str
    frequency: int
    monetary: float
    review_score: int
    installments: int
    total_payment: float

# 6. Prediction Endpoint
@app.post("/predict")
def predict(input_data: ChurnInput):
    # Safety check: Model load hua ya nahi
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded yet.")

    try:
        # Input ko DataFrame mein badlein
        input_dict = input_data.model_dump()
        df = pd.DataFrame([input_dict])

        # Feature Engineering: Training ke hisaab se renaming
        df = df.rename(columns={
            'review_score': 'avg_review',
            'installments': 'avg_installments'
        })

        # Unnecessary column drop karein
        customer_id = df['customer_id'].iloc[0]
        df = df.drop(columns=['customer_id'], errors='ignore')

        # Transform (Preprocessing)
        transformed_data = preprocessor.transform(df)

        # Prediction logic (Logistic Regression supports predict_proba)
        prediction = model.predict(transformed_data)
        probability = model.predict_proba(transformed_data)[:, 1]

        return {
            "status": "success",
            "customer_id": customer_id,
            "prediction": int(prediction[0]),
            "probability": round(float(probability[0]), 4)
        }

    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}