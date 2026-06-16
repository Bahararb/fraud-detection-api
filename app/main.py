from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="Fraud Detection API")

API_KEY = "fraud123"
model = None


# -------------------------
# Load model at startup
# -------------------------
@app.on_event("startup")
def load_model():
    global model

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "model", "fraud_model.pkl")

    model = joblib.load(model_path)


# -------------------------
# Input schema (Swagger fields)
# -------------------------
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


# -------------------------
# Root endpoint
# -------------------------
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


# -------------------------
# PREDICTION ENDPOINT (FIXED)
# -------------------------
@app.post("/predict")
def predict(data: Transaction, x_api_key: str = Header(None)):

    # API key check
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    try:
        # IMPORTANT FIX: use DataFrame (not numpy array)
        input_data = pd.DataFrame([{
            "Time": data.Time,
            "V1": data.V1,
            "V2": data.V2,
            "V3": data.V3,
            "V4": data.V4,
            "V5": data.V5,
            "V6": data.V6,
            "V7": data.V7,
            "V8": data.V8,
            "V9": data.V9,
            "V10": data.V10,
            "V11": data.V11,
            "V12": data.V12,
            "V13": data.V13,
            "V14": data.V14,
            "V15": data.V15,
            "V16": data.V16,
            "V17": data.V17,
            "V18": data.V18,
            "V19": data.V19,
            "V20": data.V20,
            "V21": data.V21,
            "V22": data.V22,
            "V23": data.V23,
            "V24": data.V24,
            "V25": data.V25,
            "V26": data.V26,
            "V27": data.V27,
            "V28": data.V28,
            "Amount": data.Amount
        }])

        # prediction
        prediction = model.predict(input_data)[0]

        # probability (if supported)
        probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1]

        return {
            "prediction": int(prediction),
            "fraud_probability": float(probability) if probability is not None else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))