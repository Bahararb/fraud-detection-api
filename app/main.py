from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="Fraud Detection API")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.mount("/static", StaticFiles(directory=os.path.join(base_dir, "app", "static")), name="static")

# API KEY 
API_KEY = "fraud123"

# Load model globally
model = None


@app.on_event("startup")
def load_model():
    global model

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "model", "fraud_model.pkl")

    model = joblib.load(model_path)
    print("Model loaded successfully")



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


@app.get("/")
def home():
    return FileResponse(os.path.join(base_dir, "app", "static", "index.html"))



@app.post("/predict")
def predict(data: Transaction, x_api_key: str = Header(None)):

   
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    try:

      
        input_data = pd.DataFrame([data.dict()])

        # (Optional debug)
        print("INPUT:", input_data)

        prediction = model.predict(input_data)[0]
        print("PREDICTION:", prediction)

       
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0]
            probability = float(proba[1])
        else:
            probability = 0.0

        print("PROBABILITY:", probability)

        
        return {
            "prediction": int(prediction),
            "fraud_probability": round(probability, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))