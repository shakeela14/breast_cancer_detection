# src/api.py
# Basic FastAPI scaffold to expose a /predict endpoint
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

app = FastAPI()

class Instance(BaseModel):
    features: list  # list of numeric features in same order as training columns

MODEL_PATH = "models/logistic_v1.joblib"
model_obj = None
MODEL_COLUMNS = None

@app.on_event("startup")
def load():
    global model_obj, MODEL_COLUMNS
    model_obj = joblib.load(MODEL_PATH)
    # model_obj is dict {"model": model, "scaler": scaler}
    # optionally persist columns in the saved obj or separately

@app.post("/predict")
def predict(payload: Instance):
    model = model_obj["model"]
    scaler = model_obj["scaler"]
    x = np.array(payload.features).reshape(1, -1)
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]
    proba = model.predict_proba(x_scaled)[0].max()
    label = "Benign" if pred == 1 else "Malignant"
    return {"label": label, "confidence": float(proba)}

# to run locally: uvicorn src.api:app --reload --port 8000
