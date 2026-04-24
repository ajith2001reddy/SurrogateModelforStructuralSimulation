from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
import joblib
import numpy as np
import time

# We need the model class to load weights properly
from train_surrogate import BeamSurrogateMLP

app = FastAPI(title="Structural Simulation Surrogate API")

# Global variables for model and scalers
model = None
X_scaler = None
y_scaler = None

@app.on_event("startup")
def load_assets():
    global model, X_scaler, y_scaler
    try:
        model = BeamSurrogateMLP()
        model.load_state_dict(torch.load("models/surrogate_model.pt", map_location="cpu"))
        model.eval()
        
        X_scaler = joblib.load("models/X_scaler.pkl")
        y_scaler = joblib.load("models/y_scaler.pkl")
        print("Model and scalers loaded successfully.")
    except Exception as e:
        print(f"Error loading assets: {e}")

class BeamParams(BaseModel):
    L: float
    b: float
    h: float
    F: float
    E: float
    rho: float

@app.post("/predict")
def predict(params: BeamParams):
    start_t = time.time()
    
    # 1. Structure inputs
    input_arr = np.array([[
        params.L, params.b, params.h, params.F, params.E, params.rho
    ]])
    
    # 2. Scale inputs
    x_scaled = X_scaler.transform(input_arr)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    
    # 3. Predict
    with torch.no_grad():
        preds_scaled = model(x_tensor)
        
    # 4. Inverse transform
    preds = y_scaler.inverse_transform(preds_scaled.numpy())[0]
    
    infer_time = (time.time() - start_t) * 1000 # ms
    
    return {
        "max_deflection_m": float(preds[0]),
        "max_stress_Pa": float(preds[1]),
        "natural_freq_Hz": float(preds[2]),
        "inference_time_ms": float(infer_time)
    }

@app.post("/predict_batch")
def predict_batch(batch_params: List[BeamParams]):
    start_t = time.time()
    
    # 1. Structure inputs
    input_list = [[p.L, p.b, p.h, p.F, p.E, p.rho] for p in batch_params]
    input_arr = np.array(input_list)
    
    # 2. Scale inputs
    x_scaled = X_scaler.transform(input_arr)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    
    # 3. Predict
    with torch.no_grad():
        preds_scaled = model(x_tensor)
        
    # 4. Inverse transform
    preds = y_scaler.inverse_transform(preds_scaled.numpy())
    
    infer_time = (time.time() - start_t) * 1000 # ms
    
    return {
        "predictions": [
            {
                "max_deflection_m": float(p[0]),
                "max_stress_Pa": float(p[1]),
                "natural_freq_Hz": float(p[2])
            } for p in preds
        ],
        "total_inference_time_ms": float(infer_time)
    }

@app.get("/health")
def health():
    return {"status": "healthy"}
