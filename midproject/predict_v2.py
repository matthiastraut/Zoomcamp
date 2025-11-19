# predict_v2.py
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from app_v2 import app as dash_app
import numpy as np
import pandas as pd
import pickle
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn

app = FastAPI(title="Election-2024")

# Mount Dash at /dash (not "/")
app.mount("/dash", WSGIMiddleware(dash_app.server))

class InputData(BaseModel):
    pct_above_bachelors: float
    race_pct_white: float
    race_pct_asian: float
    pct_bachelors_and_above: float
    workers_wo_health_ins: float
    population: float

with open("model.pkl", "rb") as f:
    xgb = pickle.load(f)

def dv(input_data):
    return pd.DataFrame([input_data])

@app.get("/")
def root():
    return {"status": "Model API is working.", "dashboard": "/dash"}

@app.post("/predict")
async def predict(input_data: Dict[str, Any]):
    y_pred = xgb.predict(dv(input_data))[0]
    return
    {
        "Trump vote share": float(np.clip(y_pred, 0, 1))
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
