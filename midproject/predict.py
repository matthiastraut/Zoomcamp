from fastapi import FastAPI
# from fastapi.middleware.wsgi import WSGIMiddleware
import numpy as np
import pandas as pd
import pickle
import uvicorn
from typing import Dict, Any, Literal
from pydantic import BaseModel

app = FastAPI(title="Election-2024")

class InputData(BaseModel):
    pct_above_bachelors: float
    race_pct_white: float
    race_pct_asian: float
    pct_bachelors_and_above: float
    workers_wo_health_ins: float
    population: float

# Loading the xgboost model with pickle
with open('model.pkl', 'rb') as f:
    xgb = pickle.load(f)
    f.close()

def dv(input_data):
    input_df = pd.DataFrame(index=input_data.keys(), data=list(input_data.values()))
    input_df = input_df.T
    return input_df

@app.get("/")
def root():
    return {"status": "App is working."}

@app.post("/predict")
async def predict(input_data: Dict[str, Any]):
    y_pred = xgb.predict(dv(input_data))[0]

    return {
        "Trump vote share:": np.minimum(np.maximum(0, y_pred), 1)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
