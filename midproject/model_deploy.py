from fastapi import FastAPI
import uvicorn
import pickle
import sklearn
from typing import Dict, Any

app = FastAPI(title="prediction-engine")

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(data):
    result = pipeline.predict_proba(data)[0, 1]
    return float(result)

@app.post("/predict")
def predict_single(data: Dict[str, Any]):
    prob = predict_single(data)

    return {
        "probability": prob
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
