# predict_v2.py
from fastapi import FastAPI
import pandas as pd
import pickle
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
import threading
import requests
import streamlit as st

# ---------------------------
# FastAPI backend
# ---------------------------

app = FastAPI()

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

#@app.get("/")
#def root():
#    return {"status": "Model API is working.", "dashboard": "/dash"}

@app.post("/predict")
async def predict(input_data: Dict[str, Any]):
    y_pred = xgb.predict(dv(input_data))[0]
    return
    {
        "Trump vote share": float(np.clip(y_pred, 0, 1))
    }

def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=9696, log_level="info")


# ---------------------------
# Streamlit dashboard
# ---------------------------

def run_streamlit():
    st.set_page_config(page_title="Prediction Dashboard", layout="centered")
    st.title("Prediction Dashboard")

    default_values = {
        "pct_above_bachelors": 0.1,
        "race_pct_white": 0.2,
        "race_pct_asian": 0.0,
        "pct_bachelors_and_above": 0.3,
        "workers_wo_health_ins": 0.9,
        "population": 10_000_000,
    }

    pct_above_bachelors = st.slider(
        "Pct Above Bachelors", 0.0, 1.0, default_values["pct_above_bachelors"], 0.01
    )
    race_pct_white = st.slider(
        "Race Pct White", 0.0, 1.0, default_values["race_pct_white"], 0.01
    )
    race_pct_asian = st.slider(
        "Race Pct Asian", 0.0, 1.0, default_values["race_pct_asian"], 0.01
    )
    pct_bachelors_and_above = st.slider(
        "Pct Bachelors and Above", 0.0, 1.0, default_values["pct_bachelors_and_above"], 0.01
    )
    workers_wo_health_ins = st.slider(
        "Workers without Health Insurance", 0.0, 1.0, default_values["workers_wo_health_ins"], 0.01
    )
    population = st.slider(
        "Population", 0, 20_000_000, default_values["population"], 100_000
    )

    if st.button("Submit"):
        payload = {
            "pct_above_bachelors": float(pct_above_bachelors),
            "race_pct_white": float(race_pct_white),
            "race_pct_asian": float(race_pct_asian),
            "pct_bachelors_and_above": float(pct_bachelors_and_above),
            "workers_wo_health_ins": float(workers_wo_health_ins),
            "population": float(population),
        }

        try:
            api_url = "http://0.0.0.0:9696/predict"
            r = requests.post(api_url, json=payload)
            if r.status_code == 200:
                st.success(f"Predicted Trump vote share: {r.json()['Trump vote share']}")
            else:
                st.error(f"API error: {r.status_code}")
        except Exception as e:
            st.error(f"Connection error: {e}")
    else:
        st.info("Adjust the sliders and click Submit to get a prediction.")


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    # Start FastAPI in a background thread
    threading.Thread(target=run_fastapi, daemon=True).start()

    # Start Streamlit UI
    run_streamlit()
