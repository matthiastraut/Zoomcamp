# main.py
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from pydantic import BaseModel

import dash
from dash import dcc, html

############################################################
# 1. FASTAPI BACKEND
############################################################
app = FastAPI()

class InputData(BaseModel):
    pct_above_bachelors: float
    race_pct_white: float
    race_pct_asian: float
    pct_bachelors_and_above: float
    workers_wo_health_ins: float
    population: float

@app.post("/predict")
def predict(data: InputData):
    # Dummy example — replace with your model
    result = (
        data.pct_above_bachelors
        + data.race_pct_white
        + data.race_pct_asian
        + data.pct_bachelors_and_above
        + data.workers_wo_health_ins
        + data.population / 1_000_000
    )
    return {"prediction": result}


############################################################
# 2. DASH FRONTEND (attached to FastAPI)
############################################################

dash_app = dash.Dash(__name__, requests_pathname_prefix="/dashboard/")

dash_app.layout = html.Div([
    html.H2("FastAPI + Dash Dashboard"),

    dcc.Slider(id="population", min=0, max=20_000_000, step=250_000,
               value=1_000_000, tooltip={"always_visible": True}),
    dcc.Slider(id="population", min=0, max=20_000_000, step=250_000,
               value=1_000_000, tooltip={"always_visible": True}),
    html.Br(),
    html.Button("Submit", id="submit-btn", n_clicks=0),
    html.Div(id="output")
])

@dash_app.callback(
    dash.Output("output", "children"),
    dash.Input("submit-btn", "n_clicks"),
    [dash.State("population", "value")]
)
def run_prediction(n_clicks, population):
    if n_clicks == 0:
        return "Awaiting input..."

    payload = {
        "pct_above_bachelors": 0.1,
        "race_pct_white": 0.2,
        "race_pct_asian": 0.0,
        "pct_bachelors_and_above": 0.3,
        "workers_wo_health_ins": 0.9,
        "population": population
    }

    import requests
    r = requests.post("http://localhost:8000/predict", json=payload)
    return f"Prediction: {r.json()['prediction']}"


############################################################
# 3. MOUNT DASH INSIDE FASTAPI
############################################################

app.mount("/dashboard", WSGIMiddleware(dash_app.server))


############################################################
# 4. Run with:
#     uvicorn main:app --reload --port 8000
############################################################
