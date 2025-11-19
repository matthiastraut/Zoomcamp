# app_v3.py
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Any, Literal
from pydantic import BaseModel

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

server = app.server

# Loading the xgboost model with pickle
with open('model.pkl', 'rb') as f:
    xgb = pickle.load(f)
    f.close()

def dv(input_data):
    input_df = pd.DataFrame(index=input_data.keys(), data=list(input_data.values()))
    input_df = input_df.T
    return input_df

def predict(input_data: Dict[str, Any]):
    y_pred = xgb.predict(dv(input_data))[0]
    return y_pred

default_values = {
    "pct_above_bachelors": 0.1,
    "race_pct_white": 0.2,
    "race_pct_asian": 0.0,
    "pct_bachelors_and_above": 0.3,
    "workers_wo_health_ins": 0.9,
    "population": 10_000_000,
}

def slider(id, value, minv=0, maxv=1, step=0.01):
    return html.Div(
        [
            html.Label(id.replace("_", " ").title(), style={"fontWeight": "500"}),
            dbc.Input(
                id=id,
                type="range",
                min=minv,
                max=maxv,
                step=step,
                value=value,
                className="form-range",
            ),
            html.Br(),
        ],
        style={"margin": "10px 0"},
    )

app.layout = html.Div(
    [
        html.H2("Prediction Dashboard", style={"textAlign": "center"}),

        slider("pct_above_bachelors", default_values["pct_above_bachelors"]),
        slider("race_pct_white", default_values["race_pct_white"]),
        slider("race_pct_asian", default_values["race_pct_asian"]),
        slider("pct_bachelors_and_above", default_values["pct_bachelors_and_above"]),
        slider("workers_wo_health_ins", default_values["workers_wo_health_ins"]),
        slider(
            "population",
            default_values["population"],
            minv=0,
            maxv=20_000_000,
            step=100_000,
        ),

        html.Button("Submit", id="submit-btn", n_clicks=0, style={"margin": "20px 0"}),

        html.H3("Predicted Trump vote share (vs Harris in 2024):"),
        html.Div(id="prediction-output", style={"fontSize": "24px", "color": "blue"}),
    ],
    style={"width": "350px", "margin": "auto"},
)

@app.callback(
    Output("prediction-output", "children"),
    Input("submit-btn", "n_clicks"),
    [
        State("pct_above_bachelors", "value"),
        State("race_pct_white", "value"),
        State("race_pct_asian", "value"),
        State("pct_bachelors_and_above", "value"),
        State("workers_wo_health_ins", "value"),
        State("population", "value"),
    ],
)
def submit_data(n_clicks, pct_above_bachelors, race_pct_white, race_pct_asian,
                pct_bachelors_and_above, workers_wo_health_ins, population):

    if n_clicks == 0:
        return "Awaiting input..."

    payload = {
        "pct_above_bachelors": float(pct_above_bachelors),
        "race_pct_white": float(race_pct_white),
        "race_pct_asian": float(race_pct_asian),
        "pct_bachelors_and_above": float(pct_bachelors_and_above),
        "workers_wo_health_ins": float(workers_wo_health_ins),
        "population": float(population),
    }

    return predict(payload)

if __name__ == "__main__":
    app.run(port=8050)