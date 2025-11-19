from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    pct_above_bachelors: float
    race_pct_white: float
    race_pct_asian: float
    pct_bachelors_and_above: float
    workers_wo_health_ins: float
    population: int

@app.post("/predict")
async def predict(data: InputData):
    # Example model: just echo back what was sent
    return {"received": data.dict()}