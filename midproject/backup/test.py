# test_fastapi.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "FastAPI + Uvicorn are working!"}