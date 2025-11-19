# run_app.py
# from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware
from dash_app import create_dash_app
import test_predict  # import FastAPI app with /predict
import uvicorn

app = test_predict.app  # reuse your FastAPI app

# attach Dash at /dashboard
dash_app = create_dash_app(app)
app.mount("/dashboard", WSGIMiddleware(dash_app.server))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)