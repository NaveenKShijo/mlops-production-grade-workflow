from fastapi import FastAPI, Request 
from fastapi.responses import JSONResponse
import pandas as pd
from predict import predict, load_artifacts
import os


app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_dir = os.getenv("SM_MODEL_DIR", os.path.join(BASE_DIR, "models"))

model, scaler = load_artifacts(model_dir)

@app.get("/home")
def get_home():
    return {'response':'This is home'}

@app.get("/ping")    
def ping():
    """ Sagemaker health check """
    return JSONResponse(status_code = 200, content = {"status": "Healthy"})

@app.post("/invocations")
async def invocations(request: Request):
    """Sagemaker prediction endpoint"""
    body = await request.json()
    input_df = pd.DataFrame([body])
    result = predict(model, scaler, input_df)
    return JSONResponse(content = {"Prediction": result}, status_code = 200)

    # Inference app/test disagree — main.py:30 returns {"Prediction": [...]}, but test_inference.py:40-43 asserts the response is a bare list. Test fails if enabled. 
    # To solve this:
    # return JSONResponse(content = result, status_code=200)