from fastapi import FastAPI
from pydantic import BaseModel

from app.model.main import predict_pipeline
from app.model.model import __version__ as model_version

app = FastAPI()

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    language: str

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    pred = predict_pipeline(payload.text)
    pred = (pred > 0.5).float().squeeze()
    out = 'Negative' if pred == 0 else 'Positive'
    
    return {"rating": out}