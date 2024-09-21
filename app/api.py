from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.model.main import predict_pipeline
from app.model.main import __version__ as model_version

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    rating: str

@app.get("/")
def home():
    # return {"health_check": "OK", "model_version": model_version}
    return FileResponse('app/static/index.html')


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    pred = predict_pipeline(payload.text)
    pred = (pred > 0.5).float().squeeze()
    out = 'Negative' if pred == 0 else 'Positive'
    
    return {"rating": out}