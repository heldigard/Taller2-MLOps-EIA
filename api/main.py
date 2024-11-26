from fastapi import FastAPI
import pandas as pd
from pycaret.classification import predict_model
from model import predict, HeartData, encoder, scaler, model

app = FastAPI()

@app.post("/predict")
def get_prediction(data: HeartData):
    data_dict = data.dict()
    prediction = predict(data_dict)
    return {"prediction": prediction}
@app.get("/")
def read_root():
    return {"message": "Heart Attack Prediction API"}