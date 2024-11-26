from fastapi import FastAPI
from model import predict_sklearn, predict_pycaret, HeartData
import pandas as pd

app = FastAPI()

@app.post("/predict/")
async def predict(data: HeartData):
    input_df = pd.DataFrame([data.dict()])
    sklearn_prediction = predict_sklearn(input_df)
    pycaret_prediction = predict_pycaret(input_df)
    return {
        "sklearn_prediction": {
            "target": int(sklearn_prediction["label"]),
            "score": float(sklearn_prediction["score"]),
        },
        "pycaret_prediction": {
            "target": int(pycaret_prediction["label"]),
            "score": float(pycaret_prediction["score"]),
        },
    }
@app.get("/")
def read_root():
    return {"message": "Heart Attack Prediction API"}
