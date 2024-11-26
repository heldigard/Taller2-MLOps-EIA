import joblib
from pycaret.classification import load_model, predict_model
import os
from pydantic import BaseModel
import pandas as pd

# Construir las rutas de los archivos
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sklearn_model_path = os.path.join(base_path, "mejor_modelo_pipeline.joblib")
pycaret_model_path = os.path.join(base_path, "mejor_modelo_pipeline")

# Cargar el modelo de scikit-learn
sklearn_model = joblib.load(sklearn_model_path)

# Cargar el modelo de pycaret
pycaret_model = load_model(pycaret_model_path)

def predict_sklearn(data: pd.DataFrame):
    prediction = sklearn_model.predict(data)
    prediction_proba = sklearn_model.predict_proba(data)
    return {
        "label": int(prediction[0]),
        "score": float(prediction_proba[0][1])
    }

def predict_pycaret(data: pd.DataFrame):
    prediction = predict_model(pycaret_model, data=data)
    return {
        "label": prediction["prediction_label"][0],
        "score": prediction["prediction_score"][0]
    }

class HeartData(BaseModel):
    age: int
    sex: int
    cp: int
    trtbps: int
    chol: int
    fbs: int
    restecg: int
    thalachh: int
    exng: int
    oldpeak: float
    slp: int
    caa: int
    thall: int