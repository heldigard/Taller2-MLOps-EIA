from pycaret.classification import load_model, predict_model
import joblib
import os
from pydantic import BaseModel
import pandas as pd

# Construir las rutas de los archivos
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_path, "mejor_modelo_pipeline.joblib")
scaler_path = os.path.join(base_path, "scaler.joblib")
encoder_path = os.path.join(base_path, "encoder.joblib")

# Load the model
model = joblib.load(model_path)
# Load the scaler and encoder
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)

def preprocess_data(data):
    data_df = pd.DataFrame([data])
    # Asegúrate de que los nombres de las características coincidan
    expected_features = encoder.get_feature_names_out()
    data_df = data_df.reindex(columns=expected_features, fill_value=0)
    # Aplica el codificador a las columnas categóricas
    data_df = encoder.transform(data_df)
    # Aplica el escalador a los datos
    data_df = scaler.transform(data_df)
    return data_df

def predict(data):
    data_df = preprocess_data(data)
    return predict_model(model, data=data_df)

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