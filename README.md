# 2_Taller2-MLOps-EIA

Taller 2 - MLOPs - EIA

## Integrantes

* Linda Castaño
* Eldigardo Camacho

## Contenido del Taller

En este repositorio se encuentra los siguientes archivos:

* 1_CRISP-DM_Exploracion.ipynb: notebook utilizado para el analisis y exploracion de los datos
* 2_CRISP-DM_Entrenamiento.ipynb: notebook utilizado para el entrenamiento del modelo con scikit-learn y pycaret.
* heart.csv: base de datos
* Taller2_MLOps.pdf: documento con las preguntas del taller resueltas.
* mejor_modelo_pipeline.joblib: el modelo entrenado con scikit learn en formato joblib.
* mejor_modelo_pipeline.pkl: el modelo entrenado en formato pycaret.

## API

En la carpeta 'api', se encuentra el código de la aplicación FastAPI que se encarga de implementar el endpoint /predict, para realizar la perdición.

### Ejemplo

Para hacer la inferencia con los siguiente datos:
```
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trtbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalachh": 150,
  "exng": 0,
  "oldpeak": 2.3,
  "slp": 0,
  "caa": 0,
  "thall": 1
}
```

Se logra la respuesta:

```
{
  "sklearn_prediction": {
    "label": 1,
    "score": 0.82
  },
  "pycaret_prediction": {
    "label": 1,
    "score": 1
  }
}
```
