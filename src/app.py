from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model khi app khởi động
model = joblib.load("artifacts/model.joblib")

class InputData(BaseModel):
    features: list[float]

@app.get("/")
def healthcheck():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)
    return {"prediction": float(prediction[0])}
