from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np
import torch

model_uri = "file:./mlruns/0/1c0961b168f846c4830dd85b4eee6c6f/artifacts/model"
model = mlflow.pyfunc.load_model(model_uri=model_uri)

app = FastAPI()

class CustomerFeatures(BaseModel):
    gender: float
    SeniorCitizen: float
    Partner: float
    Dependents: float
    tenure: float
    PhoneService: float
    MultipleLines: float
    InternetService: float
    OnlineSecurity: float
    OnlineBackup: float
    DeviceProtection: float
    TechSupport: float
    StreamingTV: float
    StreamingMovies: float
    Contract: float
    PaperlessBilling: float
    PaymentMethod: float
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API"}

@app.post("/predict/")
def predict_customer_churn(data: CustomerFeatures):
    input_array = np.array([[v for v in data.model_dump().values()]])

    # Predict using MLflow model
    prediction = model.predict(input_array)

    return {"churn_probability": float(prediction[0])}