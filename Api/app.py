from fastapi import FastAPI
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# ------------------------------------------------
# App Initialization
# ------------------------------------------------
app = FastAPI(
    title="FedEx Shipment Delivery Prediction API",
    description="Predicts whether a shipment will be delayed using ANN",
    version="1.0"
)

# ------------------------------------------------
# Load Model & Scaler
# ------------------------------------------------
model = load_model("shipment_delivery_model.h5")
scaler = joblib.load("scaler.pkl")

# ------------------------------------------------
# Feature Order (IMPORTANT)
# ------------------------------------------------
FEATURE_COLUMNS = [
    'Year', 'Month', 'DayofMonth',
    'Actual_Shipment_Time',
    'Planned_Shipment_Time',
    'Planned_Delivery_Time',
    'Planned_TimeofTravel',
    'Shipment_Delay',
    'Distance',
    'Shipment_Duration',
    'Speed',
    'Month_sin', 'Month_cos',
    'DayOfWeek_sin', 'DayOfWeek_cos',
    'Delay_Flag',
    'Carrier_Name_freq',
    'Source_freq',
    'Destination_freq'
]

# ------------------------------------------------
# Health Check
# ------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "FedEx ANN Prediction API is running 🚀"}

# ------------------------------------------------
# Prediction Endpoint
# ------------------------------------------------
@app.post("/predict")
def predict_delivery(data: dict):
    """
    Predict shipment delivery status.
    """

    # Convert input JSON to DataFrame
    df = pd.DataFrame([data])

    # Ensure correct feature order
    df = df[FEATURE_COLUMNS]

    # Scale input
    scaled_input = scaler.transform(df)

    # Predict
    probability = model.predict(scaled_input)[0][0]
    prediction = int(probability > 0.5)

    return {
        "prediction": prediction,
        "probability": round(float(probability), 4),
        "status": "Delayed" if prediction == 1 else "On-Time"
    }
