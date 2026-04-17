import sys
import os
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import clean_features

app = FastAPI(title="Diabetic Readmission Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and template row on startup
model = None
template_row = None

@app.on_event("startup")
def load_artifacts():
    global model, template_row
    model = joblib.load("models/best_model.pkl")
    raw_df = pd.read_csv("data/raw/diabetic_data.csv")
    template_df = clean_features(raw_df.copy())
    template_row = template_df.drop(columns=["readmitted"]).iloc[[0]].copy()

class PatientData(BaseModel):
    age: str
    race: str
    gender: str
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int

@app.post("/predict")
def predict(data: PatientData):
    global model, template_row
    input_df = template_row.copy()
    
    # Update categorical fields
    if "age" in input_df.columns:
        input_df.at[input_df.index[0], "age"] = data.age
    if "race" in input_df.columns:
        input_df.at[input_df.index[0], "race"] = data.race
    if "gender" in input_df.columns:
        input_df.at[input_df.index[0], "gender"] = data.gender

    # Update numeric fields
    updates = {
        "time_in_hospital": data.time_in_hospital,
        "num_lab_procedures": data.num_lab_procedures,
        "num_procedures": data.num_procedures,
        "num_medications": data.num_medications,
        "number_outpatient": data.number_outpatient,
        "number_emergency": data.number_emergency,
        "number_inpatient": data.number_inpatient,
        "number_diagnoses": data.number_diagnoses,
    }

    for col, val in updates.items():
        if col in input_df.columns:
            input_df.at[input_df.index[0], col] = val

    # Apply exactly the same cleaning used in training
    input_df = clean_features(input_df)

    # Align columns with what the trained model expects
    expected_cols = model.named_steps["preprocessor"].feature_names_in_
    input_df = input_df[expected_cols]

    # Predict
    prob = float(model.predict_proba(input_df)[0][1])
    pred = int(model.predict(input_df)[0])
    
    risk_band = "Low"
    if prob >= 0.15 and prob < 0.35:
        risk_band = "Moderate"
    elif prob >= 0.35:
        risk_band = "High"

    return {
        "probability": prob,
        "prediction": pred,
        "risk_band": risk_band
    }

# Serve static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("static/index.html")
