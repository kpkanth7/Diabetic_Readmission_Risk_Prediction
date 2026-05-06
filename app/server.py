import sys
import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.preprocess import clean_features

app = FastAPI(title="Diabetic Readmission Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load(os.path.join(BASE_DIR, "models", "best_model.pkl"))
template_row = joblib.load(os.path.join(BASE_DIR, "models", "template_row.pkl"))

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
    input_df = template_row.copy()

    for col, val in [("age", data.age), ("race", data.race), ("gender", data.gender)]:
        if col in input_df.columns:
            input_df.at[input_df.index[0], col] = val

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

    input_df = clean_features(input_df)

    expected_cols = model.named_steps["preprocessor"].feature_names_in_
    input_df = input_df[expected_cols]

    prob = float(model.predict_proba(input_df)[0][1])
    pred = int(model.predict(input_df)[0])

    risk_band = "Low"
    if prob >= 0.15 and prob < 0.35:
        risk_band = "Moderate"
    elif prob >= 0.35:
        risk_band = "High"

    return {"probability": prob, "prediction": pred, "risk_band": risk_band}

# Local dev: serve static files
static_dir = os.path.join(BASE_DIR, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def read_index():
        return FileResponse(os.path.join(static_dir, "index.html"))
