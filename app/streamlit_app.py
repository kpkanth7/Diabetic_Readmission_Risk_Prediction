import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import joblib

from src.preprocess import clean_features

st.set_page_config(
    page_title="Diabetic Readmission Risk Predictor",
    layout="wide"
)


@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")


@st.cache_data
def load_template_row():
    raw_df = pd.read_csv("data/raw/diabetic_data.csv")
    template_df = clean_features(raw_df.copy())
    template_row = template_df.drop(columns=["readmitted"]).iloc[[0]].copy()
    return template_row


model = load_model()
template_row = load_template_row()

st.title("Diabetic Readmission Risk Predictor")
st.markdown(
    "Predict whether a diabetic patient is at risk of **hospital readmission within 30 days** "
    "using a machine learning model trained on the UCI Diabetes 130-US Hospitals dataset."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Demographics")

    age_options = [
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
    ]
    race_options = ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other", "Unknown"]
    gender_options = ["Male", "Female", "Unknown"]

    age = st.selectbox("Age group", age_options, index=7)
    race = st.selectbox("Race", race_options, index=0)
    gender = st.selectbox("Gender", gender_options, index=0)

with col2:
    st.subheader("Hospital Encounter Details")

    time_in_hospital = st.slider("Days in hospital", 1, 14, 3)
    num_lab_procedures = st.slider("Number of lab procedures", 1, 100, 40)
    num_procedures = st.slider("Number of procedures", 0, 10, 1)
    num_medications = st.slider("Number of medications", 1, 50, 10)
    number_outpatient = st.slider("Outpatient visits", 0, 20, 0)
    number_emergency = st.slider("Emergency visits", 0, 20, 0)
    number_inpatient = st.slider("Inpatient visits", 0, 20, 0)
    number_diagnoses = st.slider("Number of diagnoses", 1, 16, 5)

st.divider()

if st.button("Predict Readmission Risk", use_container_width=True):
    input_df = template_row.copy()

    # Update categorical fields
    if "age" in input_df.columns:
        input_df.at[input_df.index[0], "age"] = age
    if "race" in input_df.columns:
        input_df.at[input_df.index[0], "race"] = race
    if "gender" in input_df.columns:
        input_df.at[input_df.index[0], "gender"] = gender

    # Update numeric fields
    updates = {
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
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

    prob_pct = round(prob * 100, 2)

    if prob < 0.15:
        risk_band = "Low"
    elif prob < 0.35:
        risk_band = "Moderate"
    else:
        risk_band = "High"

    st.subheader("Prediction Result")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric("Predicted Class", "High Risk" if pred == 1 else "Low Risk")

    with metric2:
        st.metric("Readmission Probability", f"{prob_pct}%")

    with metric3:
        st.metric("Risk Band", risk_band)

    st.progress(min(prob, 1.0))

    if pred == 1:
        st.error("This patient profile shows elevated risk of hospital readmission within 30 days.")
    else:
        st.success("This patient profile appears to be lower risk for 30-day hospital readmission.")

    st.markdown("### Input Summary")

    summary_df = pd.DataFrame({
        "Feature": [
            "Age group",
            "Race",
            "Gender",
            "Days in hospital",
            "Lab procedures",
            "Procedures",
            "Medications",
            "Outpatient visits",
            "Emergency visits",
            "Inpatient visits",
            "Diagnoses",
        ],
        "Value": [
            age,
            race,
            gender,
            time_in_hospital,
            num_lab_procedures,
            num_procedures,
            num_medications,
            number_outpatient,
            number_emergency,
            number_inpatient,
            number_diagnoses,
        ]
    })

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "This application is my fun portfolio project only and should not be used for real clinical decision-making."
)
