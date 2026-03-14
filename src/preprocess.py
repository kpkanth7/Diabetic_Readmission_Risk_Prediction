import pandas as pd

TARGET_COL = "readmitted"
DROP_COLS = ["encounter_id", "patient_nbr"]

NUMERIC_COLS = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop(columns=DROP_COLS, errors="ignore")
    df = df.replace("?", pd.NA)

    # Make sure expected numeric columns are numeric
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Everything else except target becomes categorical string if object-like
    for col in df.columns:
        if col == TARGET_COL:
            continue
        if col not in NUMERIC_COLS:
            df[col] = df[col].fillna("Unknown").astype(str)

    # Fill numeric missing values with median
    for col in NUMERIC_COLS:
        if col in df.columns:
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = df[col].fillna(median_val)

    return df

def preprocess_data(df: pd.DataFrame):
    df = df.copy()

    df[TARGET_COL] = df[TARGET_COL].apply(lambda x: 1 if x == "<30" else 0)

    df = clean_features(df)

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    return X, y
