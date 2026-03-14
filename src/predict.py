import joblib
import pandas as pd

def load_model():
    return joblib.load("models/best_model.pkl")

def predict(data):

    model = load_model()

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return prediction, probability
