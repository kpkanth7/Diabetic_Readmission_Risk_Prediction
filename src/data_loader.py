import pandas as pd

def load_data(path="data/raw/diabetic_data.csv"):
    df = pd.read_csv(path)
    return df
