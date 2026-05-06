import joblib
from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model

def main():

    print("Loading dataset...")

    df = load_data()

    print("Preprocessing...")

    X, y = preprocess_data(df)

    print("Training model...")

    model, X_test, y_test = train_model(X, y)

    print("Evaluating model...")

    evaluate_model(model, X_test, y_test)

    joblib.dump(X.iloc[[0]].copy(), "models/template_row.pkl")

    print("Model training complete.")

if __name__ == "__main__":
    main()
