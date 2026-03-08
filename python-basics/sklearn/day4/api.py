from fastapi import FastAPI
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI()


@app.post("/predict")
def predict_loan(data: dict):

    # Convert incoming JSON to DataFrame
    df = pd.DataFrame([data])

    # Scale the data∏
    df_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[:, 1][0]

    # Convert numeric prediction to human decision
    decision = "Safe to give loan" if prediction == 1 else "Risky to give loan"

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "decision": decision
    }
