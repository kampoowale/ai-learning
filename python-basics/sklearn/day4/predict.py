import joblib
import pandas as pd

# 1. Load the saved model and scaler
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

# 2. Example new applicant data (you will replace this later)
new_data = pd.DataFrame([{
    "Age": 32,
    "Income": 45000,
    "LoanAmount": 12000,
    "CreditScore": 710,
    "PreviousDefaults": 0
}])

# 3. Scale the new data
new_data_scaled = scaler.transform(new_data)

# 4. Predict class and probability
prediction = model.predict(new_data_scaled)[0]
probability = model.predict_proba(new_data_scaled)[:, 1][0]

# 5. Convert numeric prediction to human decision
decision = "Safe to give loan" if prediction == 1 else "Risky to give loan"

print("Prediction:", prediction)
print("Probability:", probability)
print("Decision:", decision)
