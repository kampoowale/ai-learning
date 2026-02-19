# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset from CSV
df = pd.read_csv("loan.csv")

# Optional: view first few rows
print(df.head())

# Features and target
X = df[["Age", "Income", "LoanAmount", "CreditScore", "PreviousDefaults"]]
y = df["LoanApproved"]

# Train-test split (keep names for later reference)
X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
    X, y, df["Name"], test_size=0.3, random_state=42
)

# Scale features (optional for Random Forest, but good practice)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on test data
y_pred = model.predict(X_test_scaled)
# probability of loan approval
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Show predictions by Name
results = pd.DataFrame({
    "Name": names_test.values,
    "PredictedApproval": y_pred,
    "ProbabilityOfApproval": y_prob
})

# Map 1/0 to decision
results["Decision"] = results["PredictedApproval"].map(
    {1: "Safe to give loan", 0: "Risky to give loan"})

print(results)
