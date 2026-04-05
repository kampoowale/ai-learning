# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load dataset from CSV
df = pd.read_csv("loan.csv")
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Income"] = df["Income"].fillna(df["Income"].median())
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
df["CreditScore"] = df["CreditScore"].fillna(df["CreditScore"].median())
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


rf = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on test data
y_pred = model.predict(X_test_scaled)
# probability of loan approval
y_prob = model.predict_proba(X_test_scaled)[:, 1]


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_prob))

# Show predictions by Name
results = pd.DataFrame({
    "Name": names_test.values,
    "PredictedApproval": y_pred,
    "ProbabilityOfApproval": y_prob
})

# Map 1/0 to decision

# This replaces:
# 1 → "Safe to give loan"
# 0 → "Risky to give loan"


results["Decision"] = results["PredictedApproval"].map(
    {1: "Safe to give loan", 0: "Risky to give loan"})

print(results)

# Save the trained model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the trained model
joblib.dump(model, "loan_model.pkl")

# Save the scaler too
joblib.dump(scaler, "scaler.pkl")

model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

'''
What comes after evaluation
Once you see the metrics, the next steps follow naturally:
Hyperparameter tuning (to improve accuracy and F1)
Pipeline (to avoid scaling errors and simplify deployment)
Explainability (SHAP) (to understand why the model approves/rejects)
Retraining and saving the final pipeline
'''

'''
Why this order matters
Improving a model without evaluating it is like repairing a car without checking what’s broken. Evaluation tells you:
whether the model is underfitting
whether it needs more depth
whether it needs more trees
whether scaling is helping
whether features are informative
whether the dataset is balanced
'''
