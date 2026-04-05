import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import joblib

# Load dataset from CSV
df = pd.read_csv("loan.csv")
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Income"] = df["Income"].fillna(df["Income"].median())
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
df["CreditScore"] = df["CreditScore"].fillna(df["CreditScore"].median())

print(df.head())

# Features and target
X = df[["Age", "Income", "LoanAmount", "CreditScore", "PreviousDefaults"]]
y = df["LoanApproved"]

# Train-test split (keep names for later reference)
X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
    X, y, df["Name"], test_size=0.3, random_state=42
)

# ---- STEP 1: Build a pipeline (scaler + model) ----
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42))
])

# Optional: hyperparameter tuning on the pipeline
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 5, 10, 20],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_pipeline = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Use best_pipeline for predictions
y_pred = best_pipeline.predict(X_test)
y_prob = best_pipeline.predict_proba(X_test)[:, 1]

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

results["Decision"] = results["PredictedApproval"].map(
    {1: "Safe to give loan", 0: "Risky to give loan"}
)

print(results)

# ---- Save the final pipeline ----
joblib.dump(best_pipeline, "loan_pipeline.pkl")
print("Saved pipeline as loan_pipeline.pkl")
