import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# ---------------------------
# 1️⃣ Load and clean data
# ---------------------------
df = pd.read_csv("data-students-score.csv")

# Keep only rows with Age and Score
df = df.dropna(subset=["Age", "Score"])

# Features (X) and target (y)
X = df[["Age", "Score"]]        # 2D features
y = df["Score"] >= 70           # True/False target

# ---------------------------
# 2️⃣ Scale features
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# 3️⃣ Train model on all data
# ---------------------------
model = LogisticRegression()
model.fit(X_scaled, y)

# ---------------------------
# 4️⃣ Predict on the same data (since dataset is tiny)
# ---------------------------
y_pred = model.predict(X_scaled)

# ---------------------------
# 5️⃣ Evaluate
# ---------------------------
conf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", conf_matrix)


# The confusion matrix gives counts for each category in a 2×2 table for binary classification:
# For your matrix:
# [[1, 0],
# [1, 2]]
# TN = 1 → 1 student who actually failed and was correctly predicted as failing
# FP = 0 → 0 students who actually failed but were incorrectly predicted as passing
# FN = 1 → 1 student who actually passed but was incorrectly predicted as failing
# TP = 2 → 2 students who actually passed and were correctly predicted as passing
# So yes — each number is literally a count of students in that category.

accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)

# ---------------------------
# 6️⃣ Predict random/new data
# ---------------------------
random_students = [
    [23, 56],  # low score
    [3, 75],   # weird age
    [25, 90],  # high score
    [22, 69]   # borderline fail
]

random_scaled = scaler.transform(random_students)
predictions = model.predict(random_scaled)
print("Predictions for random data:", predictions)
