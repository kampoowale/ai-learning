import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Load data
df = pd.read_csv("data-students-score.csv")

# Clean data
df = df.dropna(subset=["Age", "Score"])
df = df.dropna(axis=1, how="all")

# Features and target
X = df[["Age", "Score"]]  # [[2, 90,],[28,80]]
y = df["Score"] >= 70  # [true,true]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# Scale data (fit only on training data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = model.predict(X_test_scaled)

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

# Predict on a new student
student = [[28, 75]]   # Age = 3, Score = 75
student_scaled = scaler.transform(student)

print("\nNew student prediction:")
print("Probabilities [Fail, Pass]:", model.predict_proba(student_scaled))
print("Final decision:", model.predict(student_scaled))


# Convert scaled data back to arrays
X_train_arr = X_train_scaled
y_train_arr = y_train.values

# Create grid of points
age_range = np.linspace(X_train_arr[:, 0].min() - 1,
                        X_train_arr[:, 0].max() + 1, 100)
score_range = np.linspace(X_train_arr[:, 1].min() - 1,
                          X_train_arr[:, 1].max() + 1, 100)

xx, yy = np.meshgrid(age_range, score_range)
grid = np.c_[xx.ravel(), yy.ravel()]

# Predict probabilities on grid
probs = model.predict_proba(grid)[:, 1]
probs = probs.reshape(xx.shape)

# Plot decision boundary
plt.contour(xx, yy, probs, levels=[0.5], colors="black")

# Plot training points
plt.scatter(X_train_arr[y_train_arr == False, 0],
            X_train_arr[y_train_arr == False, 1],
            color="red", label="Fail")

plt.scatter(X_train_arr[y_train_arr == True, 0],
            X_train_arr[y_train_arr == True, 1],
            color="green", label="Pass")

plt.xlabel("Age (scaled)")
plt.ylabel("Score (scaled)")
plt.title("Decision Boundary (Logistic Regression)")
plt.legend()
plt.show()
