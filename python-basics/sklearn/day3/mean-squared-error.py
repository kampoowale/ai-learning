from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
import pandas as pd
from sklearn.metrics import f1_score

df = pd.read_csv("data-students-score.csv")
df = df.dropna(subset=["Age", "Score"])
df = df.dropna(axis=1, how="all")

# Features and target
X = df[["Age", "Score"]]
y = df["Score"] >= 70

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

# For example:
# Actual values: [10, 5, 8]
# Predicted values: [8, 5, 10]
# Step-by-step:
# Errors: [10−8, 5−5, 8−10] → [2, 0, -2]
# Squared errors: [4, 0, 4]
# MSE = (4 + 0 + 4)/3 = 8/3 ≈ 2.67

# Lower MSE → predictions are closer to actual values. lowest is 0
