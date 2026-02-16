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

# Load data
df = pd.read_csv("data-students-score.csv")
df = df.dropna(subset=["Age", "Score"])
df = df.dropna(axis=1, how="all")

# Features and target
X = df[["Age", "Score"]]
y = df["Score"] >= 70

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)
# X is age
# y is pass or fail

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion matrix', conf_matrix)
# Create a heatmap
plt.figure(figsize=(5, 4))

# Create a heatmap
# plt.figure(figsize=(5, 4))
# figsize=(5, 4) sets the size of the figure in inches:
# 5 inches wide
# 4 inches tall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("F1-score:", f1)
print("Precision:", precision)
print("Recall:", recall)


# What does F1 = 0.67 mean?
# The model is not terrible
# But not reliable
# It is biased toward catching passes, even at the cost of mistakes
# In simple terms:
# “The model prefers to be safe than sorry.”

# Precision → “Can I trust PASS predictions?”
# Recall → “Did I catch all PASS students?”


# Recall = 1.0 means:
# The model caught 100% of the students who actually passed


# Precision = 0.5 means:
# For every 2 students predicted as PASS
# Only 1 actually passed
# The other 1 actually failed

# Put both together (very important)
# model behavior is:
# “I will make sure I never miss a passing student,
# but I don’t mind wrongly passing some failing students.”

# This draws a heatmap (colored grid) from your confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred False", "Pred True"],
            yticklabels=["Actual False", "Actual True"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()
