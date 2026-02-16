# Concept in simple terms:
# It’s like asking 100 trees what the answer should be and taking a vote
# (for classification) or average (for regression).

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Dataset
X = [[1], [2], [3], [4], [5]]
y = ["No", "No", "Yes", "Yes", "Yes"]

# Train Random Forest
model = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
model.fit(X, y)

# Predict
y_pred = model.predict(X)
print("Predictions:", y_pred)

# Accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)

# Feature importance
print("Feature importance:", model.feature_importances_)
