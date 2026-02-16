from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Dataset
X = [[1], [2], [3], [4], [5]]
y = ["No", "No", "Yes", "Yes", "Yes"]

# Train classifier
model = DecisionTreeClassifier(max_depth=2)
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Evaluate
accuracy = accuracy_score(y, y_pred)
print("Predictions:", y_pred)
print("Accuracy:", accuracy)

Z = [[10]]
zResult = ["No"]
Z_pred = model.predict(Z)

# Evaluate
accuracyZ = accuracy_score(zResult, Z_pred)
print("Z Predictions:", Z_pred)
print("Z Accuracy:", accuracyZ)
