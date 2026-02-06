import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("data-students-score.csv")
print("DF", df)

# Remove row if score is empty
df = df.dropna(subset=["Score"])
print("DF", df)

# remove empty row
df = df.dropna(how="all")
print("DF", df)

# remove empty column
df = df.dropna(axis=1, how="all")

print("DF", df)

# Feature (X) and Target (y)
X = df[["Age", "Score"]]      # 2D
y = df["Score"] >= 70  # True/False

print("Value of x : ", X)
print("Calue of y : ", y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# What this does
# X → your features (Age & Score)
# y → your target (Passed True/False)
# test_size=0.2 → 20% of the data will be used for testing, the rest (80%) for training
# random_state=42 → ensures that the split is the same every time you run the code (for reproducibility)

# Model
model = LogisticRegression()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

random = [[3, 6]]
prediction = model.predict(random)
print("Prediction random data", prediction)


print("Prediction : ", X_test, y_pred)
# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
