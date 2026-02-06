import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("data-students-score.csv")

df = df.dropna(how="all")

print("Remove empty row", df)

df = df.dropna(axis=1, how="all")

print("Remove empty column", df)

df = df.dropna(subset=["Age", "Score"])
print("Remove empty Age and Score row", df)

X = df[["Age", "Score"]]
y = df["Score"] >= 70

print("X and y", X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("Prediction", X_test, prediction)
