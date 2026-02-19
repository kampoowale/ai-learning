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

conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion matrix', conf_matrix)

precision = precision_score(y_test, y_pred)

# The recall is intuitively the ability of the classifier to find all the positive samples.
# The best value is 1 and the worst value is 0.


recall = recall_score(y_test, y_pred)

# F1-score tells you how good your model really is when Precision and Recall disagree.
f1 = f1_score(y_test, y_pred)

print("F1-score:", f1)
print("Precision:", precision)
print("Recall:", recall)


# F1-score tells you how good your model really is when Precision and Recall disagree


# Recall = 1
# → “I caught all real True cases (even if I made mistakes)”
# Precision
# → “When I say True, how often am I correct?”
# Precision means: out of all predicted True, how many were actually True
# High precision = few wrong Trues (few false positives)

# F1-score
# → “Overall quality when I must balance Recall and Precision”

# F1 in machine learning does not directly tell you how good your model is overall,
# but it is a metric that combines precision and recall into a single number,
# giving a sense of balance between them.
