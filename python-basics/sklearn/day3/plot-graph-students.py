import numpy as np
import matplotlib.pyplot as plt

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
