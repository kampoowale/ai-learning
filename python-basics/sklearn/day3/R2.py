# In simple terms:
# R² = 1 → perfect model (explains all variance)
# R² = 0 → model does no better than predicting the mean
# R² < 0 → model is worse than just predicting the mean

from sklearn.metrics import r2_score

# Actual and predicted values
y_actual = [3, 5, 7]
y_pred = [2.5, 5, 8]

# Using sklearn
r2 = r2_score(y_actual, y_pred)
print("R² (using sklearn):", r2)

# Manual calculation
y_mean = sum(y_actual) / len(y_actual)
sse = sum((yi - yhat)**2 for yi, yhat in zip(y_actual, y_pred))
sst = sum((yi - y_mean)**2 for yi in y_actual)
r2_manual = 1 - sse / sst
print("R² (manual calculation):", r2_manual)
