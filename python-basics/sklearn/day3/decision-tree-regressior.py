from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Sample data
X = [[1], [2], [3], [4], [5]]
y = [1.2, 1.9, 3.1, 3.9, 5.1]

# Train model
model = DecisionTreeRegressor(max_depth=2)
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Evaluate
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("MSE:", mse)
print("R²:", r2)
print("Predict", y_pred)

predict = model.predict([[6]])

print("Predict 6", predict)
