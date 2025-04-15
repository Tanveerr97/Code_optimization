
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # feature (years of experience)
y = np.array([1000, 1500, 2000, 2500, 3000])  # target (salary)

# Create and fit the model
model = LinearRegression()
model.fit(x, y)

# Get parameters
slope = model.coef_[0]
intercept = model.intercept_



# Predict values
y_pred = model.predict(x)

# Plot
plt.scatter(x, y, color='blue', label='Actual')
plt.plot(x, y_pred, color='red', label='Predicted Line')
plt.xlabel('X (e.g., Years of Experience)')
plt.ylabel('Y (e.g., Salary)')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()


new_x = np.array([6, 7, 8]).reshape(-1, 1)
new_y_pred = model.predict(new_x)

print(new_y_pred)