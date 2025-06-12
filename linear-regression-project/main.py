# Linear Regression Example using scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
data = fetch_california_housing()
X = data.data
y = data.target
feature_names = data.feature_names

# 2. Create a DataFrame for easy exploration
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("📊 Dataset Head:")
print(df.head())

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📈 Mean Squared Error: {mse:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# 7. Visualize predictions vs actual values
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # line of perfect prediction
plt.grid(True)
plt.show()
