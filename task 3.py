# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Load the dataset
df = pd.read_csv("C:\\Users\\Divya T C\\Downloads\\Housing.csv")  # Verify the path

# 3. Preprocess the data
df.columns = df.columns.str.strip().str.lower()  # Standardizing column names to lowercase
df.dropna(inplace=True)  # Remove missing values

# Print column names to verify
print("Columns in dataset:", df.columns.tolist())

# 4. Define features and target (Ensure correct column names)
X = df[['area']]  # Ensure correct lowercase column name
y = df['price']   # Target variable

# 5. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Predict on test set
y_pred = model.predict(X_test)

# 8. Evaluate the model
print("\nEvaluation Metrics:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 9. Plotting the Regression Line
plt.scatter(X_test, y_test, color='blue', label='Actual Price')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Line')
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Simple Linear Regression: Area vs Price")
plt.legend()
plt.grid(True)
plt.show()

# 10. Print Coefficients
print("\nIntercept (b0):", model.intercept_)
print("Coefficient (b1):",model.coef_[0])
