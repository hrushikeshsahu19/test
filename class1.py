import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)

# Display the first few rows of the dataset
print(X.head())
print(y.head())

# Check for missing values
print(X.isnull().sum())

# Basic statistics of the dataset
print(X.describe())

# Correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = X.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def lcs_length(X, Y):
    m = len(X)
    n = len(Y)

    # Create a 2D table to store lengths of longest common subsequence
    lcs_table = [[0] * (n + 1) for _ in range(m + 1)]

    # Build the lcs_table in bottom-up fashion
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])

    # lcs_table[m][n] contains the length of LCS for X[0..m-1], Y[0..n-1]
    return lcs_table[m][n]

def lcs(X, Y):
    m = len(X)
    n = len(Y)
    lcs_table = [[0] * (n + 1) for _ in range(m + 1)]

    # Build lcs_table in bottom-up fashion
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])

    # Following code to print LCS
    index = lcs_table[m][n]

    # Create a character array to store the lcs string
    lcs_str = [""] * (index + 1)
    lcs_str[index] = ""

    # Start from the right-most-bottom-most corner and
    # one by one store characters in lcs_str[]
    i = m
    j = n
    while i > 0 and j > 0:

        # If current character in X[] and Y are same, then
        # current character is part of LCS
        if X[i - 1] == Y[j - 1]:
            lcs_str[index - 1] = X[i - 1]
            i -=  index- 1


# Display the shapes of the resulting datasets
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Print the model coefficients
print("Model coefficients:", model.coef_)
print("Model intercept:", model.intercept_)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Plot predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()

# Example prediction
example_data = X_test.iloc[0].values.reshape(1, -1)
predicted_price = model.predict(example_data)
print("Predicted price for the example data:", predicted_price)

# Function to multiply two numbers
def multiply_numbers(num1, num2):
    return num1 * num2

# Function to calculate HCF using Euclidean algorithm
def calculate_hcf(num1, num2):
    while num2:
        num1, num2 = num2, num1 % num2
    return num1

