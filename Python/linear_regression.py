from memory_profiler import profile
import numpy as np
import time
from sklearn.linear_model import LinearRegression
import pandas as pd

# Load data
path = "/Users/ancavanciupopescu/Desktop/Classes/CS 32/Final project/Data/linear_regression_data.csv"
data = pd.read_csv(path).iloc[:, 1:]
y = data.iloc[:, 0].values
X = data.iloc[:, 1:].values

# Linear regression from scratch
@profile
def linear_regression_from_scratch(X, y):
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])
    beta_hat = np.linalg.inv(X_b.T @ X_b) @ (X_b.T @ y)
    return beta_hat

# Scikit-Learn implementation
@profile
def linear_regression_scikit_learn(X, y):
    model = LinearRegression()
    model.fit(X, y)
    beta_hat = np.hstack([model.intercept_, model.coef_])
    return beta_hat

# Measure execution and memory
start_time_skl = time.time()
beta_hat_skl = linear_regression_scikit_learn(X, y)
end_time_skl = time.time()
skl_time = end_time_skl - start_time_skl
start_time_scratch = time.time()
beta_hat_scratch = linear_regression_from_scratch(X, y)
end_time_scratch = time.time()
scratch_time = end_time_scratch - start_time_scratch
print("Beta hat coefficients using scikit-learn:", beta_hat_skl)
print("Execution time for scikit-learn:", skl_time, "seconds")
print("Beta hat coefficients from scratch:", beta_hat_scratch)
print("Execution time from scratch:", scratch_time, "seconds")