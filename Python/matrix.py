import numpy as np
from memory_profiler import profile
import time
import pandas as pd

# Load data
path = "/Users/ancavanciupopescu/Desktop/Classes/CS 32/Final project/Data/A_data.csv"
A = pd.read_csv(path).iloc[:, 1:].values
path = "/Users/ancavanciupopescu/Desktop/Classes/CS 32/Final project/Data/B_data.csv"
B = pd.read_csv(path).iloc[:, 1:].values

# Matrix multiplication
@profile
def multiply_matrices(A, B):
    return np.dot(A, B)

# Matrix inversion
@profile
def invert_matrix(A):
    regularization_term = 1e-5 * np.eye(A.shape[0])
    A_regularized = A + regularization_term
    return np.linalg.inv(A_regularized)

# Measure execution and memory
start_time_mult = time.time()
result_mult = multiply_matrices(A, B)
end_time_mult = time.time()
mult_time = end_time_mult - start_time_mult
start_time_inv = time.time()
result_inv = invert_matrix(A)
end_time_inv = time.time()
inv_time = end_time_inv - start_time_inv
print("Execution time for 500x500 matrix multiplication", mult_time, "seconds")
print("Execution time for 500x500 matrix inversion:", inv_time, "seconds")