import numpy as np
from memory_profiler import profile
import time
import pandas as pd

# Load data
path = "/Users/ancavanciupopescu/Desktop/Classes/CS 32/Final project/Data/loop_data.csv"
data = pd.read_csv(path, header=None)
data = pd.to_numeric(data.iloc[:, 1], errors='coerce')
data = data.to_numpy()[1:]

# Generic loop to compute geometric mean
@profile
def geometric_mean_generic(data):
    if np.any(data <= 0):
        return float("nan")
    log_sum = 0
    for num in data:
        log_sum += np.log(num)
    return np.exp(log_sum / len(data))

# Vectorized computation of geometric mean
@profile
def geometric_mean_vectorized(data):
    if np.any(data <= 0):
        return float("nan")
    log_sum = np.sum(np.log(data))
    return np.exp(log_sum / len(data))

# Measure execution and memory
start_time_generic = time.time()
result_generic = geometric_mean_generic(data)
end_time_generic = time.time()
generic_time = end_time_generic - start_time_generic
start_time_vectorized = time.time()
result_vectorized = geometric_mean_vectorized(data)
end_time_vectorized = time.time()
vectorized_time = end_time_vectorized - start_time_vectorized
print("Geometric mean from generic loop:", result_generic)
print("Execution time for generic loop:", generic_time, "seconds")
print("Geometric mean from vectorized loop:", result_vectorized)
print("Execution time for vectorized loop:", vectorized_time, "seconds")