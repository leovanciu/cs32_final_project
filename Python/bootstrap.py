from memory_profiler import profile
import numpy as np
import time
from scipy.stats import bootstrap
import pandas as pd

# Load data
path = "/Users/ancavanciupopescu/Desktop/Classes/CS 32/Final project/Data/bootstrap_data.csv"
data = pd.read_csv(path, header=None)
data = pd.to_numeric(data.iloc[:, 1], errors='coerce')
data = data.to_numpy()[1:]
 
# Bootstrap using scipy.stats
@profile
def bootstrap_scipy(data, statistic, n_resamples=10000, confidence_level=0.95):
    res = bootstrap((data,), statistic, n_resamples=n_resamples, confidence_level=confidence_level, method='percentile')
    return res.confidence_interval

# Bootstrap from scratch
@profile
def bootstrap_from_scratch(data, statistic, n_resamples=10000, alpha=0.05):
    n = len(data)
    idx = np.random.randint(0, n, (n_resamples, n))
    samples = data[idx]
    stat = np.sort(statistic(samples, axis=1))
    confidence_interval = np.percentile(stat, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return confidence_interval

# Define sample mean
def sample_mean(data, axis=None):
    return np.mean(data, axis=axis)

# Measure execution and memory
start_time_scipy = time.time()
ci_scipy = bootstrap_scipy(data, sample_mean)
end_time_scipy = time.time()
scipy_time = end_time_scipy - start_time_scipy
start_time_scratch = time.time()
ci_scratch = bootstrap_from_scratch(data, sample_mean)
end_time_scratch = time.time()
scratch_time = end_time_scratch - start_time_scratch
print("95% Confidence interval for the mean using scipy.stats:", ci_scipy)
print("Execution time for scipy.stats:", scipy_time, "seconds")
print("95% Confidence interval for the mean from scratch:", ci_scratch)
print("Execution time from scratch:", scratch_time, "seconds")