import numpy as np
from sklearn.svm import SVC
from memory_profiler import profile
import time
import pandas as pd

# Load data
path = "/Users/ancavanciupopescu/Desktop/Classes/CS 32/Final project/Data/SVM_data.csv"
data = pd.read_csv(path).iloc[:, 1:]
y = data.iloc[:, 0].values
X = data.iloc[:, 1:].values

# SVM from scratch
@profile
def svm_from_scratch(X, y, epochs=500, learning_rate=0.01, C=1.0):
    w = np.zeros(X.shape[1])
    b = 0
    for epoch in range(epochs):
        for i in range(len(y)):
            if y[i] * (np.dot(X[i], w) + b) < 1:
                w += learning_rate * ((y[i] * X[i]) + (-2 * (1/C) * w))
                b += learning_rate * y[i]
            else:
                w -= learning_rate * (2 * (1/C) * w)
    return w, b

# Scikit-Learn SVM
@profile
def svm_scikit_learn(X, y):
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X, y)
    return clf.coef_[0], clf.intercept_[0]

# Measure execution and memory
start_time_scratch = time.time()
weights_scratch, bias_scratch = svm_from_scratch(X, y)
end_time_scratch = time.time()
scratch_time = end_time_scratch - start_time_scratch
start_time_skl = time.time()
weights_skl, bias_skl = svm_scikit_learn(X, y)
end_time_skl = time.time()
skl_time = end_time_skl - start_time_skl
print("Weights and bias from scratch:", weights_scratch, bias_scratch)
print("Execution time for SVM from scratch:", scratch_time, "seconds")
print("Weights and bias from scikit-learn:", weights_skl, bias_skl)
print("Execution time for SVM with scikit-learn:", skl_time, "seconds")