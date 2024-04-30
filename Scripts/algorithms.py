from memory_profiler import memory_usage
import numpy as np
import pandas as pd
import scipy.stats as stats
import time
import csv
import math
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
import cmdstanpy

## Define algorithms
# Generic loop sum
def loop_sum(n):
    sum = 0
    for i in range(1,(n+1)):
        sum += i
    return sum

# Loop for geometric mean
def loop_geom_mean(data):
    log_sum = 0
    for num in data:
        log_sum += np.log(num)
    return np.exp(log_sum / len(data))

# Vectorized computation of geometric mean
def vectorized_geom_mean(data):
    log_sum = np.sum(np.log(data))
    return np.exp(log_sum / len(data))

# Linear regression from base
def lin_reg_base(X, y):
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])
    beta_hat = np.linalg.inv(X_b.T @ X_b) @ (X_b.T @ y)
    return beta_hat

# sklearn-Learn linear regression
def lin_reg_sklearn(X, y):
    model = LinearRegression()
    model.fit(X, y)
    beta_hat = np.hstack([model.intercept_, model.coef_])
    return beta_hat

# Matrix multiplication
def multiply_matrices(A, B):
    return A @ B

# Matrix inversion
def invert_matrix(A):
    return np.linalg.inv(A)

# scipy.stats bootstrap
def bootstrap_scipy(data, statistic, n_resamples, confidence_level=0.95):
    res = stats.bootstrap((data,), statistic, n_resamples=n_resamples, confidence_level=confidence_level, method='percentile')
    return res.confidence_interval

# Bootstrap from base
def bootstrap_base(data, statistic, n_resamples, confidence_level=0.95):
    n = len(data)
    idx = np.random.randint(0, n, (n_resamples, n))
    samples = data[idx]
    stat = statistic(samples)
    confidence_interval = np.quantile(stat, [((1 - confidence_level) / 2), (1 - (1 - confidence_level) / 2)])
    return confidence_interval

def sample_mean(data):
    return sum(data)/len(data)

# Metropolis_hastings from scratch
def metropolis_hastings(X, y, num_samples, beta_0=np.zeros(11), proposal_sd=1, sigma=1):
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    current_beta = beta_0
    samples = [current_beta]
    
    Xb = X.dot(current_beta)
    current_likelihood = np.sum(stats.norm.logpdf(y, Xb, sigma))
    current_prior = np.sum(stats.norm.logpdf(current_beta, 0, 10))

    for i in range(num_samples):
        proposed_beta = np.random.normal(current_beta, proposal_sd)
        Xb_proposed = X.dot(proposed_beta)
        proposed_likelihood = np.sum(stats.norm.logpdf(y, Xb_proposed, sigma))
        proposed_prior = np.sum(stats.norm.logpdf(proposed_beta, 0, 10))
        
        p_accept = np.exp((proposed_likelihood + proposed_prior) - (current_likelihood + current_prior))
        
        if np.random.rand() < p_accept:
            current_beta = proposed_beta
            current_likelihood = proposed_likelihood
            current_prior = proposed_prior
        
        samples.append(current_beta)
    
    return np.array(samples)

# MCMC using Stan
model_file =  "Algorithms/model.stan"
model = cmdstanpy.CmdStanModel(stan_file=model_file)

def MCMC_stan(X, y, num_samples):
    intercept = np.ones((10**3, 1))
    X = np.hstack((intercept, X))
    stan_data = {'N': 10**3, 'K': 11, 'y': y, 'X': X}
    fit = model.sample(data=stan_data, chains=1, iter_sampling=num_samples)
    samples = fit.stan_variable('beta')
    return samples

# sklearn-Learn SVM
def svm_sklearn(X, y):
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X, y)
    return clf.coef_[0], clf.intercept_[0]

# SVM from base
def svm_base(X, y, epochs=500, learning_rate=0.01, C=1.0):
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


## Run algorithms and record execution time and memory
def main():
    start_all = time.time()
    # Load simulated data
    path = "/Data/linear_regression_data.csv"
    data = pd.read_csv(path).iloc[:, 1:]
    path = "/Data/A_data.csv"
    A = pd.read_csv(path).iloc[:, 1:].values
    path = "/Data/B_data.csv"
    B = pd.read_csv(path).iloc[:, 1:].values
    path_results = "/Results/Results_python.csv"
    
    with open(path_results, 'w', newline='') as file:
        # Load file where we record the results
        writer = csv.writer(file)
        writer.writerow(['Algorithm', 'n', 'Time', 'Memory'])

        for log_n in range(2, 7):
            # Select subset of data
            print(log_n)
            n = int(10**log_n)
            n_loop = int(n/10)
            sqrt_n = int(math.floor(math.sqrt(n)))
            ye3 = data.iloc[:(10**3), 0].values
            y = data.iloc[:n, 0].values
            X = data.iloc[:n, 1:11].values
            Xe3 = data.iloc[:(10**3), 1:11].values
            X_SVM = data.iloc[:sqrt_n, 1:11].values
            f = data.iloc[:sqrt_n, 11].values
            A_small = A[:sqrt_n,:sqrt_n]
            B_small = B[:sqrt_n,:sqrt_n]

            # Record execution time and memory
            duration = np.empty((10,13))
            mem_usage = np.empty((10,13))
            for run in range(10):
                print(run)
                start_time = time.time()
                execute = loop_sum(n)
                end_time = time.time()
                duration[run, 0] = end_time - start_time
                mem_usage[run, 0] = memory_usage((loop_sum, (n,)), max_usage=True)

                start_time = time.time()
                execute = loop_geom_mean(y)
                end_time = time.time()
                duration[run, 1] = end_time - start_time
                mem_usage[run, 1] = memory_usage((loop_geom_mean, (y,)), max_usage=True)

                start_time = time.time()
                execute = vectorized_geom_mean(y)
                end_time = time.time()
                duration[run, 2] = end_time - start_time
                mem_usage[run, 2] = memory_usage((vectorized_geom_mean, (y,)), max_usage=True)

                start_time = time.time()
                execute = multiply_matrices(A_small, B_small)
                end_time = time.time()
                duration[run, 3] = end_time - start_time
                mem_usage[run, 3] = memory_usage((multiply_matrices, (A_small, B_small)), max_usage=True)

                start_time = time.time()
                execute = invert_matrix(A_small)
                end_time = time.time()
                duration[run, 4] = end_time - start_time
                mem_usage[run, 4] = memory_usage((invert_matrix, (A_small,)), max_usage=True)

                start_time = time.time()
                execute = lin_reg_sklearn(X, y)
                end_time = time.time()
                duration[run, 5] = end_time - start_time
                mem_usage[run, 5] = memory_usage((lin_reg_sklearn, (X, y)), max_usage=True)

                start_time = time.time()
                execute = lin_reg_base(X, y)
                end_time = time.time()
                duration[run, 6] = end_time - start_time
                mem_usage[run, 6] = memory_usage((lin_reg_base, (X, y)), max_usage=True)
              
                start_time = time.time()
                execute = bootstrap_scipy(ye3, sample_mean, n_loop)
                end_time = time.time()
                duration[run, 7] = end_time - start_time
                mem_usage[run, 7] = memory_usage((bootstrap_scipy, (ye3, sample_mean, n_loop)), max_usage=True)

                start_time = time.time()
                execute = bootstrap_base(ye3, sample_mean, n_loop)
                end_time = time.time()
                duration[run, 8] = end_time - start_time
                mem_usage[run, 8] = memory_usage((bootstrap_base, (ye3, sample_mean, n_loop)), max_usage=True)

                start_time = time.time()
                execute = svm_sklearn(X_SVM, f)
                end_time = time.time()
                duration[run, 9] = end_time - start_time
                mem_usage[run, 9] = memory_usage((svm_sklearn, (X_SVM, f)), max_usage=True)

                start_time = time.time()
                execute = svm_base(X_SVM, f)
                end_time = time.time()
                duration[run, 10] = end_time - start_time
                mem_usage[run, 10] = memory_usage((svm_base, (X_SVM, f)), max_usage=True)

                start_time = time.time()
                execute = metropolis_hastings(Xe3, ye3, n_loop)
                end_time = time.time()
                duration[run, 11] = end_time - start_time
                mem_usage[run, 11] = memory_usage((metropolis_hastings, (Xe3, ye3, n_loop)), max_usage=True)

                start_time = time.time()
                execute = MCMC_stan(Xe3, ye3, n_loop)
                end_time = time.time()
                duration[run, 12] = end_time - start_time
                mem_usage[run, 12] = memory_usage((MCMC_stan, (Xe3, ye3, n_loop)), max_usage=True)

            print(mem_usage)
            print(duration)
            algo_names = ['loop_sum','loop_geom_mean', 'vectorized_geom_mean', 'matrix_multiplication', 'matrix_inversion', 'linear_regression_package',
                      'linear_regression_base', 'bootstrap_package', 'bootstrap_base', 'svm_package', 'svm_base', 'Metropolis_Hastings', 'MCMC_stan']
            for i in range(13):
                writer.writerow([algo_names[i], n,  np.median(duration[:,i]), np.median(mem_usage[:,i])])
    end_all = time.time()
    all_time = start_all-end_all
    print(all_time)
if __name__ == '__main__':
    main()