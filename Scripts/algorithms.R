library(bench)
library(boot)
library(e1071)
library(rstan)
library(MASS)
library(dplyr)
set.seed(32)
setwd("cs32_final_project/") # Put project file path here
start <- Sys.time()

## Define algorithms
# Generic loop sum
loop_sum <- function(n) {
  sum <- 0
  for (i in 1:n) {
    sum <- sum + 1
  }
  return(sum)
}

# Loop for geometric mean
loop_geom_mean <- function(data) {
  log_sum <- 0
  n <- length(data)
  for (num in data) {
    log_sum <- log_sum + log(num)
  }
  return(exp(log_sum / n))
}

# Vectorized computation of geometric mean
vectorized_geom_mean <- function(data) {
  return(exp(mean(log(data))))
}

# Linear regression using lm()
lin_reg_package <- function(X, y) {
  model <- lm(y ~ X + 0)
  return(coef(model))
}

# Linear regression from base
lin_reg_base <- function(X, y) {
  Xb <- cbind(1, X)
  beta_hat <- solve(t(Xb) %*% Xb) %*% (t(Xb) %*% y)
  return(betas)
}

# Matrix multiplication function
multiply_matrices <- function(A, B) {
  return(A %*% B)
}

# Matrix inversion function
invert_matrix <- function(A) {
  return(solve(A))
}

# Bootstrap from boot
bootstrap_package <- function(data, statistic, R) {
  result <- boot(data, statistic = statistic, R = R)
  ci <- boot.ci(result, type = "perc")$percent[4:5]
  return(ci)
}

# Bootstrap from base
bootstrap_base <- function(data, statistic, B) {
  n <- length(data)
  results <- replicate(B, statistic(sample(data, size = length(data), replace = TRUE)))
  ci <- quantile(results, probs = c(0.025, 0.975))
  return(ci)
}


mean_boot <- function(data, indices) {
  mean(data[indices])
}

mean_function <- function(data) {
  mean(data)
}

# SVM using e1071 package
svm_e1071 <- function(X, y) {
  model <- svm(x = X, y = as.factor(y), kernel = "linear", type = "C-classification", scale = FALSE)
  list(coefficients = t(as.vector(model$coefs) %*% model$SV), intercept = -model$rho)
}

# Simple linear SVM from base
svm_base <- function(X, y, epochs = 100, learning_rate = 0.01, lambda = 0.01) {
  n_samples <- nrow(X)
  n_features <- ncol(X)
  weights <- matrix(0, nrow = n_features, ncol = 1)
  intercept <- 0
  
  for (epoch in 1:epochs) {
    for (i in 1:n_samples) {
      if (y[i] * (crossprod(weights, X[i, ]) + intercept) < 1) {
        weights <- weights + learning_rate * ((y[i] * matrix(X[i, ], ncol = 1)) - (2 * lambda * weights))
        intercept <- intercept + learning_rate * y[i]
      } else {
        weights <- weights - learning_rate * (2 * lambda * weights)
      }
    }
  }
  list(coefficients = weights, intercept = intercept)
}

# Metropolis_hastings for linear regression from base
metropolis_hastings <- function(X, y, num_samples, beta_0 = rep(0,11), proposal_sd=1, sigma=1) {
  X <- cbind(1, X)
  current_beta <- beta_0
  samples <- list(current_beta)
    
  Xb <- X %*% current_beta
  current_likelihood <- sum(dnorm(y, Xb, sigma, log = TRUE))
  current_prior <- sum(dnorm(current_beta, 0, 10, log = TRUE))
  
  for (i in 1:num_samples) {
    proposed_beta <- mvrnorm(1, current_beta, diag(rep(proposal_sd, length(beta_0))))
    Xb_proposed <- X %*% proposed_beta
    proposed_likelihood <- sum(dnorm(y, Xb_proposed, sigma, log = TRUE))
    proposed_prior <- sum(dnorm(proposed_beta, 0, 10, log = TRUE))
    
    p_accept <- exp((proposed_likelihood + proposed_prior) - (current_likelihood + current_prior))
    
    if (runif(1) < p_accept) {
      current_beta <- proposed_beta
      current_likelihood <- proposed_likelihood
      current_prior <- proposed_prior
    }
    
    samples[[i + 1]] <- current_beta
  }
  
  
  return(do.call(rbind, samples))
}

# Hamiltonian MCMC for linear regression using Stan
model_file =  "/Scripts/model.stan"
model <- stan_model(model_file)

MCMC_stan <- function(X, y, num_samples) {
  X <- cbind(1, X)
  stan_data <- list(N = 1e3, K = 11, y = y, X = X)
  fit <- sampling(model, data = stan_data, chains = 1, iter = num_samples, refresh = 0)
  samples <- extract(fit)$beta
  return(samples)
}

## Run algorithms and record execution time and memory
# Load simulated data
A <- as.matrix(read.csv("Data/A.csv")[,-1])
B <- as.matrix(read.csv("Data/B.csv")[,-1])
data <- read.csv("Data/data.csv")[,-1]
results_list <- list()
algo_names <- c('loop_sum','loop_geom_mean', 'vectorized_geom_mean', 'matrix_multiplication', 'matrix_inversion', 'linear_regression_package',
              'linear_regression_base', 'bootstrap_package', 'bootstrap_base', 'svm_package', 'svm_base', 'Metroplis-Hastings_base', 'MCMC_stan')


for (log_n in 2:6) {
  # Slice subset of data
  n <- 10^log_n
  n_loop <- n / 10
  sqrt_n <- floor(sqrt(n))
  ye3 <- data[1:1e3,1]
  y <- data[1:n,1]
  X <- as.matrix(data[1:n, 2:11])
  Xe3 <- as.matrix(data[1:1e3, 2:11])
  X_SVM <- as.matrix(data[1:sqrt_n, 2:11])
  f <- data[1:sqrt_n, 12]
  A_small <- A[1:sqrt_n, 1:sqrt_n]
  B_small <- B[1:sqrt_n, 1:sqrt_n]
  
  # Perform benchmarks
  result <- bench::mark(
    loop_sum = loop_sum(n),
    loop_geom_mean = loop_geom_mean(y),
    vectorized_geom_mean = vectorized_geom_mean(y),
    matrix_multiplication = multiply_matrices(A_small, B_small),
    matrix_inversion = invert_matrix(A_small),
    lin_reg_package = lin_reg_package(X, y),
    lin_reg_base = lin_reg_base(X, y),
    bootstrap_package = bootstrap_package(ye3, mean_boot, n_loop),
    bootstrap_base = bootstrap_base(ye3, mean_function, n_loop),
    svm_e1071 = svm_e1071(X_SVM, f),
    svm_base = svm_base(X_SVM, f),
    metropolis_hastings = metropolis_hastings(Xe3, ye3, n_loop),
    MCMC_stan = MCMC_stan(Xe3, ye3, n_loop),
    iterations = 10,
    check = FALSE
  )
  
  # Store time and memory results in a list of dataframes
  result <- result %>%
    mutate(n = n) %>%
    select(expression, median, mem_alloc, n) %>%
    mutate(
      Algorithm = as.character(expression),
      Time = as.numeric(median),
      Memory = as.numeric(mem_alloc)/1024^2
    ) %>%
    select(Algorithm, n, Time, Memory)
  
  result$Algorithm <-c('loop_sum','loop_geom_mean', 'vectorized_geom_mean', 'matrix_multiplication', 'matrix_inversion', 'linear_regression_package',
                                       'linear_regression_base', 'bootstrap_package', 'bootstrap_base', 'svm_package', 'svm_base', 'Metropolis-Hastings', 'MCMC_stan')
  
  results_list[[length(results_list) + 1]] <- result
}

# Save results in csv file
all_results <- bind_rows(results_list)
path_results = "Results/Results_R.csv"
write.csv(all_results, path_results, row.names = FALSE)

# Measure total time
end <- Sys.time()
end-start