library(e1071)
library(bench)
setwd("/Users/ancavanciupopescu/Desktop/Classes/CS 32/Final project/Data/")
set.seed(32)
data <- read.csv("SVM_data.csv")[,-1]
y <- data[,1]
X <- as.matrix(data[,2:20])

# SVM using e1071 package
svm_package <- function(X, y) {
  model <- svm(x = X, y = as.factor(y), kernel = "linear", type = "C-classification", scale = FALSE)
  list(coefficients = t(as.vector(model$coefs) %*% model$SV), intercept = -model$rho)
}

# Simple linear SVM from scratch
svm_from_scratch <- function(X, y, epochs = 100, learning_rate = 0.01, lambda = 0.01) {
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

# Measure execution time and memory
results <- bench::mark(
  package = { package_result <- svm_package(X, y) },
  scratch = { scratch_result <- svm_from_scratch(X, y) },
  iterations = 10,
  check = FALSE
)
cat("Execution time for e1071:", results$total_time[1], "\n")
cat("Memory allocation for e1071:", results$mem_alloc[1]/1e6, "\n")
cat("Coefficients from package:\n", package_result$coefficients, "\nIntercept from package:", package_result$intercept, "\n")
cat("Execution time from scratch:", results$total_time[2], "\n")
cat("Memory allocation from scratch:", results$mem_alloc[2]/1e6, "\n")
cat("Coefficients from scratch:\n", scratch_result$coefficients, "\nIntercept from scratch:", scratch_result$intercept, "\n")
