library(bench)
setwd("/Users/ancavanciupopescu/Desktop/Classes/CS 32/Final project/Data/")
data <- read.csv("linear_regression_data.csv")[,-1]
y <- data[,1]
X <- as.matrix(data[,2:6])

# Linear regression using lm()
linear_regression_package <- function(X, y) {
  model <- lm(y ~ X + 0)
  return(coef(model))
}

# Linear regression from scratch
linear_regression_scratch <- function(X, y) {
  Xb <- cbind(1, X)
  betas <- solve(t(Xb) %*% Xb) %*% (t(Xb) %*% y)
  return(betas)
}


# Measure execution time and memory
results <- bench::mark(
  package = { betas_package <- linear_regression_package(X, y) },
  scratch = { betas_scratch <- linear_regression_scratch(X, y) },
  iterations = 10,
  check = FALSE
)
cat("Execution time for lm:", results$total_time[1], "\n")
cat("Memory allocation for lm:", results$mem_alloc[1]/1e6, "\n")
cat("Coefficients from package:", betas_package, "\n")
cat("Execution time from scratch:", results$total_time[2], "\n")
cat("Memory allocation from scratch:", results$mem_alloc[2]/1e6, "\n")
cat("Coefficients from scratch:", betas_scratch, "\n")
