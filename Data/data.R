set.seed(32)
setwd("/Users/ancavanciupopescu/Desktop/Classes/CS 32/Final project/Data/")

# Generate linear regression data
X <- matrix(rnorm(1e4 * 5), ncol = 5)
y <- 2 + 3 * X[,1] + 4 * X[,2] + 5 * X[,3] + 6 * X[,4] + 7 * X[,5] + rnorm(1e4)
data <- cbind(y, X)
write.csv(data, "linear_regression_data.csv")

# Generate bootstrap data
data <- rnorm(1e4)
write.csv(data, "bootstrap_data.csv")

# Generate geometric mean data
data <- runif(1e4, 0.1, 1)
write.csv(data, "loop_data.csv")

# Generate matrix data
A <- matrix(runif(500 * 500), ncol = 500)
B <- matrix(runif(500 * 500), ncol = 500)
write.csv(A, "A_data.csv")
write.csv(A, "B_data.csv")

# Generate SVM data
X <- matrix(rnorm(2000), nrow = 100, ncol = 20)
y <- sample(c(-1, 1), 100, replace = TRUE)
data <- cbind(y, X)
write.csv(data, "SVM_data.csv")
