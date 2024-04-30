set.seed(32)
setwd("/Data")

# Generate linear regression data
betas <- runif(10, -10, 10)
X <- matrix(rnorm(1e6 * 10), ncol = 10)
y <- 100 + X[,1:10] %*% betas + rnorm(1e6)
f <- sample(c(-1, 1), 1e6, replace = TRUE)
data <- cbind(y, X, f)
write.csv(data, "data.csv")

# Generate matrix data
A <- matrix(runif(1e6), ncol = 1e3)
B <- matrix(runif(1e6), ncol = 1e3)
write.csv(A, "A.csv")
write.csv(A, "B.csv")