set.seed(32)

# Generate linear regression data
betas <- runif(10, -10, 10)
X <- matrix(rnorm(1e6 * 10), ncol = 10)
y <- 100 + X[,1:10] %*% betas + rnorm(1e6)
f <- sample(c(-1, 1), 1e6, replace = TRUE)
data <- cbind(y, X, f)
write.csv(data, "Data/data.csv")

# Generate matrix data
A <- matrix(runif(1e6), ncol = 1e3)
B <- matrix(runif(1e6), ncol = 1e3)
write.csv(A, "Data/A.csv")
write.csv(B, "Data/B.csv")