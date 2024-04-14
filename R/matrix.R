library(bench)
setwd("/Users/ancavanciupopescu/Desktop/Classes/CS 32/Final project/Data/")
A <- as.matrix(read.csv("A_data.csv")[,-1])
B <- as.matrix(read.csv("B_data.csv")[,-1])

# Matrix multiplication function
multiply_matrices <- function(A, B) {
  return(A %*% B)
}

# Matrix inversion function
invert_matrix <- function(A) {
  n <- dim(A)[1]
  regularization_term <- diag(1e-5, n, n)
  A_regularized <- A + regularization_term
  return(solve(A_regularized))
}

# Measure execution time and memory
results <- bench::mark(
  multiply = { mult_result <- multiply_matrices(A, B) },
  invert = { inv_result <- invert_matrix(A) },
  iterations = 10,
  check = FALSE
)
cat("Execution time for 500x500 matrix multiplication:", results$total_time[1], "\n")
cat("Memory allocation for 500x500 matrix multiplication:", results$mem_alloc[1]/1e6, "\n")
cat("Execution time from 500x500 matrix inversion:", results$total_time[2], "\n")
cat("Memory allocation from 500x500 matrix inversion:", results$mem_alloc[2]/1e6, "\n")