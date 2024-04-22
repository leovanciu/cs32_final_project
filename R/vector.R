library(bench)
setwd("/Users/ancavanciupopescu/Desktop/Classes/CS 32/Final project/Data/")
data <- read.csv("loop_data.csv")[,2]

# Generic loop to compute geometric mean
geometric_mean_loop <- function(data) {
  if (any(data <= 0)) return(NA)
  log_sum <- 0
  n <- length(data)
  for (num in data) {
    log_sum <- log_sum + log(num)
  }
  return(exp(log_sum / n))
}

# Vectorized computation of geometric mean
geometric_mean_vectorized <- function(data) {
  if (any(data <= 0)) return(NA)
  return(exp(mean(log(data))))
}

# Measure execution time and memory
results <- bench::mark(
  loop = { loop_result <- geometric_mean_loop(data) },
  vectorized = { vectorized_result <- geometric_mean_vectorized(data) },
  iterations = 10,
  check = FALSE
)
cat("Geometric mean from generic loop:", loop_result, "\n")
cat("Execution time for generic loop:", results$total_time[1], "\n")
cat("Memory allocation for generic loop:", results$mem_alloc[1]/1e6, "\n")
cat("Geometric mean from vectorized loop:", vectorized_result, "\n")
cat("Execution time for vectorized loop:", results$total_time[2], "\n")
cat("Memory allocation for vectorized loop:", results$mem_alloc[2]/1e6, "\n")
