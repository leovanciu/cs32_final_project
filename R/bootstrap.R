library(boot)
library(bench)
setwd("/Users/ancavanciupopescu/Desktop/Classes/CS 32/Final project/Data/")
data <- read.csv("bootstrap_data.csv")[,2]
set.seed(32)

# Bootstrap using boot package
bootstrap_package <- function(data, statistic, R) {
  result <- boot(data, statistic, R = R)
  ci <- boot.ci(result, type = "perc")$percent[4:5]
  return(ci)
}

# Bootstrap from scratch
bootstrap_scratch <- function(data, statistic, R) {
  n <- length(data)
  results <- numeric(R)
  for (i in 1:R) {
    resampled <- sample(data, replace = TRUE)
    results[i] <- statistic(resampled)
  }
  ci <- quantile(results, probs = c(0.025, 0.975))
  return(ci)
}

# Define sample mean
statistic_boot <- function(data, indices) {
  mean(data[indices])
}

statistic_function <- function(data) {
  mean(data)
}

# Measure execution time and memory
results <- bench::mark(
  package = { ci_package <- bootstrap_package(data, statistic_boot, 1000) },
  scratch = { ci_scratch <- bootstrap_scratch(data, statistic_function, 1000) },
  iterations = 10,
  check = FALSE
)
cat("Execution time for boot:", results$total_time[1], "\n")
cat("Memory allocation for boot:", results$mem_alloc[1]/1e6, "\n")
cat("95% Confidence interval for the mean using boot:", ci_package[1], "-", ci_package[2], "\n")
cat("Execution time from scratch:", results$total_time[2], "\n")
cat("Memory allocation from scratch:", results$mem_alloc[2]/1e6, "\n")
cat("95% Confidence interval for the mean from scratch:", ci_scratch[1], "-", ci_scratch[2], "\n")