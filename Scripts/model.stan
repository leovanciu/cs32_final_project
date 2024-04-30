data {
int N; // Number of data points
int K; // Number of predictors
matrix[N, K] X; // Predictor matrix
vector[N] y; // Response variable
}
parameters {
vector[K] beta; // Regression coefficients 
real sigma; // Standard deviation of the residuals
}

model {
// Priors
beta ~ normal(0, 10); // Priors for the coefficients
sigma ~ cauchy(0, 5); // Prior for the standard deviation

// Likelihood
y ~ normal(X * beta , sigma);
}

