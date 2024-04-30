data {
int N; // Sample size
int K; // Number of predictors
matrix[N, K] X; // Covariates
vector[N] y; // Outcome
}
parameters {
vector[K] beta;
real sigma;
}

model {
// Priors
beta ~ normal(0, 10);
sigma ~ cauchy(0, 5);

// Likelihood
y ~ normal(X * beta , sigma);
}

