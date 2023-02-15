## Empirical Bayesian Estimation of Gaussian Matrices

### What is it?

An R package for denoising Gaussian means with empirical Bayes $g$-modeling.
The general model is as follows:

$$
\theta_i \sim_{iid} g \quad (\subseteq \mathbb{R}^p)
$$

$$
X_{ij} \mid \theta_{ij} \sim_{indep.} N(\theta_{ij}, 1)
$$

$$
L_{ij} \leq X_{ij} \leq R_{ij}
$$

The data is represented with matrices:

$$
\theta = \begin{bmatrix}
\theta_{11} & \dots & \theta_{1p} \\
\theta_{21} & \dots & \theta_{2p} \\
\vdots & \ddots & \vdots \\
\theta_{n1} & \dots & \theta_{np} \\
\end{bmatrix}
\qquad
X = \begin{bmatrix}
X_{11} & \dots & X_{1p} \\
X_{21} & \dots & X_{2p} \\
\vdots & \ddots & \vdots \\
X_{n1} & \dots & X_{np} \\
\end{bmatrix}
$$

$$
L = \begin{bmatrix}
L_{11} & \dots & L_{1p} \\
L_{21} & \dots & L_{2p} \\
\vdots & \ddots & \vdots \\
L_{n1} & \dots & L_{np} \\
\end{bmatrix}
\qquad
R = \begin{bmatrix}
R_{11} & \dots & R_{1p} \\
R_{21} & \dots & R_{2p} \\
\vdots & \ddots & \vdots \\
R_{n1} & \dots & R_{np} \\
\end{bmatrix}
$$

The bounds $L_{ij}$ and $R_{ij}$ are assumed to be known and constant.
When $L_{ij} = R_{ij}$ there is a direct (noisy) measurement of $\theta_{ij}$, if $L_{ij} < R_{ij}$ then there is a censored measurement of $\theta_{ij}$.
This structure is commonly referred to as partially interval censored data and it allows for any combination of observed measurements and left-, right-, and interval-censored measurements.


### What does it do?

This package provides an object `EBayesMat` (Empirical Bayes Gaussian Matrix estimate) that estimates the prior, $g$ over a user-specified grid `gr` and then computes the posterior mean or $\ell_1$ mediod as estimates for $\theta$. By default `gr` is set using the exemplar method so the grid is the maximum likelihood estimate for each $\theta_{ij}$.

Suppose $p = 1$ and there is no censoring, then the basic utility is:

```r
library(EBayesMat)

# create noisy measurements
n <- 100
t <- sample(c(0, 5), size = n, replace = TRUE, prob = c(0.8, 0.2))
x <- t + stats::rnorm(n)

# fit g-model with default prior grid
res1 <- EBayesMat(x)

# measure performance of estimated posterior mean
mean((t - fitted(res1))^2)
```

Next we can look at a more complicated example with $p = 10$:

```r
library(EBayesMat)

# create noisy measurements (low rank structure)
n <- 1000; p <- 10; r <- 3
u <- matrix(sample(1:5, size = n*r, replace = TRUE), n, r)
v <- matrix(stats::rexp(p*r), p, r)
x <- tcrossprod(u, v) + matrix(stats::rnorm(n*p), n, p)

# assume we can't accurately measure x < 1 but we know theta > 0
L <- ifelse(x < 1, 0, x)
R <- ifelse(x < 1, 1, x)

# fit g-model with default prior grid
res2 <- EBayesMat(x)
res3 <- EBayesMat(L, R)

# oberve that the censoring affects the fitted range 
range(fitted(res2))
range(fitted(res3))

# fit censored data with a different grid (large and random not MLE)
res4 <- EBayesMat(
    L = L,
    R = R,
    gr = sapply(1:p, function(j) stats::runif(1e+4, min = min(L[,j]), max = max(R[,j]))),
    algorithm = "EM"
)

# compute posterior mean and L1mediod given new data
# we can also predict based on partially interval-censored observations
y <- matrix(stats::rexp(5*p, rate = 0.5), 5, p)
predict(res4, y) # posterior mean
predict(res4, y, method = "L1mediod") # posterior L1-mediod
```


### How do install it?

Until the package is available on CRAN, it can be installed directly from GitHub:

```r
remotes::install_github("barbehenna/EBayesMat")
```


### Who wrote it?

Alton Barbehenn


### What license?

GPL (>= 3)
