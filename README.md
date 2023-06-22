## Empirical Bayesian Estimation of Censored Gaussian (Tobit) Matrices

<!-- badges: start -->
[![R-CMD-check](https://github.com/barbehenna/ebTobit/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/barbehenna/ebTobit/actions/workflows/R-CMD-check.yaml)
[![CRAN status](https://www.r-pkg.org/badges/version/ebTobit)](https://CRAN.R-project.org/package=ebTobit)
[![CRAN RStudio mirror downloads](https://cranlogs.r-pkg.org/badges/last-month/ebTobit?color=blue)](https://r-pkg.org/pkg/ebTobit)
[![CRAN RStudio mirror downloads](https://cranlogs.r-pkg.org/badges/grand-total/ebTobit?color=blue)](https://r-pkg.org/pkg/ebTobit)
[![Lifecycle: stable](https://img.shields.io/badge/lifecycle-stable-brightgreen.svg)](https://lifecycle.r-lib.org/articles/stages.html#stable)
<!-- badges: end -->

### What is it?

An R package for denoising censored, Gaussian means with empirical Bayes $g$-modeling.
The general model is as follows:

$$
\theta_i \sim_{iid} g \quad (\subseteq \mathbb{R}^p)
$$

$$
X_{ij} \mid \theta_{ij} \sim_{indep.} N(\theta_{ij}, \sigma^2)
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

The bounds $L_{ij}$ and $R_{ij}$ are assumed to be known.
When $L_{ij} = R_{ij}$ there is a direct (noisy) measurement of $\theta_{ij}$, if $L_{ij} < R_{ij}$ then there is a censored measurement of $\theta_{ij}$.
This structure is commonly referred to as partially interval censored data and it allows for any combination of observed measurements and left-, right-, and interval-censored measurements.

We use a Tobit likelihood for each measurement:

$$
P(L, R \mid \theta) = \begin{cases}
\phi_{\sigma} ( L - \theta ) & L = R \\
\Phi_{\sigma} ( R - \theta ) - \Phi_{\sigma} ( L - \theta ) & L < R
\end{cases}
$$

where the standard Gaussian likelihood is used when there is a direct Gaussian measurement (ie $L = X = R$) and a Gaussian probability is used when there is a censored Gaussian measurement (ie $L < R$).


### What does it do?

This package provides an object `ebTobit` (Empirical Bayes model with Tobit likelihood) that estimates the prior, $g$ over a user-specified grid `gr` and then computes the posterior mean or $\ell_1$ mediod as estimates for $\theta$.
In one dimension, the $\ell_1$ mediod is the median.
By default `gr` is set using the exemplar method so the grid is the maximum likelihood estimate for each $\theta_{ij}$.
When the censoring interval is finite, the maximum likelihood estimate for each $\theta_{ij}$ is $0.5 ( L_{ij} + R_{ij} )$

Suppose $p = 1$ and there is no censoring, then the basic utility is:

```r
library(ebTobit)

# create noisy measurements
n <- 100
t <- sample(c(0, 5), size = n, replace = TRUE, prob = c(0.8, 0.2))
x <- t + stats::rnorm(n)

# fit g-model with default prior grid
res1 <- ebTobit(x)

# measure performance of estimated posterior mean
mean((t - fitted(res1))^2)
```

Next we can look at a more complicated example with $p = 10$:

```r
library(ebTobit)

# create noisy measurements (low rank structure)
n <- 1000; p <- 10
t <- matrix(stats::rgamma(n*p, shape = 5, rate = 1), n, p)
x <- t + matrix(stats::rnorm(n*p), n, p)

# assume we can't accurately measure x < 1 but we know theta > 0
L <- ifelse(x < 1, 0, x)
R <- ifelse(x < 1, 1, x)

# fit g-model with default prior grid
res2 <- ebTobit(x)
res3 <- ebTobit(L, R)

# oberve that the censoring affects the fitted range 
range(fitted(res2))
range(fitted(res3))

# fit censored data with a different grid (large and random not MLE)
res4 <- ebTobit(
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

This package is available on CRAN. It can also be installed directly from GitHub:

```r
remotes::install_github("barbehenna/ebTobit")
```


### Data

This R package also includes a real bile acid `data.frame` taken directly from Lei et al. (2018) (https://doi.org/10.1096/fj.201700055R) via https://github.com/WandeRum/GSimp (https://doi.org/10.1371/journal.pcbi.1005973). The bile acid data contains measurements of 34 bile acids for 198 patients; no missing values are present in the data. In our modeling, we assume the bile acid values are independent log-normal measurements.

```r
data(BileAcid, package = "ebTobit") # attach the bile acid data
```


### Who wrote it?

[Alton Barbehenn](https://github.com/barbehenna) and [Sihai Dave Zhao](https://github.com/sdzhao)


### What license?

GPL (>= 3)
