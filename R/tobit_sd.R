#' Maximum Likelihood Estimator for a Single Standard Deviation Parameter
#'
#' Use standard numerical optimization methods to maximize the log-likelihood of
#' the given problem. If all of the data is passed in, this method computes the
#' global estimate of standard deviation. By passing in a subset of the data,
#' more specific estimates can be made (ex column-specific standard deviations).
#'
#' @param L matrix of lower bounds on observations (n x p)
#' @param R matrix of upper bounds on observations (n x p)
#' @param mu matrix of known means (n x p)
#' @param interval a vector containing the end-points of the interval defining
#' the convex search space (default: \code{c(1e-4, 1e+2)})
#' @param tol the desired accuracy
#'
#' @return a list containing estimate (maximum) and log-likelihood (objective)
#' @importFrom stats optimize dnorm pnorm
#' @export
#'
#' @examples
#' set.seed(1)
#' n = 100; p = 5; r = 2
#' U.true = matrix(stats::rexp(n*r), n, r)
#' V.true = matrix(sample(x = c(1,4,7),
#'                        size = p*r, 
#'                        replace = TRUE, 
#'                        prob = c(0.7, 0.2, 0.1)), 
#'                 p, r)
#' TH = tcrossprod(U.true, V.true)
#' X = TH + matrix(stats::rnorm(n*p, sd = 1), n, p)
#' ldl <- 0.1 # lower detection limit, known to be non-negative
#' L <- ifelse(X < ldl, 0, X)
#' R <- ifelse(X < ldl, ldl, X)
#' tobit_sd_mle(L, R, mu = TH)
tobit_sd_mle <- function(L, R, mu = matrix(mean(L+R)/2, nrow(L), ncol(L)), 
                         interval = c(1e-4, 100), tol = .Machine$double.eps^0.25) {
    # set-up
    L <- L - mu
    R <- R - mu
    
    # fit MLE
    stats::optimize(
        f = function(s) sum(ifelse(L == R,
                                   stats::dnorm(L, sd = s, log = TRUE),
                                   log(pmax(stats::pnorm(R/s) - stats::pnorm(L/s), .Machine$double.xmin)))),
        interval = interval,
        maximum = TRUE,
        tol = tol
    )
}


#' Fit Tobit Standard Deviation via Maximum Likelihood
#'
#' Fit the matrix of standard deviations given censored observations current
#' mean estimates. Currently there are four models for S implemented: global,
#' column-specific, row-specific, and rank-1.
#'
#' @param L matrix of lower bounds on observations (n x p)
#' @param R matrix of upper bounds on observations (n x p)
#' @param mu matrix of known means (n x p)
#' @param sd.structure structure imposed on noise level estimates, must be one
#' of: "global", "column", "row", or "rank1"
#' @param interval a vector containing the end-points of the interval defining
#' the convex search space (default: \code{c(1e-4, 1e+2)})
#' @param tol the desired accuracy
#' @param maxiter early stopping condition
#'
#' @return matrix of maximum likelihood estimates for each observation's 
#' standard deviation (n x p)
#' @export
#'
#' @examples
#' set.seed(1)
#' n = 100; p = 5; r = 2
#' U.true = matrix(stats::rexp(n*r), n, r)
#' V.true = matrix(sample(x = c(1,4,7),
#'                        size = p*r, 
#'                        replace = TRUE, 
#'                        prob = c(0.7, 0.2, 0.1)), 
#'                 p, r)
#' TH = tcrossprod(U.true, V.true)
#' X = TH + matrix(stats::rnorm(n*p, sd = 1), n, p)
#' ldl <- 0.1 # lower detection limit, known to be non-negative
#' L <- ifelse(X < ldl, 0, X)
#' R <- ifelse(X < ldl, ldl, X)
#' 
#' tobit_sd(L, R, mu = TH)
#' tobit_sd(L, R, mu = TH, sd.structure = "column")
tobit_sd <- function(L, R, mu = matrix(colMeans(L+R)/2, nrow(L), ncol(L), byrow = TRUE), 
                     sd.structure = "global", interval = c(1e-4, 1e+2),
                     tol = .Machine$double.eps^0.25, maxiter = 1000) {
    # set-up
    n <- nrow(L)
    p <- ncol(L)

    # fit mle
    switch (sd.structure,
            "global" = matrix(tobit_sd_mle(L, R, mu = mu, interval = interval, tol = tol)$maximum, n, p),
            "column" = matrix(sapply(1:p, function(j) tobit_sd_mle(L[,j], R[,j], mu = mu[,j], interval = interval, tol = tol)$maximum), n, p, byrow = TRUE),
            "row" = matrix(sapply(1:n, function(i) tobit_sd_mle(L[i,], R[i,], mu = mu[i,], interval = interval, tol = tol)$maximum), n, p, byrow = FALSE),
            "rank1" = {
                sr <- lapply(1:n, function(i) tobit_sd_mle(L[i,], R[i,], mu[i,], interval = interval, tol = tol))
                loss.old <- sum(sapply(sr, "[[", "objective"))
                sr <- sapply(sr, "[[", "maximum")
                
                # alternate fitting rows and columns given the other
                for (iter in seq_len(maxiter)) {
                    sc <- sapply(1:p, function(j) tobit_sd_mle(L[,j]/sr, R[,j]/sr, mu = mu[,j]/sr, interval = interval, tol = tol)$maximum)
                    sr <- lapply(1:n, function(i) tobit_sd_mle(L[i,]/sc, R[i,]/sc, mu[i,]/sc, interval = interval, tol = tol))
                    loss <- sum(sapply(sr, "[[", "objective"))
                    sr <- sapply(sr, "[[", "maximum")
                    
                    # numerical optimization with high tolerance allows for jumping
                    # around optima, here we stop once the likelihood stops increasing
                    if (loss < loss.old || abs(loss - loss.old) < tol * abs(loss.old)) break
                    loss.old <- loss
                }
                
                tcrossprod(sr, sc)
            },
            stop("invalid sd.structure value")
    )
}
