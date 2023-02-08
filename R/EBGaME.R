#' Create and Fit an EBGaME Object for Matrix Estimation
#'
#' Fit and estimate the nonparametric maximum likelihood estimator in R^p
#' (p >= 1) when the likelihood is Gaussian and possibly interval censored. If
#' p = 1, then \code{L}, \code{R}, and \code{gr} may be vectors (they are
#' immediately converted into matrices internally).
#'
#' To use a custom fitting algorithm, define a function \code{MyAlg} that
#' takes in an n x m likelihood matrix: P_ij = P(X_i | theta = t_j) and returns
#' a vector of estimated prior weights for t_j. Once \code{MyAlg} is defined,
#' fit the prior by using \code{algorithm = "MyAlg"} or use the function
#' itself \code{algorithm = MyAlg}.
#'
#' Alternative fitting algorithms "ConvexPrimal"and "ConvexDual" have been
#' (wrappers of \code{REBayes::KWPrimal} and \code{REBayes::KWDual},
#' respectively) included and can be used if MOSEK and \code{REBayes} are
#' properly installed.
#'
#' @param L n x p matrix of lower bounds on observations
#' @param R n x p matrix of upper bounds on observations
#' @param gr m x p matrix of grid points
#' @param algorithm method to fit prior, either a function or function name
#' @param ... further arguments passed into fitting method such as \code{rtol}
#' and \code{maxiter}, see for example \code{\link{EM}}
#'
#' @return a fitted \code{EBGaME} object containing at least the prior weights,
#' corresponding grid/support points, and likelihood matrix relating the grid to
#' the observations
#' @export
#'
#' @examples
#' set.seed(1)
#' n <- 100
#' p <- 5
#' r <- 2
#' U.true <- matrix(stats::rexp(n*r), n, r)
#' V.true <- matrix(sample(x = c(1,4,7), size = p*r, replace = TRUE, prob = c(0.7, 0.2, 0.1)), p, r)
#' TH <- tcrossprod(U.true, V.true)
#' X <- TH + matrix(stats::rnorm(n*p), n, p)
#'
#' # fit uncensored method
#' fit1 <- EBGaME(X)
#'
#' # fit left-censored method
#' ldl <- 1 # lower and upper detection limits
#' udl <- Inf
#' L <- ifelse(X < ldl, 0, ifelse(X <= udl, X, udl))
#' R <- ifelse(X < ldl, ldl, ifelse(X <= udl, X, Inf))
#' fit2 <- EBGaME(L, R)
EBGaME <- function(L, R = L, gr = (R+L)/2, algorithm = "EM", ...) {
    # allow vector inputs when p = 1
    if (is.vector(L) & is.vector(R) & is.vector(gr)) {
        L <- matrix(L, ncol = 1)
        R <- matrix(R, ncol = 1)
        gr <- matrix(gr, ncol = 1)
    }

    # basic checks
    stopifnot(is.matrix(L))
    stopifnot(all(dim(L) == dim(R)))
    stopifnot(all(L <= R))
    stopifnot(is.matrix(gr))
    stopifnot(ncol(gr) == ncol(L))

    # set-up
    n <- nrow(L)
    m <- nrow(gr)

    # get fitting algorithm
    if (is.character(algorithm)) {
        algorithm <- match.fun(algorithm)
    }

    # define full likelihood matrix
    # P_ij = prod_{k=1}^m P(X_ik | theta = t_jk)
    lik <- likMat(L = L, R = R, gr = gr)

    # fit prior
    prior <- algorithm(lik, ...)

    # return fit
    new_EBGaME(
        prior = prior,
        gr = gr,
        lik = lik
    )
}




#' Create a new EBGaME object
#'
#' Validate the provided elements and populate the object. Future extensions may
#' be loosened to allow non-numeric \code{gr} but it is currently required for
#' posterior mean and mediod calculations.
#'
#' @param prior numeric vector of non-negative weights (sums to one)
#' @param gr numeric matrix of support points
#' @param lik numeric matrix of likelihoods
#' @return an \code{EBGaME} object containing at least the prior weights,
#' corresponding grid/support points, and likelihood matrix relating the grid to
#' the observations
#' @export
new_EBGaME <- function(prior, gr, lik) {
    # basic checks
    stopifnot(is.vector(prior))
    stopifnot(is.numeric(prior))
    stopifnot(is.matrix(gr))
    stopifnot(is.numeric(gr))
    stopifnot(is.matrix(lik))
    stopifnot(is.numeric(lik))
    stopifnot(ncol(lik) == nrow(gr))
    stopifnot(all(lik >= 0))
    stopifnot(length(prior) == nrow(gr))
    stopifnot(all(prior >= 0))
    stopifnot(abs(sum(prior) - 1) <= sqrt(.Machine$double.eps))

    # create object
    structure(list(
        prior = prior,
        gr = gr,
        lik = lik
    ), class = "EBGaME")
}
