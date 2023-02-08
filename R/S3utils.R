#### S3 methods to help work with EBGaME objects ####


#' Validate EBGaME Object
#' @param object any R object
#' @export
is.EBGaME <- function(object) {
    all(inherits(object, "EBGaME") &
            is.numeric(object$prior) &
            is.vector(object$prior) &
            is.numeric(object$lik) &
            is.matrix(object$lik) &
            is.numeric(object$gr) &
            is.matrix(object$gr) &
            ncol(object$lik) == nrow(object$gr) &
            length(object$prior) == nrow(object$gr) &
            all(object$lik >= 0) &
            all(object$prior >= 0) &
            abs(sum(object$prior) - 1) <= sqrt(.Machine$double.eps)
    )
}


#' Marginal Log-likelihood of an EBGaME object
#' @param object an object inheriting from class \code{\link{EBGaME}}
#' @param ... not used
#' @export
logLik.EBGaME <- function(object, ...) with(object, sum(log(lik %*% prior)))


#' Compute Posterior Mean of an EBGaME object
#' @param object an object inheriting from class \code{\link{EBGaME}}
#' @export
posterior_mean.EBGaME <- function(object)
    with(object, (lik %*% (prior * gr)) / drop(lik %*% prior))


#' Compute the Posterior L1 Mediod of an EBGaME object
#'
#' The posterior L1 mediod is defined as \\arg\\min_y E |y - t|_1 where the
#' expectation is taken over the posterior t|X=x. Here the posterior L1 mediod
#' is evaluated for each of the observations used to fit \code{object}.
#'
#' @param object an object inheriting from class \code{\link{EBGaME}}
#'
#' @importFrom stats optimize
#' @export
posterior_L1mediod.EBGaME <- function(object) {
    # set-up
    n = nrow(object$lik)
    p = ncol(object$gr)
    med = matrix(nrow = n, ncol = p)

    # compute each L1 mediod
    for (i in 1:n) {
        # L1 loss is separable, so we can compute each term separately
        for (j in 1:p) {
            med[i,j] = stats::optimize(
                function(t) with(object, sum(prior * lik[i,] * abs(t - gr[,j]))),
                interval = range(object$gr[,j])
            )$minimum
        }
    }

    med
}


#' Fitted Estimates of an EBGaME object
#'
#' Compute either the posterior mean (default) or posterior L1 mediod which
#' corresponds to the posterior median in one-dimension.
#'
#' @param object an object inheriting from class \code{\link{EBGaME}}
#' @param method either "mean" or "L1mediod" corresponding to the S3 methods:
#' \code{posterior_*.EBGaME()}
#' @param ... not used
#' @export
fitted.EBGaME <- function(object, method = "mean", ...)
    match.fun(paste0("posterior_",method,".EBGaME"))(object)
