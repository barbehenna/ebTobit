#### S3 methods to help work with EBayesMat objects ####


#' Validate EBayesMat Object
#' @param object any R object
#' @export
is.EBayesMat <- function(object) {
    all(inherits(object, "EBayesMat") &
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


#' Marginal Log-likelihood of an EBayesMat object
#' @param object an object inheriting from class \code{\link{EBayesMat}}
#' @param ... not used
#' @export
logLik.EBayesMat <- function(object, ...) with(object, sum(log(lik %*% prior)))


#' Compute Posterior Mean of an EBayesMat object
#' @param object an object inheriting from class \code{\link{EBayesMat}}
#' @export
posterior_mean.EBayesMat <- function(object)
    with(object, (lik %*% (prior * gr)) / drop(lik %*% prior))


#' Compute the Posterior L1 Mediod of an EBayesMat object
#'
#' The posterior L1 mediod is defined as \\arg\\min_y E |y - t|_1 where the
#' expectation is taken over the posterior t|X=x. Here the posterior L1 mediod
#' is evaluated for each of the observations used to fit \code{object}.
#'
#' @param object an object inheriting from class \code{\link{EBayesMat}}
#'
#' @importFrom stats optimize
#' @export
posterior_L1mediod.EBayesMat <- function(object) {
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

    drop(med)
}


#' Compute Posterior Mode of an EBayesMat object
#' @param object an object inheriting from class \code{\link{EBayesMat}}
#' @export
posterior_mode.EBayesMat <- function(object)
    with(object, gr[apply(sweep(lik, MARGIN = 2, STATS = prior, FUN = "*"), MARGIN = 1, FUN = "which.max"), ])


#' Fitted Estimates of an EBayesMat object
#'
#' Compute either the posterior mean (default) or posterior L1 mediod which
#' corresponds to the posterior median in one-dimension.
#'
#' @param object an object inheriting from class \code{\link{EBayesMat}}
#' @param method either "mean", "L1mediod", or "mode" corresponding to the 
#' methods: \code{posterior_*.EBayesMat()}
#' @param ... not used
#' @export
fitted.EBayesMat <- function(object, method = "mean", ...)
    match.fun(paste0("posterior_",method,".EBayesMat"))(object)


#' Fitted Estimates of an EBayesMat object
#'
#' Compute either the posterior mean (default) or posterior L1 mediod which
#' corresponds to the posterior median in one-dimension.
#'
#' @param object an object inheriting from class \code{\link{EBayesMat}}
#' @param L n x p matrix of lower bounds on observations
#' @param R n x p matrix of upper bounds on observations
#' @param method either "mean", "L1mediod", or "mode" corresponding to the 
#' methods: \code{posterior_*.EBayesMat()}
#' @param ... not used
#' @export
predict.EBayesMat <- function(object, L, R = L, method = "mean", ...) {
    # allow vector inputs when p = 1
    if (is.vector(L) & is.vector(R)) {
        L <- matrix(L, ncol = 1)
        R <- matrix(R, ncol = 1)
    }

    # basic checks
    stopifnot(is.EBayesMat(object))
    stopifnot(is.matrix(L))
    stopifnot(all(dim(L) == dim(R)))
    stopifnot(all(L <= R))

    # likelihood of with new observations
    new_lik <- likMat(L, R, object$gr)

    # compute posterior statistic
    new_obj <- new_EBayesMat(object$prior, object$gr, new_lik)
    match.fun(paste0("posterior_",method,".EBayesMat"))(new_obj)
}
