#### S3 methods to help work with ebTobit objects and helper functions ####


#' Validate ebTobit Object
#' @param object any R object
#' @returns boolean: TRUE if the object is a valid \code{\link{ebTobit}} object
#' @export
is.ebTobit <- function(object) {
    all(inherits(object, "ebTobit") &
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


#' Marginal Log-likelihood of an ebTobit object
#' @param object an object inheriting from class \code{\link{ebTobit}}
#' @param ... not used
#' @returns log likelihood for the fitted empirical Bayes model in \code{object}
#' @export
logLik.ebTobit <- function(object, ...) with(object, sum(log(lik %*% prior)))


#' Compute Posterior Mean of an ebTobit object
#' @param object an object inheriting from class \code{\link{ebTobit}}
#' @returns numeric matrix of posterior means for the fitted empirical Bayes
#' model in \code{object}
#' @export
posterior_mean.ebTobit <- function(object)
    with(object, (lik %*% (prior * gr)) / drop(lik %*% prior))


#' Compute the Posterior L1 Mediod of an ebTobit object
#'
#' The posterior L1 mediod is defined as \\arg\\min_y E |y - t|_1 where the
#' expectation is taken over the posterior t|X=x. Here the posterior L1 mediod
#' is evaluated for each of the observations used to fit \code{object}.
#'
#' @param object an object inheriting from class \code{\link{ebTobit}}
#' @returns numeric matrix of posterior L1 mediods for the fitted empirical
#' Bayes model in \code{object}
#' 
#' @importFrom stats optimize
#' @export
posterior_L1mediod.ebTobit <- function(object) {
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


#' Compute Posterior Mode of an ebTobit object
#' @param object an object inheriting from class \code{\link{ebTobit}}
#' @returns numeric matrix of posterior modes for the fitted empirical
#' Bayes model in \code{object}
#' @export
posterior_mode.ebTobit <- function(object)
    with(object, gr[apply(sweep(lik, MARGIN = 2, STATS = prior, FUN = "*"), MARGIN = 1, FUN = "which.max"), ])


#' Fitted Estimates of an ebTobit object
#'
#' Compute either the posterior mean (default) or posterior L1 mediod which
#' corresponds to the posterior median in one-dimension.
#'
#' @param object an object inheriting from class \code{\link{ebTobit}}
#' @param method either "mean", "L1mediod", or "mode" corresponding to the 
#' methods: \code{posterior_*.ebTobit()}
#' @param ... not used
#' @returns matrix containing the posterior estimates for measurements in the
#' fit empirical Bayes model \code{object}
#' @export
fitted.ebTobit <- function(object, method = "mean", ...)
    match.fun(paste0("posterior_",method,".ebTobit"))(object)


#' Fitted Estimates of an ebTobit object
#'
#' Compute either the posterior mean (default) or posterior L1 mediod which
#' corresponds to the posterior median in one-dimension.
#'
#' @param object an object inheriting from class \code{\link{ebTobit}}
#' @param L n x p matrix of lower bounds on observations
#' @param R n x p matrix of upper bounds on observations
#' @param s1 a single numeric standard deviation or an n x p matrix of standard
#' deviations
#' @param method either "mean", "L1mediod", or "mode" corresponding to the 
#' methods: \code{posterior_*.ebTobit()}
#' @param ... not used
#' @returns matrix of posterior estimates for new observations under the
#' provided, pre-fit empirical Bayes model \code{object}
#' @export
predict.ebTobit <- function(object, L, R = L, s1 = 1, method = "mean", ...) {
    # allow vector inputs when p = 1
    if (is.vector(L) & is.vector(R)) {
        L <- matrix(L, ncol = 1)
        R <- matrix(R, ncol = 1)
    }
    
    # expand s1 to match L
    if (length(s1) == 1) {
        s1 <- matrix(s1, nrow = nrow(L), ncol = ncol(L))
    }

    # basic checks
    stopifnot(is.ebTobit(object))
    stopifnot(is.matrix(L))
    stopifnot(all(dim(L) == dim(R)))
    stopifnot(all(L <= R))
    stopifnot(all(dim(s1) == dim(L)))
    stopifnot(all(s1 > 0))
    
    # likelihood of with new observations
    new_lik <- likMat(L = L, R = R, gr = object$gr, s1 = s1)

    # compute posterior statistic
    new_obj <- new_ebTobit(object$prior, object$gr, new_lik)
    match.fun(paste0("posterior_",method,".ebTobit"))(new_obj)
}
