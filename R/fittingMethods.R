#### Alternative fitting methods (depends on REBayes and MOSEK)

#' Convex Optimization of the Kiefer-Wolfowitz NPMLE
#'
#' This method only works if there is a working installation of \code{REBayes}
#' available. See the \code{REBayes} package and corresponding papers for more
#' implementation details.
#'
#' The matrix \code{A} is structured as follows: A_ij = P(X_i | theta = t_j),
#' where X_i is the i'th observation and t_j is the j'th set of
#' parameters/grid-point.
#'
#' @param A numeric matrix likelihoods
#' @param ... further arguments passed to \code{Rmosek} such as \code{rtol}
#' @return a vector containing the fitted prior
#' @export
ConvexPrimal <- function(A, ...) 
    REBayes::KWPrimal(A = A, 
                      d = rep(1, ncol(A)), 
                      w = rep(1 / nrow(A), nrow(A)), ...)$f

#' @inherit ConvexPrimal
#' @export
ConvexDual <- function(A, ...) 
    REBayes::KWDual(A = A, 
                    d = rep(1, ncol(A)), 
                    w = rep(1 / nrow(A), nrow(A)), ...)$f
