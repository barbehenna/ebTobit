#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]


//' Nonparametric Maximum Likelihood via Expectation Maximization
//'
//' Compute the nonparametric maximum likelihood estimate given a likelihood
//' matrix. The matrix A is structured so that A_\{ij\} = f(X_i | theta_j) for
//' some grid of potential parameter values theta_1, ..., theta_p and
//' observations X_1, ..., X_n. The parameters, theta_j, can be multidimensional
//' because all that is required is the likelihood. Convergence is achieved when
//' the relative improvements of the log-likelihood is below the provided
//' tolerance level.
//'
//' @param A numeric matrix likelihoods
//' @param maxiter early stopping condition
//' @param tol convergence tolerance
//' @return the estimated prior distribution (a vector of masses corresponding
//' to the columns of A)
//'
//' @examples
//' set.seed(1)
//' t = sample.int(5, size = 1000, replace = TRUE)
//' x = t + stats::rnorm(1000)
//' gr = seq(from = min(x), to = max(x), length.out = 500)
//' A = stats::dnorm(outer(x, gr, "-"))
//' EM(A)
//'
//' \dontrun{
//' all.equal(
//'     REBayes::KWPrimal(A = A, d = rep(1, 500), w = rep(1/1000, 1000))$f,
//'     drop(EM(A, maxiter = 1e+9, tol = 1e-16)) # EM alg converges slowly
//' )
//' }
//'
//' @useDynLib EBGaME
//' @importFrom Rcpp evalCpp
//' @export
// [[Rcpp::export]]
arma::vec EM(const arma::mat& A, int maxiter = 1000, double tol = 1e-6) {
    arma::vec g = arma::ones(A.n_cols) / A.n_cols;
    arma::vec f = A * g;
    double loglik = 0;
    double loglik_old = arma::datum::log_min * A.n_rows;
    bool conv = false;

    for(int i = 0; i < maxiter; i++) {
        g = A.t() * (1 / f) % g / A.n_rows;
        f = A * g;

        loglik = arma::sum(arma::log(f));
        if(loglik - loglik_old < tol) {
            conv = true;
            break;
        }

        loglik_old = loglik;
    }

    if (!conv) {
        Rcpp::warning("EM algorithm failed to fully converge: consider increasing maxiter or decreasing tol.");
    }

    return g;
}
