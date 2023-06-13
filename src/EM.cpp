#include <cmath>
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
//' @param rtol convergence tolerance: abs(loss_new - loss_old)/abs(loss_old)
//' @return the estimated prior distribution (a vector of masses corresponding
//' to the columns of A)
//'
//' @examples
//' set.seed(1)
//' t = sample(c(0,5), size = 100, replace = TRUE)
//' x = t + stats::rnorm(100)
//' gr = seq(from = min(x), to = max(x), length.out = 50)
//' A = stats::dnorm(outer(x, gr, "-"))
//' EM(A)
//' \dontrun{
//' # compare to solution from rmosek (requires additional library installation):
//' all.equal(
//'     REBayes::KWPrimal(A = A, d = rep(1, 50), w = rep(1/100, 100))$f,
//'     EM(A, maxiter = 1e+6, rtol = 1e-16), # EM alg converges slowly
//'     tolerance = 0.01
//' )
//' }
//' @useDynLib ebTobit
//' @importFrom Rcpp evalCpp
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector EM(const arma::mat& A, int maxiter = 1e+4, double rtol = 1e-6) {
    arma::vec g = arma::ones(A.n_cols) / A.n_cols;
    arma::vec f = A * g;
    double loglik = 0;
    double loglik_old = arma::datum::log_min * A.n_rows;
    bool conv = false;

    for(int i = 0; i < maxiter; i++) {
        g = A.t() * (1 / f) % g / A.n_rows;
        f = A * g;

        loglik = arma::sum(arma::log(f));
        if(loglik - loglik_old < rtol * abs(loglik_old)) {
            conv = true;
            break;
        }

        loglik_old = loglik;
    }

    if (!conv) {
        Rcpp::warning("EM algorithm failed to fully converge: consider increasing maxiter or decreasing rtol.");
    }

    Rcpp::NumericVector out = Rcpp::wrap(g);
    out.attr("dim") = R_NilValue;
    return out;
}
