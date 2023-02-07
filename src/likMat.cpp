// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>
#include <Rcpp.h>
#include <cmath>

// Compute the Standard Gaussian pdf
inline double dnorm_cpp(double x) {
    return exp(-0.5 * x * x) * M_SQRT1_2 * M_2_SQRTPI / 2;
}

// Compute the difference in Standard Gaussian cdfs
inline double phi_cpp(double x, double y) {
    return (erf(x * M_SQRT1_2) - erf(y * M_SQRT1_2)) / 2;
}

//' Helper Function - generate likelihood for pair (L,R) and mean gr
//'
//' Compute P(L_i, R_i | theta = t_k) for observations (L_i, R_i) and grid of
//' mean t_k.
//'
//' @param L numeric vector of lower bounds
//' @param R numeric vector of upper bounds
//' @param gr numeric vector of means
//' @return the likelihood under partial interval censoring
//'
//' @examples
//' \dontrun{
//' # set-up
//' p = 15
//' gr = stats::rnorm(p)
//' L = R = stats::rnorm(p)
//' missing.idx = sample.int(n = p, size = p/5)
//' L[missing.idx] = L[missing.idx] - stats::runif(length(missing.idx), 0, 1)
//' R[missing.idx] = R[missing.idx] + stats::runif(length(missing.idx), 0, 1)
//'
//' # R solution
//' lik = prod(ifelse(
//'            L == R,
//'            stats::dnorm(L-gr),
//'            stats::pnorm(R-gr) - stats::pnorm(L-gr)))
//'
//' # Compare R to RcppParallel method
//' all.equal(lik, lik_GaussianPIC(L, R, gr))
//' }
//' @useDynLib EBGaME
//' @importFrom Rcpp evalCpp
//' @export
// [[Rcpp::export]]
double lik_GaussianPIC(Rcpp::NumericVector L, Rcpp::NumericVector R, Rcpp::NumericVector gr) {
    double out = 1.0;
    L = L - gr;
    R = R - gr;

    for (int i = 0; i < L.length(); i++) {
        if(L[i] == R[i]) {
            out = out * dnorm_cpp(L[i]);
        } else {
            out = out * phi_cpp(R[i], L[i]);
        }
    }

    return out;
}




// Generic C++ likelihood calculator
// ---- Internal use only ----
template <typename InputIteratorR, typename InputIteratorL, typename InputIteratorG>
inline double lik_calc(InputIteratorR beginR, InputIteratorR endR,
                       InputIteratorL beginL, InputIteratorG beginG) {

    // value to return
    double rval = 1;

    // set iterators to beginning of ranges
    InputIteratorR itr = beginR;
    InputIteratorL itl = beginL;
    InputIteratorG itg = beginG;

    // for each input item
    while (itr != endR) {
        // take the value and increment the iterator
        double r = *itr++;
        double l = *itl++;
        double g = *itg++;

        // accumulate likelihood
        if (r == l) {
            rval *= dnorm_cpp(r-g);
        } else {
            rval *= phi_cpp(r-g, l-g);
        }
    }

    return rval;
}


// Enable parallel calculation of the likelihood by row of likelihood matrix
// (sample/observation). Serial calculation of likelihood within row (grid points).
struct LikMat : public RcppParallel::Worker {

    // Input matrices
    const RcppParallel::RMatrix<double> L;
    const RcppParallel::RMatrix<double> R;
    const RcppParallel::RMatrix<double> G;

    // Output matrix
    RcppParallel::RMatrix<double> lik;

    // Initialize from Rcpp input and output matrixes
    LikMat(const Rcpp::NumericMatrix L, const Rcpp::NumericMatrix R,
           const Rcpp::NumericMatrix G, const Rcpp::NumericMatrix lik)
        : L(L), R(R), G(G), lik(lik) {}

    // function call operator that work for the specified range (begin/end)
    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; i++) {
            for (std::size_t j = 0; j < G.nrow(); j++) {
                // rows we will operate on
                RcppParallel::RMatrix<double>::Row l = L.row(i);
                RcppParallel::RMatrix<double>::Row r = R.row(i);
                RcppParallel::RMatrix<double>::Row g = G.row(j);

                // calculate likelihood P(L[i,], R[i,] | g[j,])
                lik(i, j) = lik_calc(r.begin(), r.end(), l.begin(), g.begin());
            }
        }
    }

};


//' Helper Function - generate likelihood matrix
//'
//' Compute a matrix L whose entries are L[i,k] = P(L_i, R_i | theta = t_k) for
//' observations (L_i, R_i) and grid of means t_k.
//'
//' @param L n x p matrix of lower bounds
//' @param R n x p matrix of upper bounds
//' @param gr m x p matrix of candidate means
//' @return the n x m likelihood matrix under partial interval censoring
//'
//' @examples
//' \dontrun{
//' # set-up
//' n = 100; m = 50; p = 5
//' gr = matrix(stats::rnorm(m*p), m, p)
//' L = R = matrix(stats::rnorm(n*p), n, p)
//' missing.idx = sample.int(n = n*p, size = p*p)
//' L[missing.idx] = L[missing.idx] - stats::runif(p, 0, 1)
//'
//' # R solution
//' lik = matrix(nrow = n, ncol = m)
//' for (i in 1:n) {
//'     for(k in 1:m) {
//'         lik[i,k] = prod(ifelse(
//'             L[i,] == R[i,],
//'             stats::dnorm(L[i,]-gr[k,]),
//'             stats::pnorm(R[i,]-gr[k,]) - stats::pnorm(L[i,]-gr[k,])
//'         ))
//'     }
//' }
//'
//' # Compare R to RcppParallel method
//' all.equal(lik, likMat(L, R, gr))
//' }
//' @useDynLib EBGaME
//' @importFrom Rcpp evalCpp
//' @import RcppParallel
//' @export
// [[Rcpp::export]]
Rcpp::NumericMatrix likMat(Rcpp::NumericMatrix L, Rcpp::NumericMatrix R, Rcpp::NumericMatrix gr) {

    // allocate the matrix we will return
    Rcpp::NumericMatrix out(L.nrow(), gr.nrow());

    // create the worker
    LikMat lik(L, R, gr, out);

    // call worker with parallelFor
    // parallel over rows in out
    RcppParallel::parallelFor(0, L.nrow(), lik);

    return out;
}
