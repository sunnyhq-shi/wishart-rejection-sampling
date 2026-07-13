// [[Rcpp::depends(RcppArmadillo, BH)]]
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <chrono>

using namespace Rcpp;
using namespace arma;

// -------------------------------------------------------
// Notation:
//   ldetxbarg  = (1/n) * sum_i log|X_i|  (log geometric mean of determinants)
//   ldet_muhat = log|muhat|
//   ldet_mu0   = log|mu0|
//   muhat      = (n*xbar + kappa*mu0) / (n+kappa)
//   bound      = (p-1)/2  (domain lower boundary for alpha)
//   lambda     = beta*eta + n*(ldet_muhat - ldetxbarg) + kappa*(ldet_muhat - ldet_mu0)
//   nu_star    = ahat*lambda + 1  (covering Gamma shape)
// -------------------------------------------------------

// -------------------------------------------------------
// Random generators
// -------------------------------------------------------
std::mt19937 rng(std::random_device{}());
std::uniform_real_distribution<double> unif(0.0, 1.0);

// -------------------------------------------------------
// Input validation helpers
//
// These centralize the checks needed to keep the EM mode
// finder and rejection sampler from hanging, crashing, or
// silently returning nonsense on malformed input. They are
// called at the top of every exported entry point, before
// any heavy computation or container allocation, so that
// bad input always produces a clean, informative R-level
// error (via Rcpp::stop) rather than an infinite loop, a
// low-level Armadillo exception, or a silently wrong answer.
// -------------------------------------------------------

/**
 * @brief Small but numerically meaningful offset used to nudge alpha
 * strictly inside its domain (away from the (p-1)/2 boundary) when
 * clamping EM/Newton bracket endpoints.
 *
 * NOTE: an earlier version used std::numeric_limits<double>::min()
 * (~2.2e-308) for this purpose. For any bound of realistic magnitude
 * (e.g. bound = 1.0), adding that value is a strict no-op in double
 * precision (it is far smaller than machine epsilon relative to the
 * bound), so the clamp silently failed to move the iterate off the
 * boundary. That let the Newton bracket collapse onto the boundary
 * exactly, which then tripped the "a must be > (p-1)/2" domain checks
 * in digammap/trigammap/lgammap after many EM iterations of otherwise
 * legitimate convergence. 1e-8 is comfortably above machine epsilon
 * for the alpha magnitudes this model operates at.
 */
static const double BOUND_EPS = 1e-8;

/**
 * @brief Check X for the fully-degenerate case: every observation
 * numerically identical to every other observation.
 *
 * log-det is strictly concave on the positive-definite cone. Jensen's
 * inequality for a strictly concave function applied to n equally-weighted
 * points is an equality if and only if ALL n points coincide -- not if
 * merely some subset of them happen to match. Concretely: a dataset of
 * n=10 observations where exactly one pair is an exact duplicate (the
 * other 8 distinct) still gives lambda_raw comfortably positive (barely
 * different from the all-distinct case); only when all 10 collapse to a
 * single repeated value does lambda_raw hit exactly 0. An earlier version
 * of this check incorrectly rejected data as soon as ANY two slices
 * coincided, which both mischaracterized the actual failure condition and
 * would reject perfectly well-posed datasets that happen to contain a
 * coincidental (or legitimately duplicated, e.g. from rounding/recording)
 * tie among otherwise-varied observations.
 *
 * This check is therefore intentionally narrow: it only fires when every
 * single slice is numerically identical to the first, which is exactly
 * the condition under which lambda_raw = 0 and the improper prior's
 * posterior for alpha becomes unidentified in its upper tail. It exits
 * as soon as it finds any slice that differs, so it is cheap (O(1) in
 * the typical case, O(n) only in the fully-degenerate case).
 */
inline void check_not_fully_degenerate(const arma::cube& X) {
    if (X.n_slices < 2) return;
    const arma::mat& X0 = X.slice(0);
    for (int i = 1; i < (int)X.n_slices; i++)
        if (arma::any(arma::vectorise(X.slice(i) != X0)))
            return;  // found a slice that differs -- not degenerate
    stop("X contains %d observations that are ALL numerically identical to "
         "one another. Check for a data-loading error (e.g. the same observation "
         "repeated across all slices).", (int)X.n_slices);
}

/** @brief Stop with a clear message if x is NaN or +/-Inf. */
inline void check_finite(double x, const char* name) {
    if (!std::isfinite(x))
        stop("%s must be finite (got %f)", name, x);
}

/** @brief Stop if any element of M is NaN or +/-Inf. */
inline void check_finite_mat(const arma::mat& M, const char* name) {
    if (!M.is_finite())
        stop("%s contains non-finite (NaN/Inf) entries", name);
}

/** @brief Stop unless M is exactly p x p. */
inline void check_square_dims(const arma::mat& M, int p, const char* name) {
    if ((int)M.n_rows != p || (int)M.n_cols != p)
        stop("%s must be %d x %d, got %d x %d", name, p, p, (int)M.n_rows, (int)M.n_cols);
}

/**
 * @brief Stop unless p >= 2 and n >= 2.
 *
 * p = 1 (scalar "Wishart", i.e. a chi-squared model) and n = 1 (a single
 * observation) are outside the intended use of this model -- p = 1 makes
 * several multivariate-specific quantities degenerate, and n = 1 leaves
 * the improper prior's (alpha-bound)^-1 singularity with nothing to
 * overcome it (see the mode_alphaEM_improper boundary-convergence check).
 * Rather than let those cases surface later as a cryptic numerical
 * failure, they are rejected immediately with a clear message.
 */
inline void check_n_p(int n, int p) {
    if (p < 2) stop("p must be >= 2 (got %d); p = 1 is not a supported use case for this model", p);
    if (n < 2) stop("n must be >= 2 (got %d); need at least two observations", n);
}

/** @brief Stop unless nsamp >= 1. Checked before any sample container is allocated. */
inline void check_nsamp(int nsamp) {
    if (nsamp < 1) stop("nsamp must be >= 1 (got %d)", nsamp);
}

/**
 * @brief Validate a p x p matrix intended to be used as mu0 (or xbar):
 * finite, correctly sized, symmetric, and positive definite.
 */
inline void check_pd_matrix(const arma::mat& M, int p, const char* name) {
    check_finite_mat(M, name);
    check_square_dims(M, p, name);
    if (arma::norm(M - M.t(), "fro") > 1e-8 * std::max(1.0, arma::norm(M, "fro")))
        stop("%s must be symmetric", name);
    arma::mat L;
    if (!arma::chol(L, M))
        stop("%s must be positive definite", name);
}

/**
 * @brief Validate the proper-prior hyperparameters (beta, eta, kappa, mu0)
 * shared by mode_alphaEM(), rejection_sampler(), and wishart_inference().
 * Throws with a specific, actionable message on the first violated
 * condition. Does nothing when improper = true.
 */
inline void check_proper_prior_params(int p, double beta, double eta,
                                       double kappa, const arma::mat& mu0,
                                       bool improper) {
    if (improper) return;

    check_finite(beta,  "beta");
    check_finite(eta,   "eta");
    check_finite(kappa, "kappa");
    if (beta <= 1.0)  stop("beta must be > 1 for proper prior (got %f)", beta);
    if (eta  <= 0.0)  stop("eta must be > 0 for proper prior (got %f)", eta);
    if (kappa < 1.0)  stop("kappa must be >= 1 for proper prior (got %f)", kappa);

    check_pd_matrix(mu0, p, "mu0");

    double bound     = (p - 1.0) / 2.0;
    double prior_mode = (beta - 1.0) / (beta * eta);
    if (!(prior_mode > bound))
        stop("prior mode (beta-1)/(beta*eta) = %.4f must be > (p-1)/2 = %.4f. "
             "Decrease eta or increase beta.", prior_mode, bound);
}

// -------------------------------------------------------
// Forward declarations
// -------------------------------------------------------
double lfafun_improper(double a, int n, int p,
                       const arma::mat& xbar, double ldetxbarg);

double lfafun_proper(double a, int n, int p,
                     double ldet_muhat, double ldetxbarg,
                     double ldet_mu0, double beta,
                     double eta, double kappa);

double acpt_rate_unified(double ahat, int p, int n,
                         const arma::mat& xbar, double ldetxbarg,
                         const arma::mat& mu0, double mxlfa,
                         double beta, double eta, double kappa,
                         double lambda, double nu_star, bool improper);

Rcpp::NumericVector mode_alphaEM_improper(int n, int p,
                                          const arma::mat& xbar,
                                          double ldetxbarg,
                                          double tol, bool prnt,
                                          int max_em_iter, int max_nr_iter);

Rcpp::NumericVector mode_alphaEM_proper(int n, int p,
                                        const arma::mat& xbar,
                                        double ldetxbarg,
                                        const arma::mat& mu0,
                                        double beta, double eta, double kappa,
                                        double tol, bool prnt,
                                        int max_em_iter, int max_nr_iter);

Rcpp::List rejection_sample_unified(double ahat, int p, int n,
                                     const arma::mat& xbar, double ldetxbarg,
                                     const arma::mat& mu0, double mxlfa,
                                     double beta, double eta, double kappa,
                                     double lambda, double nu_star,
                                     bool improper, int nsamp);

// -------------------------------------------------------
// Test
// -------------------------------------------------------

/**
 * @brief Simple hello-world test to verify the library loaded correctly.
 * @return 0
 */
// [[Rcpp::export]]
int test() {
    Rcout << "Hello from C++" << std::endl;
    return 0;
}

// -------------------------------------------------------
// Wishart sampler
// -------------------------------------------------------

/**
 * @brief Generate n draws from a Wishart_p(nu, Sigma) distribution.
 *
 * Uses the Bartlett decomposition: if L is the lower Cholesky factor of Sigma,
 * then A = L * T satisfies W = A * A^T ~ Wishart_p(nu, Sigma), where T is
 * lower triangular with T(i,i) ~ sqrt(Chi^2(nu - i)) and T(i,j) ~ N(0,1)
 * for i > j.
 *
 * @param n     Number of draws
 * @param p     Dimension of the matrix
 * @param nu    Degrees of freedom, must be > p - 1
 * @param Sigma Scale matrix (p x p), must be positive definite
 * @param Wa    Output cube of size (p, p, n); filled in place
 * @return      true on success, false if Cholesky decomposition of Sigma fails
 */
bool rwishart(int n, int p, double nu, arma::mat Sigma, arma::cube& Wa) {

    if (nu <= p - 1.0) stop("nu must be larger than p-1");

    arma::mat L;
    bool ok = arma::chol(L, Sigma);
    if (!ok) return false;

    arma::mat rtSig = L.t();
    Wa.set_size(p, p, n);

    for (int k = 0; k < n; k++) {
        arma::mat T(p, p, arma::fill::zeros);
        for (int i = 0; i < p; i++) {
            T(i, i) = std::sqrt(R::rchisq(nu - i));
            for (int j = 0; j < i; j++)
                T(i, j) = R::rnorm(0.0, 1.0);
        }
        arma::mat A = rtSig * T;
        Wa.slice(k) = A * A.t();
    }
    return true;
}

/**
 * @brief R-exported version of the Wishart sampler.
 *
 * Identical to the internal 5-argument version but returns the cube
 * directly and throws on failure rather than returning false, making
 * it suitable for direct use from R.
 *
 * @param n     Number of draws
 * @param p     Dimension of the matrix
 * @param nu    Degrees of freedom, must be > p - 1
 * @param Sigma Scale matrix (p x p), must be positive definite
 * @return      Cube of size (p, p, n) containing the Wishart draws
 * @throws      Rcpp::exception if nu <= p-1 or Sigma is not positive definite
 */
// [[Rcpp::export]]
arma::cube rwishart(int n, int p, double nu, const arma::mat& Sigma) {

    check_n_p(n, p);
    check_finite(nu, "nu");
    check_finite_mat(Sigma, "Sigma");
    check_square_dims(Sigma, p, "Sigma");
    if (nu <= p - 1.0) stop("nu must be larger than p - 1");

    arma::mat L;
    bool ok = arma::chol(L, Sigma);
    if (!ok) stop("Sigma must be positive definite");

    arma::mat rtSig = L.t();
    arma::cube Wa(p, p, n);

    for (int k = 0; k < n; k++) {
        arma::mat T(p, p, arma::fill::zeros);
        for (int i = 0; i < p; i++) {
            T(i, i) = std::sqrt(R::rchisq(nu - i));
            for (int j = 0; j < i; j++)
                T(i, j) = R::rnorm(0.0, 1.0);
        }
        arma::mat A = rtSig * T;
        Wa.slice(k) = A * A.t();
    }
    return Wa;
}

// -------------------------------------------------------
// Basic helpers
// -------------------------------------------------------

/**
 * @brief Log of the multivariate Gamma function.
 *
 * Defined as:
 *   log Gamma_p(a) = p*(p-1)/4 * log(pi) + sum_{i=1}^{p} log Gamma(a - (i-1)/2)
 *
 * Required for the normalizing constant of the Wishart and inverse-Wishart
 * distributions. The domain requires a > (p-1)/2.
 *
 * @param a Argument, must be > (p-1)/2
 * @param p Dimension
 * @return  log Gamma_p(a)
 * @throws  Rcpp::exception if a <= (p-1)/2
 */
double lgammap(double a, int p) {
    if (a <= (p - 1.0) / 2.0) stop("a must be > (p-1)/2");
    double lgp = p * (p - 1.0) * 0.25 * log(M_PI);
    for (int i = 0; i < p; i++)
        lgp += R::lgammafn(a - i / 2.0);
    return lgp;
}

/**
 * @brief R-exported wrapper for lgammap.
 * @param a Argument, must be > (p-1)/2
 * @param p Dimension
 * @return  log Gamma_p(a)
 * @throws  Rcpp::exception if a <= (p-1)/2
 */
// [[Rcpp::export]]
double lgammap_export(double a, int p) { return lgammap(a, p); }

/**
 * @brief Multivariate digamma function (first derivative of log Gamma_p).
 *
 * Defined as:
 *   psi_p(a) = d/da log Gamma_p(a) = sum_{i=1}^{p} psi(a - (i-1)/2)
 *
 * where psi is the standard digamma function. Appears in the score
 * equation and EM surrogate Q'.
 *
 * @param a Argument, must be > (p-1)/2
 * @param p Dimension
 * @return  psi_p(a)
 * @throws  Rcpp::exception if a <= (p-1)/2
 */
double digammap(double a, int p) {
    if (a <= (p - 1.0) / 2.0) stop("a must be > (p-1)/2");
    double d = 0.0;
    for (int i = 0; i < p; i++)
        d += R::digamma(a - i / 2.0);
    return d;
}

/**
 * @brief Multivariate trigamma function (second derivative of log Gamma_p).
 *
 * Defined as:
 *   psi_p'(a) = d^2/da^2 log Gamma_p(a) = sum_{i=1}^{p} psi'(a - (i-1)/2)
 *
 * where psi' is the standard trigamma function. Appears in Q'' for the
 * Newton step in the EM mode finder.
 *
 * @param a Argument, must be > (p-1)/2
 * @param p Dimension
 * @return  psi_p'(a)
 * @throws  Rcpp::exception if a <= (p-1)/2
 */
double trigammap(double a, int p) {
    if (a <= (p - 1.0) / 2.0) stop("a must be > (p-1)/2");
    double d = 0.0;
    for (int i = 0; i < p; i++)
        d += R::trigamma(a - i / 2.0);
    return d;
}

/**
 * @brief Log determinant of a positive definite matrix.
 *
 * Uses Armadillo's log_det for numerical stability. Throws a runtime
 * error (rather than Rcpp::exception) so it can be caught internally
 * by wishart_inference's try/catch block and returned as an error list.
 *
 * @param X Square positive definite matrix
 * @return  log|X|
 * @throws  std::runtime_error if determinant is non-positive
 */
// [[Rcpp::export]]
double ldet(const arma::mat& X) {
    double val, sign;
    bool ok = arma::log_det(val, sign, X);
    if (!ok || sign <= 0.0)
        throw std::runtime_error("Determinant is non-positive");
    return val;
}

/**
 * @brief Compute sufficient statistics for Wishart observations.
 *
 * Returns the sample mean matrix xbar and the log geometric mean
 * of determinants ldetxbarg, the sufficient statistics for (alpha, Sigma)
 * under X_i ~ Wishart_p(2*alpha, Sigma):
 *
 *   xbar      = (1/n) * sum_{i=1}^n X_i
 *   ldetxbarg = (1/n) * sum_{i=1}^n log|X_i|
 *
 * Note: ldetxbarg is the log geometric mean of |X_i|, not log|xbar|.
 * By Jensen's inequality ldetxbarg <= log|xbar| always.
 *
 * @param X Cube of n Wishart draws, dimensions (p, p, n)
 * @return  List with:
 *            - xbar:      p x p sample mean matrix
 *            - ldetxbarg: scalar log geometric mean of determinants
 * @throws  std::runtime_error via ldet() if any slice X_i is not positive
 *          definite. More likely when nu is close to p-1.
 */
// [[Rcpp::export]]
Rcpp::List wishart_stats(const arma::cube& X) {
    check_n_p((int)X.n_slices, (int)X.n_rows);
    if (X.n_cols != X.n_rows)
        stop("X slices must be square, got %d x %d", (int)X.n_rows, (int)X.n_cols);
    if (!X.is_finite())
        stop("X contains non-finite (NaN/Inf) entries");
    check_not_fully_degenerate(X);

    arma::mat xbar    = arma::mean(X, 2);
    double ldetxbarg  = 0.0;
    for (unsigned int i = 0; i < X.n_slices; i++)
        ldetxbarg += ldet(arma::mat(X.slice(i)));
    ldetxbarg /= X.n_slices;
    return Rcpp::List::create(
        Rcpp::Named("xbar")      = xbar,
        Rcpp::Named("ldetxbarg") = ldetxbarg
    );
}

/**
 * @brief Compute f_p(a) = a * (p*log(a) - psi_p(a)).
 *
 * Appears in Q'(alpha|alpha^t) via:
 *   Q'(alpha|alpha^t) = n*f_p(alpha)/alpha + (beta-1)/alpha
 *                       + f_p(kappa*alpha)/alpha - c^t
 *
 * Bounds on f_p(a)/a are used to derive the EM bracketing interval:
 *   p/(2*a)  <  f_p(a)/a  <  p*(p+1)/2 / (a - (p-1)/2)
 *
 * @param a Argument, must be > (p-1)/2
 * @param p Dimension
 * @return  f_p(a) = a*(p*log(a) - psi_p(a))
 */
double fpfun(double a, int p) {
    return a * (p * std::log(a) - digammap(a, p));
}

// -------------------------------------------------------
// IMPROPER PRIOR
//
//   p(alpha, mu) propto (alpha - (p-1)/2)^{-1} * |mu|^{-(p+1)/2}
// -------------------------------------------------------

/**
 * @brief Log unnormalized marginal posterior of alpha under the improper prior.
 *
 * The improper prior p(alpha,mu) propto (alpha-(p-1)/2)^{-1} * |mu|^{-(p+1)/2}
 * after marginalizing over mu gives:
 *
 *   log f*(alpha) = log Gamma_p(n*alpha)
 *                  - n * log Gamma_p(alpha)
 *                  - n*p*alpha*log(n)
 *                  - log(alpha - (p-1)/2)
 *                  - lambda * alpha
 *
 * where lambda = n*(log|xbar| - ldetxbarg) >= 0 by Jensen's inequality,
 * clamped to 1e-10 for numerical stability.
 *
 * Returns R_NegInf when alpha <= (p-1)/2 (outside domain).
 *
 * @param a         Shape parameter alpha, must be > (p-1)/2
 * @param n         Number of Wishart observations
 * @param p         Dimension
 * @param xbar      Sample mean matrix (p x p)
 * @param ldetxbarg Log geometric mean of determinants (1/n)*sum log|X_i|
 * @return          log f*(alpha), or R_NegInf if alpha <= (p-1)/2
 */
// [[Rcpp::export]]
double lfafun_improper(double a, int n, int p,
                       const arma::mat& xbar, double ldetxbarg) {

    check_n_p(n, p);
    check_finite_mat(xbar, "xbar");
    check_square_dims(xbar, p, "xbar");
    check_finite(ldetxbarg, "ldetxbarg");

    double bound = (p - 1.0) / 2.0;
    if (!std::isfinite(a)) return R_NegInf;
    if (a <= bound) return R_NegInf;

    double lambda_raw = n * (ldet(xbar) - ldetxbarg);
    double lambda     = std::max(lambda_raw, 1e-10);

    return lgammap(n * a, p)
           - n * lgammap(a, p)
           - n * p * a * std::log((double)n)
           - std::log(a - bound)
           - lambda * a;
}

/**
 * @brief First derivative of EM surrogate Q(alpha|alpha^t) — improper prior.
 *
 * The surrogate Q is the expected complete-data log posterior given alpha^t.
 * Its derivative with respect to alpha is:
 *
 *   Q'(alpha|alpha^t) = n*f_p(alpha)/alpha - 1/(alpha-(p-1)/2) - c^t
 *
 * where c^t = n*(-p*log(2) + log|2*n*alpha^t*xbar| - psi_p(n*alpha^t) - ldetxbarg)
 * is constant in alpha. The root of Q' gives the EM update.
 *
 * @param a         Current alpha value
 * @param a0        Current EM iterate alpha^t
 * @param n         Number of Wishart observations
 * @param p         Dimension
 * @param xbar      Sample mean matrix (p x p)
 * @param ldetxbarg Log geometric mean of determinants
 * @return          Q'(alpha|alpha^t)
 */
double Q1fun_improper(double a, double a0, int n, int p,
                      const arma::mat& xbar, double ldetxbarg) {

    double c = n * (-p * std::log(2.0)
                    + ldet(2.0 * n * a0 * xbar)
                    - digammap(n * a0, p)
                    - ldetxbarg);

    return n * fpfun(a, p) / a
           - 1.0 / (a - (p - 1.0) / 2.0)
           - c;
}

/**
 * @brief Second derivative of EM surrogate Q(alpha|alpha^t) — improper prior.
 *
 * Obtained by differentiating Q'(alpha|alpha^t):
 *
 *   Q''(alpha|alpha^t) = n*p/alpha - n*psi_p'(alpha) + 1/(alpha-(p-1)/2)^2
 *
 * Note c^t drops out since it is constant in alpha. Used in the Newton step:
 *   alpha_new = alpha - Q'(alpha|alpha^t) / Q''(alpha|alpha^t)
 *
 * Does not depend on alpha^t. If Q'' >= 0 (locally convex surrogate),
 * Newton is redirected to the boundary of [alow, aup] that Q' points toward.
 *
 * @param a Current Newton point alpha
 * @param n Number of Wishart observations
 * @param p Dimension
 * @return  Q''(alpha|alpha^t)
 */
double Q2fun_improper(double a, int n, int p) {
    double denom = a - (p - 1.0) / 2.0;
    return n * p / a
           - n * trigammap(a, p)
           + 1.0 / (denom * denom);
}

/**
 * @brief EM algorithm to find the posterior mode of alpha — improper prior.
 *
 * At each iteration, maximizes the surrogate Q(alpha|alpha^t) via Newton's
 * method on Q' within the bracket [alow, aup] derived from bounds on f_p:
 *
 *   alow = (n-2) / (2*c^t)  +  (p-1)/2
 *   aup  = (n*p*(p+1) - 2) / (2*c^t)  +  (p-1)/2
 *
 * If the actual log posterior lfafun_improper does not increase, bisects
 * back toward the previous iterate to enforce EM monotonicity.
 *
 * @param n         Number of Wishart observations
 * @param p         Dimension
 * @param xbar      Sample mean matrix (p x p)
 * @param ldetxbarg Log geometric mean of determinants
 * @param tol       Convergence tolerance on max(|a - aold|, |lfa - lfold|)
 * @param prnt      If true, print iterate and bounds at each EM step
 * @return          NumericVector {ahat, log f*(ahat)}
 * @throws          Rcpp::exception if c^t <= 0
 */
Rcpp::NumericVector mode_alphaEM_improper(int n, int p,
                                          const arma::mat& xbar,
                                          double ldetxbarg,
                                          double tol        = 1e-6,
                                          bool   prnt       = false,
                                          int max_em_iter   = 1000,
                                          int max_nr_iter   = 100) {

    double lambda_raw = n * (ldet(xbar) - ldetxbarg);

    // By Jensen's inequality (concavity of log-det), lambda_raw >= 0 always,
    // with equality iff the observations X_i are exactly proportional to
    // one another (e.g. duplicated data). In that degenerate case, alpha's
    // upper tail is not identified by the data at all: instead of failing
    // loudly, EM can "converge" (successive iterates differ by less than
    // tol) at an astronomically large, numerically meaningless alpha,
    // because the log-posterior is essentially flat out to enormous
    // values. Catch the root cause directly rather than let that happen.
    if (lambda_raw <= 1e-8)
        stop("lambda = n*(log|xbar| - mean(log|X_i|)) is numerically zero "
             "or negative (%.3e). By Jensen's inequality this is always "
             ">= 0, with equality only when the observations in X are "
             "exactly identical to one another (merely proportional "
             "observations, e.g. X and 2*X, do NOT trigger this -- log-det "
             "is strictly concave, so any genuine difference between "
             "observations keeps this quantity strictly positive). The "
             "improper prior's posterior for alpha has no well-defined "
             "upper tail in this regime -- check X for duplicated or "
             "near-duplicated observations.", lambda_raw);

    double lambda     = std::max(lambda_raw, 1e-10);
    double bound      = (p - 1.0) / 2.0;

    // initial guess based on moment approximation. When lambda is tiny
    // (e.g. n small, or xbar nearly matching the geometric mean of
    // determinants so Jensen's gap collapses), this raw formula can send
    // the starting point wildly outside the valid domain (even negative),
    // which then poisons every downstream log-determinant / digamma call.
    // Clamp to a safe, finite, interior starting point instead.
    double a = bound + (((p + 1.0) * (p + 1.0) + 2.0) * n - 12.0) / (8.0 * lambda);
    if (!std::isfinite(a) || a <= bound || lambda <= 1e-8)
        a = bound + std::max(1.0, p / 2.0);

    double lfa = lfafun_improper(a, n, p, xbar, ldetxbarg);
    double dev = 999.0;

    int em_iter = 0;
    for (; em_iter < max_em_iter; em_iter++) {

        if (dev <= tol) break;

        double aold  = a;
        double lfold = lfa;

        double c = n * (-p * std::log(2.0)
                        + ldet(2.0 * n * aold * xbar)
                        - digammap(n * aold, p)
                        - ldetxbarg);

        if (c <= 0.0) stop("c^t is non-positive in improper mode finder");

        double alow = (n - 2.0) / (2.0 * c) + bound;
        double aup  = (n * p * (p + 1.0) - 2.0) / (2.0 * c) + bound;

        alow = std::max(alow, bound + BOUND_EPS);
        aup  = std::max(aup,  alow  + BOUND_EPS);

        if (prnt)
            Rcpp::Rcout << "em_iter=" << em_iter << "  a=" << aold
                        << "  bounds=[" << alow << ", " << aup << "]" << std::endl;

        // Newton on Q' starting from current iterate, capped at max_nr_iter
        // to guarantee termination even if Newton oscillates across a very
        // wide bracket without reaching the tolerance.
        double newa      = aold;
        double inner_dev = 999.0;

        for (int nr_iter = 0; nr_iter < max_nr_iter && inner_dev > tol; nr_iter++) {
            double q1   = Q1fun_improper(newa, aold, n, p, xbar, ldetxbarg);
            double q2   = Q2fun_improper(newa, n, p);
            if (q2 >= 0.0) { newa = (q1 > 0.0) ? aup : alow; break; }
            double next = newa - q1 / q2;
            next        = std::max(next, alow + BOUND_EPS);
            next        = std::min(next, aup);
            inner_dev   = std::abs(next - newa);
            newa        = next;
        }

        a   = newa;
        lfa = lfafun_improper(a, n, p, xbar, ldetxbarg);

        // EM safeguard: bisect if log posterior did not increase
        if (lfa < lfold) {
            double atry  = (aold + a) / 2.0;
            double lftry = lfafun_improper(atry, n, p, xbar, ldetxbarg);
            if (lftry > lfold) { a = atry; lfa = lftry; }
            else                { a = aold; lfa = lfold; }
        }

        dev = std::max(std::abs(a - aold), std::abs(lfa - lfold));
    }

    if (dev > tol && em_iter >= max_em_iter)
        stop("mode_alphaEM (improper prior) failed to converge within "
             "max_em_iter = %d iterations (last alpha = %.6f, "
             "|change| = %.3e > tol = %.3e). Try increasing max_em_iter, "
             "or check whether n and p are large enough to be informative.",
             max_em_iter, a, dev, tol);

    // The improper prior has a (alpha-bound)^-1 singularity at the domain
    // boundary, so log f*(alpha) -> +Inf as alpha -> bound+. If EM has
    // converged essentially onto the boundary rather than to a genuine
    // interior stationary point, the posterior does not have a well-defined
    // mode for this (n, p) combination -- this is a property of the model
    // and data, not a solvable numerical issue, so it is reported as such.
    if (a - bound < 1e-4)
        stop("Posterior mode (improper prior) converged to the domain "
             "boundary alpha = (p-1)/2 = %.4f (alpha - bound = %.3e) rather "
             "than an interior point. The improper prior is singular at "
             "this boundary, so no proper posterior mode exists here -- "
             "n = %d observations is likely too few relative to p = %d. "
             "Consider using more data or switching to the proper prior "
             "(supply beta, eta, kappa, mu0).", bound, a - bound, n, p);

    return Rcpp::NumericVector::create(a, lfa);
}

// -------------------------------------------------------
// PROPER PRIOR
//
//   alpha ~ Gamma(beta, beta*eta)  truncated at (p-1)/2
//   mu|alpha ~ inv-Wishart(2*kappa*alpha, 2*kappa*alpha*mu0)
// -------------------------------------------------------

/**
 * @brief Log unnormalized marginal posterior of alpha under the proper prior.
 *
 * The proper prior alpha ~ Gamma(beta, beta*eta) (truncated at (p-1)/2) and
 * mu|alpha ~ inv-Wishart(2*kappa*alpha, 2*kappa*alpha*mu0), after marginalizing
 * over mu, gives:
 *
 *   log f*(alpha) = g_p(alpha) + (beta-1)*log(alpha) - lambda*alpha
 *
 * where:
 *   g_p(alpha) = log Gamma_p((n+k)*alpha)
 *                - n*log Gamma_p(alpha)
 *                - log Gamma_p(k*alpha)
 *                + alpha*p*(k*log(k) - (n+k)*log(n+k))
 *
 *   lambda = beta*eta + n*(ldet_muhat - ldetxbarg) + kappa*(ldet_muhat - ldet_mu0)
 *   muhat  = (n*xbar + kappa*mu0) / (n+kappa)
 *
 * Returns R_NegInf outside the domain alpha > (p-1)/2.
 *
 * @param a          Shape parameter alpha, must be > (p-1)/2
 * @param n          Number of Wishart observations
 * @param p          Dimension
 * @param ldet_muhat log|muhat| where muhat = (n*xbar + kappa*mu0)/(n+kappa)
 * @param ldetxbarg  Log geometric mean of determinants (1/n)*sum log|X_i|
 * @param ldet_mu0   log|mu0|
 * @param beta       Gamma prior shape, must be > 1
 * @param eta        Gamma prior rate parameter
 * @param kappa      inv-Wishart prior strength, must be >= 1
 * @return           log f*(alpha), or R_NegInf if outside domain
 */
// [[Rcpp::export]]
double lfafun_proper(double a, int n, int p,
                     double ldet_muhat, double ldetxbarg,
                     double ldet_mu0, double beta,
                     double eta, double kappa) {

    check_n_p(n, p);
    check_finite(ldet_muhat, "ldet_muhat");
    check_finite(ldetxbarg,  "ldetxbarg");
    check_finite(ldet_mu0,   "ldet_mu0");
    check_finite(beta,  "beta");
    check_finite(eta,   "eta");
    check_finite(kappa, "kappa");

    double bound = (p - 1.0) / 2.0;
    double nk    = n + kappa;

    if (!std::isfinite(a)) return R_NegInf;
    if (a         <= bound) return R_NegInf;
    if (kappa * a <= bound) return R_NegInf;
    if (nk    * a <= bound) return R_NegInf;

    double gp = lgammap(nk * a,    p)
              - n * lgammap(a,     p)
              - lgammap(kappa * a, p)
              + a * p * (kappa * std::log(kappa) - nk * std::log(nk));

    double lambda = beta * eta
                  + n     * (ldet_muhat - ldetxbarg)
                  + kappa * (ldet_muhat - ldet_mu0);

    return gp + (beta - 1.0) * std::log(a) - a * lambda;
}

/**
 * @brief EM constant c^t for the proper prior surrogate Q'.
 *
 * Derived from E[log|mu| | alpha^t, x] using the inv-Wishart expectation:
 *
 *   c^t = (n+k)*(-psi_p((n+k)*alpha^t) + p*log((n+k)*alpha^t) + log|muhat|)
 *         + beta*eta - n*ldetxbarg - kappa*log|mu0|
 *
 * c^t is constant with respect to alpha (depends only on alpha^t and data),
 * and appears as the linear coefficient in Q'(alpha|alpha^t).
 *
 * @param at         Current EM iterate alpha^t
 * @param n          Number of Wishart observations
 * @param p          Dimension
 * @param ldet_muhat log|muhat|
 * @param ldetxbarg  Log geometric mean of determinants
 * @param ldet_mu0   log|mu0|
 * @param beta       Gamma prior shape
 * @param eta        Gamma prior rate
 * @param kappa      inv-Wishart prior strength
 * @return           c^t
 */
double compute_ct(double at, int n, int p,
                  double ldet_muhat, double ldetxbarg,
                  double ldet_mu0, double beta,
                  double eta, double kappa) {

    double nk = n + kappa;
    return nk * (-digammap(nk * at, p)
                  + p * std::log(nk * at)
                  + ldet_muhat)
           + beta * eta
           - n     * ldetxbarg
           - kappa * ldet_mu0;
}

/**
 * @brief First derivative of EM surrogate Q(alpha|alpha^t) — proper prior.
 *
 * Derived by differentiating Q(alpha|alpha^t) with respect to alpha:
 *
 *   Q'(alpha|alpha^t) = n*f_p(alpha)/alpha + (beta-1)/alpha
 *                       + f_p(kappa*alpha)/alpha - c^t
 *
 * Expanding:
 *   = n*p*log(alpha) - n*psi_p(alpha)
 *     + (beta-1)/alpha
 *     + kappa*p*log(kappa*alpha) - kappa*psi_p(kappa*alpha)
 *     - c^t
 *
 * @param a          Current alpha value
 * @param at         Current EM iterate alpha^t
 * @param n          Number of Wishart observations
 * @param p          Dimension
 * @param ldet_muhat log|muhat|
 * @param ldetxbarg  Log geometric mean of determinants
 * @param ldet_mu0   log|mu0|
 * @param beta       Gamma prior shape
 * @param eta        Gamma prior rate
 * @param kappa      inv-Wishart prior strength
 * @return           Q'(alpha|alpha^t)
 */
double Q1fun_proper(double a, double at, int n, int p,
                    double ldet_muhat, double ldetxbarg,
                    double ldet_mu0, double beta,
                    double eta, double kappa) {

    double ct = compute_ct(at, n, p, ldet_muhat, ldetxbarg,
                           ldet_mu0, beta, eta, kappa);

    return n * p * std::log(a) - n * digammap(a, p)
           + (beta - 1.0) / a
           + kappa * p * std::log(kappa * a) - kappa * digammap(kappa * a, p)
           - ct;
}

/**
 * @brief Second derivative of EM surrogate Q(alpha|alpha^t) — proper prior.
 *
 * Obtained by differentiating Q'(alpha|alpha^t):
 *
 *   Q''(alpha|alpha^t) = (n+kappa)*p/alpha
 *                        - (beta-1)/alpha^2
 *                        - n*psi_p'(alpha)
 *                        - kappa^2 * psi_p'(kappa*alpha)
 *
 * c^t drops out since it is constant in alpha. Used in the Newton step:
 *   alpha_new = alpha - Q'(alpha|alpha^t) / Q''(alpha|alpha^t)
 *
 * Does not depend on alpha^t. If Q'' >= 0 (locally convex surrogate),
 * Newton is redirected to the boundary of [alow, aup] that Q' points toward.
 *
 * @param a     Current Newton point alpha
 * @param n     Number of Wishart observations
 * @param p     Dimension
 * @param kappa inv-Wishart prior strength
 * @param beta  Gamma prior shape
 * @return      Q''(alpha|alpha^t)
 */
double Q2fun_proper(double a, int n, int p,
                    double kappa, double beta) {

    return (n + kappa) * p / a
           - (beta - 1.0) / (a * a)
           - n             * trigammap(a,         p)
           - kappa * kappa * trigammap(kappa * a, p);
}

/**
 * @brief EM algorithm to find the posterior mode of alpha — proper prior.
 *
 * At each iteration, maximizes the surrogate Q(alpha|alpha^t) via Newton's
 * method on Q' within the bracket [alow, aup] derived from bounds on f_p:
 *
 *   alow = ((n+1)*p/2 + (beta-1)) / c^t
 *   aup  = ((n+kappa)*p*(p+1)/2 + (beta-1)) / c^t  +  (p-1)/2
 *
 * alow is clamped to (p-1)/2 if it falls below the domain boundary.
 * If the actual log posterior lfafun_proper does not increase, bisects
 * back toward the previous iterate to enforce EM monotonicity.
 *
 * @param n     Number of Wishart observations
 * @param p     Dimension
 * @param xbar  Sample mean matrix (p x p)
 * @param ldetxbarg Log geometric mean of determinants
 * @param mu0   Prior center matrix (p x p)
 * @param beta  Gamma prior shape, must be > 1
 * @param eta   Gamma prior rate parameter
 * @param kappa inv-Wishart prior strength, must be >= 1
 * @param tol   Convergence tolerance on max(|a-aold|, |lfa-lfold|)
 * @param prnt  If true, print iterate and bounds at each EM step
 * @return      NumericVector {ahat, log f*(ahat)}
 * @throws      Rcpp::exception if c^t <= 0
 */
Rcpp::NumericVector mode_alphaEM_proper(int n, int p,
                                        const arma::mat& xbar,
                                        double ldetxbarg,
                                        const arma::mat& mu0,
                                        double beta, double eta, double kappa,
                                        double tol        = 1e-6,
                                        bool   prnt       = false,
                                        int max_em_iter   = 1000,
                                        int max_nr_iter   = 100) {

    double nk         = n + kappa;
    arma::mat muhat   = (n * xbar + kappa * mu0) / nk;
    double ldet_muhat = ldet(muhat);
    double ldet_mu0   = ldet(mu0);
    double bound      = (p - 1.0) / 2.0;

    // initial guess: midpoint between domain boundary and upper bound
    // derived from bounds on f_p in Q'(alpha|alpha^t) = 0
    double lambda_init = beta * eta
                       + n     * (ldet_muhat - ldetxbarg)
                       + kappa * (ldet_muhat - ldet_mu0);

    double a_upper = (beta - 1.0 + (n * p * p + n * p + p * p) / 2.0) / lambda_init
                     + bound;

    double a = (bound + a_upper) / 2.0;
    // Guard against lambda_init <= 0 (or near the numerical floor) or
    // otherwise non-finite a_upper, which can happen for adversarial/
    // degenerate data-prior combinations and would otherwise send the
    // starting point outside the valid domain, or absurdly far from it
    // (mirroring the same failure mode fixed in the improper-prior finder).
    if (!std::isfinite(a) || a <= bound || lambda_init <= 1e-8)
        a = bound + std::max(1.0, p / 2.0);

    double lfa = lfafun_proper(a, n, p, ldet_muhat, ldetxbarg,
                               ldet_mu0, beta, eta, kappa);
    double dev = 999.0;

    int em_iter = 0;
    for (; em_iter < max_em_iter; em_iter++) {

        if (dev <= tol) break;

        double aold  = a;
        double lfold = lfa;

        double ct = compute_ct(aold, n, p, ldet_muhat, ldetxbarg,
                               ldet_mu0, beta, eta, kappa);

        if (ct <= 0.0) stop("c^t is non-positive in proper mode finder");

        double alow = ((n + 1)     * p / 2.0             + (beta - 1.0)) / ct;
        double aup  = ((n + kappa) * p * (p + 1.0) / 2.0 + (beta - 1.0)) / ct + bound;

        alow = std::max(alow, bound + BOUND_EPS);
        aup  = std::max(aup,  alow  + BOUND_EPS);

        if (prnt)
            Rcpp::Rcout << "em_iter=" << em_iter << "  a=" << aold
                        << "  bounds=[" << alow << ", " << aup << "]" << std::endl;

        // Newton on Q' starting from current iterate, capped at max_nr_iter
        double newa      = aold;
        double inner_dev = 999.0;

        for (int nr_iter = 0; nr_iter < max_nr_iter && inner_dev > tol; nr_iter++) {
            double q1   = Q1fun_proper(newa, aold, n, p,
                                       ldet_muhat, ldetxbarg, ldet_mu0,
                                       beta, eta, kappa);
            double q2   = Q2fun_proper(newa, n, p, kappa, beta);
            if (q2 >= 0.0) { newa = (q1 > 0.0) ? aup : alow; break; }
            double next = newa - q1 / q2;
            next        = std::max(next, alow + BOUND_EPS);
            next        = std::min(next, aup);
            inner_dev   = std::abs(next - newa);
            newa        = next;
        }

        a   = newa;
        lfa = lfafun_proper(a, n, p, ldet_muhat, ldetxbarg,
                            ldet_mu0, beta, eta, kappa);

        // EM safeguard: bisect if log posterior did not increase
        if (lfa < lfold) {
            double atry  = (aold + a) / 2.0;
            double lftry = lfafun_proper(atry, n, p, ldet_muhat, ldetxbarg,
                                         ldet_mu0, beta, eta, kappa);
            if (lftry > lfold) { a = atry; lfa = lftry; }
            else                { a = aold; lfa = lfold; }
        }

        dev = std::max(std::abs(a - aold), std::abs(lfa - lfold));
    }

    if (dev > tol && em_iter >= max_em_iter)
        stop("mode_alphaEM (proper prior) failed to converge within "
             "max_em_iter = %d iterations (last alpha = %.6f, "
             "|change| = %.3e > tol = %.3e). Try increasing max_em_iter, "
             "or check whether the prior and data are in extreme conflict.",
             max_em_iter, a, dev, tol);

    // If EM converges essentially onto the truncation boundary rather than
    // an interior stationary point, the truncation at (p-1)/2 is binding:
    // the unconstrained posterior maximum lies at or below the boundary.
    // This is a legitimate but unusual configuration (typically a strong
    // prior-data conflict, or n too small relative to p) worth flagging
    // rather than silently returning a boundary value as if it were an
    // ordinary interior mode.
    if (a - bound < 1e-4)
        stop("Posterior mode (proper prior) converged to the truncation "
             "boundary alpha = (p-1)/2 = %.4f (alpha - bound = %.3e) rather "
             "than an interior point, indicating the truncation is binding. "
             "This usually means the prior and data are in strong conflict, "
             "or n = %d is too small relative to p = %d. Consider revisiting "
             "the prior (beta, eta) or gathering more data.", bound, a - bound, n, p);

    return Rcpp::NumericVector::create(a, lfa);
}

// -------------------------------------------------------
// Exported unified mode finder
// -------------------------------------------------------

/**
 * @brief Find the posterior mode of alpha (exported, unified interface).
 *
 * Dispatches to mode_alphaEM_improper or mode_alphaEM_proper depending
 * on the improper flag. This is the only exported mode-finding function;
 * the two internal implementations are not exported directly.
 *
 * @param n        Number of Wishart observations
 * @param p        Dimension
 * @param xbar     Sample mean matrix (p x p)
 * @param ldetxbarg Log geometric mean of determinants
 * @param mu0      Prior center matrix (p x p); ignored if improper = true
 * @param improper If true, use improper prior; if false, use proper prior
 * @param beta     Gamma prior shape, must be > 1 (ignored if improper)
 * @param eta      Gamma prior rate (ignored if improper)
 * @param kappa    inv-Wishart prior strength, must be >= 1 (ignored if improper)
 * @param tol      Convergence tolerance
 * @param prnt     If true, print EM iterates and bounds
 * @return         NumericVector {ahat, log f*(ahat)}
 */
// [[Rcpp::export]]
Rcpp::NumericVector mode_alphaEM(int n, int p,
                                  const arma::mat& xbar,
                                  double ldetxbarg,
                                  Rcpp::Nullable<arma::mat> mu0_ = R_NilValue,
                                  double beta        = 0.0,
                                  double eta         = 1.0,
                                  double kappa       = 0.0,
                                  double tol         = 1e-6,
                                  bool   prnt        = false,
                                  int    max_em_iter = 1000,
                                  int    max_nr_iter = 100) {

    check_n_p(n, p);
    check_finite_mat(xbar, "xbar");
    check_square_dims(xbar, p, "xbar");
    check_finite(ldetxbarg, "ldetxbarg");
    check_finite(tol, "tol");
    if (tol <= 0.0) stop("tol must be > 0 (got %f)", tol);
    if (max_em_iter < 1) stop("max_em_iter must be >= 1 (got %d)", max_em_iter);
    if (max_nr_iter < 1) stop("max_nr_iter must be >= 1 (got %d)", max_nr_iter);

    // auto-detect improper prior: beta=0, eta=1, kappa=0, mu0=NULL
    bool improper = (beta == 0.0 && eta == 1.0 && kappa == 0.0 && mu0_.isNull());

    // resolve mu0: default to I_p if not provided
    arma::mat mu0 = mu0_.isNull()
                    ? arma::eye(p, p)
                    : arma::mat(Rcpp::as<arma::mat>(mu0_));

    check_proper_prior_params(p, beta, eta, kappa, mu0, improper);

    if (improper)
        return mode_alphaEM_improper(n, p, xbar, ldetxbarg,
                                     tol, prnt, max_em_iter, max_nr_iter);
    else
        return mode_alphaEM_proper(n, p, xbar, ldetxbarg, mu0,
                                   beta, eta, kappa,
                                   tol, prnt, max_em_iter, max_nr_iter);
}

// -------------------------------------------------------
// Integration of f* (unified)
// -------------------------------------------------------

/**
 * @brief Numerically integrate the unnormalized posterior f*(alpha).
 *
 * Integrates f*(alpha) / f*(ahat) over (lower, infinity) using
 * Gauss-Kronrod quadrature (61-point rule). The integrand is
 * reparameterized as b = alpha - ahat to center around the mode,
 * improving numerical stability.
 *
 * Used to compute the theoretical acceptance rate of the rejection sampler.
 *
 * @param lower      Domain lower boundary (p-1)/2
 * @param n          Number of Wishart observations
 * @param p          Dimension
 * @param xbar       Sample mean matrix (p x p)
 * @param ldetxbarg  Log geometric mean of determinants
 * @param ldet_muhat log|muhat|
 * @param ldet_mu0   log|mu0|
 * @param mxlfa      log f*(ahat), used to center the integrand
 * @param ahat       Posterior mode (center of integration)
 * @param beta       Gamma prior shape (ignored if improper)
 * @param eta        Gamma prior rate (ignored if improper)
 * @param kappa      inv-Wishart prior strength (ignored if improper)
 * @param improper   If true use improper prior, else proper prior
 * @param tol        Integration tolerance
 * @return           Integral of f*(alpha)/f*(ahat) over (lower, infinity)
 */
double integrate_fstar_unified(double lower, int n, int p,
                                const arma::mat& xbar, double ldetxbarg,
                                double ldet_muhat, double ldet_mu0,
                                double mxlfa, double ahat,
                                double beta, double eta, double kappa,
                                bool improper, double tol = 1e-6) {

    using boost::math::quadrature::gauss_kronrod;

    auto integrand = [&](double b) -> double {
        double a = ahat + b;
        if (a <= lower) return 0.0;
        double logf;
        if (improper)
            logf = lfafun_improper(a, n, p, xbar, ldetxbarg) - mxlfa;
        else
            logf = lfafun_proper(a, n, p, ldet_muhat, ldetxbarg,
                                 ldet_mu0, beta, eta, kappa) - mxlfa;
        if (!std::isfinite(logf)) return 0.0;
        return std::exp(logf);
    };

    // integrate from max(lower-ahat, -500) to infinity
    const double LOG_TAIL_CUTOFF = -500.0;
    double b_lower = std::max(lower - ahat, LOG_TAIL_CUTOFF);
    double error;

    return gauss_kronrod<double, 61>::integrate(
        integrand, b_lower,
        std::numeric_limits<double>::infinity(),
        15, tol, &error
    );
}

// -------------------------------------------------------
// Theoretical acceptance rate (unified)
// -------------------------------------------------------

/**
 * @brief Compute the theoretical acceptance rate of the rejection sampler.
 *
 * The covering density is Gamma(nu_star, lambda) truncated to (lower, infinity).
 * The acceptance rate is:
 *
 *   acpt = integral(f*(alpha)) * g_trunc(ahat)
 *
 * where g_trunc(ahat) = g(ahat) / P(X > lower) is the truncated Gamma
 * density at the mode, and integral(f*(alpha)) is computed via
 * integrate_fstar_unified.
 *
 * @param ahat     Posterior mode
 * @param p        Dimension
 * @param n        Number of Wishart observations
 * @param xbar     Sample mean matrix (p x p)
 * @param ldetxbarg Log geometric mean of determinants
 * @param mu0      Prior center matrix (p x p)
 * @param mxlfa    log f*(ahat)
 * @param beta     Gamma prior shape (ignored if improper)
 * @param eta      Gamma prior rate (ignored if improper)
 * @param kappa    inv-Wishart prior strength (ignored if improper)
 * @param lambda   Covering Gamma rate parameter
 * @param nu_star  Covering Gamma shape parameter (= ahat*lambda + 1)
 * @param improper If true use improper prior, else proper prior
 * @return         Theoretical acceptance rate in (0, 1]
 */
double acpt_rate_unified(double ahat, int p, int n,
                          const arma::mat& xbar, double ldetxbarg,
                          const arma::mat& mu0, double mxlfa,
                          double beta, double eta, double kappa,
                          double lambda, double nu_star, bool improper) {

    double lower      = (p - 1.0) / 2.0;
    double scale      = 1.0 / lambda;

    arma::mat muhat   = improper ? xbar
                                 : arma::mat((n * xbar + kappa * mu0) / (n + kappa));
    double ldet_muhat = ldet(muhat);
    double ldet_mu0   = improper ? 0.0 : ldet(mu0);

    double integral = integrate_fstar_unified(lower, n, p, xbar, ldetxbarg,
                                               ldet_muhat, ldet_mu0,
                                               mxlfa, ahat,
                                               beta, eta, kappa, improper);

    // truncated Gamma density at mode: g(ahat) / P(X > lower)
    double log_p_tail = R::pgamma(lower, nu_star, scale, 0, 1);  // log P(X > lower)
    double log_g_mode = R::dgamma(ahat, nu_star, scale, 1) - log_p_tail;

    return std::exp(std::log(integral) + log_g_mode);
}

// -------------------------------------------------------
// Exported rejection sampler wrapper
// -------------------------------------------------------

/**
 * @brief R-exported rejection sampler for joint posterior of (alpha, mu).
 *
 * Requires the posterior mode ahat and log f*(ahat) = mxlfa from
 * mode_alphaEM(), plus the covering Gamma parameters lambda and nu_star.
 * Prior type is auto-detected from parameters:
 *   Improper: beta=0, eta=1, kappa=0, mu0=NULL  (defaults)
 *   Proper:   beta>1, kappa>=1, mu0 provided
 *
 * @param ahat     Posterior mode from mode_alphaEM()[1]
 * @param mxlfa    log f*(ahat) from mode_alphaEM()[2]
 * @param lambda   Covering Gamma rate (= beta*eta + n*log|muhat/xbar_g| + ...)
 * @param nu_star  Covering Gamma shape (= ahat*lambda + 1)
 * @param p        Dimension
 * @param n        Number of Wishart observations
 * @param xbar     Sample mean matrix (p x p)
 * @param ldetxbarg Log geometric mean of determinants
 * @param mu0      Prior center matrix; NULL for improper prior
 * @param beta     Gamma prior shape; 0.0 = improper, must be > 1 if proper
 * @param eta      Gamma prior rate; default 1.0
 * @param kappa    inv-Wishart prior strength; 0.0 = improper, must be >= 1 if proper
 * @param nsamp    Number of posterior samples (default 10000)
 * @return         List with alpha_sample, mu_sample, empirical_acpt_rate,
 *                 theoretical_acpt_rate
 */
// [[Rcpp::export]]
Rcpp::List rejection_sampler(double ahat, double mxlfa,
                              double lambda, double nu_star,
                              int p, int n,
                              const arma::mat& xbar, double ldetxbarg,
                              Rcpp::Nullable<arma::mat> mu0_ = R_NilValue,
                              double beta  = 0.0,
                              double eta   = 1.0,
                              double kappa = 0.0,
                              int    nsamp = 10000) {

    // Validate before any Rcpp container is allocated: some Rcpp allocation
    // failures (e.g. a negative-length NumericVector) raise an R-level
    // condition directly rather than a catchable C++ exception, which would
    // otherwise slip past a caller's try/catch (including wishart_inference's).
    check_n_p(n, p);
    check_nsamp(nsamp);
    check_finite(ahat, "ahat");
    check_finite(mxlfa, "mxlfa");
    check_finite(lambda, "lambda");
    check_finite(nu_star, "nu_star");
    if (lambda   <= 0.0) stop("lambda must be > 0 (got %f)", lambda);
    if (nu_star  <= 0.0) stop("nu_star must be > 0 (got %f)", nu_star);
    if (ahat <= (p - 1.0) / 2.0)
        stop("ahat must be > (p-1)/2 = %.4f (got %f)", (p - 1.0) / 2.0, ahat);
    check_finite_mat(xbar, "xbar");
    check_square_dims(xbar, p, "xbar");
    check_finite(ldetxbarg, "ldetxbarg");

    // auto-detect improper prior
    bool improper = (beta == 0.0 && eta == 1.0 && kappa == 0.0 && mu0_.isNull());

    arma::mat mu0 = mu0_.isNull()
                    ? arma::eye(p, p)
                    : arma::mat(Rcpp::as<arma::mat>(mu0_));

    check_proper_prior_params(p, beta, eta, kappa, mu0, improper);

    return rejection_sample_unified(ahat, p, n, xbar, ldetxbarg,
                                     mu0, mxlfa, beta, eta, kappa,
                                     lambda, nu_star, improper, nsamp);
}

/**
 * @brief Rejection sampler for joint posterior of (alpha, mu) — unified.
 *
 * Samples alpha from f*(alpha) using a truncated Gamma(nu_star, lambda)
 * covering density, then draws mu | alpha, x from its conditional
 * inv-Wishart posterior:
 *
 *   Improper: mu|alpha,x ~ inv-Wishart(2*n*alpha,     2*alpha*n*xbar)
 *   Proper:   mu|alpha,x ~ inv-Wishart(2*(n+k)*alpha, 2*alpha*(n*xbar+k*mu0))
 *
 * Alpha proposals are drawn via the inverse CDF method to guarantee
 * alpha > (p-1)/2 without wasting draws on boundary rejections.
 * The empirical acceptance rate counts only rejections from the
 * covering condition, not boundary truncation.
 *
 * @param ahat     Posterior mode of alpha
 * @param p        Dimension
 * @param n        Number of Wishart observations
 * @param xbar     Sample mean matrix (p x p)
 * @param ldetxbarg Log geometric mean of determinants
 * @param mu0      Prior center matrix (p x p)
 * @param mxlfa    log f*(ahat)
 * @param beta     Gamma prior shape (ignored if improper)
 * @param eta      Gamma prior rate (ignored if improper)
 * @param kappa    inv-Wishart prior strength (ignored if improper)
 * @param lambda   Covering Gamma rate (= beta*eta + n*log|muhat/xbar_g| + ...)
 * @param nu_star  Covering Gamma shape (= ahat*lambda + 1)
 * @param improper If true use improper prior, else proper prior
 * @param nsamp    Number of posterior samples to draw
 * @return         List with:
 *                   - alpha_sample:          vector of length nsamp
 *                   - mu_sample:             cube of size (p, p, nsamp)
 *                   - empirical_acpt_rate:   nsamp / total_draws
 *                   - theoretical_acpt_rate: from acpt_rate_unified
 */
Rcpp::List rejection_sample_unified(double ahat, int p, int n,
                                     const arma::mat& xbar, double ldetxbarg,
                                     const arma::mat& mu0, double mxlfa,
                                     double beta, double eta, double kappa,
                                     double lambda, double nu_star,
                                     bool improper, int nsamp = 10000) {

    // guard against invalid inputs that would cause infinite loop
    if (!std::isfinite(ahat))
        stop("rejection_sample_unified: ahat is not finite");
    if (!std::isfinite(mxlfa))
        stop("rejection_sample_unified: mxlfa is not finite — did you extract mode_alphaEM()[2] correctly?");
    if (!std::isfinite(lambda) || lambda <= 0.0)
        stop("rejection_sample_unified: lambda must be finite and positive");
    if (!std::isfinite(nu_star) || nu_star <= 0.0)
        stop("rejection_sample_unified: nu_star must be finite and positive");

    arma::mat muhat   = improper ? xbar
                                 : arma::mat((n * xbar + kappa * mu0) / (n + kappa));
    double ldet_muhat = ldet(muhat);
    double ldet_mu0   = improper ? 0.0 : ldet(mu0);

    double lower = (p - 1.0) / 2.0;
    double scale = 1.0 / lambda;

    // truncated Gamma proposal constants
    double p_low         = R::pgamma(lower, nu_star, scale, 1, 0);
    double log_g_at_mode = R::dgamma(ahat, nu_star, scale, 1) - std::log(1.0 - p_low);
    double log_M         = -log_g_at_mode;  // log of covering constant M = 1/g_trunc(ahat)

    long long accepted    = 0;
    long long total_draws = 0;
    const long long max_draws = static_cast<long long>(nsamp) * 100000LL;  // safety cap: 100000x nsamp

    Rcpp::NumericVector alpha_sample(nsamp);
    arma::cube          mu_sample(p, p, nsamp);

    while (accepted < nsamp) {
        total_draws++;

        if (total_draws > max_draws)
            stop("rejection_sample_unified: exceeded %lld draws — "
                 "check that the covering bound M is valid",
                 max_draws);

        // draw alpha from Gamma(nu_star, lambda) truncated to (lower, inf)
        double u      = R::unif_rand();
        double a_prop = R::qgamma(p_low + u * (1.0 - p_low), nu_star, scale, 1, 0);

        // evaluate log f*(a_prop) - log f*(ahat)
        double log_f;
        if (improper)
            log_f = lfafun_improper(a_prop, n, p, xbar, ldetxbarg) - mxlfa;
        else
            log_f = lfafun_proper(a_prop, n, p, ldet_muhat, ldetxbarg,
                                  ldet_mu0, beta, eta, kappa) - mxlfa;

        if (!std::isfinite(log_f)) continue;

        // acceptance step
        double log_g      = R::dgamma(a_prop, nu_star, scale, 1) - std::log(1.0 - p_low);
        double log_accept = log_f - log_M - log_g;

        if (std::log(R::runif(0.0, 1.0)) < log_accept) {

            // draw mu | alpha, x via inv-Wishart = inverse of Wishart draw
            double nk_eff = improper ? (double)n : (n + kappa);
            double nu_mu  = 2.0 * nk_eff * a_prop;
            arma::mat psi = improper
                            ? arma::mat(2.0 * a_prop * n * xbar)
                            : arma::mat(2.0 * a_prop * (n * xbar + kappa * mu0));
            arma::mat psi_inv = arma::inv_sympd(psi);

            arma::cube W;
            if (!rwishart(1, p, nu_mu, psi_inv, W)) continue;

            alpha_sample[accepted]    = a_prop;
            mu_sample.slice(accepted) = arma::inv_sympd(W.slice(0));
            accepted++;
        }
    }

    double accept_rate = static_cast<double>(nsamp) / total_draws;

    double theoretical_rate = acpt_rate_unified(ahat, p, n, xbar, ldetxbarg,
                                                  mu0, mxlfa, beta, eta, kappa,
                                                  lambda, nu_star, improper);

    return Rcpp::List::create(
        Rcpp::Named("alpha_sample")          = alpha_sample,
        Rcpp::Named("mu_sample")             = mu_sample,
        Rcpp::Named("empirical_acpt_rate")   = accept_rate,
        Rcpp::Named("theoretical_acpt_rate") = theoretical_rate
    );
}

// -------------------------------------------------------
// Main inference function (unified)
// -------------------------------------------------------

/**
 * @brief Bayesian inference for the Wishart shape parameter alpha.
 *
 * Given X_1,...,X_n ~ iid Wishart_p(2*alpha, Sigma), computes the
 * posterior of (alpha, mu) where mu = alpha*Sigma is the mean matrix,
 * under either an improper or proper prior:
 *
 *   Improper: p(alpha,mu) propto (alpha-(p-1)/2)^{-1} * |mu|^{-(p+1)/2}
 *   Proper:   alpha ~ Gamma(beta, beta*eta) truncated at (p-1)/2
 *             mu|alpha ~ inv-Wishart(2*kappa*alpha, 2*kappa*alpha*mu0)
 *
 * The algorithm:
 *   1. Compute sufficient statistics (xbar, ldetxbarg)
 *   2. Find posterior mode ahat via EM (Newton-within-EM)
 *   3. Construct covering Gamma(nu_star, lambda) with nu_star = ahat*lambda+1
 *   4. Draw (alpha, mu) jointly via rejection sampling
 *
 * @param X        Cube of n Wishart observations, dimensions (p, p, n)
 * @param improper If true, use improper prior; if false, use proper prior
 * @param mu0      Prior center matrix (p x p); ignored if improper = true
 * @param beta     Gamma prior shape, must be > 1 (ignored if improper)
 * @param eta      Gamma prior rate parameter (ignored if improper)
 * @param kappa    inv-Wishart prior strength, must be >= 1 (ignored if improper)
 * @param nsamp    Number of posterior samples
 * @return         List with two sublists:
 *                   results:
 *                     - alpha_samples:         vector of length nsamp
 *                     - mu_samples:            cube of size (p, p, nsamp)
 *                     - ahat:                  posterior mode of alpha
 *                     - theoretical_acpt_rate: theoretical acceptance rate
 *                     - empirical_acpt_rate:   empirical acceptance rate
 *                   statistics:
 *                     - xbar:                  p x p sample mean
 *                     - muhat:                 p x p posterior center of mu
 *                     - log_det_geometric_mean: ldetxbarg
 *                     - cover_shape:           nu_star
 *                     - cover_rate:            lambda
 * @throws         Rcpp::exception if beta <= 1 or kappa < 1 (proper prior)
 */
// [[Rcpp::export]]
Rcpp::List wishart_inference(arma::cube& X,
                              Rcpp::Nullable<arma::mat> mu0_ = R_NilValue,
                              double beta        = 0.0,
                              double eta         = 1.0,
                              double kappa       = 0.0,
                              int    nsamp       = 10000,
                              double tol         = 1e-6,
                              bool   prnt        = false,
                              int    max_em_iter = 1000,
                              int    max_nr_iter = 100) {

    int n = X.n_slices;
    int p = X.n_rows;

    check_n_p(n, p);
    if ((int)X.n_cols != p)
        stop("X slices must be square: got %d x %d", p, (int)X.n_cols);
    if (!X.is_finite())
        stop("X contains non-finite (NaN/Inf) entries");
    check_not_fully_degenerate(X);
    check_nsamp(nsamp);
    check_finite(tol, "tol");
    if (tol <= 0.0) stop("tol must be > 0 (got %f)", tol);
    if (max_em_iter < 1) stop("max_em_iter must be >= 1 (got %d)", max_em_iter);
    if (max_nr_iter < 1) stop("max_nr_iter must be >= 1 (got %d)", max_nr_iter);

    // auto-detect improper prior: beta=0, eta=1, kappa=0, mu0=NULL
    bool improper = (beta == 0.0 && eta == 1.0 && kappa == 0.0 && mu0_.isNull());

    // resolve mu0: default to I_p if not provided
    arma::mat mu0 = mu0_.isNull()
                    ? arma::eye(p, p)
                    : arma::mat(Rcpp::as<arma::mat>(mu0_));

    check_proper_prior_params(p, beta, eta, kappa, mu0, improper);

    try {
        auto start = std::chrono::high_resolution_clock::now();

        // sufficient statistics
        arma::mat xbar   = arma::mean(X, 2);
        double ldetxbarg = 0.0;
        for (int i = 0; i < n; i++)
            ldetxbarg += ldet(arma::mat(X.slice(i)));
        ldetxbarg /= n;

        // mode finding
        Rcpp::NumericVector a_star;
        double ldet_muhat, ldet_mu0_val;

        if (improper) {
            a_star       = mode_alphaEM_improper(n, p, xbar, ldetxbarg,
                                                  tol, prnt, max_em_iter, max_nr_iter);
            ldet_muhat   = ldet(xbar);
            ldet_mu0_val = 0.0;
        } else {
            a_star = mode_alphaEM_proper(n, p, xbar, ldetxbarg, mu0,
                                         beta, eta, kappa,
                                         tol, prnt, max_em_iter, max_nr_iter);
            arma::mat muhat = (n * xbar + kappa * mu0) / (n + kappa);
            ldet_muhat   = ldet(muhat);
            ldet_mu0_val = ldet(mu0);
        }

        double ahat  = a_star[0];
        double mxlfa = a_star[1];

        // covering Gamma parameters
        double lambda;
        if (improper) {
            double lambda_raw = n * (ldet(xbar) - ldetxbarg);
            lambda = std::max(lambda_raw, 1e-10);
        } else {
            lambda = beta * eta
                   + n     * (ldet_muhat - ldetxbarg)
                   + kappa * (ldet_muhat - ldet_mu0_val);
        }
        double nu_star = ahat * lambda + 1.0;

        // rejection sampling
        Rcpp::List sampling = rejection_sample_unified(
            ahat, p, n, xbar, ldetxbarg, mu0, mxlfa,
            beta, eta, kappa, lambda, nu_star, improper, nsamp
        );

        Rcpp::NumericVector alpha_sample = sampling["alpha_sample"];
        arma::cube          mu_sample    = sampling["mu_sample"];
        double empirical_acpt            = sampling["empirical_acpt_rate"];
        double theoretical_acpt          = sampling["theoretical_acpt_rate"];

        arma::mat muhat_out = improper
                              ? xbar
                              : arma::mat((n * xbar + kappa * mu0) / (n + kappa));

        auto end       = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        return Rcpp::List::create(
            Rcpp::Named("results") = Rcpp::List::create(
                Rcpp::Named("alpha_samples")         = alpha_sample,
                Rcpp::Named("mu_samples")            = mu_sample,
                Rcpp::Named("ahat")                  = ahat,
                Rcpp::Named("theoretical_acpt_rate") = theoretical_acpt,
                Rcpp::Named("empirical_acpt_rate")   = empirical_acpt
            ),
            Rcpp::Named("statistics") = Rcpp::List::create(
                Rcpp::Named("xbar")                   = xbar,
                Rcpp::Named("muhat")                  = muhat_out,
                Rcpp::Named("log_det_geometric_mean") = ldetxbarg,
                Rcpp::Named("cover_shape")            = nu_star,
                Rcpp::Named("cover_rate")             = lambda,
                Rcpp::Named("elapsed_seconds")        = elapsed
            )
        );

    } catch (const std::exception& e) {
        Rcpp::Rcout << e.what() << std::endl;
        return Rcpp::List::create(Rcpp::Named("error") = e.what());
    }
}