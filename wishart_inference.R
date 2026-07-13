# -------------------------------------------------------
# Bayesian Inference for the Wishart Distribution
#
# This script demonstrates posterior inference for alpha
# and mu in the model X_i ~ Wishart_p(2*alpha, Sigma), 
# i=1,...,n, using rejection sampling under two prior 
# specifications:
#
#   Improper: p(alpha, mu) propto (alpha-(p-1)/2)^{-1} 
#             * |mu|^{-(p+1)/2}
#
#   Proper:   alpha ~ Gamma(beta, beta*eta) truncated 
#             at (p-1)/2
#             mu|alpha ~ inv-Wishart(2*kappa*alpha, 
#             2*kappa*alpha*mu0)
#
# The posterior mode is found via a Newton-within-EM 
# algorithm, and joint samples of (alpha, mu) are drawn 
# via rejection sampling with a truncated Gamma covering 
# density.
#
# REQUIREMENT: wishart_final.cpp must be in the same  
# directory where this script is running. The C++ file  
# is compiled once via sourceCpp() and does not need to  
# be recompiled unless modified.
# -------------------------------------------------------

library(Rcpp)
library(RcppArmadillo)
sourceCpp("wishart_inference.cpp")

# Verify the library loaded correctly。
# Should print "Hello from C++"
test()

# =======================================================
# SECTION 1: Setup
# =======================================================
# Set the true parameters for data generation.
# In a real application, only the data X is available.

set.seed(6)
p     <- 3       # dimension of each matrix observation
nu    <- 10       # true degrees of freedom (= 2*alpha)
a     <- nu / 2  # true alpha
Sigma <- diag(p) # true scale matrix
n     <- 10      # number of observed matrices

# =======================================================
# SECTION 2: Generate Data
# =======================================================
# Generate n draws from Wishart_p(2*alpha, Sigma).
#
# For high dimensional matrices, the sampler could produce 
# non-positive-definite matrices. We loop until we obtain  
# a valid sample where all matrices have positive definite 
# determinant.

success <- FALSE
while (!success) {
  X    <- rwishart(n, p, nu, Sigma)
  stat <- tryCatch(wishart_stats(X), error = function(e) NULL)
  if (!is.null(stat)) success <- TRUE
}

# Extract sufficient statistics:
#   xbar      = (1/n) * sum X_i         (sample mean matrix)
#   ldetxbarg = (1/n) * sum log|X_i|    (log geometric mean of determinants)
stat      <- wishart_stats(X)
xbar      <- stat$xbar
cat("Sample mean matrix:\n")
print(xbar)
ldetxbarg <- stat$ldetxbarg
cat("Sample log geometric mean of determinants: ", ldetxbarg)

# =======================================================
# SECTION 3: Posterior Inference
# =======================================================
# Two approaches are available:
#
#   (A) PACKAGED  — wishart_inference() does everything  
#                   in one call: sufficient statistics,   
#                   mode finding, and sampling.
#
#   (B) MANUAL    — call each step separately for full   
#                   control: wishart_stats(), then        
#                   mode_alphaEM(), then rejection_sampler().
#
# Both approaches support improper and proper priors.
# The prior type is auto-detected from the parameters:
#   - Improper: omit mu0, beta, eta, kappa (use defaults)
#   - Proper:   provide mu0, beta > 1, kappa >= 1
# =======================================================

# -------------------------------------------------------
# (A) PACKAGED: wishart_inference()
# -------------------------------------------------------
#
# INPUT:
#   X         — p x p x n array of Wishart observations (required)
#   mu0       — p x p prior center matrix (omit for improper prior)
#   beta      — Gamma prior shape, must be > 1 (proper prior only)
#   eta       — Gamma prior rate (proper prior only)
#   kappa     — inv-Wishart prior strength >= 1 (proper prior only)
#   nsamp     — number of posterior samples (default 10000)
#   prnt      — print EM iterations if TRUE (default FALSE)
#
# OUTPUT: a list with two sublists:
#   $results
#     $alpha_samples         — vector of nsamp posterior draws of alpha
#     $mu_samples            — p x p x nsamp array of posterior draws of mu
#     $ahat                  — posterior mode of alpha
#     $theoretical_acpt_rate — theoretical acceptance rate of sampler
#     $empirical_acpt_rate   — empirical acceptance rate of sampler
#   $statistics
#     $xbar                  — p x p sample mean matrix
#     $muhat                 — p x p posterior center of mu
#     $log_det_geometric_mean— ldetxbarg
#     $cover_shape           — nu_star (covering Gamma shape)
#     $cover_rate            — lambda  (covering Gamma rate)
#     $elapsed_seconds       — wall-clock time for full inference
#
# -------------------------------------------------------
# IMPROPER PRIOR (default)
#
# The improper prior p(alpha,mu) propto 
# (alpha-(p-1)/2)^{-1} * |mu|^{-(p+1)/2} is used when 
# no prior parameters are provided. This is NOT the same 
# as the proper prior with beta=0, eta=1, kappa=0 — those 
# are simply the function defaults that trigger the improper 
# path internally.
# -------------------------------------------------------
res_imp <- wishart_inference(X, nsamp=50000)

# optional: verbose — print EM iterations and convergence
# res_imp <- wishart_inference(X, prnt = TRUE)

cat("--- Improper Prior ---\n")
cat("true a:               ", a, "\n")
cat("posterior mode:       ", res_imp$results$ahat, "\n")
cat("95% Interval:               ", quantile(res_imp$results$alpha_samples, c(0.025, 0.975)), "\n")
cat("theoretical acpt rate:", res_imp$results$theoretical_acpt_rate, "\n")
cat("empirical acpt rate:  ", res_imp$results$empirical_acpt_rate, "\n")
cat("elapsed seconds:      ", res_imp$statistics$elapsed_seconds, "\n")

# Grid search: independent check of the EM mode.
# The grid must start above (p-1)/2 = 0.5.
# Shrink the upper bound to the region of interest for
# finer resolution with the same number of points.
alpha_grid <- seq(1.05, 10, by = 0.001)
post_imp   <- sapply(alpha_grid, function(av) {
  tryCatch(exp(lfafun_improper(av, n, p, xbar, ldetxbarg)),
           error = function(e) NA)
})
a_grid_imp <- alpha_grid[which.max(post_imp)]
cat("grid search mode:     ", a_grid_imp, "\n")
cat("difference:           ", abs(res_imp$results$ahat - a_grid_imp), "\n")

# Plot: posterior samples, density and covering Gamma
lambda_imp        <- res_imp$statistics$cover_rate
nu_star_imp       <- res_imp$statistics$cover_shape
alpha_samples_imp <- res_imp$results$alpha_samples

h         <- hist(alpha_samples_imp, breaks = 50, plot = FALSE)
n_samp    <- length(alpha_samples_imp)
bin_width <- diff(h$breaks)[1]
h_prop    <- h$counts / n_samp

norm_const    <- sum(diff(alpha_grid) *
                       (post_imp[-1] + post_imp[-length(post_imp)]) / 2,
                     na.rm = TRUE)
post_imp_prop <- (post_imp / norm_const) * bin_width

cover_imp <- dgamma(alpha_grid, nu_star_imp, rate = lambda_imp) /
  dgamma(res_imp$results$ahat, nu_star_imp, rate = lambda_imp)
cover_imp_prop <- cover_imp * max(post_imp_prop, na.rm = TRUE)

plot(h$mids, h_prop, type = "n",
     main = sprintf("Improper Prior: Posterior samples of a, density and cover\nn=%d, p=%d, true a=%.3f",
                    n, p, a),
     xlab = expression(alpha), ylab = "proportion",
     xlim = c(min(alpha_samples_imp) - 0.1, max(alpha_samples_imp) + 0.1),
     ylim = c(0, max(h_prop) * 1.1))
rect(h$breaks[-length(h$breaks)], 0,
     h$breaks[-1], h_prop, col = "grey90", border = "grey70")
lines(alpha_grid, post_imp_prop,  col = "steelblue", lwd = 2)
lines(alpha_grid, cover_imp_prop, col = "purple",    lwd = 2, lty = 3)
abline(v = res_imp$results$ahat,  col = "darkgreen", lwd = 2, lty = 2)
legend("topright",
       legend = c("posterior samples", "posterior density",
                  "covering Gamma",
                  sprintf("EM mode = %.3f", res_imp$results$ahat)),
       col = c("grey70", "steelblue", "purple", "darkgreen"),
       lty = c(1, 1, 3, 2), lwd = c(8, 2, 2, 2))

# -------------------------------------------------------
# PROPER PRIOR
#
# Specify beta > 1, kappa >= 1, and mu0.
# The prior mode of alpha is (beta-1)/(beta*eta), which  
# must be > (p-1)/2 for an interior mode. Here we place  
# the prior mode at 3, while the true alpha is 2, to 
# demonstrate prior-likelihood tension.
#
# mu0 is the prior center for mu = 2*alpha*Sigma.
# Here we set mu0 = 4*I since true mu = 2*2*I = 4*I.
# -------------------------------------------------------
beta  <- 50                # large beta = concentrated prior
eta   <- (beta-1)/(6*beta) # prior mode = (beta-1)/(beta*eta) = 3
kappa <- 1                 # prior strength on mu
mu0   <- diag(10, p)        # prior center for mu (= true mu)

res_pro <- wishart_inference(X, mu0 = mu0, beta = beta,
                             eta = eta, kappa = kappa, nsamp = 5000)

cat("--- Proper Prior ---\n")
cat("prior mode:           ", (beta - 1) / (beta * eta), "\n")
cat("true a:               ", a, "\n")
cat("posterior mode:       ", res_pro$results$ahat, "\n")
cat("posterior mean:       ", mean(res_pro$results$alpha_samples), "\n")
cat("95% CI:               ", quantile(res_pro$results$alpha_samples, c(0.025, 0.975)), "\n")
cat("theoretical acpt rate:", res_pro$results$theoretical_acpt_rate, "\n")
cat("empirical acpt rate:  ", res_pro$results$empirical_acpt_rate, "\n")
cat("elapsed seconds:      ", res_pro$statistics$elapsed_seconds, "\n")

ldet_muhat     <- ldet((n * xbar + kappa * mu0) / (n + kappa))
ldet_mu0       <- log(det(mu0))
# the grid needs to cover the entire range of samples between
min(res_pro$results$alpha_samples)
max(res_pro$results$alpha_samples)

alpha_grid_pro <- seq(1.05, 10, by = 0.001)

post_pro <- sapply(alpha_grid_pro, function(av) {
  tryCatch(exp(lfafun_proper(av, n, p, ldet_muhat, ldetxbarg,
                             ldet_mu0, beta, eta, kappa)),
           error = function(e) NA)
})
a_grid_pro <- alpha_grid_pro[which.max(post_pro)]
cat("grid search mode:     ", a_grid_pro, "\n")
cat("difference:           ", abs(res_pro$results$ahat - a_grid_pro), "\n")

# Plot 1: likelihood, prior and posterior normalized at  
# their own modes to visualize prior-likelihood tension.
prior_pro <- sapply(alpha_grid_pro, function(av) {
  if (av <= (p - 1) / 2) return(NA)
  dgamma(av, beta, rate = beta * eta) /
    pgamma((p - 1) / 2, beta, rate = beta * eta, lower = FALSE)
})
lik_pro <- sapply(alpha_grid_pro, function(av) {
  tryCatch(exp(n * p * av * log(av) -
                 n * lgammap_export(av, p) +
                 n * av * ldetxbarg -
                 n * av * ldet(xbar) -
                 n * av * p),
           error = function(e) NA)
})
post_pro_norm  <- post_pro  / max(post_pro,  na.rm = TRUE)
prior_pro_norm <- prior_pro / max(prior_pro, na.rm = TRUE)
lik_pro_norm   <- lik_pro   / max(lik_pro,   na.rm = TRUE)

plot(alpha_grid_pro, post_pro_norm,
     type = "l", col = "steelblue", lwd = 2,
     main = sprintf("Proper Prior: Likelihood, prior and posterior\nn=%d, p=%d, true a=%.3f, beta=%.3f, eta=%.3f, kappa=%.3f",
                    n, p, a, beta, eta, kappa),
     xlab = expression(alpha),
     ylab = "density (normalized at own mode)",
     ylim = c(0, 1.1))
lines(alpha_grid_pro, lik_pro_norm,   col = "orange", lwd = 2)
lines(alpha_grid_pro, prior_pro_norm, col = "purple", lwd = 2, lty = 2)
abline(v = res_pro$results$ahat, col = "darkgreen", lwd = 2, lty = 2)
legend("topright",
       legend = c("posterior", "likelihood", "prior",
                  sprintf("Em mode = %.3f", res_pro$results$ahat)),
       col = c("steelblue", "orange", "purple", "darkgreen"),
       lty = c(1, 1, 2, 2), lwd = 2)

# Plot 2: posterior samples, density and covering Gamma
lambda_pro        <- res_pro$statistics$cover_rate
nu_star_pro       <- res_pro$statistics$cover_shape
alpha_samples_pro <- res_pro$results$alpha_samples

h_pro     <- hist(alpha_samples_pro, breaks = 50, plot = FALSE)
n_samp    <- length(alpha_samples_pro)
bin_width <- diff(h_pro$breaks)[1]
h_prop    <- h_pro$counts / n_samp

norm_const    <- sum(diff(alpha_grid_pro) *
                       (post_pro[-1] + post_pro[-length(post_pro)]) / 2,
                     na.rm = TRUE)
post_pro_prop <- (post_pro / norm_const) * bin_width

cover_pro <- dgamma(alpha_grid_pro, nu_star_pro, rate = lambda_pro) /
  dgamma(res_pro$results$ahat, nu_star_pro, rate = lambda_pro)
cover_pro_prop <- cover_pro * max(post_pro_prop, na.rm = TRUE)

plot(h_pro$mids, h_prop, type = "n",
     main = sprintf("Proper Prior: Posterior samples of a, density and cover\nn=%d, p=%d, true a=%.3f, beta=%.3f, eta=%.3f, kappa=%.3f",
                    n, p, a, beta, eta, kappa),
     xlab = expression(alpha), ylab = "proportion",
     xlim = c(min(alpha_samples_pro) - 0.1, max(alpha_samples_pro) + 0.1),
     ylim = c(0, max(h_prop) * 1.1))
rect(h_pro$breaks[-length(h_pro$breaks)], 0,
     h_pro$breaks[-1], h_prop, col = "grey90", border = "grey70")
lines(alpha_grid_pro, post_pro_prop,  col = "steelblue", lwd = 2)
lines(alpha_grid_pro, cover_pro_prop, col = "purple",    lwd = 2, lty = 3)
abline(v = res_pro$results$ahat,      col = "darkgreen", lwd = 2, lty = 2)
legend("topright",
       legend = c("posterior samples", "posterior density",
                  "covering Gamma",
                  sprintf("EM mode = %.3f", res_pro$results$ahat)),
       col = c("grey70", "steelblue", "purple", "darkgreen"),
       lty = c(1, 1, 3, 2), lwd = c(8, 2, 2, 2))

# -------------------------------------------------------
# (B) MANUAL: step-by-step control
# -------------------------------------------------------
# Use this approach when you want full control over each
# step, or when you want to call the mode finder and 
# sampler independently without rerunning the whole pipeline.
#
# The three steps are:
#   1. wishart_stats()     — compute sufficient statistics
#   2. mode_alphaEM()      — find the posterior mode
#   3. rejection_sampler() — draw posterior samples
# -------------------------------------------------------

# Step 1: sufficient statistics (already computed above)
# xbar, ldetxbarg

# -------------------------------------------------------
# Step 2: mode_alphaEM()
#
# INPUT:
#   n, p        — sample size and dimension
#   xbar        — sample mean matrix from wishart_stats()
#   ldetxbarg   — log geometric mean of determinants
#   mu0         — prior center matrix (omit for improper)
#   beta, eta, kappa — prior parameters (omit for improper)
#   prnt        — print EM iterations if TRUE
#
# OUTPUT: NumericVector {ahat, log f*(ahat)}
# -------------------------------------------------------

# improper (omit mu0 — uses defaults)
ahat_imp <- mode_alphaEM(n, p, xbar, ldetxbarg)
cat("improper mode:", ahat_imp[1], "  log f*(ahat):", ahat_imp[2], "\n")

# proper (provide mu0 and prior parameters)
ahat_pro <- mode_alphaEM(n, p, xbar, ldetxbarg,
                         mu0 = mu0, beta = beta, eta = eta, kappa = kappa)
cat("proper mode:  ", ahat_pro[1], "  log f*(ahat):", ahat_pro[2], "\n")

# optional: verbose — print EM iterates
cat("\n--- EM verbose ---\n")
ahat_verb <- mode_alphaEM(n, p, xbar, ldetxbarg,
                          mu0 = mu0, beta = beta, eta = eta, kappa = kappa,
                          prnt = TRUE)

# -------------------------------------------------------
# Step 3: rejection_sampler()
#
# Requires ahat and mxlfa from mode_alphaEM(), plus the
# covering Gamma parameters lambda and nu_star.
#
# INPUT:
#   ahat, mxlfa  — mode and log f*(mode) from mode_alphaEM()
#   lambda       — covering Gamma rate
#   nu_star      — covering Gamma shape (= ahat*lambda + 1)
#   p, n         — dimension and sample size
#   xbar         — sample mean matrix
#   ldetxbarg    — log geometric mean of determinants
#   mu0          — prior center matrix (omit for improper)
#   beta, eta, kappa — prior parameters (omit for improper)
#   nsamp        — number of posterior samples (default 10000)
#
# OUTPUT: list with alpha_sample, mu_sample,
#         empirical_acpt_rate, theoretical_acpt_rate
# -------------------------------------------------------

# improper: compute lambda and nu_star manually
ahat_imp_val  <- ahat_imp[1]   # posterior mode
mxlfa_imp_val <- ahat_imp[2]   # log f*(ahat)

lambda_raw  <- n * (ldet(xbar) - ldetxbarg)
lambda_imp  <- max(lambda_raw, 1e-10)
nu_star_imp <- ahat_imp_val * lambda_imp + 1

samp_imp <- rejection_sampler(ahat_imp_val, mxlfa_imp_val,
                              lambda_imp, nu_star_imp,
                              p, n, xbar, ldetxbarg)

cat("--- Manual Improper Sampler ---\n")
cat("empirical acpt rate:  ", samp_imp$empirical_acpt_rate, "\n")
cat("theoretical acpt rate:", samp_imp$theoretical_acpt_rate, "\n")

# proper: compute lambda and nu_star manually
ahat_pro_val  <- ahat_pro[1]
mxlfa_pro_val     <- ahat_pro[2]
muhat         <- (n * xbar + kappa * mu0) / (n + kappa)
lambda_pro_m  <- beta * eta +
  n     * (ldet(muhat) - ldetxbarg) +
  kappa * (ldet(muhat) - log(det(mu0)))
nu_star_pro_m <- ahat_pro_val * lambda_pro_m + 1

samp_pro <- rejection_sampler(ahat_pro_val, mxlfa_pro_val,
                              lambda_pro_m, nu_star_pro_m,
                              p, n, xbar, ldetxbarg,
                              mu0 = mu0, beta = beta,
                              eta = eta, kappa = kappa,
                              nsamp = 10000)

cat("--- Manual Proper Sampler ---\n")
cat("empirical acpt rate:  ", samp_pro$empirical_acpt_rate, "\n")
cat("theoretical acpt rate:", samp_pro$theoretical_acpt_rate, "\n")


# =======================================================
# SECTION 4: Error Handling Demonstrations
# =======================================================
# The C++ backend validates its inputs and its own internal
# convergence at every layer. This section deliberately
# triggers each category of error so you can see the
# messages you'll get back, and confirm error handling
# behaves consistently before you rely on it. Every call
# below is wrapped in tryCatch() so this script keeps
# running; in normal use you'd just let the error propagate
# (or check for wishart_inference()'s $error field -- see
# 4H below).
#
# demo_error() is a small helper that runs an expression,
# prints whichever error it raises, and returns invisibly.
# =======================================================

demo_error <- function(label, expr) {
  cat("\n---", label, "---\n")
  result <- tryCatch(expr, error = function(e) {
    cat("Caught error:\n ", conditionMessage(e), "\n")
    invisible(NULL)
  })
  invisible(result)
}

## ---- 4A. Structural limits: p and n ----
# This model requires p >= 2 (a scalar "Wishart" is just a
# chi-squared and outside its scope) and n >= 2 (a single
# observation can't identify alpha under the improper prior).
demo_error("p = 1 rejected", {
  rwishart(10, 1, 6, matrix(2, 1, 1))
})
demo_error("n = 1 rejected", {
  X_bad <- array(diag(3), dim = c(3, 3, 1))
  wishart_inference(X_bad, nsamp = 100)
})

## ---- 4B. Sample-count validity: nsamp ----
demo_error("nsamp = 0 rejected", {
  wishart_inference(X, nsamp = 0)
})
demo_error("nsamp negative rejected", {
  wishart_inference(X, nsamp = -100)
})

## ---- 4C. Proper-prior hyperparameter validity ----
# beta must be > 1, eta must be > 0, kappa must be >= 1,
# and all three must be finite.
demo_error("beta <= 1 rejected", {
  wishart_inference(X, mu0 = diag(p), beta = 1, eta = 1, kappa = 1, nsamp = 100)
})
demo_error("eta <= 0 rejected (previously slipped through silently!)", {
  wishart_inference(X, mu0 = diag(p), beta = 5, eta = 0, kappa = 1, nsamp = 100)
})
demo_error("kappa < 1 rejected", {
  wishart_inference(X, mu0 = diag(p), beta = 5, eta = 1, kappa = 0.5, nsamp = 100)
})
demo_error("non-finite beta rejected", {
  wishart_inference(X, mu0 = diag(p), beta = Inf, eta = 1, kappa = 1, nsamp = 100)
})

## ---- 4D. Prior-data compatibility: prior mode must be interior ----
# The prior mode (beta-1)/(beta*eta) must exceed (p-1)/2, or
# there is no valid interior starting point for this prior.
demo_error("prior mode <= (p-1)/2 rejected", {
  wishart_inference(X, mu0 = diag(p), beta = 5, eta = 10, kappa = 1, nsamp = 100)
})

## ---- 4E. mu0 validity: dimensions and positive-definiteness ----
demo_error("mu0 with wrong dimensions rejected", {
  wishart_inference(X, mu0 = diag(2), beta = 50, eta = (50 - 1) / (6 * 50), kappa = 1, nsamp = 100)
})
demo_error("mu0 not positive definite rejected", {
  bad_mu0 <- matrix(c(1, 2, 0, 2, 1, 0, 0, 0, 1), 3, 3)
  wishart_inference(X, mu0 = bad_mu0, beta = 50, eta = (50 - 1) / (6 * 50), kappa = 1, nsamp = 100)
})

## ---- 4F. Non-finite data ----
demo_error("NaN in xbar rejected", {
  bad_xbar <- xbar
  bad_xbar[1, 1] <- NaN
  mode_alphaEM(n, p, bad_xbar, ldetxbarg)
})

## ---- 4G. Exact data degeneracy ----
# log-det is strictly concave, so lambda = n*(log|xbar| -
# mean(log|X_i|)) is exactly zero if and only if ALL n
# observations are numerically identical to one another.
demo_error("ALL n observations identical is a degenerate case -- rejected", {
  X_all_same <- array(X[, , 1], dim = c(p, p, n))
  wishart_inference(X_all_same, nsamp = 100)
})

## ---- 4H. rwishart argument validity ----
demo_error("rwishart: nu <= p-1 rejected", {
  rwishart(5, 3, 2, diag(3))
})
demo_error("rwishart: non-positive-definite Sigma rejected", {
  bad_Sigma <- matrix(c(1, 2, 2, 1), 2, 2)  # not PD
  rwishart(5, 2, 5, bad_Sigma)
})

## ---- 4I. EM convergence safety net ----
# max_em_iter is a real cap (not just accepted and ignored).
# Setting it artificially low on perfectly good data shows
# the diagnostic you get instead of an infinite loop.
demo_error("artificially low max_em_iter triggers a clear non-convergence error", {
  mode_alphaEM(n, p, xbar, ldetxbarg, max_em_iter = 1)
})

## ---- 4J. wishart_inference()'s error-list convention ----
# Parameter misspecification (4B-4H above -- bad nsamp, bad
# priors, bad mu0, malformed/duplicated data) always fails
# fast with an ordinary R error, even when called through
# wishart_inference(). That's deliberate: those are mistakes
# you want to see immediately, not buried in a return value.
#
# Genuine numerical/convergence problems that can only be
# detected *during* the computation are handled differently:
# wishart_inference() catches them internally and returns
# list(error = "...") instead of raising, so a batch job
# iterating over many datasets doesn't die on one bad case.
# mode_alphaEM()/rejection_sampler() (the "manual workflow"
# functions) always raise directly -- see 4I above for the
# same underlying failure via mode_alphaEM().
cat("\n--- wishart_inference()'s catch-and-return convention ---\n")
r <- wishart_inference(X, nsamp = 100, max_em_iter = 1) 
cat("wishart_inference() did NOT raise; it returned:\n")
str(r)

