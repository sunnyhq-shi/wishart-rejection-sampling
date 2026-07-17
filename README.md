# wishartinference

Bayesian inference for the shape parameter of the Wishart distribution.

Given data $X_1, \dots, X_n \overset{iid}{\sim} \text{Wishart}_p(2\alpha, \Sigma)$, this
package provides posterior inference for the shape parameter $\alpha$ and the
mean matrix $\mu = \alpha\Sigma$, under two prior specifications:

- **Improper prior**:
  $$p(\alpha, \mu) \propto (\alpha - (p-1)/2)^{-1} \cdot |\mu|^{-(p+1)/2}$$

- **Proper prior**:
  $$\alpha \sim \text{Gamma}(\beta, \beta\eta) \text{ truncated at } (p-1)/2, \qquad
  \mu \mid \alpha \sim \text{inv-Wishart}(2\kappa\alpha, 2\kappa\alpha\mu_0)$$

The posterior mode is found via a Newton-within-EM algorithm, and joint
samples of $(\alpha, \mu)$ are drawn via rejection sampling with a truncated
Gamma covering density. The core computation is implemented in C++
(RcppArmadillo + Boost) for speed.

## Installation

```r
install.packages(
  "wishartinference",
  repos = c("https://sunnyhq-shi.r-universe.dev", "https://cloud.r-project.org")
)
```

This installs a precompiled binary where available — no local C++ compiler
or Rtools setup required.

## Quick start

```r
library(wishartinference)

# Simulate data from a known truth
set.seed(6)
p <- 3; nu <- 10; Sigma <- diag(p); n <- 10
X <- rwishart(n, p, nu, Sigma)

# Full pipeline: sufficient stats, EM mode, rejection sampling
res <- wishart_inference(X, nsamp = 5000)

res$results$ahat                                    # posterior mode
quantile(res$results$alpha_samples, c(0.025, 0.975)) # 95% interval
```

## Improper vs. proper prior

The prior type is auto-detected from the arguments you supply:

| | Improper | Proper |
|---|---|---|
| `mu0` | omit | supply (p x p, positive definite) |
| `beta` | omit | must be > 1 |
| `eta` | omit | must be > 0 |
| `kappa` | omit | must be >= 1 |

```r
# Improper prior (default)
res_imp <- wishart_inference(X, nsamp = 5000)

# Proper prior
beta  <- 50
eta   <- (beta - 1) / (6 * beta)   # places the prior mode at alpha = 3
kappa <- 1
mu0   <- diag(10, p)

res_pro <- wishart_inference(X, mu0 = mu0, beta = beta, eta = eta,
                              kappa = kappa, nsamp = 5000)
```

## Step-by-step (manual) workflow

For full control over each stage, or to reuse the mode/sampler independently:

```r
stat <- wishart_stats(X)                          # sufficient statistics
ahat <- mode_alphaEM(n, p, stat$xbar, stat$ldetxbarg)  # posterior mode

lambda   <- max(n * (ldet(stat$xbar) - stat$ldetxbarg), 1e-10)
nu_star  <- ahat[1] * lambda + 1

samp <- rejection_sampler(ahat[1], ahat[2], lambda, nu_star,
                           p, n, stat$xbar, stat$ldetxbarg)
```

## Exported functions

| Function | Purpose |
|---|---|
| `wishart_inference()` | Full pipeline: sufficient statistics, EM mode finding, rejection sampling |
| `mode_alphaEM()` | Find the posterior mode of alpha via Newton-within-EM |
| `rejection_sampler()` | Draw joint posterior samples of `(alpha, mu)` given a mode |
| `wishart_stats()` | Compute sufficient statistics (`xbar`, `ldetxbarg`) from data |
| `rwishart()` | Generate draws from a Wishart distribution |
| `lfafun_improper()` / `lfafun_proper()` | Evaluate the log unnormalized posterior of alpha |
| `lgammap_export()` | Log of the multivariate Gamma function |
| `ldet()` | Log determinant of a positive definite matrix |

See `?function_name` for full argument documentation on any of the above.

## Error handling

Bad inputs (invalid `nsamp`, malformed priors, non-positive-definite
matrices, degenerate data, etc.) raise an informative R-level error
immediately. Genuine numerical non-convergence during computation is instead
caught internally by `wishart_inference()`, which returns
`list(error = "...")` rather than stopping — useful when iterating over many
datasets in a batch job. `mode_alphaEM()` and `rejection_sampler()`, the
manual-workflow functions, always raise directly.

```r
r <- wishart_inference(X, nsamp = 100, max_em_iter = 1)
if (!is.null(r$error)) {
  message("Inference failed: ", r$error)
}
```

## Model constraints

- `p >= 2` and `n >= 2` are required throughout; `p = 1` (a scalar
  "Wishart," i.e. chi-squared) and `n = 1` are outside the intended scope of
  this model.
- Under the improper prior, `n * (log|xbar| - mean(log|X_i|))` must be
  strictly positive; this fails only when all `n` observations are
  numerically identical.
- Under the proper prior, the prior mode `(beta - 1) / (beta * eta)` must
  exceed `(p - 1) / 2`.

## Authors

- Phil Everson — `peverso1@swarthmore.edu`
- Hanqi Shi (maintainer) — `hshi1@swarthmore.edu`

Developed at Swarthmore College.

## License

MIT — see `LICENSE`.