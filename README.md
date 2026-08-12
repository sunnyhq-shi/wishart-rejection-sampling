# wishartinference

Bayesian inference for the shape parameter of the Wishart distribution —
and its p = 1 reduction, the Gamma distribution.

Given data $X_1, \dots, X_n \overset{iid}{\sim} \text{Wishart}_p(2\alpha, \Sigma)$, this
package provides posterior inference for the shape parameter $\alpha$ and the
mean matrix $\mu = \alpha\Sigma$. The posterior mode is found via a
Newton-within-EM algorithm, and joint samples of $(\alpha, \mu)$ are drawn
via rejection sampling with a Gamma covering density. The core computation
is implemented in C++ (RcppArmadillo + Boost) for speed.

The package has two independent backends, covered in their own sections
below — there is no unifying dispatcher, so use whichever matches your data:

- **`wishart_inference()`** — matrix data, `p >= 2`
- **`gamma_inference()`** — scalar data, `p = 1` (the exact p = 1 reduction
  of the same model, so scalar observations never need to be wrapped in
  trivial `1 x 1` matrices)

A full walkthrough of both backends — improper and proper priors, the
manual step-by-step workflow, and the package's error-handling
conventions — is in `demo/wishart_demo.R`.

## Installation

```r
install.packages(
  "wishartinference",
  repos = c("https://sunnyhq-shi.r-universe.dev", "https://cloud.r-project.org")
)
```

This installs a precompiled binary where available — no local C++ compiler
or Rtools setup required.

---

## Wishart model (p >= 2)

- **Improper prior**:
  $$p(\alpha, \mu) \propto (\alpha - (p-1)/2)^{-1} \cdot |\mu|^{-(p+1)/2}$$

- **Proper prior**:
  $$(\alpha - (p-1)/2) \sim \text{Gamma}(\beta, \text{rate} = \beta\eta), \qquad
  \mu \mid \alpha \sim \text{inv-Wishart}(2\kappa\alpha, 2\kappa\alpha\mu_0)$$

### Quick start

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

### Improper vs. proper prior

The prior type is auto-detected from the arguments you supply. `beta = 0,
eta = 1, kappa = 0` (the function defaults) is not merely a placeholder
flag — it is mathematically identical to the improper prior above.

| | Improper | Proper |
|---|---|---|
| `mu0` | omit | supply (`p x p`, positive definite) |
| `beta` | omit (0) | `>= 0` |
| `eta` | omit (1) | `>= 0` |
| `kappa` | omit (0) | `>= 1` |

```r
beta  <- 50
eta   <- (beta - 1) / (beta * (3 - (p - 1) / 2))  # places the prior mode at alpha = 3
kappa <- 1
mu0   <- diag(10, p)

res_pro <- wishart_inference(X, mu0 = mu0, beta = beta, eta = eta,
                              kappa = kappa, nsamp = 5000)
```

### Manual step-by-step workflow

For full control over each stage, or to reuse the mode/sampler independently:

```r
stat <- wishart_stats(X)                               # sufficient statistics
ahat <- mode_alphaEM(n, p, stat$xbar, stat$ldetxbarg)   # posterior mode

lambda   <- max(n * (ldet(stat$xbar) - stat$ldetxbarg), 1e-10)
nu_star  <- ahat[1] * lambda + 1

samp <- rejection_sampler(ahat[1], ahat[2], lambda, nu_star,
                           p, n, stat$xbar, stat$ldetxbarg)
```

### Exported functions

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

### Constraints

- `p >= 2` and `n >= 2` are required.
- Under the improper prior, `n * (log|xbar| - mean(log|X_i|))` must be
  strictly positive; fails only when all `n` observations are numerically
  identical.

---

## Gamma model (p = 1)

The exact `p = 1` case of the model above: $X_i \sim \text{Gamma}(\alpha,
\alpha/\mu)$. Same priors, same EM-then-rejection-sampling approach, just
scalar inputs/outputs instead of matrices.

### Quick start

```r
set.seed(11)
n <- 20; a <- 3
x <- rgamma(n, a, a)   # mean 1

res <- gamma_inference(x, nsamp = 5000)

res$results$ahat                                    # posterior mode
quantile(res$results$alpha_samples, c(0.025, 0.975)) # 95% interval

# Proper prior
res_pro <- gamma_inference(x, mu0 = 1, beta = 5, eta = 0.5, kappa = 2, nsamp = 5000)
```

### Exported functions

| Function | Purpose |
|---|---|
| `gamma_inference()` | Full pipeline for scalar data |
| `mode_alphaEM_gamma()` | Posterior mode via EM |
| `rejection_sampler_gamma()` | Joint posterior samples of `(alpha, mu)` given a mode |
| `lfafun_gamma()` | Log unnormalized posterior of alpha |

### Differences from the matrix case

- `kappa = 0` is allowed even with `beta > 0` here (proper prior on
  `alpha`, flat prior on `mu`) — the matrix backend requires `kappa >= 1`
  whenever the prior isn't fully improper.
- **Structural edge case**: at `n = 2` with a fully improper prior
  (`beta = 0, kappa = 0`), the posterior has no interior mode for *any*
  dataset — its supremum sits exactly at the boundary `alpha = 0`.
  `gamma_inference()` raises an informative error here rather than
  returning a degenerate result. Any `beta > 0`, any `kappa > 0`, or
  `n > 2` resolves it. (The matrix model has no analogous failure — an
  interior mode is always guaranteed there at `n >= 2`.)

---

## Error handling

Bad inputs (invalid `nsamp`, malformed priors, non-positive-definite
matrices, degenerate data, etc.) raise an informative R-level error
immediately, for both backends. Genuine numerical non-convergence is
instead caught internally by `wishart_inference()` / `gamma_inference()`,
which return `list(error = "...")` rather than stopping — useful when
iterating over many datasets in a batch job. The manual-workflow functions
(`mode_alphaEM()`, `rejection_sampler()`, and their `_gamma` counterparts)
always raise directly.

```r
r <- wishart_inference(X, nsamp = 100, max_em_iter = 1)
if (!is.null(r$error)) {
  message("Inference failed: ", r$error)
}
```

## Authors

- Phil Everson — `peverso1@swarthmore.edu`
- Hanqi Shi (maintainer) — `hshi1@swarthmore.edu`

Developed at Swarthmore College.

## License

MIT — see `LICENSE`.
