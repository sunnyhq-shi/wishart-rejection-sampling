# Bayesian Inference for the Wishart Shape Parameter

Posterior inference for the shape parameter of a Wishart distribution,

$$X_i \sim \text{Wishart}_p(2\alpha, \Sigma), \qquad i = 1, \dots, n,$$

under two prior specifications for $(\alpha, \mu)$ where $\mu = 2\alpha\Sigma$:

- **Improper prior:** $p(\alpha,\mu) \propto (\alpha-\tfrac{p-1}{2})^{-1}|\mu|^{-(p+1)/2}$
- **Proper prior:** $\alpha \sim \text{Gamma}(\beta, \beta\eta)$ truncated at $(p-1)/2$, with $\mu \mid \alpha \sim \text{inv-Wishart}(2\kappa\alpha,\, 2\kappa\alpha\mu_0)$

The posterior mode of $\alpha$ is located with a Newton-within-EM algorithm, and joint samples of $(\alpha, \mu)$ are drawn by rejection sampling against a truncated Gamma covering density.

## Repository contents

| File | Role |
|---|---|
| `wishart_inference.cpp` | Core implementation (RcppArmadillo). All statistics, mode-finding, and sampling logic lives here. |
| `wishart_inference.R` | Demonstration script. Simulates data, calls into the C++ backend, checks results against a grid search, and produces diagnostic plots. |

The C++ file is the base — it's a self-contained set of `Rcpp::export`-ed functions with no package scaffolding required. The R script is a runnable demo, not a dependency of the C++ code.

## Requirements

- R (≥ 4.0 recommended)
- R packages: `Rcpp`, `RcppArmadillo`
- A C++ compiler with **Boost** headers available (the `BH` R package provides these automatically, or a system Boost install works too)

Install the R package dependencies once:

```r
install.packages(c("Rcpp", "RcppArmadillo", "BH"))
```

## Getting the code

You don't need `git` installed to get these two files into an R session — `download.file()` pulls them straight from GitHub's raw content server:

```r
dir.create("wishart-inference", showWarnings = FALSE)

download.file(
  "https://raw.githubusercontent.com/sunnyhq-shi/wishart-rejection-sampling
/main/wishart_inference.cpp",
  destfile = "wishart-inference/wishart_inference.cpp"
)
download.file(
  "https://raw.githubusercontent.com/sunnyhq-shi/wishart-rejection-sampling
/main/wishart_inference.R",
  destfile = "wishart-inference/wishart_inference.R"
)

setwd("wishart-inference")
```

Other ways to get it, if you prefer:

- **Clone the whole repo** (requires `git` on your system, run from a terminal, not R):
  ```bash
  git clone https://github.com/sunnyhq-shi/wishart-rejection-sampling.git
  ```
- **Download the ZIP** — on the repo's GitHub page, use *Code → Download ZIP*, then unzip locally.
- **From R with `usethis`** (if installed), which wraps the same idea as the ZIP download:
  ```r
  usethis::create_from_github("sunnyhq-shi/wishart-rejection-sampling", destdir = ".")
  ```

Once you have both files in the same working directory, everything else happens inside R.

## Quick start

```r
library(Rcpp)
library(RcppArmadillo)
sourceCpp("wishart_inference.cpp")   # compiles once; recompiles only if the file changes

test()  # should print "Hello from C++"
```

`sourceCpp()` compiles the `.cpp` file locally the first time it's sourced in a session (or whenever it changes) — there's no separate build step or package install needed.

### Simulate data and run inference

```r
p <- 3; nu <- 10; n <- 10
Sigma <- diag(p)

X <- rwishart(n, p, nu, Sigma)
stat <- wishart_stats(X)     # xbar, ldetxbarg

# Improper prior (default)
res <- wishart_inference(X, nsamp = 50000)
res$results$ahat                       # posterior mode
quantile(res$results$alpha_samples, c(0.025, 0.975))  # 95% interval

# Proper prior: supply mu0, beta > 1, kappa >= 1
res_pro <- wishart_inference(
  X, mu0 = diag(10, p),
  beta = 50, eta = (50 - 1) / (6 * 50), kappa = 1,
  nsamp = 5000
)
```

`wishart_inference()` is the packaged, one-call interface. For step-by-step control (compute statistics, find the mode, then sample independently), call `wishart_stats()`, `mode_alphaEM()`, and `rejection_sampler()` directly — see the fully worked examples, plotting code, and a full error-handling walkthrough in `wishart_inference.R`.

## What the demo script covers

`wishart_inference.R` is organized into:

1. **Setup** — true parameters and dimensions for a simulated dataset
2. **Data generation** — draws from `Wishart_p(2*alpha, Sigma)`, retrying until all slices are numerically positive definite
3. **Posterior inference** — both the packaged (`wishart_inference()`) and manual (`wishart_stats()` → `mode_alphaEM()` → `rejection_sampler()`) workflows, for both priors, with grid-search cross-checks of the EM mode and diagnostic plots (posterior samples vs. density vs. covering Gamma, and — for the proper prior — likelihood/prior/posterior overlaid to show prior-data tension)
4. **Error handling demonstrations** — deliberately triggers every category of input validation (bad dimensions, invalid hyperparameters, degenerate data, non-convergence, etc.) so you can see the exact error messages before relying on them in your own code

## Notes on the model

- Auto-detection of prior type: omitting `mu0`, `beta`, `eta`, `kappa` (or leaving them at their defaults) triggers the **improper** prior path; supplying `mu0` with `beta > 1` and `kappa >= 1` triggers the **proper** prior path.
- The proper prior requires the prior mode $(\beta-1)/(\beta\eta)$ to lie strictly above $(p-1)/2$, or there's no valid interior starting point.
- `wishart_inference()` catches genuine numerical/convergence failures internally and returns `list(error = "...")` rather than raising, so a batch job over many datasets won't die on one bad case. The lower-level `mode_alphaEM()` / `rejection_sampler()` functions always raise directly — useful when you want failures to surface immediately during development.


