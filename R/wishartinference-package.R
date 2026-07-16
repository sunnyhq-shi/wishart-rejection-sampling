#' wishartinference: Bayesian Inference for the Wishart Distribution Shape Parameter
#'
#' Posterior inference for the shape parameter alpha and mean matrix mu
#' in the model X_i ~ Wishart_p(2*alpha, Sigma).
#'
#' @useDynLib wishartinference, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @keywords internal
"_PACKAGE"
