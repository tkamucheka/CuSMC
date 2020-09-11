// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// Silence Eigen warning messages
#pragma clang diagnostic ignored "-Wunknown-pragmas"

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>
#include <statistics.hpp>

using namespace Rcpp;

//' @useDynLib CuSMC

// via the depends attribute we tell Rcpp to create hooks for
// RcppEigen so that the build process will know what to do
//
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]

//' MultiVariateNormal Distribution
//'
//' @param  mu            [vector]: Mean vector.
//' @param  sigma         [matrix]: Covariant matrix.
//' @return draws         [vector]: Vector of real random draws from distribution
//' @export
//' @examples
//' mu = c(0, 0)
//' sigma = matrix(c(1, 0, 0, 1), nrow = 2)
//' CuSMC::MVN(mu, sigma)
// [[Rcpp::export]]
Eigen::VectorXd
MVN(Eigen::VectorXd mu, Eigen::MatrixXd sigma)
{
  Eigen::VectorXd draws(mu.rows());
  MultiVariateNormalDistribution MVN(mu, sigma);
  MVN.sample(draws, sigma, 200);
  return draws;
}

//' MultiVariateNormal Probability Density Function
//'
//' @param x             [vector]: x.
//' @param mu            [vector]: Mean vector.
//' @param sigma         [matrix]: Covariant matrix.
//' @return              [numeric]: Real value
//' @export
//' @examples
//' x = c(0, 0)
//' mu = c(0, 0)
//' sigma = matrix(c(1, 0, 0, 1), nrow = 2)
//' CuSMC::MVNPDF(x, mu, sigma)
// [[Rcpp::export]]
double MVNPDF(Eigen::VectorXd x, Eigen::VectorXd mu, Eigen::MatrixXd sigma)
{
  int n = mu.rows();
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(n, n);
  MultiVariateNormalDistribution MVN(mu, sigma);
  return MVN.pdf(x, F);
}
