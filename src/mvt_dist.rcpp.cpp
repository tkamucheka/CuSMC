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

//' MultiVariate T Distribution
//'
//' @param  mu            [vector]: Location vector.
//' @param  sigma         [matrix]: Dispersion matrix.
//' @param nu             [float]: degrees of freedom.
//' @return draws         [vector]: Vector of random draws from distribution
//' @export
// [[Rcpp::export]]
Eigen::VectorXd MVT(Eigen::VectorXd mu, Eigen::MatrixXd sigma, float nu)
{
  unsigned N = mu.rows();
  Eigen::VectorXd draws = Eigen::VectorXd::Zero(N);
  MultiVariateTStudentDistribution MVT(mu, sigma, nu);
  
  // Find the eigen vectors of the covariance matrix
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>
      eigen_solver(sigma);
  Eigen::MatrixXd eigenvectors = eigen_solver.eigenvectors().real();

  // Find the eigenvalues of the covariance matrix
  Eigen::MatrixXd eigenvalues = eigen_solver.eigenvalues().real().asDiagonal();

  // Find the transformation matrix
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(eigenvalues);
  Eigen::MatrixXd sqrt_eigenvalues = es.operatorSqrt();
  Eigen::MatrixXd Q = eigenvectors * sqrt_eigenvalues;

  MVT.sample(draws, Q, 200);
  return draws;
}

//' MultiVariate T Probability Density Function
//'
//' @param mu            [vector]: Location vector.
//' @param sigma         [matrix]: Dispersion matrix.
//' @param nu             [float]: degrees of freedom.
//' @export
// [[Rcpp::export]]
double MVTPDF(Eigen::VectorXd x, Eigen::VectorXd mu, Eigen::MatrixXd sigma, float nu)
{
  MultiVariateTStudentDistribution MVT(mu, sigma, nu);
  int n = mu.rows();
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(n, n);
  return MVT.pdf(x,F);
}
