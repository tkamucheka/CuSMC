// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// Silence Eigen warning messages
#pragma clang diagnostic ignored "-Wunknown-pragmas"

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>

// local includes
#include "../inst/include/samplers.hpp"

using namespace Rcpp;

//' @useDynLib CuSMC

// via the depends attribute we tell Rcpp to create hooks for
// RcppEigen so that the build process will know what to do
//
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]

//' Metropolis Hastings Sampler
//'
//' @param  w           [vector]: Weights
//' @param  N           [matrix]: Number of weights
//' @param  B           [vector]: Bias term
//' @return a           [vector]: Ancestors
//' @export
//' @examples
//' w = c(0, 0)
//' N = 2
//' B = 10
//' CuSMC::metropolis_hastings(w, N, B)
// [[Rcpp::export]]
Eigen::VectorXd metropolis_hastings(Eigen::VectorXd w, int N, int B)
{
  // Array of new ancestors
  unsigned *ancestors = new unsigned[N * 2];
  Eigen::VectorXd a(N);

  // Internal function expects a vector of weight vectors
  Eigen::VectorXd *weights = new Eigen::VectorXd[1];
  weights[0] = w;

  Sampler::metropolis_hastings(ancestors, weights, N, 1, B);

#pragma omp parallel for
  for (int i = 0; i < N; ++i)
    a[i] = ancestors[N + i];

  delete[] ancestors;
  delete[] weights;

  return a;
}
