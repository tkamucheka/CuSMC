// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>

#include "../inst/include/particle_filter.hpp"

using namespace Rcpp;

//' @useDynLib CuMCMC

// via the depends attribute we tell Rcpp to create hooks for
// RcppEigen so that the build process will know what to do
//
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]

struct env_t
{
  unsigned N, d, timeSteps;
  Eigen::VectorXd *y_t;
  Eigen::MatrixXd F, G, C0;
  Eigen::VectorXd m0;
} ENV;

//' Run simulations
//'
//' @param N integer: Number of particles is solution.
//' @param d integer: Number of parameters.
//' @param timeSteps integer: Total time steps.
//' @param Y matrix: Input observations.
//' @param m0 vector: Initial parameters at t=0
//' @param C0 matrix: Initial covariant matrix at t=0
//' @param F matrix: covariant matrix for scaling particle samples 
//' @param sampler integer: Resampling sampler
//' @param distribution integer: Distribution for sampling particles 
// [[Rcpp::export]]
List run(unsigned &N, unsigned &d, unsigned &timeSteps,
         Eigen::MatrixXd Y, Eigen::VectorXd m0, Eigen::MatrixXd C0, Eigen::MatrixXd F, int sampler, int distribution) {
  
  // Setup Environment
  // Dimensions
  ENV.d = d;
  ENV.N = N;
  ENV.timeSteps = timeSteps;
  // Input Data
  ENV.m0 = m0; // Eigen::VectorXd::Zero(d);
  ENV.C0 = C0; // Eigen::MatrixXd::Identity(d, d);
  ENV.F  = F;  //Eigen::MatrixXd::Identity(d, d);
  ENV.G  = Eigen::MatrixXd::Identity(d, d);
  
  ENV.y_t = new Eigen::VectorXd[ENV.timeSteps];

  Eigen::VectorXd **post_x_t, *w_t;
  unsigned *a_t;
  
  // Initialize variables
  post_x_t = new Eigen::VectorXd *[ENV.timeSteps];
  w_t = new Eigen::VectorXd[ENV.timeSteps];
  a_t = new unsigned[ENV.timeSteps * N];
  
  for (unsigned t = 0; t < ENV.timeSteps; ++t)
  {
    ENV.y_t[t] = Y.col(t);
    post_x_t[t] = new Eigen::VectorXd[N];
    w_t[t] = Eigen::VectorXd::Zero(N);
    
    for (unsigned i = 0; i < N; ++i)
      post_x_t[t][i] = Eigen::VectorXd::Zero(ENV.d);
  }
  
  // Run particle filter
  float runtime;
  particle_filter(post_x_t, w_t, a_t, ENV.y_t, ENV.F, ENV.G, ENV.m0, ENV.C0,
                  ENV.d, N, ENV.timeSteps, runtime);

  // Build return object
  List ret;
  // ret["ancestors"] = a_t;
  // ret["weights"] = w_t;
  // ret["posterior_x"] = post_x_t;
  
  return ret;
}
