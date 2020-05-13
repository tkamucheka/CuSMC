// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
// Silence Eigen warning messages
#pragma clang diagnostic ignored "-Wunknown-pragmas"

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>
#include <functional>
#include <map>
#include <string>
#include <assert.h>

#include <particle_filter.hpp>

using namespace Rcpp;

//' @useDynLib CuSMC

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
  float df;
} ENV;

// typedef std::function<void(unsigned *, Eigen::VectorXd *, int, unsigned)> resampler_t;
// samplers_t Samplers;
// Samplers["metropolis"] = metropolis;
// distributions_t Distributions;

//' Run simulations
//'
//' @param N            [integer]: Number of particles is solution.
//' @param d            [integer]: Number of parameters.
//' @param timeSteps    [integer]: Total time steps.
//' @param Y            [matrix]: Input observations.
//' @param m0           [vector]: Initial parameters at t=0
//' @param C0           [matrix]: Initial covariant matrix at t=0
//' @param F            [matrix]: Covariant matrix for scaling particle samples
//' @param sampler      [string]: Resampling sampler
//' @param distribution [string]: Distribution for sampling particles
//' @export
// [[Rcpp::export]]
List run(unsigned &N, unsigned &d, unsigned &timeSteps,
         Eigen::MatrixXd Y, Eigen::VectorXd m0, Eigen::MatrixXd C0, Eigen::MatrixXd F,
         float df, std::string resampler, std::string distribution)
{
  // Initialize Sampler and Distribution
  // assert(Samplers.find(sampler_opt) != Samplers.end());
  // resampler_f resampler = Samplers[sampler_opt];

  // assert(Distributions.find(distribution_opt) != Distributions.end());
  // distributionCreator_f distCreator = Distributions[distribution_opt];

  // Setup Environment
  // Dimensions
  ENV.d = d;
  ENV.N = N;
  ENV.timeSteps = timeSteps;
  // Input Data
  ENV.m0 = m0; // Eigen::VectorXd::Zero(d);
  ENV.C0 = C0; // Eigen::MatrixXd::Identity(d, d);
  ENV.F = F;   //Eigen::MatrixXd::Identity(d, d);
  ENV.G = Eigen::MatrixXd::Identity(d, d);
  ENV.df = df;

  // Initialize variables
  ENV.y_t = new Eigen::VectorXd[ENV.timeSteps];

  Eigen::VectorXd **post_x_t, *w_t;
  unsigned *a_t;

  post_x_t = new Eigen::VectorXd *[ENV.timeSteps];
  w_t = new Eigen::VectorXd[ENV.timeSteps];
  a_t = new unsigned[ENV.timeSteps * N];

  // Initialiaze Sampler and Distribution
  // Resampler sampler = Sampler::getSampler(opt_sampler);
  // Distribution distCreator = StatisticalDistribution.getCreator(opt_distribution);

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
                  ENV.d, N, ENV.timeSteps, runtime, resampler, distribution, ENV.df);

  // Build return object
  List ret;
  // ret["ancestors"] = a_t;
  // ret["weights"] = w_t;
  // ret["posterior_x"] = post_x_t;

  return ret;
}