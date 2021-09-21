// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
// Silence Eigen warning messages
#pragma clang diagnostic ignored "-Wunknown-pragmas"

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppArmadillo.h>
#include <RcppEigen.h>
#include <functional>
#include <map>
#include <string>
#include <assert.h>

#include <particle_filter.hpp>
#include <io.hpp>

using namespace Rcpp;

//' @useDynLib CuSMC

// via the depends attribute we tell Rcpp to create hooks for
// RcppEigen so that the build process will know what to do
//
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

struct env_t
{
  unsigned N, d, timeSteps;
  Eigen::VectorXd *y_t;
  Eigen::MatrixXd F, G, V, W, C0;
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
//' @param G            [matrix]: Covariant matrix for scaling particle samples
//' @param V            [matrix]: Covariant matrix for noise??
//' @param W            [matrix]: Covariant matrix for noise??
//' @param df           [float]:  Degrees of freedom (for MVT)
//' @param resampler      [string]: Resampling sampler
//' @param distribution [string]: Distribution for sampling particles
//' @export
// [[Rcpp::export]]
List run(unsigned &N, unsigned &d, unsigned &timeSteps,
         Eigen::MatrixXd Y,
         Eigen::VectorXd m0, Eigen::MatrixXd C0, Eigen::MatrixXd F, Eigen::MatrixXd G,
         Eigen::MatrixXd V, Eigen::MatrixXd W,
         float df, std::string resampler, std::string distribution, unsigned p = 0)
{
  assert(p < N);
  // Setup Environment
  // Dimensions
  ENV.d = d;
  ENV.N = N;
  ENV.timeSteps = timeSteps;
  // Input Data
  ENV.m0 = m0; // Eigen::VectorXd::Zero(d);
  ENV.C0 = C0; // Eigen::MatrixXd::Identity(d, d);
  ENV.F = F;   //Eigen::MatrixXd::Identity(d, d);
  ENV.G = G;
  ENV.V = V;
  ENV.W = W;
  ENV.df = df;

  // // Initialize variables
  ENV.y_t = new Eigen::VectorXd[ENV.timeSteps];

  Eigen::VectorXd **post_x_t, *w_t;
  unsigned *a_t;

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

  // // Run particle filter
  float runtime;
  particle_filter(
      post_x_t, w_t, a_t, ENV.y_t,
      ENV.F, ENV.G,
      ENV.m0, ENV.C0,
      ENV.V, ENV.W,
      N, ENV.d, ENV.timeSteps, runtime, resampler, distribution, ENV.df);
  writeOutput(ENV.y_t, w_t, post_x_t, N, ENV.d, ENV.timeSteps, p);

  // Convert Eigen::VectorXd *w_t to Rcpp::NumericMatrix w
  Rcpp::NumericMatrix w(timeSteps,N);
  for (int i = 0; i < timeSteps; i++) {
    SEXP s = Rcpp::wrap(w_t[i]);
    Rcpp::NumericVector v(s);
    w(i,_) = v;
  }

  // Convert Eigen::VectorXd **post_x_t to arma::cube theta
  arma::cube theta = arma::cube(timeSteps, N, d);
  for (arma::uword i = 0; i < theta.n_rows; ++i)
    for (arma::uword j = 0; j < theta.n_cols; ++j)
      for (arma::uword k = 0 ; k < theta.n_slices; ++k) 
        theta(i,j,k)= post_x_t[i][j][k];

  return Rcpp::List::create(Rcpp::Named("weights") = w,
                            Rcpp::Named("posterior_x") = theta);
}
/*
//' Simulations sim: y_t is generated
//'
//' @param N            [integer]: Number of particles is solution.
//' @param d            [integer]: Number of parameters.
//' @param timeSteps    [integer]: Total time steps.
//' @param m0           [vector]: Initial parameters at t=0
//' @param C0           [matrix]: Initial covariant matrix at t=0
//' @param F            [matrix]: Covariant matrix for scaling particle samples
//' @param sampler      [string]: Resampling sampler
//' @param distribution [string]: Distribution for sampling particles
//' @export
// [[Rcpp::export]]
List sim(unsigned &N, unsigned &d, unsigned &timeSteps,
         Eigen::VectorXd m0, Eigen::MatrixXd C0, Eigen::MatrixXd F,
         float df, std::string resampler, std::string distribution)
{
  // Setup Environment
  // Dimensions
  ENV.d = d;
  ENV.N = N;
  ENV.timeSteps = timeSteps;
  // Input Data
  ENV.m0 = m0;
  ENV.C0 = C0;
  ENV.F = F;
  ENV.G = Eigen::MatrixXd::Identity(ENV.d, ENV.d);
  ENV.df = df;

  // // Initialize variables
  Eigen::VectorXd *prior_x_t, **post_x_t, *w_t;
  unsigned *a_t;

  ENV.y_t = new Eigen::VectorXd[ENV.timeSteps];
  prior_x_t = new Eigen::VectorXd[ENV.timeSteps];
  post_x_t = new Eigen::VectorXd *[ENV.timeSteps];
  w_t = new Eigen::VectorXd[ENV.timeSteps];
  a_t = new unsigned[ENV.timeSteps * N];

  for (unsigned t = 0; t < ENV.timeSteps; ++t)
  {
    // ENV.y_t[t] = Y.col(t);
    ENV.y_t[t] = Eigen::VectorXd::Zero(ENV.d);
    prior_x_t[t] = Eigen::VectorXd::Zero(ENV.d);
    post_x_t[t] = new Eigen::VectorXd[N];
    w_t[t] = Eigen::VectorXd::Zero(N);

    for (unsigned i = 0; i < N; ++i)
      post_x_t[t][i] = Eigen::VectorXd::Zero(ENV.d);
  }

  //Noise matrices
  Eigen::MatrixXd E_1 = Eigen::MatrixXd::Identity(ENV.d,ENV.d);
  Eigen::MatrixXd E_2 = Eigen::MatrixXd::Identity(ENV.d,ENV.d);

  // // Run particle filter
  float runtime;
  generateInput(prior_x_t, ENV.y_t, ENV.F, ENV.G, E_1, E_2, N, ENV.d, ENV.timeSteps);
  particle_filter(post_x_t, w_t, a_t, ENV.y_t, ENV.F, ENV.G, ENV.m0, ENV.C0,
                  ENV.d, N, ENV.timeSteps, runtime, resampler, distribution, ENV.df);
  writeOutput_ysim(prior_x_t, ENV.y_t, w_t, post_x_t, N, ENV.d, ENV.timeSteps);

  // Build return object
  List ret;
  // ret["ancestors"] = a_t;
  // ret["weights"] = w_t;
  // ret["posterior_x"] = post_x_t;

  return ret;
}
*/

/*
//' Simulations step
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
List step(unsigned &N, unsigned &d, unsigned &timeSteps,
          Eigen::MatrixXd Y, Eigen::MatrixXd F, float df,
          std::string resampler, std::string distribution)
{
  // Setup Environment
  // Dimensions
  ENV.d = d;
  ENV.N = N;
  ENV.timeSteps = timeSteps;
  // Input Data
  ENV.m0 = Eigen::VectorXd::Zero(d);
  ENV.C0 = Eigen::MatrixXd::Identity(d, d);
  ENV.F = F; //Eigen::MatrixXd::Identity(d, d);
  ENV.G = Eigen::MatrixXd::Identity(d, d);
  ENV.df = df;

  // // Initialize variables
  ENV.y_t = new Eigen::VectorXd[ENV.timeSteps];

  Eigen::VectorXd **post_x_t, *w_t;
  unsigned *a_t;

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

  // // Run particle filter
  float runtime;
  particle_filter_step(post_x_t, w_t, a_t, ENV.y_t, ENV.F, ENV.G, ENV.m0, ENV.C0,
                       ENV.d, N, ENV.timeSteps, runtime, resampler, distribution, ENV.df);

  // Build return object
  List ret;
  // ret["ancestors"] = a_t;
  // ret["weights"] = w_t;
  // ret["posterior_x"] = post_x_t;

  return ret;
}
*/
