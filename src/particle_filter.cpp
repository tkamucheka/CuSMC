#ifndef __PARTICLE_FILTER_CPP
#define __PARTICLE_FILTER_CPP

#include "../inst/include/particle_filter.hpp"

void particle_filter(
    Eigen::VectorXd **post_x_t,
    Eigen::VectorXd *w_t, unsigned *a_t,
    Eigen::VectorXd *y_t,
    Eigen::MatrixXd F, Eigen::MatrixXd G,
    Eigen::MatrixXd V, Eigen::MatrixXd W,
    Eigen::VectorXd m0, Eigen::MatrixXd C0,
    dim_t N, dim_t d, dim_t timeSteps,
    float df,
    std::string resampler_opt,
    std::string distribution_opt,
    float &runtime)
{
  Timer timer;
  // 1. Initialize particles x_t (theta) and weights w_t (omega)
  Rcpp::Rcout << "Initializing thetas and omegas at t[0]... " << std::flush;
  startTime(&timer);
  initialize(post_x_t, w_t,
             m0, C0, N, d, 0, df,
             distribution_opt);
  stopTime(&timer);
  Rcpp::Rcout << "Done. " << elapsedTime(timer) << " s" << std::endl;

  // 2. MCMC time marching
  Rcpp::Rcout << "Simulating... " << std::flush;
  startTime(&timer);
  // Eigen::MatrixXd E = Eigen::MatrixXd::Identity(d, d); // Hack
  MCMC(post_x_t, w_t, a_t, y_t,
       F, G, V, W, N, d, timeSteps, df,
       resampler_opt, distribution_opt);
  stopTime(&timer);
  runtime = elapsedTime(timer);
  Rcpp::Rcout << "Done. " << elapsedTime(timer) << " s" << std::endl;
}

void particle_filter_step(Eigen::VectorXd **post_x_t,
                          Eigen::VectorXd *w_t, unsigned *a_t,
                          Eigen::VectorXd *y_t,
                          Eigen::MatrixXd F, Eigen::MatrixXd G,
                          Eigen::VectorXd m0, Eigen::MatrixXd C0,
                          //  Eigen::MatrixXd *sigmaV, Eigen::MatrixXd *sigmaW,
                          dim_t d, dim_t N, dim_t timeSteps, float &runtime,
                          std::string resampler_opt, std::string distribution_opt, float df)
{
  Timer timer;
  // 1. MCMC time stepping
  Rcpp::Rcout << "Stepping... " << std::flush;
  startTime(&timer);
  Eigen::MatrixXd E = Eigen::MatrixXd::Identity(d, d); // Hack
  MCMC_step(post_x_t, w_t, a_t, y_t, F, G, N, d, timeSteps, resampler_opt, distribution_opt, df);
  stopTime(&timer);
  runtime = elapsedTime(timer);
  Rcpp::Rcout << "Done. " << elapsedTime(timer) << " s" << std::endl;
}

#endif