#ifndef __PARTICLE_FILTER_HPP
#define __PARTICLE_FILTER_HPP

// #include <Eigen/Dense>
#include <sys/time.h>
#include <time.h>

#include "types.hpp"
#include "utility.hpp"
#include "mcmc.hpp"

void particle_filter(Eigen::VectorXd **post_x_t,
                     Eigen::VectorXd *w_t, unsigned *a_t,
                     Eigen::VectorXd *y_t,
                     Eigen::MatrixXd F, Eigen::MatrixXd G,
                     Eigen::VectorXd m0, Eigen::MatrixXd C0,
                     //  Eigen::MatrixXd *sigmaV, Eigen::MatrixXd *sigmaW,
                     dim_t d, dim_t N, dim_t timeSteps, float &runtime);

#endif

#ifndef __PARTICLE_FILTER_CPP
#define __PARTICLE_FILTER_CPP

void particle_filter(Eigen::VectorXd **post_x_t,
                     Eigen::VectorXd *w_t, unsigned *a_t,
                     Eigen::VectorXd *y_t,
                     Eigen::MatrixXd F, Eigen::MatrixXd G,
                     Eigen::VectorXd m0, Eigen::MatrixXd C0,
                     //  Eigen::MatrixXd *sigmaV, Eigen::MatrixXd *sigmaW,
                     dim_t d, dim_t N, dim_t timeSteps, float &runtime)
{
  Timer timer;
  // 1. Initialize particles x_t (theta) and weights w_t (omega)
  //Rcpp::Rcout << "Initializing thetas and omegas at t[0]... " << std::flush;
  startTime(&timer);
  initialize(post_x_t, w_t, m0, C0, N, d, 0);
  stopTime(&timer);
  //Rcpp::Rcout << "Done. " << elapsedTime(timer) << " s" << std::endl;

  // 2. MCMC time marching
  //Rcpp::Rcout << "Simulating... " << std::flush;
  startTime(&timer);
  Eigen::MatrixXd E = Eigen::MatrixXd::Identity(d, d); // Hack
  MCMC(post_x_t, w_t, a_t, y_t, E, F, N, d, timeSteps);
  stopTime(&timer);
  runtime = elapsedTime(timer);
  //Rcpp::Rcout << "Done. " << elapsedTime(timer) << " s" << std::endl;
}

#endif