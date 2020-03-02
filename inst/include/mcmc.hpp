#ifndef __MCMC_HPP
#define __MCMC_HPP

#include <assert.h>
#include <random>
// #include <Eigen/Dense>
#include <RcppEigen.h>
#include <math.h>
#include <omp.h>

#include "types.hpp"
#include "statistics.hpp"
#include "linear_algebra.hpp"
#include "mvn.hpp"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

void generateInput(Eigen::VectorXd *prior_x_t, Eigen::VectorXd *y_t,
                   Eigen::MatrixXd &F, Eigen::MatrixXd &G, Eigen::MatrixXd &I_1,
                   Eigen::MatrixXd &I_2, dim_t N, dim_t d, dim_t timeSteps);
void initialize(Eigen::VectorXd **x_t, Eigen::VectorXd *w_t,
                Eigen::VectorXd m0, Eigen::MatrixXd C0,
                unsigned N, unsigned d, unsigned t);
void MCMC(Eigen::VectorXd **post_x_t, Eigen::VectorXd *w_t, unsigned *a_t,
          Eigen::VectorXd *y_t, Eigen::MatrixXd &F, Eigen::MatrixXd &I, const dim_t N,
          const dim_t d, const dim_t timeSteps);

#endif


#ifndef __MCMC_CPP
#define __MCMC_CPP

void generateInput(Eigen::VectorXd *prior_x_t, Eigen::VectorXd *y_t,
                   Eigen::MatrixXd &F, Eigen::MatrixXd &G, Eigen::MatrixXd &I_1,
                   Eigen::MatrixXd &I_2, dim_t N, dim_t d, dim_t timeSteps)
{
  // Noise e_t
  Eigen::VectorXd e_t(d);
  // = (Eigen::VectorXd *)malloc(sizeof(Eigen::VectorXd) * timeSteps);

  // Noise eps_t
  Eigen::VectorXd eps_t(d);
  // = (Eigen::VectorXd *)malloc(sizeof(Eigen::VectorXd) * timeSteps);

  // Initialize distributions
  Eigen::MatrixXd I_0 = Eigen::MatrixXd::Identity(d, d);
  // I_0 << 1, 0, 0, 0, 1, 0, 0, 0, 1;

  MultiVariateNormalDistribution MVNx(Eigen::VectorXd::Zero(d), I_0);
  MultiVariateNormalDistribution MVN1(Eigen::VectorXd::Zero(d), 0.001 * I_1);
  MultiVariateNormalDistribution MVN2(Eigen::VectorXd::Zero(d), 0.001 * I_2);

  // Calculate x_t[0][N]
  MVNx.sample(prior_x_t[0], 200);

  // Generate input data y_t[t][i]
  for (unsigned t = 1; t < timeSteps; ++t)
  {
    // Generate noise e & eps
    MVN1.sample(e_t, 200);
    MVN2.sample(eps_t, 200);

    prior_x_t[t] = (G * prior_x_t[t - 1]) + eps_t;
    y_t[t] = (F * prior_x_t[t]) + e_t;
  }
}

void initialize(Eigen::VectorXd **x_t, Eigen::VectorXd *w_t,
                Eigen::VectorXd m0, Eigen::MatrixXd C0,
                unsigned N, unsigned d, unsigned t)
{

  // Eigen::VectorXd mean(d);
  // mean.setZero();

  MultiVariateNormalDistribution mvn(m0, C0);

  // Initialize theta
#pragma omp parallel for
  for (unsigned i = 0; i < N; ++i)
    mvn.sample(x_t[0][i], 200);

  // Initialize omega
  w_t[0].fill(1 / double(N));
}

void MCMC(Eigen::VectorXd **post_x_t, Eigen::VectorXd *w_t, unsigned *a_t,
          Eigen::VectorXd *y_t, Eigen::MatrixXd &E, Eigen::MatrixXd &F,
          const dim_t N, const dim_t d, const dim_t timeSteps)
{
  // Solve Covariant Matrix for determinant & inverse
  double E_det = E.determinant();
  Eigen::MatrixXd E_inv = E.inverse();
  Eigen::MatrixXd Q(d, d);
  eigenSolver(Q, E);

  double sqrt2pi = std::sqrt(2 * M_PI);
  double norm = 1.0f / (std::pow(sqrt2pi, (d / 2.0f)) * std::pow(E_det, -0.5));

  for (unsigned t = 1; t < timeSteps; ++t)
  {
    // Calculate ancestors with Metropolis
    metropolis(a_t, w_t, N, t);

    // Propagate particles
    propagate_K(post_x_t, a_t, Q, N, d, t);

    // Resample weights
    reweight_G(w_t, y_t, post_x_t, norm, E_inv, E, F, N, d, t);
  }
}
#endif