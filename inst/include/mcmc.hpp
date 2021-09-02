#ifndef __MCMC_HPP
#define __MCMC_HPP

// Global headers
#include <assert.h>
#include <random>
#include <RcppEigen.h>
#include <math.h>
#include <omp.h>
#include <map>
#include <string>

// Local headers
#include "types.hpp"
#include "linear_algebra.hpp"
#include "statistics.hpp"
#include "samplers.hpp"
// #include "distributions/mvn_dist.hpp"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

static resamplers_t Resamplers;
static distributions_t Distributions;

void generateInput(
    Eigen::VectorXd *prior_x_t, Eigen::VectorXd *y_t,
    Eigen::MatrixXd &F, Eigen::MatrixXd &G, Eigen::MatrixXd &I_1, Eigen::MatrixXd &I_2,
    dim_t N, dim_t d, dim_t timeSteps);

void initialize(
    Eigen::VectorXd **x_t, Eigen::VectorXd *w_t,
    Eigen::VectorXd m0, Eigen::MatrixXd C0,
    unsigned N, unsigned d, unsigned t, float df, std::string distribution_opt);

void propagate_K(
    Eigen::VectorXd **post_x_t, unsigned *a_t,
    const Eigen::MatrixXd G, const Eigen::MatrixXd Q_w,
    const dim_t N, const dim_t d, const dim_t t,
    const float df,
    std::string distribution_opt);

void reweight_G(
    Eigen::VectorXd *w_t, const Eigen::VectorXd *y_t,
    Eigen::VectorXd **post_x_t,
    const Eigen::MatrixXd &F,
    const Eigen::MatrixXd &V_det,
    const Eigen::MatrixXd &V_inv,
    const dim_t N, const dim_t d, const dim_t t,
    const float df,
    std::string distributions_opt);

void MCMC(
    Eigen::VectorXd **post_x_t,
    Eigen::VectorXd *w_t, unsigned *a_t,
    Eigen::VectorXd *y_t,
    const Eigen::MatrixXd &F, const Eigen::MatrixXd &G,
    const Eigen::matrixXd &V, const Eigen::MatrixXd &W,
    const dim_t N, const dim_t d, const dim_t timeSteps, const float df,
    std::string resampler_opt,
    std::string distribution_opt);

void MCMC_step(
    Eigen::VectorXd **post_x_t,
    Eigen::VectorXd *w_t, unsigned *a_t,
    Eigen::VectorXd *y_t,
    const Eigen::MatrixXd &F, const Eigen::MatrixXd &G,
    const Eigen::matrixXd &V, const Eigen::MatrixXd &W,
    const dim_t N, const dim_t d, const dim_t timeSteps, const float df,
    std::string resampler_opt,
    std::string distribution_opt);

#endif