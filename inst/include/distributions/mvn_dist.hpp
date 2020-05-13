#ifndef __MVN_DIST_HPP
#define __MVN_DIST_HPP

// Global headers
#include <RcppEigen.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <random>

// Local headers
#include "../types.hpp"
#include "../support.cuh"

void metropolis(unsigned *a_t, Eigen::VectorXd *w_t, int N, unsigned t);
void propagate_K(Eigen::VectorXd **post_x_t, unsigned *a_t,
                 const Eigen::MatrixXd Q, const dim_t N, const dim_t d,
                 const dim_t t, const float df = 0.0f);
void reweight_G(Eigen::VectorXd *w_t, const Eigen::VectorXd *y_t,
                Eigen::VectorXd **post_x_t, const double norm,
                const Eigen::MatrixXd &E_inv, const Eigen::MatrixXd E, const Eigen::MatrixXd F,
                const dim_t N, const dim_t d, const dim_t t, const float df = 0.0f);

#endif