#ifdef __GPU

#ifndef __MVN_DIST_HPP
#define __MVN_DIST_HPP

// Global headers
#include <RcppEigen.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <time.h>
#include <stdlib.h>

// Local headers
#include "../types.hpp"
#include "../support.cuh"

void mvn_pdf_kernel_wrapper(Eigen::VectorXd &w_t,
                            const Eigen::VectorXd *y,
                            Eigen::VectorXd **post_x_t,
                            const double norm,
                            const Eigen::MatrixXd &E_inv,
                            const Eigen::MatrixXd &F,
                            const dim_t N, const dim_t d,
                            const dim_t t);

void mvn_sample_kernel_wrapper(Eigen::VectorXd **post_x_t, 
                               unsigned *a_t, const Eigen::MatrixXd G, 
                               const Eigen::MatrixXd Q,
                               const dim_t N, const dim_t d, const dim_t t);

void mvn_sample_kernel_wrapper(Eigen::VectorXd *post_x_t,
                            const Eigen::VectorXd mu,
                            const Eigen::MatrixXd Q,
                            const dim_t N, const dim_t d);                               

void mvt_pdf_kernel_wrapper(Eigen::VectorXd &w_t,
                            const Eigen::VectorXd *y_t,
                            Eigen::VectorXd **post_x_t,
                            const Eigen::MatrixXd &E_inv,
                            const Eigen::MatrixXd &F,
                            const double norm,
                            const dim_t N, const dim_t d,
                            const dim_t t,
                            const float df);

void mvt_sample_kernel_wrapper(
    Eigen::VectorXd **post_x_t,
    unsigned *a_t,
    const Eigen::MatrixXd G,
    const Eigen::MatrixXd Q,
    const dim_t N, const dim_t d, const dim_t t,
    const float df);

#endif

#endif