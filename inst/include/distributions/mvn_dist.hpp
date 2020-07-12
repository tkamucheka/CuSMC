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

void mvn_sample_kernel_wrapper(Eigen::VectorXd &draws,
                               const Eigen::VectorXd &mu,
                               const Eigen::MatrixXd &Q,
                               const dim_t d);
void mvn_pdf_kernel_wrapper(double *w,
                            const Eigen::VectorXd &y,
                            const Eigen::VectorXd &mu,
                            const Eigen::MatrixXd &E_inv,
                            const Eigen::MatrixXd &F,
                            const double norm,
                            const dim_t d);

#endif

#endif