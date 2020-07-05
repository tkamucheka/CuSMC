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

void mvn_sample_kernel_wrapper(double Eigen::VectorXd *draws,
                               double Eigen::VectorXd *mu, const Eigen::MatrixXd Q,
                               const dim_t d);
void mvn_pdf_kernel_wrapper(const Eigen::VectroXd *y,
                            Eigen::VectorXd *mu,
                            const Eigen::MatrixXd &E_Inv,
                            const Eigen::MatrixXd F,
                            const double norm,
                            const dim_t d)

#endif