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
                     dim_t d, dim_t N, dim_t timeSteps, float &runtime,
                     std::string resampler_opt, std::string distribution_opt, float df);

#endif