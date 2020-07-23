#ifndef __IO_HPP
#define __IO_HPP

#include <Eigen/Dense>
#include <fstream>

#include "types.hpp"

void writeOutput(Eigen::VectorXd(*prior_x_t), Eigen::VectorXd *y_t,
                 Eigen::VectorXd *w_t, Eigen::VectorXd **post_x_t,
                 const dim_t N, const dim_t d, const dim_t timeSteps);

#endif