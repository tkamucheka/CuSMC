#ifndef LINEAR_ALGEBRA_HPP
#define LINEAR_ALGEBRA_HPP

#include <iostream>
#include <random>
#include <RcppEigen.h>

// Eigen Solver
void eigenSolver(Eigen::MatrixXd &I_sol, const Eigen::MatrixXd &sigma);

#endif