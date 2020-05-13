#ifndef LINEAR_ALGEBRA_CPP
#define LINEAR_ALGEBRA_CPP

#include <linear_algebra.hpp>

void eigenSolver(Eigen::MatrixXd &I_sol, Eigen::MatrixXd &sigma)
{
  // Find the eigen vectors of the covariance matrix
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(sigma);
  Eigen::MatrixXd eigenvectors = eigen_solver.eigenvectors().real();

  // Find the eigenvalues of the covariance matrix
  Eigen::MatrixXd eigenvalues = eigen_solver.eigenvalues().real().asDiagonal();

  // Find the transformation matrix
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(eigenvalues);
  Eigen::MatrixXd sqrt_eigenvalues = es.operatorSqrt();
  I_sol = eigenvectors * sqrt_eigenvalues;
}

#endif