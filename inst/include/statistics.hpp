#ifndef __STATISTICS_HPP
#define __STATISTICS_HPP

#include <cmath>
#include <vector>
#include <random>
#include <omp.h>

#include "types.hpp"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

class StatisticalDistribution
{
public:
  StatisticalDistribution();
  ~StatisticalDistribution();

  // Distribution functions
  // Probability Density Function
  double pdf(const double &x);
  // Cumulative Distribution Function
  double cdf(const double &x);

  // Inverse cumulative distribution functions
  double inv_pdf(const double &quantile);

  // Descriptive stats
  double mean();
  double var();
  double stdev();

  // Random draw function
  void sample(const std::vector<double> &uniform_draws,
              std::vector<double> &dist_draws);
};

class StandardNormalDistribution : public StatisticalDistribution
{
public:
  double mu;    // mean of distribution
  double sigma; // standard deviation of distribution

  StandardNormalDistribution(const double &mu = 0, const double &sigma = 1);
  virtual ~StandardNormalDistribution();

  // Distribution functions
  double pdf(const double &x) const;
  double cdf(const double &x) const;

  // Inverse cumulative distribution function
  double inv_cdf(const double &quantile) const;

  // Descriptive stats
  double mean() const;  // = 0
  double var() const;   // = 1
  double stdev() const; // = 1

  // Random draw function
  void sample(const std::vector<double> &uniform_draws,
              std::vector<double> &dist_draws) const;
};

class MultiVariateNormalDistribution : StatisticalDistribution
{
public:
  Eigen::VectorXd mu;    // mean of distribution
  Eigen::MatrixXd sigma; // standard deviation of distribution

  MultiVariateNormalDistribution(const Eigen::VectorXd &mu,
                                 const Eigen::MatrixXd &sigma);
  ~MultiVariateNormalDistribution();

  // Distribution functions
  double pdf(const Eigen::VectorXd &x) const;
  double pdf(const Eigen::VectorXd &y,
             const Eigen::VectorXd &x,
             const Eigen::MatrixXd &s) const;
  double cdf() const;

  // Inverse cumulative distribution function
  double inv_cdf(const double &quantile) const;

  // Descriptive stats
  Eigen::VectorXd mean() const;
  Eigen::VectorXd var() const;
  Eigen::VectorXd stdev() const;

  // Random draw function
  void sample(Eigen::VectorXd &dist_draws,
              const unsigned int n_iterations);

  void sample(Eigen::VectorXd &dist_draws,
              const Eigen::VectorXd &x,
              const unsigned int n_iterations);
};

#endif

#ifndef __STATISTICS_CPP
#define __STATISTICS_CPP

// Base class constructor/destructor
StatisticalDistribution::StatisticalDistribution() {}
StatisticalDistribution::~StatisticalDistribution() {}

// Standard Normal Distribution ===============================================

// Standard Normal Distribution constructor/destructor
StandardNormalDistribution::StandardNormalDistribution(const double &m,
                                                       const double &s)
{
  mu = m;
  sigma = s;
}
StandardNormalDistribution::~StandardNormalDistribution() {}

// Standard Normal Distribution probability density function
// double StandardNormalDistribution::pdf(const double &x) const
// {
//   return exp(-0.5f * x * x) / sqrt(2.0f * M_PI);
// }

// Standard Normal Distribution probability density function
double StandardNormalDistribution::pdf(const double &x) const
{
  return exp(-1.0f * (x - mu) * (x - mu) / (2.0f * sigma * sigma)) /
    (sigma * sqrt(2 * M_PI));
}

// Standard Normal Distrbution cumulative distribution function
double StandardNormalDistribution::cdf(const double &x) const
{
  // Integral from a to x
  const double neg_inf = -1.0f; // Dependant on distribution
  const double N = 1e5;         // Tune for accuracy/speed
  const double c = (x - neg_inf) / N;
  double sum = 0;
  
  for (int k = 1; k < N - 1; ++k)
    sum += pdf(neg_inf + k * c);
  
  return c * ((pdf(x) + pdf(neg_inf)) / 2 + sum);
}

// Inverse cumulative distribution function (aka the probit function)
double StandardNormalDistribution::inv_cdf(const double &quantile) const
{
  // This is the Beasley-Springer-Moro algorithm which can
  // be found in Glasserman [2004]. We won't go into the
  // details here, so have a look at the reference for more info
  static double a[4] = {2.50662823884,
                        -18.61500062529,
                        41.39119773534,
                        -25.44106049637};
  
  static double b[4] = {-8.47351093090,
                        23.08336743743,
                        -21.06224101826,
                        3.13082909833};
  
  static double c[9] = {0.3374754822726147,
                        0.9761690190917186,
                        0.1607979714918209,
                        0.0276438810333863,
                        0.0038405729373609,
                        0.0003951896511919,
                        0.0000321767881768,
                        0.0000002888167364,
                        0.0000003960315187};
  
  if (quantile >= 0.5 && quantile <= 0.92)
  {
    double num = 0.0;
    double denom = 1.0;
    
    for (int i = 0; i < 4; i++)
    {
      num += a[i] * pow((quantile - 0.5), 2 * i + 1);
      denom += b[i] * pow((quantile - 0.5), 2 * i);
    }
    return num / denom;
  }
  else if (quantile > 0.92 && quantile < 1)
  {
    double num = 0.0;
    
    for (int i = 0; i < 9; i++)
    {
      num += c[i] * pow((log(-log(1 - quantile))), i);
    }
    return num;
  }
  else
  {
    return -1.0 * inv_cdf(1 - quantile);
  }
}

// Expectation/mean
double StandardNormalDistribution::mean() const { return 0.0; }

// Variance
double StandardNormalDistribution::var() const { return 1.0; }

// Standard Deviation
double StandardNormalDistribution::stdev() const { return 1.0; }

// Obtain a sequence of random draws from this distribution
void StandardNormalDistribution::sample(const std::vector<double> &uniform_draws,
                                        std::vector<double> &dist_draws) const
{
  // The simplest method to calculate this is with the Box-Muller method,
  // which has been used procedurally in many other articles on QuantStart.com
  
  // Check that the uniform draws and dist_draws are the same size and
  // have an even number of elements (necessary for B-M)
  if (uniform_draws.size() != dist_draws.size())
  {
    Rcpp::Rcout << "Draws vectors are of unequal size in standard normal dist." << std::endl;
    return;
  }
  
  // Check that uniform draws have an even number of elements (necessary for B-M)
  if (uniform_draws.size() % 2 != 0)
  {
    Rcpp::Rcout << "Uniform draw vector size not an even number." << std::endl;
    return;
  }
  
  // Slow, but easy to implement
  for (int i = 0; i < uniform_draws.size() / 2; i++)
  {
    dist_draws[2 * i] = sqrt(-2.0 * log(uniform_draws[2 * i])) * sin(2 * M_PI * uniform_draws[2 * i + 1]);
    dist_draws[2 * i + 1] = sqrt(-2.0 * log(uniform_draws[2 * i])) * cos(2 * M_PI * uniform_draws[2 * i + 1]);
  }
  
  return;
}

// Multi-Variate Normal Normal Distribution ====================================

MultiVariateNormalDistribution::MultiVariateNormalDistribution(const Eigen::VectorXd &m,
                                                               const Eigen::MatrixXd &s)
{
  mu = m;
  sigma = s;
};
MultiVariateNormalDistribution::~MultiVariateNormalDistribution(){};

// Distribution functions
double MultiVariateNormalDistribution::pdf(const Eigen::VectorXd &x) const
{
  dim_t n = x.rows();
  // Eigen::VectorXd x_mu = x - mu;
  double sqrt2pi = std::sqrt(2 * M_PI);
  double quadform = (x - mu).transpose() * sigma.inverse() * (x - mu);
  double norm = 1.0f / (std::pow(sqrt2pi, (n / 2.0f)) *
                        std::pow(sigma.determinant(), -0.5));

  return norm * exp(-0.5 * quadform);
};

double MultiVariateNormalDistribution::pdf(const Eigen::VectorXd &y,
                                           const Eigen::VectorXd &x,
                                           const Eigen::MatrixXd &E) const
{
  // Consider reimplementing following mthod below:
  //
  // function pdf = mvnpdf(x, mu, sigma)
  //    [d, p] = size(x);
  //    % mu can be a scalar, a 1xp vector or a nxp matrix
  //    if nargin == 1, mu = 0; end
  //    if all(size(mu) == [ 1, p ]), mu = repmat(mu, [ d, 1 ]); end
  //    if nargin < 3
  //      pdf = (2 * pi) ^ (-p / 2) * exp(-sumsq(x - mu, 2) / 2);
  //    else
  //      r = chol(sigma);
  //      pdf = (2 * pi) ^ (-p / 2) * exp(-sumsq((x - mu) / r, 2) / 2) / prod(diag(r));
  // end
  
  // dim_t n = x.rows();
  // double sqrt2pi = std::sqrt(2 * M_PI);
  // double norm = std::pow(sqrt2pi, -n) * std::pow(E.determinant(), -0.5);
  // 
  // double quadform = (y - x).transpose() * E.inverse() * (y - x);
  // 
  // return norm * exp(-0.5 * quadform);
  return 0.0f;
};

double MultiVariateNormalDistribution::cdf() const { return 0.0f; };

// Inverse cumulative distribution function
double MultiVariateNormalDistribution::inv_cdf(const double &quantile) const { return 0.0f; };

// Descriptive stats
Eigen::VectorXd MultiVariateNormalDistribution::mean() const { return mu; };
// Eigen::VectorXd MultiVariateNormalDistribution::var() const { return mu; };
Eigen::VectorXd MultiVariateNormalDistribution::stdev() const { return sigma; };

// Random draw function
void MultiVariateNormalDistribution::sample(Eigen::VectorXd &dist_draws,
                                            const unsigned int n_iterations)
{
  // Generator
  std::random_device randomDevice{};
  std::mt19937 generator{randomDevice()};
  // Uniform Distribution between 0 and 1
  std::uniform_real_distribution<double> U{0, 1};
  
  std::normal_distribution<double> N{0, 1};
  
  dim_t n = mu.rows();
  
  // Generate x from the N(0, I) distribution
  Eigen::VectorXd x(n);
  Eigen::VectorXd sum(n);
  sum.setZero();
  
  for (unsigned int i = 0; i < n_iterations; i++)
  {
#pragma omp parallel for
    for (unsigned j = 0; j < n; ++j)
      x[j] = N(generator);
    
    // x.setRandom();
    x = 0.5 * (x + Eigen::VectorXd::Ones(n));
    sum = sum + x;
  }
  sum = sum - (static_cast<double>(n_iterations) / 2) * Eigen::VectorXd::Ones(n);
  x = sum / (std::sqrt(static_cast<double>(n_iterations) / 12));
  
  // x[0] = N(generator);
  // x[1] = N(generator);
  
  // Find the eigen vectors of the covariance matrix
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>
    eigen_solver(sigma);
  Eigen::MatrixXd eigenvectors = eigen_solver.eigenvectors().real();
  
  // Find the eigenvalues of the covariance matrix
  Eigen::MatrixXd eigenvalues = eigen_solver.eigenvalues().real().asDiagonal();
  
  // Find the transformation matrix
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(eigenvalues);
  Eigen::MatrixXd sqrt_eigenvalues = es.operatorSqrt();
  Eigen::MatrixXd Q = eigenvectors * sqrt_eigenvalues;
  
  dist_draws = (Q * x) + mu;
};

void MultiVariateNormalDistribution::sample(Eigen::VectorXd &dist_draws,
                                            const Eigen::VectorXd &x,
                                            const unsigned int n_iterations)
{
  // Find the eigen vectors of the covariance matrix
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(sigma);
  Eigen::MatrixXd eigenvectors = eigen_solver.eigenvectors().real();
  
  // Find the eigenvalues of the covariance matrix
  Eigen::MatrixXd eigenvalues = eigen_solver.eigenvalues().real().asDiagonal();
  
  // Find the transformation matrix
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(eigenvalues);
  Eigen::MatrixXd sqrt_eigenvalues = es.operatorSqrt();
  Eigen::MatrixXd Q = eigenvectors * sqrt_eigenvalues;
  
  dist_draws = (Q * x) + mu;
};

#endif