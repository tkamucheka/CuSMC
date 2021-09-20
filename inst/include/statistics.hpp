//#ifndef __GPU

#ifndef __STATISTICS_HPP
#define __STATISTICS_HPP

// Global includes
#include <cmath>
#include <vector>
#include <random>
#include <omp.h>
#include <RcppEigen.h>

// Local includes
#include <types.hpp>
// #include "distributions/mvn_dist.hpp"
// TODO: Correct API in mvt_dist and include
// #include "distributions/mvt_dist.hpp"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

// Distribution initializations options object
struct distParams_t
{
  Eigen::VectorXd **post_x_t;
  unsigned *a_t;
  Eigen::VectorXd mu;
  Eigen::MatrixXd sigma;
  double sigma_det;
  Eigen::MatrixXd sigma_inv;
  float nu = 0.0f;
  Eigen::MatrixXd Q;
  Eigen::MatrixXd Q_w;
  int N, d, t;
};

class StatisticalDistribution
{
public:
  StatisticalDistribution();
  virtual ~StatisticalDistribution() = default;

  // Return class instance
  template <class M, class S>
  void init(const M, const S) {}
  template <class M, class S, class N>
  void init(const M, const S, const N) {}
  static StatisticalDistribution *getInstance() { return new StatisticalDistribution(); }

  // Distribution functions
  // Probability Density Function
  virtual double pdf(const double &x) const { return 0.0f; }
  virtual double pdf(const Eigen::VectorXd &x) const { return 0.0f; }
  virtual double pdf(const Eigen::VectorXd &x, const Eigen::MatrixXd &F) const { return 0.0f; }
  virtual double getNorm() const { return 0.0f; }
  // Cumulative Distribution Function
  double cdf(const double &x) const;

  // Inverse cumulative distribution functions
  double inv_pdf(const double &quantile);

  // Descriptive stats
  double mean();
  double var();
  double stdev();

  // Random draw function
  virtual void sample(const std::vector<double> &uniform_draws,
                      std::vector<double> &dist_draws) const;
  // MVN & MVT
  virtual void sample(Eigen::VectorXd &dist_draws,
                      const unsigned int n_iterations) const;
  virtual void sample(Eigen::VectorXd &dist_draws,
                      const Eigen::MatrixXd &x,
                      const unsigned int n_iterations) const;
};

class StandardNormalDistribution : public StatisticalDistribution
{
private:
  double mu;    // mean of distribution
  double sigma; // standard deviation of distribution

public:
  StandardNormalDistribution();
  StandardNormalDistribution(const double &mu = 0, const double &sigma = 1);
  virtual ~StandardNormalDistribution();

  // Return class instance
  // static StandardNormalDistribution getInstance() { return new StandardNormalDistribution(); };
  static StandardNormalDistribution *getInstance(const distParams_t);
  void init(const double &mu, const double &sigma)
  {
    this->mu = mu;
    this->sigma = sigma;
  };

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

class MultiVariateNormalDistribution : public StatisticalDistribution
{
private:
  Eigen::VectorXd mu;    // mean of distribution
  Eigen::MatrixXd sigma; // standard deviation of distribution

public:
  distParams_t params;

  MultiVariateNormalDistribution();
  MultiVariateNormalDistribution(const Eigen::VectorXd &mu,
                                 const Eigen::MatrixXd &sigma);
  ~MultiVariateNormalDistribution();

  // Return class instance
  static MultiVariateNormalDistribution *getInstance(const distParams_t);
  // static MultiVariateNormalDistribution *getInstance(const Eigen::VectorXd &mu,
  //                                                    const Eigen::MatrixXd &sigma);
  void init(const Eigen::VectorXd &mu, const Eigen::MatrixXd &sigma)
  {
    this->mu = mu;
    this->sigma = sigma;
  };

  // Distribution functions
  Eigen::VectorXd pdf_cu(const Eigen::VectorXd *y, //pdf_cu
                         const Eigen::VectorXd **post_x_t, 
                         const Eigen::MatrixXd &F) const;
  double pdf(const Eigen::VectorXd &x) const;
  double pdf(const Eigen::VectorXd &x, const Eigen::MatrixXd &F) const;
  // double pdf(const Eigen::VectorXd &y,
  //            const Eigen::VectorXd &x,
  //            const Eigen::MatrixXd &s) const;
  double getNorm() const;
  double cdf() const;

  // Inverse cumulative distribution function
  double inv_cdf(const double &quantile) const;

  // Descriptive stats
  Eigen::VectorXd mean() const;
  Eigen::VectorXd var() const;
  Eigen::VectorXd stdev() const;

  // Random draw function
  void sample(Eigen::VectorXd &dist_draws,
              const Eigen::MatrixXd &Q,
              const unsigned int n_iterations) const;
  void sample(
    Eigen::VectorXd **post_x_t,
    unsigned *a_t,
    const Eigen::MatrixXd G,
    const Eigen::MatrixXd Q,
    const dim_t N,
    const dim_t d,
    const dim_t t);
};

class MultiVariateTStudentDistribution : public StatisticalDistribution
{
private:
  Eigen::VectorXd mu;    // mean of distribution
  Eigen::MatrixXd sigma; // standard deviation of distribution
  float nu;

public:
  distParams_t params;
  
  MultiVariateTStudentDistribution();
  MultiVariateTStudentDistribution(const Eigen::VectorXd &mu,
                                   const Eigen::MatrixXd &sigma,
                                   const float &nu);
  ~MultiVariateTStudentDistribution();

  // Return class instance
  static MultiVariateTStudentDistribution *getInstance(const distParams_t);
  // static MultiVariateTStudentDistribution *getInstance(const Eigen::VectorXd &mu,
  //                                                      const Eigen::MatrixXd &sigma,
  //                                                      const float &nu);
  void init(const Eigen::VectorXd &mu, const Eigen::MatrixXd &sigma, const float &nu)
  {
    this->mu = mu;
    this->sigma = sigma;
    this->nu = nu;
  }

  // Distribution functions
  Eigen::VectorXd pdf_cu(const Eigen::VectorXd *y,
                         const Eigen::VectorXd **post_x_t,
                         const Eigen::MatrixXd &F) const;
  double pdf(const Eigen::VectorXd &y, const Eigen::MatrixXd &F) const;
  double pdf(const Eigen::VectorXd &y) const;
  // double pdf(const Eigen::VectorXd &y,
  //            const Eigen::VectorXd &x,
  //            const Eigen::MatrixXd &E) const;
  double getNorm() const;
  double cdf() const;

  // Inverse cumulative distribution function
  double inv_cdf(const double &quantile) const;

  // Descriptive stats
  Eigen::VectorXd mean() const;
  Eigen::VectorXd stdev() const;
  float dfree() const;

  // Random draw function
  void sample(Eigen::VectorXd &dist_draws,
              const Eigen::MatrixXd Q,
              const unsigned int n_iterations) const;
  void sample( //Cuda sampling
    Eigen::VectorXd **post_x_t,
    unsigned *a_t,
    const Eigen::MatrixXd G,
    const Eigen::MatrixXd Q,
    const dim_t N, const dim_t d, const dim_t t) const;
};

#endif // __STATISTICS_HPP
//#endif // __GPU