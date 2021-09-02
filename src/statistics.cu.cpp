#ifdef __GPU

#ifndef __STATISTICS_CPP
#define __STATISTICS_CPP

#include "../inst/include/statistics.hpp"
#include "../inst/include/distributions/mvn_dist.hpp"

// Base class constructor/destructor
// StatisticalDistribution::StatisticalDistribution() {}
// StatisticalDistribution::~StatisticalDistribution() {}

// Standard Normal Distribution ===============================================

// Standard Normal Distribution constructor/destructor
StandardNormalDistribution::StandardNormalDistribution(const double &m,
                                                       const double &s)
{
  mu = m;
  sigma = s;
}
StandardNormalDistribution::~StandardNormalDistribution() {}

// Return class instance
StandardNormalDistribution *StandardNormalDistribution::getInstance(const distParams_t dist)
{
  // StandardNormalDistribution N(mu, sigma);
  return new StandardNormalDistribution(dist.mu[0], dist.sigma(0, 0));
}

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
}
MultiVariateNormalDistribution::~MultiVariateNormalDistribution(){};

// Return class instance
MultiVariateNormalDistribution *MultiVariateNormalDistribution::getInstance(const distParams_t params)
{
  // MultiVariateNormalDistribution MVN(mu, sigma);
  MultiVariateNormalDistribution MVN = new MultiVariateNormalDistribution(params.mu, params.sigma);
  MVN->params = params;

  return MVN;
}

// Distribution functions
Eigen::VectorXd MultiVariateNormalDistribution::pdf_cu(const Eigen::VectorXd &y, const Eigen::MatrixXd &F) const
{
  // unsigned int n = mu.rows();
  // double sqrt2pi = std::sqrt(2 * M_PI);
  double norm = this->getNorm();

  Eigen::VectorXd w = Eigen::VectorXd::Zero(this->params.N);
  mvn_pdf_kernel_wrapper(w, y, mu, sigma_inv, F, norm,
                         this->params.N, this->params.d, this->params.t);

  // DEBUG:
  // Rcpp::Rcout << "PDF: " << w << std::endl;

  return w;
};

double MultiVariateNormalDistribution::pdf() {}

//Calculate constant norm
double
MultiVariateNormalDistribution::getNorm() const
{
  int n = this->mu.rows();

  double sqrt2pi = std::sqrt(2 * M_PI);
  return 1.0f / (std::pow(sqrt2pi, n) *
                 std::pow(sigma_det, 0.5));
}

double MultiVariateNormalDistribution::cdf() const { return 0.0f; };

// Inverse cumulative distribution function
double MultiVariateNormalDistribution::inv_cdf(const double &quantile) const { return 0.0f; };

// Descriptive stats
Eigen::VectorXd MultiVariateNormalDistribution::mean() const { return mu; };
// Eigen::VectorXd MultiVariateNormalDistribution::var() const { return mu; };
Eigen::VectorXd MultiVariateNormalDistribution::stdev() const { return sigma; };

// Random draw function
void MultiVariateNormalDistribution::sample(
    Eigen::VectorXd **post_x_t,
    unsigned *a_t,
    const Eigen::MatrixXd G,
    const Eigen::MatrixXd Q,
    const dim_t N,
    const dim_t d,
    const dim_t t)
{
  mvn_sample_kernel_wrapper(post_x_t, a_t, G, Q, N, d, t);
};

// Multi-Variate T Student Distribution ====================================

MultiVariateTStudentDistribution::MultiVariateTStudentDistribution(
    const Eigen::VectorXd &m,
    const Eigen::MatrixXd &s,
    const float &nu)
{
  this->mu = m;    //location vector
  this->sigma = s; //scale matrix
  this->nu = nu;   //degree of freedom
};
MultiVariateTStudentDistribution::~MultiVariateTStudentDistribution(){};

// Return class instance
MultiVariateTStudentDistribution *MultiVariateTStudentDistribution::getInstance(const distParams_t params)
{
  // MultiVariateTStudentDistribution MVT(mu, sigma, nu);
  return new MultiVariateTStudentDistribution(params.mu, params.sigma, params.nu);
}

// Distribution functions
Eigen::VectorXd MultiVariateTStudentDistribution::pdf_cu(const Eigen::VectorXd &y, const Eigen::MatrixXd &F) const
{
  double norm = this->getNorm();

  Eigen::VectorXd w = Eigen::VectorXd::Zero(this->params.N);
  mvt_pdf_kernel_wrapper(w, y, mu, sigma_inv, F, norm,
                         this->params.N, this->params.d, this->params.t,
                         df);

  // DEBUG:
  // Rcpp::Rcout << "PDF: " << w << std::endl;

  return w;
};

double MultiVariateTStudentDistribution::pdf(const Eigen::VectorXd &y,
                                             const Eigen::VectorXd &x,
                                             const Eigen::MatrixXd &E) const
{
  return 0.0f;
};

//Calculate constant norm
double MultiVariateTStudentDistribution::getNorm() const
{
  int n = this->mu.rows();
  double pi_df = M_PI * this->nu;
  double norm1 = std::pow(this->nu, (-0.5 * n)) * std::pow(this->sigma_det, -0.5);
  double norm2 = tgamma(0.5 * (this->nu + n)) / tgamma(0.5 * this->nu);

  return norm1 * norm2;
};

double MultiVariateTStudentDistribution::cdf() const { return 0.0f; };

// Inverse cumulative distribution function
double MultiVariateTStudentDistribution::inv_cdf(const double &quantile) const { return 0.0f; };

// Descriptive stats
Eigen::VectorXd MultiVariateTStudentDistribution::mean() const { return mu; }; //note that this is location vector, not the mean vector!

Eigen::VectorXd MultiVariateTStudentDistribution::stdev() const { return sigma; }; //note that this is scale matrix, not the covarriance matrix which is df/(df-2)*sigma

float MultiVariateTStudentDistribution::dfree() const { return nu; };

// Random draw function
void MultiVariateTStudentDistribution::sample(
    Eigen::VectorXd **post_x_t,
    unsigned *a_t,
    const Eigen::MatrixXd Q,
    const dim_t N, const dim_t d, const dim_t t,
    const float df) const
{
  mvt_sample_kernel_wrapper(post_x_t, a_t, Q, N, d, t, df);
};

#endif

#endif