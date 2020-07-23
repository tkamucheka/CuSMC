#ifndef __MCMC_CPP
#define __MCMC_CPP

#include <mcmc.hpp>

void generateInput(Eigen::VectorXd *prior_x_t, Eigen::VectorXd *y_t,
                   Eigen::MatrixXd &F, Eigen::MatrixXd &G, Eigen::MatrixXd &I_1,
                   Eigen::MatrixXd &I_2, dim_t N, dim_t d, dim_t timeSteps)
{
  // Noise e_t
  Eigen::VectorXd e_t(d);
  // = (Eigen::VectorXd *)malloc(sizeof(Eigen::VectorXd) * timeSteps);

  // Noise eps_t
  Eigen::VectorXd eps_t(d);
  // = (Eigen::VectorXd *)malloc(sizeof(Eigen::VectorXd) * timeSteps);

  // Initialize distributions
  Eigen::MatrixXd I_0 = Eigen::MatrixXd::Identity(d, d);
  // I_0 << 1, 0, 0, 0, 1, 0, 0, 0, 1;

  MultiVariateNormalDistribution MVNx(Eigen::VectorXd::Zero(d), I_0);
  MultiVariateNormalDistribution MVN1(Eigen::VectorXd::Zero(d), 0.001 * I_1);
  MultiVariateNormalDistribution MVN2(Eigen::VectorXd::Zero(d), 0.001 * I_2);

  // Calculate x_t[0][N]
  MVNx.sample(prior_x_t[0], 200);

  // Generate input data y_t[t][i]
  for (unsigned t = 1; t < timeSteps; ++t)
  {
    // Generate noise e & eps
    MVN1.sample(e_t, 200);
    MVN2.sample(eps_t, 200);

    prior_x_t[t] = (G * prior_x_t[t - 1]) + eps_t;
    y_t[t] = (F * prior_x_t[t]) + e_t;
  } 
}

template <typename K, typename V>
void print_map(std::map<K, V> const &m)
{
  for (auto it = m.cbegin(); it != m.cend(); ++it)
  {
    Rcpp::Rcout << "{" << (*it).first << ": "
                << "}\n";
  }
}

void initialize(std::string distribution_opt, Eigen::VectorXd **x_t, Eigen::VectorXd *w_t,
                Eigen::VectorXd m0, Eigen::MatrixXd C0,
                unsigned N, unsigned d, unsigned t, float df)
{
  // Distributions
  Distributions["normal"] =
      [](distParams_t params) { return StandardNormalDistribution::getInstance(params); };
  Distributions["mvn"] =
      [](distParams_t params) { return MultiVariateNormalDistribution::getInstance(params); };
  Distributions["mvt"] =
      [](distParams_t params) { return MultiVariateTStudentDistribution::getInstance(params); };

  // assert(Distributions.find(distribution_opt) != Distributions.end());
  // Eigen::VectorXd mean(d);
  // mean.setZero();

  // print_map(Distributions);

  // MultiVariateNormalDistribution mvn(m0, C0); %Quan Mai changed to distribution_opt
  distParams_t params = {m0, C0, df};
  StatisticalDistribution *dist = Distributions[distribution_opt](params);

  // Initialize theta
#pragma omp parallel for
  for (unsigned i = 0; i < N; ++i)
    dist->sample(x_t[0][i], 200);

  // Initialize omega
  w_t[0].fill(1 / double(N));
}

void propagate_K(std::string distribution_opt, Eigen::VectorXd **post_x_t, unsigned *a_t,
                 const Eigen::MatrixXd Q, const dim_t N, const dim_t d,
                 const dim_t t, const float df)
{
  // Distributions
  Distributions["normal"] =
      [](distParams_t params) { return StandardNormalDistribution::getInstance(params); };
  Distributions["mvn"] =
      [](distParams_t params) { return MultiVariateNormalDistribution::getInstance(params); };
  Distributions["mvt"] =
      [](distParams_t params) { return MultiVariateTStudentDistribution::getInstance(params); };

// Initialize distribution with shuffled(theta) x_t
#pragma omp parallel for
  for (unsigned i = 0; i < N; ++i)
  {
    // `distCreator` initializes a new StatisticalDistribution object with the
    // given arguments. The actual is distribution used is selected based on
    // user input. The underlying function call may look like this:
    //
    // MultiVariateNormalDistribution mvn(post_x_t[t - 1][a_t[t * N + i]],
    //                                    Eigen::MatrixXd::Identity(d, d));
    //
    // NOTE: In the future it may be worthwhile to pass the arguments in a
    // `distOptions` object because different distribution may functin with
    // different sets of parameters.

    // Shuffle (thetas) x_t in the indices
    // and initialize new distribution
    distParams_t params = {post_x_t[t - 1][a_t[t * N + i]],
                           Eigen::MatrixXd::Identity(d, d),
                           df};

    StatisticalDistribution *dist = Distributions[distribution_opt](params);

    // Sample new (thetas) x_t
    dist->sample(post_x_t[t][i], 200);

    delete dist;
  }
}

void reweight_G(std::string distributions_opt, Eigen::VectorXd *w_t, const Eigen::VectorXd *y_t,
                Eigen::VectorXd **post_x_t, const double norm,
                const Eigen::MatrixXd &E_inv, const Eigen::MatrixXd E, const Eigen::MatrixXd F,
                const dim_t N, const dim_t d, const dim_t t, const float df)
{
  // Distributions
  Distributions["normal"] =
      [](distParams_t params) { return StandardNormalDistribution::getInstance(params); };
  Distributions["mvn"] =
      [](distParams_t params) { return MultiVariateNormalDistribution::getInstance(params); };
  Distributions["mvt"] =
      [](distParams_t params) { return MultiVariateTStudentDistribution::getInstance(params); };

#pragma omp parallel for
  for (unsigned i = 0; i < N; ++i)
  {
    // `distCreator` initializes a new StatisticalDistribution object with the
    // given arguments. The actual is distribution used is selected based on
    // user input. The underlying function call may look like this:
    //
    // MultiVariateNormalDistribution mvn(F * post_x_t[t][i], E);
    //
    // NOTE: In the future it may be worthwhile to pass the arguments in a
    // `distOptions` object because different distribution may functin with
    // different sets of parameters.

    // Initialize MVN distribution
    // mean  = x[t], covariance matrix sigma = E
    distParams_t params = {post_x_t[t][i], E, df};
    StatisticalDistribution *dist = Distributions[distributions_opt](params);

    // Get new weights from probality density function
    w_t[t][i] = dist->pdf(y_t[t], F);
    
    delete dist;
  }
}

void MCMC(Eigen::VectorXd **post_x_t, Eigen::VectorXd *w_t, unsigned *a_t,
          Eigen::VectorXd *y_t, Eigen::MatrixXd &E, Eigen::MatrixXd &F,
          const dim_t N, const dim_t d, const dim_t timeSteps,
          std::string resampler_opt, std::string distribution_opt, const float df)
{
  // Initialize Resamplers and Distributions
  // Resamplers;
  Resamplers["metropolis"] = [](unsigned *a_t, Eigen::VectorXd *w_t, int N, unsigned t, int B = 10) {
    Sampler::metropolis_hastings(a_t, w_t, N, t, B);
  };

  // Distributions
  Distributions["normal"] =
      [](distParams_t params) { return StandardNormalDistribution::getInstance(params); };
  Distributions["mvn"] =
      [](distParams_t params) { return MultiVariateNormalDistribution::getInstance(params); };
  Distributions["mvt"] =
      [](distParams_t params) { return MultiVariateTStudentDistribution::getInstance(params); };

  // Assert availability of resampler and distribution
  // assert(Resamplers.find(resampler_opt) != Resamplers.end());
  // assert(Distributions.find(distribution_opt) != Distributions.end());

  resampler_f resampler = Resamplers[resampler_opt];

  // Solve Covariant Matrix for determinant & inverse
  //double E_det = E.determinant();
  Eigen::MatrixXd E_inv = E.inverse();
  Eigen::MatrixXd Q(d, d);
  eigenSolver(Q, E);

  // Get norm from distribution
  distParams_t params = {Eigen::VectorXd::Zero(d), E, df};
  StatisticalDistribution *dist = Distributions[distribution_opt](params);
  double norm = dist->getNorm();

  int B = 10;
  for (unsigned t = 1; t < timeSteps; ++t)
  {
    // Calculate ancestors with Metropolis
    resampler(a_t, w_t, N, t, B);

    // Propagate particles
    propagate_K(distribution_opt, post_x_t, a_t, Q, N, d, t, df);

    // Resample weights
    reweight_G(distribution_opt, w_t, y_t, post_x_t, norm, E_inv, E, F, N, d, t, df);
  }
}

void MCMC_step(Eigen::VectorXd **post_x_t, Eigen::VectorXd *w_t, unsigned *a_t,
               Eigen::VectorXd *y_t, Eigen::MatrixXd &E, Eigen::MatrixXd &F,
               const dim_t N, const dim_t d, const dim_t timeSteps,
               std::string resampler_opt, std::string distribution_opt, const float df)
{
  // Initialize Resamplers and Distributions
  // Resamplers;
  Resamplers["metropolis"] = [](unsigned *a_t, Eigen::VectorXd *w_t, int N, unsigned t, int B = 10) {
    Sampler::metropolis_hastings(a_t, w_t, N, t, B);
  };

  // Distributions
  Distributions["normal"] =
      [](distParams_t params) { return StandardNormalDistribution::getInstance(params); };
  Distributions["mvn"] =
      [](distParams_t params) { return MultiVariateNormalDistribution::getInstance(params); };
  Distributions["mvt"] =
      [](distParams_t params) { return MultiVariateTStudentDistribution::getInstance(params); };

  // Assert availability of resampler and distribution
  // assert(Resamplers.find(resampler_opt) != Resamplers.end());
  // assert(Distributions.find(distribution_opt) != Distributions.end());

  resampler_f resampler = Resamplers[resampler_opt];

  // Solve Covariant Matrix for determinant & inverse
  //double E_det = E.determinant();
  Eigen::MatrixXd E_inv = E.inverse();
  Eigen::MatrixXd Q(d, d);
  eigenSolver(Q, E);

  // Get norm from distribution
  distParams_t params = {Eigen::VectorXd::Zero(d), E, df};
  StatisticalDistribution *dist = Distributions[distribution_opt](params);
  double norm = dist->getNorm();

  int B = 10;
  for (unsigned t = 1; t < timeSteps; ++t)
  {
    // Calculate ancestors with Metropolis
    resampler(a_t, w_t, N, t, B);

    // Propagate particles
    propagate_K(distribution_opt, post_x_t, a_t, Q, N, d, t, df);

    // Resample weights
    reweight_G(distribution_opt, w_t, y_t, post_x_t, 5.0, E_inv, E, F, N, d, t, df);
  }
}

#endif