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

  Eigen::MatrixXd E1 = 0.001 * I_1;
  Eigen::MatrixXd E2 = 0.001 * I_2;

  MultiVariateNormalDistribution MVNx(Eigen::VectorXd::Zero(d), I_0);
  MultiVariateNormalDistribution MVN1(Eigen::VectorXd::Zero(d), I_1);
  MultiVariateNormalDistribution MVN2(Eigen::VectorXd::Zero(d), I_2);

  // Calculate x_t[0][N]
  MVNx.sample(prior_x_t[0], I_0, 200);

  // Generate input data y_t[t][i]
  for (unsigned t = 1; t < timeSteps; ++t)
  {
    // Generate noise e & eps
    MVN1.sample(e_t, E1, 200);
    MVN2.sample(eps_t, E2, 200);

    prior_x_t[t] = (G * prior_x_t[t - 1]) + eps_t;
    y_t[t] = (F * prior_x_t[t]) + e_t;
  }
}

void initialize(
    Eigen::VectorXd **x_t, Eigen::VectorXd *w_t,
    Eigen::VectorXd m0, Eigen::MatrixXd C0,
    unsigned N, unsigned d, unsigned t, float df, std::string distribution_opt)
{
  // Distributions
  Distributions["normal"] =
      [](distParams_t params)
  { return StandardNormalDistribution::getInstance(params); };
  Distributions["mvn"] =
      [](distParams_t params)
  { return MultiVariateNormalDistribution::getInstance(params); };
  Distributions["mvt"] =
      [](distParams_t params)
  { return MultiVariateTStudentDistribution::getInstance(params); };

  // assert(Distributions.find(distribution_opt) != Distributions.end());

  // MultiVariateNormalDistribution mvn(m0, C0); %Quan Mai changed to distribution_opt
  distParams_t params;
  params.mu = m0;
  params.sigma = C0;
  params.nu = df;

  // Get Q
  eigenSolver(Q, C0);

  StatisticalDistribution *dist = Distributions[distribution_opt](params);

  // Initialize thetas
#pragma omp parallel for
  for (unsigned i = 0; i < N; ++i)
    dist->sample(x_t[0][i], Q, 200);

  // Initialize omega
  w_t[0].fill(1 / double(N));

  delete dist;
}

void propagate_K(
    Eigen::VectorXd **post_x_t, unsigned *a_t,
    const Eigen::MatrixXd G, const Eigen::MatrixXd W,
    const Eigen::MatrixXd Q_w,
    const dim_t N, const dim_t d, const dim_t t,
    const float df,
    std::string distribution_opt)
{
  // Distributions
  Distributions["normal"] =
      [](distParams_t params)
  { return StandardNormalDistribution::getInstance(params); };
  Distributions["mvn"] =
      [](distParams_t params)
  { return MultiVariateNormalDistribution::getInstance(params); };
  Distributions["mvt"] =
      [](distParams_t params)
  { return MultiVariateTStudentDistribution::getInstance(params); };

  // BUG:
  // Q is already the result of eigenSolver
  //
  // // Find the eigen vectors of the covariance matrix
  // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>
  //     eigen_solver(Q);
  // Eigen::MatrixXd eigenvectors = eigen_solver.eigenvectors().real();

  // // Find the eigenvalues of the covariance matrix
  // Eigen::MatrixXd eigenvalues = eigen_solver.eigenvalues().real().asDiagonal();

  // // Find the transformation matrix
  // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(eigenvalues);
  // Eigen::MatrixXd sqrt_eigenvalues = es.operatorSqrt();
  // Eigen::MatrixXd _Q = eigenvectors * sqrt_eigenvalues;

// Initialize distribution with shuffled(theta) x_t
#ifndef __GPU

  distParams_t params;
  params.sigma = W; // sigma = W
  params.nu = df;

#pragma omp parallel for
  for (unsigned i = 0; i < N; ++i)
  {
    // `distCreator` initializes a new StatisticalDistribution object with the
    // given arguments. The actual is distribution used is selected based on
    // user input. The underlying function call may look like this:
    //
    // MultiVariateNormalDistribution mvn(post_x_t[t - 1][a_t[t * N + i]],
    //                                    G, 200);
    //
    // NOTE: In the future it may be worthwhile to pass the arguments in a
    // `distOptions` object because different distribution may functin with
    // different sets of parameters.

    // Shuffle (thetas) x_t in the indices
    // and initialize new distribution
    // mu = G * x
    params.mu = G * post_x_t[t - 1][a_t[t * N + i]];

    StatisticalDistribution *dist = Distributions[distribution_opt](params);
    // Sample new (thetas) x_t
    dist->sample(post_x_t[t][i], Q_w, 200);

    delete dist;
  }
#else

  distParams_t params;
  params.post_x_t = post_x_t;
  params.a_t = a_t;
  params.sigma = W;
  params.Q_w = Q_w;
  params.N = N;
  params.d = d;
  params.t = t;
  params.nu = df;

  StatisticalDistribution *dist = Distributions[distribution_opt](params);

  dist->sample(post_x_t, a_t, G, Q_w, N, d, t);

  delete dist;

#endif
}

void reweight_G(
    Eigen::VectorXd *w_t, const Eigen::VectorXd *y_t,
    Eigen::VectorXd **post_x_t,
    const Eigen::MatrixXd &F, const Eigen::MatrixXd &V,
    const double &V_det,
    const Eigen::MatrixXd &V_inv,
    const dim_t N, const dim_t d, const dim_t t,
    const float df,
    std::string distributions_opt)
{
  // Distributions
  Distributions["normal"] =
      [](distParams_t params)
  { return StandardNormalDistribution::getInstance(params); };
  Distributions["mvn"] =
      [](distParams_t params)
  { return MultiVariateNormalDistribution::getInstance(params); };
  Distributions["mvt"] =
      [](distParams_t params)
  { return MultiVariateTStudentDistribution::getInstance(params); };

#ifndef __GPU

  distParams_t params;
  params.sigma_det = V_det; // V matrix determinant
  params.sigma_inv = V_inv; // V matrix inverse
  params.mu = Eigen::VectorXd::Zero(d); // mean matrix = 0
  params.sigma = V; // Cov matrix SigmaV
  params.nu = df;
  //StatisticalDistribution *dist = Distributions[distributions_opt](params);

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

    // Initialize distribution
    // mean = 0, covariance matrix sigma = V
    // params.mu = post_x_t[t][i];

    // Get new weights from probality density function
    StatisticalDistribution *dist = Distributions[distributions_opt](params);
    w_t[t][i] = dist->pdf(y_t[t]-F*post_x_t[t][i]);

    delete dist;
  }

#else

  distParams_t params;
  params.post_x_t = post_x_t;
  params.a_t = a_t;
  params.sigma = V;
  params.sigma_det = V_det;
  params.sigma_inv = V_inv;
  params.df = df;
  params.Q = Q;
  params.N = N;
  params.d = d;
  params.t = t;

  StatisticalDistribution *dist = Distributions[distribution_opt](params);

  w_t[t] = dist->pdf_cu(y_t, F);

  delete dist;

#endif
}

void MCMC(
    Eigen::VectorXd **post_x_t,
    Eigen::VectorXd *w_t, unsigned *a_t,
    Eigen::VectorXd *y_t,
    const Eigen::MatrixXd &F, const Eigen::MatrixXd &G,
    const Eigen::MatrixXd V, const Eigen::MatrixXd W,
    const dim_t N, const dim_t d, const dim_t timeSteps, const float df,
    std::string resampler_opt,
    std::string distribution_opt)
{
  // Initialize Resamplers and Distributions
  // Resamplers;
  Resamplers["metropolis"] = [](unsigned *a_t, Eigen::VectorXd *w_t, int N, unsigned t, int B = 10)
  {
    Sampler::metropolis_hastings(a_t, w_t, N, t, B);
  };

  // Distributions
  Distributions["normal"] =
      [](distParams_t params)
  { return StandardNormalDistribution::getInstance(params); };
  Distributions["mvn"] =
      [](distParams_t params)
  { return MultiVariateNormalDistribution::getInstance(params); };
  Distributions["mvt"] =
      [](distParams_t params)
  { return MultiVariateTStudentDistribution::getInstance(params); };

  // Assert availability of resampler and distribution
  // assert(Resamplers.find(resampler_opt) != Resamplers.end());
  // assert(Distributions.find(distribution_opt) != Distributions.end());

  resampler_f resampler = Resamplers[resampler_opt];
  // Solve Covariant Matrix for determinant & inverse
  //double E_det = E.determinant();
  double V_det = V.determinant();
  if (V_det == 0 ) Rcpp::Rcout << "The Covariance matrix SigmaV is singular\n" << std::endl;
  Eigen::MatrixXd V_inv = V.inverse();
  eigenSolver(Q_w, W);
  // BUG: norm changes with each timestep
  // Get norm from distribution
  // distParams_t params;
  // params.mu = Eigen::VectorXd::Zero(d);
  // params.sigma = G;
  // params.nu = df;

  // StatisticalDistribution *dist = Distributions[distribution_opt](params);
  // double norm = dist->getNorm();

  int B = 10;
  for (unsigned t = 1; t < timeSteps; ++t)
  {
    // Calculate ancestors with Metropolis
    resampler(a_t, w_t, N, t, B);

    // Propagate particles
    propagate_K(post_x_t, a_t,
                G, W, Q_w,
                N, d, t, df,
                distribution_opt);

    // Resample weights
    reweight_G(w_t, y_t, post_x_t,
               F, V, V_det, V_inv,
               N, d, t, df,
               distribution_opt);
  }
}

// void MCMC_step(
//     Eigen::VectorXd **post_x_t,
//     Eigen::VectorXd *w_t, unsigned *a_t,
//     Eigen::VectorXd *y_t,
//     const Eigen::MatrixXd &F, const Eigen::MatrixXd &G,
//     const Eigen::matrixXd &V, const Eigen::MatrixXd &W,
//     const dim_t N, const dim_t d, const dim_t timeSteps, const float df,
//     std::string resampler_opt,
//     std::string distribution_opt)
// {
//   // Initialize Resamplers and Distributions
//   // Resamplers;
//   Resamplers["metropolis"] = [](unsigned *a_t, Eigen::VectorXd *w_t, int N, unsigned t, int B = 10)
//   {
//     Sampler::metropolis_hastings(a_t, w_t, N, t, B);
//   };

//   // Distributions
//   Distributions["normal"] =
//       [](distParams_t params)
//   { return StandardNormalDistribution::getInstance(params); };
//   Distributions["mvn"] =
//       [](distParams_t params)
//   { return MultiVariateNormalDistribution::getInstance(params); };
//   Distributions["mvt"] =
//       [](distParams_t params)
//   { return MultiVariateTStudentDistribution::getInstance(params); };

//   // Assert availability of resampler and distribution
//   // assert(Resamplers.find(resampler_opt) != Resamplers.end());
//   // assert(Distributions.find(distribution_opt) != Distributions.end());

//   resampler_f resampler = Resamplers[resampler_opt];

//   // Solve Covariant Matrix for determinant & inverse
//   //double E_det = E.determinant();
//   Eigen::MatrixXd E_inv = E.inverse();
//   Eigen::MatrixXd Q(d, d);
//   eigenSolver(Q, E);

//   // Get norm from distribution
//   distParams_t params;
//   params.mu = Eigen::VectorXd::Zero(d);
//   params.sigma = E;
//   params.nu = df;

//   StatisticalDistribution *dist = Distributions[distribution_opt](params);
//   double norm = dist->getNorm();

//   int B = 10;
//   for (unsigned t = 1; t < timeSteps; ++t)
//   {
//     // Calculate ancestors with Metropolis
//     resampler(a_t, w_t, N, t, B);

//     // Propagate particles
//     propagate_K(distribution_opt, post_x_t, a_t, Q, N, d, t, df);

//     // Resample weights
//     reweight_G(distribution_opt, w_t, y_t, post_x_t, norm, E_inv, E, F, N, d, t, df);
//   }
// }

#endif
