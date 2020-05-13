#ifndef __SAMPLER_CPP
#define __SAMPLER_CPP

#include <samplers.hpp>

// Metropolis-Hastings Sampler
void Sampler::metropolis_hastings(unsigned *a_t, Eigen::VectorXd *w_t, int N, unsigned t, int B)
{
  // Generator
  std::random_device randomDevice{};
  std::mt19937 generator{randomDevice()};
  // Uniform Distribution between 0 and 1
  std::uniform_real_distribution<> U_u{0, 1};
  // Descrete Uniform Distribution between 1 and N
  std::uniform_int_distribution<> U_j(0, N - 1);

  double u;
  int j, k;

#pragma omp parallel for
  for (unsigned i = 0; i < N; ++i)
  {
    k = i;

    for (unsigned n = 0; n < B; ++n)
    {
      u = U_u(generator);
      j = U_j(generator);

      if (u <= w_t[t - 1][j] / w_t[t - 1][k])
        k = j;
    }

    a_t[t * N + i] = k;
  }
}

#endif
