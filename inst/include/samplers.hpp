#ifndef __SAMPLER_HPP
#define __SAMPLER_HPP

#include <random>
#include <RcppEigen.h>

class Sampler
{
private:
  /* data */
public:
  Sampler(/* args */);
  ~Sampler();

  static void metropolis_hastings(unsigned *a_t, Eigen::VectorXd *w_t, int N, unsigned t, int B = 10);

  // Add other possible samplers here:
  // void gibbs();
};

#endif