#ifndef __TYPES_HPP
#define __TYPES_HPP

// Global headers
#include <string>
#include <map>
#include <functional>
#include <RcppEigen.h>

// Local headers
#include "statistics.hpp"

typedef unsigned dim_t;

struct opts_t
{
  int d, N, t;               // Dimensions
  std::string y_filename;    // Y Vector
  std::string m0, C0;        // Initialization x and E
  std::string indir, outdir; // Input/Output directories
};

struct device_t
{
  std::string name; // Device name
  int id,           // Device ID
      major,        // Device version major number
      minor;        // Device version minor number
};

// Resampler function
typedef std::function<void(unsigned *a_t, Eigen::VectorXd *w_t, int N, unsigned t, int B)> resampler_f;
// typedef void (*resampler_f)(unsigned *, Eigen::VectorXd *, int, unsigned, int);
typedef std::map<std::string, resampler_f> resamplers_t;

// Distribution creator function
typedef std::function<StatisticalDistribution *(distParams_t)> distributionCreator_f;
typedef std::map<std::string, distributionCreator_f> distributions_t;

#endif