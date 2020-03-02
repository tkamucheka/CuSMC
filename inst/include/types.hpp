#ifndef __TYPES_HPP
#define __TYPES_HPP

#include <string>

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

#endif