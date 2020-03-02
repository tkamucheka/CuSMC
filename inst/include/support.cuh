#ifndef __SUPPORT_CUH
#define __SUPPORT_CUH

#include <Rcpp.h>
#include <assert.h>
#include <cstdio>
#include <cuda_runtime.h>

static void HandleCUDAError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    Rprintf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    Rcpp::stop("EXIT_FAILURE");
  }
}

static void HandleCURANDError(curandStatus err, const char *file, int line) {
  if ((err) != CURAND_STATUS_SUCCESS) {
    Rprintf("Curand error %d at %s:%d\n", err, file, line);
    Rcpp::stop("EXIT_FAILURE");
  }
}


// Macros
#define CUDA_CALL(err) (HandleCUDAError(err, __FILE__, __LINE__))
#define CURAND_CALL(err) (HandleCURANDError(err, __FILE__, __LINE__))
#define FATAL(msg, ...)                                                       \
do                                                                            \
{                                                                             \
  Rprintf("%s [%s:%d] " msg "\n", __FILE__, __LINE__, ##__VA_ARGS__);     \
  Rcpp::stop("-1");                                                           \
} while (0)

#if __BYTE_ORDER != __LITTLE_ENDIAN
#error "File I/O is not implemented for this system: wrong endianness."
#endif

#endif