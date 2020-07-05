#ifdef __GPU

#ifndef __MVN_CPP
#define __MVN_CPP

#include <distributions/mvn_dist.hpp>

#define TILE_SIZE 16

__constant__ dim_t dev_d;
__constant__ dim_t dev_N;
__constant__ double dev_norm;

__global__ void setup_kernel(curandState_t *state)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < dev_N * dev_d)
    curand_init(1234LLU, idx, 0, &state[idx]);
}

__global__ void norm_rand_kernel(double *rand_x, curandState_t *randState)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < dev_N * dev_d)
    rand_x[idx] = curand_normal_double(randState + idx);
}

__global__ void mvn_sample_kernel(double *post_x_t, double *pre_x_t,
                                  double *Q, double *rand_norm_vars)
{
  __shared__ int matOffset;
  __shared__ int m;
  __shared__ int k;
  __shared__ int n;

  __shared__ double _Q[TILE_SIZE][TILE_SIZE];
  __shared__ double _x[TILE_SIZE][TILE_SIZE];

  // Get matrix id
  matOffset = blockIdx.x * dev_d;

  // Get thread x and y
  const int x = threadIdx.y;
  const int y = threadIdx.z;

  // Get row[y] and column[x] index
  int rowIdx = blockIdx.z * blockDim.z + threadIdx.z;
  int colIdx = blockIdx.y * blockDim.y + threadIdx.y;

  m = dev_d;
  k = dev_d;
  n = 1;

  // Initialize tiles to zero
  _Q[y][x] = 0.0;
  _x[y][x] = 0.0;

  // Calculate the number of tiles required
  int const TILES = ((k - 1) / TILE_SIZE) + 1;

  // Initialize dot product and TILE_ROW
  double dotProduct = 0;
  int TILE_ROW;

  for (unsigned t = 0; t < TILES; ++t)
  {
    // Figure out row of tiles we are at
    TILE_ROW = t * TILE_SIZE;

    // Read values from matrix A
    _Q[y][x] =
        (rowIdx < m && x + TILE_ROW < k) ? Q[(rowIdx * k) + x + TILE_ROW] : 0.0;

    // Read values from matrix B
    _x[y][x] = (colIdx < n && y + TILE_ROW < k)
                   ? rand_norm_vars[matOffset + (y + TILE_ROW) * n + colIdx]
                   : 0.0;

    __syncthreads();

    for (int i = 0; i < TILE_SIZE; ++i)
      dotProduct += _Q[y][i] * _x[i][x];

    __syncthreads();
  }

  int el = matOffset + (rowIdx * n + colIdx);
  __syncthreads();

  // save dot product
  if (rowIdx < m && colIdx < n)
    post_x_t[el] = dotProduct + pre_x_t[el];
}

void mvn_sample_kernel_wrapper(double Eigen::VectorXd *draws,
                               double Eigen::VectorXd *mu, const Eigen::MatrixXd Q,
                               const dim_t d)
{
  cudaError_t cuda_ret;
  const size_t PARTICLE_SZ = sizeof(double) * d;
  const size_t COVMAT_SZ = sizeof(double) * d * d;

  double *host_draws = (double *)malloc(PARTICLE_SZ);

  // Copy over Eigen data types to arrays
  double *host_mu = (double *)malloc(PARTICLE_SZ);
  for (int i = 0; i < d; ++i)
    host_mu[i] = mu[i];

  double *host_Q = (double *)malloc(COVMAT_SZ);
  for (size_t i = 0; i < d; ++i)
    for (size_t j = 0; j < d; ++j)
      host_Q[i * d + j] = Q(i, j);

  // device memory
  double *dev_mu;
  double *dev_draws;
  double *dev_Q;
  double *dev_norm_rand;
  curandState *devPRNGStates;

  // Allocate device memory
  CUDA_CALL(cudaMalloc((void **)&devPRNGStates, sizeof(curandState) * d));
  CUDA_CALL(cudaMalloc((void **)&dev_norm_rand, VECTOR_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_mu, VECTOR_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_draws, VECTOR_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_Q, COVMAT_SZ));
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("unable to allocate memory on device");

  // Copy to device memory
  CUDA_CALL(cudaMemcpy(dev_mu, host_mu, VECTOR_SZ, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_Q, host_Q, VECTOR_SZ, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpyToSymbol(dev_d, &d, sizeof(dim_t)));
  CUDA_CALL(cudaMemcpyToSymbol(dev_d, &d, sizeof(dim_t)));

  dim3 rand_blockDim(512);
  int rand_gridDim = ceil(double(d) / double(rand_blockDim.x));

  // Setup PRNG states
  mvn_sample_setup_kernel<<<rand_gridDim, rand_blockDim>>>(devPRNGStates);
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to launch kernel: mvn_sample_setup_kernel");

  // Generate random numbers
  mvn_sample_norm_rand_kernel<<<rand_gridDim, rand_blockDim>>>(dev_norm_rand, devPRNGStates);
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to lauch kernel: mvn_sample_norm_rand_kernel");

  const unsigned BLOCK_SIZE = TILE_SIZE;

  dim3 blockDim(1, BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(1, ceil(double(d) / double(BLOCK_SIZE)),
               ceil(double(d) / double(BLOCK_SIZE)));

  mvn_sample_kernel<<<gridDim, blockDim>>>(dev_draws, dev_mu, dev_Q, dev_norm_rand);
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to launch kernel: mvn_sample_kernel");

  CUDA_CALL(cudaMemcpy(host_draws, dev_draws, VECTOR_SZ, cudaMemcpyDeviceToHost));
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to transfer data to host");

  // Return random draws
  for (int i = 0; i < d; ++i)
    draws[i] = host_draws[i];

  // Free device memory
  cudaFree(devStates);
  cudaFree(dev_mu);
  cudaFree(dev_draws);
  cudaFree(dev_Q);

  // Free host memory
  free(host_draws);
}

__global__ void mvn_pdf_kernel_y_minus_Fmu(double *dev_alpha_t, double *dev_y_t,
                                           double *dev_x_t, double *dev_F)
{
  __shared__ int matOffset;
  __shared__ int m;
  __shared__ int k;
  __shared__ int n;

  __shared__ double _F[TILE_SIZE][TILE_SIZE];
  __shared__ double _x[TILE_SIZE][TILE_SIZE];

  // Get matrix offset
  matOffset = blockIdx.x * dev_d;

  // Get thread x and y
  const int x = threadIdx.y;
  const int y = threadIdx.z;

  // Get row[y] and column[x] index
  int rowIdx = blockIdx.z * blockDim.z + threadIdx.z;
  int colIdx = blockIdx.y * blockDim.y + threadIdx.y;

  m = dev_d;
  k = dev_d;
  n = 1;

  // Initialize tiles to zero
  _F[y][x] = 0.0;
  _x[y][x] = 0.0;

  // Calculate the number of tiles required
  int const TILES = ((k - 1) / TILE_SIZE) + 1;

  // Initialize dot product and TILE_ROW
  double dotProduct = 0;
  int TILE_ROW;

  for (unsigned t = 0; t < TILES; ++t)
  {
    // Figure out row of tiles we are at
    TILE_ROW = t * TILE_SIZE;

    // Read values from matrix A
    _F[y][x] = (rowIdx < m && x + TILE_ROW < k)
                   ? dev_F[(rowIdx * k) + x + TILE_ROW]
                   : 0.0;

    // Read values from matrix B
    _x[y][x] = (colIdx < n && y + TILE_ROW < k)
                   ? dev_x_t[matOffset + (y + TILE_ROW) * n + colIdx]
                   : 0.0;

    __syncthreads();

    for (int i = 0; i < TILE_SIZE; ++i)
      dotProduct += _F[y][i] * _x[i][x];

    __syncthreads();
  }

  int el = rowIdx * n + colIdx;
  // __syncthreads();

  // save dot product
  if (blockIdx.x < dev_N && rowIdx < m && colIdx < n)
    dev_alpha_t[matOffset + el] = dev_y_t[el] - dotProduct;
}

__global__ void mvn_pdf_kernel_Einv_alpha(double *dev_Ealpha_t,
                                          double *dev_alpha_t,
                                          double *dev_E_inv)
{
  __shared__ int matOffset;
  __shared__ int m;
  __shared__ int k;
  __shared__ int n;

  __shared__ double _E_inv[TILE_SIZE][TILE_SIZE];
  __shared__ double _alpha[TILE_SIZE][TILE_SIZE];

  // Get matrix offset
  matOffset = blockIdx.x * dev_d;

  // Get thread x and y
  const int x = threadIdx.y;
  const int y = threadIdx.z;

  // Get row[y] and column[x] index
  int rowIdx = blockIdx.z * blockDim.z + threadIdx.z;
  int colIdx = blockIdx.y * blockDim.y + threadIdx.y;

  m = dev_d;
  k = dev_d;
  n = 1;

  // Initialize tiles to zero
  _E_inv[y][x] = 0.0;
  _alpha[y][x] = 0.0;

  // Calculate the number of tiles required
  int const TILES = ((k - 1) / TILE_SIZE) + 1;

  // Initialize dot product and TILE_ROW
  double dotProduct = 0;
  int TILE_ROW;

  for (unsigned t = 0; t < TILES; ++t)
  {
    // Figure out row of tiles we are at
    TILE_ROW = t * TILE_SIZE;

    // Read values from matrix A
    _E_inv[y][x] = (rowIdx < m && x + TILE_ROW < k)
                       ? dev_E_inv[(rowIdx * k) + x + TILE_ROW]
                       : 0.0;

    // Read values from matrix B
    _alpha[y][x] = (colIdx < n && y + TILE_ROW < k)
                       ? dev_alpha_t[matOffset + (y + TILE_ROW) * n + colIdx]
                       : 0.0;

    __syncthreads();

    for (int i = 0; i < TILE_SIZE; ++i)
      dotProduct += _E_inv[y][i] * _alpha[i][x];

    __syncthreads();
  }

  int el = matOffset + (rowIdx * n + colIdx);
  // __syncthreads();

  // save dot product
  if (rowIdx < m && colIdx < n)
    dev_Ealpha_t[el] = dotProduct;
}

__global__ void mvn_pdf_kernel(double *dev_w_t, double *dev_alpha_t,
                               double *dev_Ealpha_t)
{
  __shared__ int matOffset;
  __shared__ int m;
  __shared__ int k;
  __shared__ int n;

  __shared__ double _Ealpha[TILE_SIZE][TILE_SIZE];
  __shared__ double _alpha[TILE_SIZE][TILE_SIZE];

  // Get matrix offset
  matOffset = blockIdx.x * dev_d;

  // Get thread x and y
  const int x = threadIdx.y;
  const int y = threadIdx.z;

  // Get row[y] and column[x] index
  int rowIdx = blockIdx.z * blockDim.z + threadIdx.z;
  int colIdx = blockIdx.y * blockDim.y + threadIdx.y;

  m = dev_d;
  k = dev_d;
  n = 1;

  // Initialize tiles to zero
  _Ealpha[y][x] = 0.0;
  _alpha[y][x] = 0.0;

  // Calculate the number of tiles required
  int const TILES = ((k - 1) / TILE_SIZE) + 1;

  // Initialize dot product and TILE_ROW
  double quadform = 0;
  int TILE_ROW;

  for (unsigned t = 0; t < TILES; ++t)
  {
    // Figure out row of tiles we are at
    TILE_ROW = t * TILE_SIZE;

    // Read values from matrix A
    _alpha[y][x] = (rowIdx < m && x + TILE_ROW < k)
                       ? dev_Ealpha_t[(rowIdx * k) + x + TILE_ROW]
                       : 0.0;

    // Read values from matrix B
    _Ealpha[y][x] = (colIdx < n && y + TILE_ROW < k)
                        ? dev_alpha_t[matOffset + (y + TILE_ROW) * n + colIdx]
                        : 0.0;

    __syncthreads();

    for (int i = 0; i < TILE_SIZE; ++i)
      quadform += _alpha[y][i] * _Ealpha[i][x];

    __syncthreads();
  }

  int el = blockIdx.x;
  __syncthreads();

  // save dot product

  if (rowIdx < m && colIdx < n)
    dev_w_t[el] = dev_norm * exp(-0.5 * quadform);
}

void mvn_pdf_kernel_wrapper(const Eigen::VectroXd *y,
                            Eigen::VectorXd *mu,
                            const Eigen::MatrixXd &E_Inv,
                            const Eigen::MatrixXd F,
                            const double norm,
                            const dim_t d)
{
  cudaError_t cuda_ret;
  double *dev_w;
  double *dev_y;
  double *dev_mu;
  double *dev_E_inv;
  double *dev_F;
  double *dev_Ealpha;
  double *dev_alpha;

  const size_t WEIGHT_SZ = sizeof(double);
  const size_t PARTICLE_SZ = sizeof(double) * d;
  const size_t SIGMA_SZ = sizeof(double) * d * d;

  CUDA_CALL(cudaMalloc((void **)&dev_w, WEIGHT_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_y, PARTICLE_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_mu, PARTICLE_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_E_inv, COVMAT_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_F, COVMAT_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_Ealpha, PARTICLE_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_alpha, PARTICLE_SZ));
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to allocate menory on device");

  double *host_y;
  double *host_mu;
  double *host_E_inv;
  double *host_F;

  host_y = (double *)malloc(PARTICLE_SZ);
  host_mu = (double *)malloc(PARTICLE_SZ);
  host_E_inv = (double *)malloc(COVMAT_SZ);
  host_F = (double *)malloc(COVMAT_SZ);

  for (int i = 0; i < d; ++i)
  {
    host_y[i] = y[i];
    host_mu[i] = mu[i];
  }

  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j)
    {
      host_E_inv[i * d + j] = E_inv(i, j);
      host_F[i * d + j] = F(i, j);
    }

  CUDA_CALL(cudaMemcpy(dev_y, host_y, PARTICLE_SZ, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_mu, host_mu, PARTICLE_SZ, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_E_inv, host_E_inv, COVMAT_SZ, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_F, host_F, COVMAT_SZ, cudaMemcpyHostToDevice));

  // Copy constants
  CUDA_CALL(cudaMemcpyToSymbol(dev_d, &d, sizeof(dim_t)));
  // CUDA_CALL(cudaMemcpyToSymbol(dev_N, &N, sizeof(dim_t)));
  CUDA_CALL(cudaMemcpyToSymbol(dev_norm, &norm, sizeof(double)));

  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to transfer data to device")

  const int BLOCK_SIZE = TILE_SIZE;

  dim3 blockDim(1, BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(1, ceil(double(d) / double(BLOCK_SIZE)),
               ceil(double(d) / double(BLOCK_SIZE)));

  mvn_pdf_kernel_y_minus_Fmu<<<gridDim, blockDim>>>(dev_alpha,
                                                    dev_y, dev_mu, dev_F);
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to launch kernel mvnpdf_kernel_y_minus_Fmu");

  mvn_pdf_kernel_Einv_alpha<<<gridDim, blockDim>>>(dev_Ealpha,
                                                   dev_alpha, dev_E_inv);
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to launch kernel: mvnpdf_kernel_Einv_alpha");

  mvn_pdf_kernel<<<gridDim, blockDim>>>(dev_w, dev_alpha, dev_Ealpha);
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to launch kernel: mvnpdf_kernel");

  w_t = host_w[0];

  // Reset device
  cudaDeviceReset();

  // Free device memory
  cudaFree(dev_w);
  cudaFree(dev_y);
  cudaFree(dev_mu);
  cudaFree(dev_E_inv);

  // Free host memory
  free(host_w);
  free(host_mu);
  free(host_y);
  free(host_E_inv)
}

#endif

#endif