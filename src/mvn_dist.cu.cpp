#ifdef __GPU

#ifndef __MVN_CPP
#define __MVN_CPP

#include <distributions/mvn_dist.hpp>

#define TILE_SIZE 16

__constant__ dim_t dev_d;
__constant__ double dev_norm;
__constant__ int dev_seed;

__global__ void mvn_sample_setup_kernel(curandState_t *state)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < dev_d * dev_N)
    curand_init(dev_seed, idx, 0, &state[idx]);
}

__global__ void mvn_sample_norm_rand_kernel(double *rand_x, curandState_t *randState)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < dev_d * dev_N)
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
    _Q[y][x] = (rowIdx < m && x + TILE_ROW < k)
                   ? Q[(rowIdx * k) + x + TILE_ROW]
                   : 0.0;

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

  // Return rand vector
  if (rowIdx < m && colIdx < n)
    post_x_t[el] = dotProduct + pre_x_t[el];
}

void mvn_sample_kernel_wrapper(Eigen::VectorXd **post_x_t, unsigned *a_t,
                               const Eigen::MatrixXd Q, const dim_t N, const dim_t d,
                               const dim_t t)
{
  cudaError_t cuda_ret;
  const size_t VECTOR_SZ = sizeof(double) * N * d;
  const size_t COVMAT_SZ = sizeof(double) * d * d;

  // Init random seed
  time_t t;
  srand((unsigned)time(&t));
  int seed = rand();

  double *host_draws = (double *)malloc(VECTOR_SZ);

  // shuffle pre_x_t
  Eigen::VectorXd *host_pre_x_t = new Eigen::VectorXd[N];
  for (unsigned i = 0; i < N; ++i)
  {
    host_pre_x_t[i] = Eigen::VectorXd(d);
    host_pre_x_t[i] = post_x_t[t - 1][a_t[t * N + i]];
  }

  // Copy over Eigen data types to arrays
  double *host_x_t = (double *)malloc(sizeof(double) * N * d);
  for (int y = 0; y < N; ++y)
    for (unsigned x = 0; x < d; ++x)
      host_x_t[y * d + x] = host_pre_x_t[y][x];

  double *host_Q = (double *)malloc(COVMAT_SZ);
  for (size_t i = 0; i < d; ++i)
    for (size_t j = 0; j < d; ++j)
      host_Q[i * d + j] = Q(i, j);

  // device memory
  double *dev_pre_x_t;
  double *dev_post_x_t;
  double *dev_Q;
  double *dev_norm_rand;
  curandState *devPRNGStates;

  // Allocate device memory
  CUDA_CALL(cudaMalloc((void **)&devPRNGStates, N * d * sizeof(curandState)));
  CUDA_CALL(cudaMalloc((void **)&dev_norm_rand, VECTOR_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_pre_x_t, VECTOR_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_post_x_t, VECTOR_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_Q, COVMAT_SZ));
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to allocate memory on device");

  // Copy to device memory
  CUDA_CALL(cudaMemcpy(dev_pre_x_t, host_x_t, VECTOR_SZ, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_Q, host_Q, COVMAT_SZ, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpyToSymbol(dev_seed, &seed, sizeof(int)));
  CUDA_CALL(cudaMemcpyToSymbol(dev_d, &d, sizeof(dim_t)));

  dim3 rand_blockDim(512);
  int rand_gridDim = ceil(double(N * d) / double(rand_blockDim.x));

  // Setup PRNG states
  mvn_sample_setup_kernel<<<rand_gridDim, rand_blockDim>>>(devPRNGStates);
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to launch kernel: mvn_sample_setup_kernel");

  // Generate random numbers
  mvn_sample_norm_rand_kernel<<<rand_gridDim, rand_blockDim>>>(dev_norm_rand,
                                                               devPRNGStates);
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to lauch kernel: mvn_sample_norm_rand_kernel");

  const unsigned BLOCK_SIZE = TILE_SIZE;

  dim3 blockDim(1, BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(N, ceil(double(d) / double(BLOCK_SIZE)),
               ceil(double(d) / double(BLOCK_SIZE)));

  mvn_sample_kernel<<<gridDim, blockDim>>>(dev_post_x_t, dev_mu, dev_Q,
                                           dev_norm_rand);
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to launch kernel: mvn_sample_kernel");

  CUDA_CALL(cudaMemcpy(host_x_t, dev_post_x_t, VECTOR_SZ,
                       cudaMemcpyDeviceToHost));
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to transfer data to host");

  // Return random draws
  for (unsigned i = 0; i < N; ++i)
    for (unsigned j = 0; j < d; ++j)
      post_x_t[t][i][j] = host_x_t[i * d + j];

  // Free device memory
  cudaFree(devPRNGStates);
  cudaFree(dev_pre_x_t);
  cudaFree(dev_post_x_t);
  cudaFree(dev_Q);

  // Free host memory
  free(host_x_t);
  free(host_Q);
}

__global__ void mvn_pdf_kernel_y_minus_Fmu(double *dev_alpha,
                                           double *dev_y,
                                           double *dev_x,
                                           double *dev_F)
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
  const int y = threadIdx.y;
  const int z = threadIdx.z;

  // Get row[y] and column[x] index
  int rowIdx = blockIdx.z * blockDim.z + threadIdx.z;
  int colIdx = blockIdx.y * blockDim.y + threadIdx.y;

  // Initialize dimensions
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

  // save dot product
  __syncthreads();
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

  // Initialize dimensions
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

  // save dot product
  __syncthreads();
  if (rowIdx < m && colIdx < n)
    dev_Ealpha[el] = dotProduct;
}

__global__ void mvn_pdf_kernel(double *dev_w_t,
                               double *dev_alpha_t,
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

  // Initialize dimensions
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
  if (x == 0 && y == 0)
    dev_w_t[blockIdx.x] = dev_norm * exp(-0.5 * quadform);
}

// MVN kernel wrapper
void mvn_sample_kernel_wrapper(Eigen::VectorXd **post_x_t,
                               unsigned *a_t,
                               const Eigen::MatrixXd Q,
                               const dim_t N, const dim_t d,
                               const dim_t t)
{
  cudaError_t cuda_ret;
  const size_t VECTOR_SZ = sizeof(double) * d * N;
  const size_t COVMAT_SZ = sizeof(double) * d * d;

  // Init random seed
  time_t t;
  srand((unsigned)time(&t));
  int seed = rand() % 6297;

  // shuffle pre_x_t
  Eigen::VectorXd *host_pre_x_t = new Eigen::VectorXd[N];
  for (unsigned i = 0; i < N; ++i)
  {
    host_pre_x_t[i] = Eigen::VectorXd(d);
    host_pre_x_t[i] = post_x_t[t - 1][a_t[t * N + i]];
  }

  // Copy over Eigen data types to arrays
  double *host_x_t = (double *)malloc(sizeof(double) * N * d);
  for (int y = 0; y < N; ++y)
    for (unsigned x = 0; x < d; ++x)
      host_x_t[y * d + x] = host_pre_x_t[y][x];

  double *host_Q = (double *)malloc(COVMAT_SZ);
  for (size_t i = 0; i < d; ++i)
    for (size_t j = 0; j < d; ++j)
      host_Q[i * d + j] = Q(i, j);

  // device memory
  double *dev_pre_x_t;
  double *dev_post_x_t;
  double *dev_Q;
  double *dev_norm_rand;
  curandState *devPRNGStates;

  // Allocate device memory
  CUDA_CALL(cudaMalloc((void **)&devPRNGStates, sizeof(curandState) * d * N));
  CUDA_CALL(cudaMalloc((void **)&dev_norm_rand, VECTOR_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_pre_x_t, VECTOR_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_post_x_t, VECTOR_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_Q, COVMAT_SZ));
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("unable to allocate memory on device");

  // Copy to device memory
  CUDA_CALL(cudaMemcpy(dev_pre_x_t, host_x_t, VECTOR_SZ, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_Q, host_Q, COVMAT_SZ, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpyToSymbol(dev_seed, &seed, sizeof(int)));
  CUDA_CALL(cudaMemcpyToSymbol(dev_d, &d, sizeof(dim_t)));
  CUDA_CALL(cudaMemcpyToSymbol(dev_N, &N, sizeof(dim_t)));
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to transfer data to device");

  dim3 rand_blockDim(512);
  int rand_gridDim = ceil(double(N * d) / double(rand_blockDim.x));

  // Setup PRNG states
  mvn_sample_setup_kernel<<<rand_gridDim, rand_blockDim>>>(devPRNGStates);
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to launch kernel: mvn_sample_setup_kernel");

  // Generate random numbers
  mvn_sample_norm_rand_kernel<<<rand_gridDim, rand_blockDim>>>(
      dev_norm_rand,
      devPRNGStates);
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to lauch kernel: mvn_sample_norm_rand_kernel");

  const unsigned int BLOCK_SIZE = TILE_SIZE;

  dim3 blockDim(1, BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(N, ceil(double(d) / double(BLOCK_SIZE)),
               ceil(double(d) / double(BLOCK_SIZE)));

  mvn_sample_kernel<<<gridDim, blockDim>>>(dev_post_x_t,
                                           dev_pre_x_t,
                                           dev_Q,
                                           dev_norm_rand);
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to launch kernel: mvn_sample_kernel");

  CUDA_CALL(cudaMemcpy(host_x_t, dev_post_x_t, VECTOR_SZ,
                       cudaMemcpyDeviceToHost));
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to transfer data to host");

  // Return random draws
  for (unsigned i = 0; i < N; ++i)
    for (unsigned j = 0; j < d; ++j)
      post_x_t[t][i][j] = host_x_t[i * d + j];

  // Free device memory
  cudaFree(devPRNGStates);
  cudaFree(dev_pre_x_t);
  cudaFree(dev_post_x_t);
  cudaFree(dev_Q);

  // Free host memory
  free(host_x_t);
}

// MVNPDF kernel wrapper
void mvn_pdf_kernel_wrapper(Eigen::VectorXd *w_t,
                            const Eigen::VectorXd *y_t,
                            Eigen::VectorXd **post_x_t,
                            const double norm,
                            const Eigen::MatrixXd &E_inv,
                            const Eigen::MatrixXd E,
                            const Eigen::MatrixXd F,
                            const dim_t N, const dim_t d,
                            const dim_t t)
{
  cudaError_t cuda_ret;
  double *dev_w_t;
  double *dev_x_t;
  double *dev_y_t;
  double *dev_E_inv;
  double *dev_F;
  double *dev_Ealpha_t;
  double *dev_alpha_t;

  const size_t WEIGHT_SZ = sizeof(double) * N;
  const size_t INPUT_SZ = sizeof(double) * d;
  const size_t PARTICLE_SZ = sizeof(double) * d * N;
  const size_t COVMAT_SZ = sizeof(double) * d * d;

  CUDA_CALL(cudaMalloc((void **)&dev_w_t, WEIGHT_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_y_t, INPUT_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_x_t, PARTICLE_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_E_inv, COVMAT_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_F, COVMAT_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_Ealpha_t, PARTICLE_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_alpha_t, PARTICLE_SZ));
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to allocate menory on device");

  double *host_w_t;
  double *host_y_t;
  double *host_x_t;
  double *host_E_inv;
  double *host_F;

  host_w_t = (double *)malloc(WEIGHTS_SZ);
  host_y_t = (double *)malloc(INPUT_SZ);
  host_x_t = (double *)malloc(PARTICLE_SZ);
  host_E_inv = (double *)malloc(COVMAT_SZ);
  host_F = (double *)malloc(COVMAT_SZ);

#pragma omp parallel for
  for (int j = 0; j < d; ++j)
    host_y_t[j] = y[t][j];

#pragma omp parallel for
  for (unsigned i = 0; i < N; ++i)
    for (int j = 0; j < d; ++j)
      host_x_t[i * d + j] = post_x_t[t][i][j];

#pragma omp parallel for
  for (i = 0; i < d; ++i)
    for (j = 0; j < d; ++j)
    {
      host_E_inv[i * d + j] = E_inv(i, j);
      host_F[i * d + j] = F(i, j);
    }

  CUDA_CALL(cudaMemcpy(dev_y_t, host_y_t, INPUT_SZ, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_x_t, host_x_t, PARTICLE_SZ, cudaMemcpyHostToDevice));
  CUDA_CALL(
      cudaMemcpy(dev_E_inv, host_E_inv, COVMAT_SZ, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_F, host_F, COVMAT_SZ, cudaMemcpyHostToDevice));

  // Copy constants
  CUDA_CALL(cudaMemcpyToSymbol(dev_d, &d, sizeof(dim_t)));
  CUDA_CALL(cudaMemcpyToSymbol(dev_N, &N, sizeof(dim_t)));
  CUDA_CALL(cudaMemcpyToSymbol(dev_norm, &norm, sizeof(double)));

  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to transfer data to device");

  const int BLOCK_SIZE = TILE_SIZE;

  dim3 blockDim(1, BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(N, ((d - 1) / BLOCK_SIZE) + 1, ((d - 1) / BLOCK_SIZE) + 1);

  mvn_pdf_kernel_y_minus_Fmu<<<gridDim, blockDim>>>(dev_alpha_t, dev_y_t, dev_x_t,
                                                    dev_F);
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to launch kernel mvnpdf_kernel_y_minus_Fmu");

  mvn_pdf_kernel_Einv_alpha<<<gridDim, blockDim>>>(dev_Ealpha_t, dev_alpha_t,
                                                   dev_E_inv);
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to launch kernel: mvnpdf_kernel_Einv_alpha");

  mvn_pdf_kernel<<<gridDim, blockDim>>>(dev_w_t, dev_alpha_t, dev_Ealpha_t);
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to launch kernel: mvnpdf_kernel");

  CUDA_CALL(cudaMemcpy(&host_w_t, dev_w_t, WEIGHT_SZ, cudaMemcpyDeviceToHost));
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to transfer data from device");

  // w = host_w;
  for (unsigned i = 0; i < N; ++i)
    w_t[t][i] = host_w_t[i];

  // Reset device
  cudaDeviceReset();

  // Free device memory
  cudaFree(dev_w_t);
  cudaFree(dev_y_t);
  cudaFree(dev_x_t);
  cudaFree(dev_E_inv);

  // Free host memory
  free(host_w_t);
  free(host_x_t);
  free(host_y_t);
  free(host_E_inv);
}

#endif

#endif