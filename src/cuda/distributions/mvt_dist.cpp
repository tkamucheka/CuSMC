#ifndef __MVT_CPP
#define __MVT_CPP

#include <distributions/mvt_dist.hpp>

// namespace Kernel
// {

/*
 * Computes \sum_{i}^{N} x_i y_i for x, y \in \mathbb{R}^{N}.
 */
__device__ double devVecDot(const size_t N, const double *x, const double *y)
{
  assert(N > 0);
  assert(x != NULL);
  assert(y != NULL);
  // x == y allowed

  double sum = 0;
  for (size_t i = 0; i < N; ++i)
    sum += x[i] * y[i];

  return sum;
}

/*
 * Computes z_{i} \gets x_{i} - y_{i} for x, y \in \mathbb{R}^N.
 */
__device__ void devVecMinus(const size_t N, double *z, const double *x,
                            const double *y)
{
  assert(N > 0);
  assert(x != NULL);
  assert(y != NULL);
  // x == y allowed

  for (size_t i = 0; i < N; ++i)
  {
    z[i] = x[i] - y[i];
  }
}

/*
 * Solves the lower triangular system L^T x = b for x, b \in \mathbb{R}^{N},
 * L \in \mathbb{R}^{N \times N} and L_{i, j} = 0 for j > i.
 */
__device__ void devSolveLowerTri(const size_t N, const double *L, double *x,
                                 const double *b)
{
  assert(N > 0);
  assert(L != NULL);
  assert(x != NULL);
  assert(b != NULL);
  // x == b allowed

  for (size_t i = 0; i < N; ++i)
  {
    double sum = 0.0;
    if (i > 0)
    {
      for (size_t j = 0; j <= i - 1; ++j)
      {
        sum += L[i * N + j] * x[j];
      }
    }

    x[i] = (b[i] - sum) / L[i * N + i];
  }
}

/*
 * Solves the upper triangular system L^T x = b for x, b \in \mathbb{R}^{N},
 * L \in \mathbb{R}^{N \times N} and L_{i, j} = 0 for j > i.
 */
__device__ void devSolveLowerTriT(const size_t N, const double *L, double *x,
                                  const double *b)
{
  assert(N > 0);
  assert(L != NULL);
  assert(x != NULL);
  assert(b != NULL);
  // x == b allowed

  // treat L as an upper triangular matrix U
  for (size_t i = 0; i < N; i++)
  {
    size_t ip = N - 1 - i;
    double sum = 0;
    for (size_t j = ip + 1; j < N; ++j)
    {
      sum += L[j * N + ip] * x[j];
    }

    x[ip] = (b[ip] - sum) / L[ip * N + ip];
  }
}

__device__ void matrix_mult(double *C, double *A, double *B, int r1, int c1,
                            int c2)
{
  for (int m = 0; m < r1; ++m)
    for (int k = 0; k < c2; ++k)
    {
      double dotP = 0;
      for (int n = 0; n < c1; ++n)
        dotP += A[m * c1 + n] * B[n * c2 + k];

      C[m * c2 + k] = dotP;
    }
}

__device__ void vectorAdd(double *C, double *A, double *B, int d)
{
  for (int i = 0; i < d; ++i)
    C[i] = A[i] + B[i];
}

__device__ void cholesky(double *L, double *A, int n)
{
  assert(L != NULL);

  for (int i = 0; i < n; i++)
    for (int j = 0; j < (i + 1); j++)
    {
      double s = 0;
      for (int k = 0; k < j; k++)
        s += L[i * n + k] * L[j * n + k];
      L[i * n + j] = (i == j) ? sqrt(A[i * n + i] - s)
                              : (1.0 / L[j * n + j] * (A[i * n + j] - s));
    }
}
/*
  *chi-square distribution
*/
__device__ double curand_gamma(curandState_t *localState, const double a, const double b)
{
  /* assume a > 0 */

  if (a < 1)
  {
    double u = curand_uniform_double(localState);
    return curand_gamma(localState, 1.0 + a, b) * pow(u, 1.0 / a);
  }

  {
    double x, v, u;
    double d = a - 1.0 / 3.0;
    double c = (1.0 / 3.0) / sqrt(d);

    while (1)
    {
      do
      {
        x = curand_normal_double(localState);
        v = 1.0 + c * x;
      } while (v <= 0);

      v = v * v * v;
      u = curand_uniform_double(localState);

      if (u < 1 - 0.0331 * x * x * x * x)
        break;

      if (log(u) < 0.5 * x * x + d * (1 - v + log(v)))
        break;
    }
    return b * d * v;
  }
}

__device__ double curand_chi_square(curandState_t *localState, const float nu)
{
  double chisq = 2 * curand_gamma(localState, nu / 2, 1.0);
  return chisq;
}

__constant__ dim_t dev_d;
__constant__ dim_t dev_N;
__constant__ double dev_norm;
__constant__ float dev_df;
// __constant__ double F_coeffs;

// __device__ void generate_uniform(double *result, int n,
//                                  curandState *randState) {
//   int count = 0;
//   while (count < n) {
//     double myrandf = curand_uniform(randState);
//     myrandf *= (1 - 0 + 0.999999);
//     myrandf += 0;
//     int myrand = (int)truncf(myrandf);

//     assert(myrand <= 1);
//     assert(myrand >= 0);
//     result[myrand - 0]++;
//     count++;
//   }
// }

__device__ void mvnSample(double *dist_draws, double *x, double *mu, double *Q,
                          curandStateXORWOW_t *randState)
{
  double2 rand_x;
  int n_iterations = 1;
  double *Qx = new double[dev_d * dev_d];
  // double *x = new double[dev_d];
  double *sum = new double[dev_d];

  // Generator
  // Normal Distribution N(0, 1)
  sum[0] = 0;
  sum[1] = 0;
  // for (unsigned i = 0; i < n_iterations; ++i) {
  // rand_x = curand_normal2_double(randState);
  // x[0] = rand_x.x;
  // x[1] = rand_x.y;

  //   for (unsigned j = 0; j < d; ++j) {
  //     x[j] = 0.5 * (x[j] + 1);
  //     sum[j] = sum[j] + x[j];
  //   }
  // }

  // for (unsigned i = 0; i < d; ++i) {
  //   sum[i] = sum[i] - (static_cast<double>(n_iterations) / 2) * 1;
  //   x[i] = sum[i] / (sqrt(static_cast<double>(n_iterations) / 12));
  // }

  matrix_mult(Qx, Q, x, dev_d, dev_d, 1);
  vectorAdd(dist_draws, Qx, mu, dev_d);

  free(Qx);
  // free(x);
  free(sum);

  // cholesky(L, sigma, d);
  // matrix_mult(Q, x, L, d, 1, d);
  // vectorAdd(dist_draws, Q, mu, d);
}

__global__ void setup_kernel(curandState_t *state)
{
  // int x = threadIdx.x;
  // int y = blockIdx.x * blockDim.y + threadIdx.y;
  // int idx = y * dev_d + x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < dev_N * dev_d)
    curand_init(1234LLU, idx, 0, &state[idx]);
}

__global__ void tdist_rand_kernel(double *rand_x, double *rand_chi, curandState_t *randState)
{
  // int x = threadIdx.x;
  // int y = blockIdx.x * blockDim.y + threadIdx.y;
  // int idx = y * dev_d + x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < dev_N * dev_d)
    rand_x[idx] = curand_normal_double(randState + idx);
  rand_chi[idx] = curand_chi_square(randState + idx, dev_df);
  rand_chi[idx] = std::sqrt(dev_df / rand_chi[idx]);
}

#define TILE_SIZE 16

__global__ void mvt_sample_kernel(double *post_x_t, double *pre_x_t,
                                  double *Q, double *rand_norm_vars, double *rand_tdist_vars)
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
    post_x_t[el] = rand_tdist_vars[el] * dotProduct + pre_x_t[el];
}

__global__ void mvtpdf_kernel_y_minus_Fmu(double *dev_alpha_t, double *dev_y_t, //same as mvn
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
  // printf("\nel: %d, y_el: %d", matOffset + el, el);
  if (blockIdx.x < dev_N && rowIdx < m && colIdx < n)
    dev_alpha_t[matOffset + el] = dev_y_t[el] - dotProduct;
}

__global__ void mvtpdf_kernel_Einv_alpha(double *dev_Ealpha_t, //same as mvn
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

__global__ void mvtpdf_kernel_xtra(double *dev_w_t, double *dev_alpha_t,
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

  m = 1;
  k = 2;
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
    {
      // printf("\n[%d][%d][%d]: %f, %f", y, x, i, _Ealpha[y][i], _alpha[i][x]);
      quadform += _alpha[y][i] * _Ealpha[i][x];
    }

    __syncthreads();
  }

  int el = blockIdx.x;
  __syncthreads();

  // save dot product
  double _quadform = 1.0f + quadform / dev_df;
  double power = -0.5 * (dev_d + dev_df);
  if (rowIdx < m && colIdx < n)
  {
    dev_w_t[el] = dev_norm * std::pow(_quadform, power);
  }
}

// namespace Kernel {
void metropolis(unsigned *a_t, Eigen::VectorXd *w_t, int N, unsigned t)
{
  // Generator
  std::random_device randomDevice{};
  std::mt19937 generator{randomDevice()};
  // Uniform Distribution between 0 and 1
  std::uniform_real_distribution<> U_u{0, 1};
  // Descrete Uniform Distribution between 1 and N
  std::uniform_int_distribution<> U_j(0, N - 1);

  int B = 10;

  double u;
  int j, k;

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

void propagate_K(Eigen::VectorXd **post_x_t, unsigned *a_t,
                 const Eigen::MatrixXd Q, const dim_t N, const dim_t d,
                 const dim_t t, const float df)
{
  // shuffle pre_x_t
  Eigen::VectorXd *host_pre_x_t = new Eigen::VectorXd[N];
  for (unsigned i = 0; i < N; ++i)
  {
    host_pre_x_t[i] = Eigen::VectorXd(d);
    // for (unsigned j = 0; j <)
    host_pre_x_t[i] = post_x_t[t - 1][a_t[t * N + i]];
  }

  double *host_x_t = (double *)malloc(sizeof(double) * N * d);
  for (int y = 0; y < N; ++y)
    for (unsigned x = 0; x < d; ++x)
      host_x_t[y * d + x] = host_pre_x_t[y][x];

  cudaError_t cuda_ret;
  const size_t VECTOR_SZ = sizeof(double) * N * d;
  const size_t COVMAT_SZ = sizeof(double) * d * d;

  double *dev_pre_x_t;
  double *dev_post_x_t;
  double *dev_Q;
  double *dev_norm_rand;
  double *dev_tdist_rand;
  curandState *devStates;

  /* Allocate space for prng states on device */
  CUDA_CALL(cudaMalloc((void **)&devStates, N * d * sizeof(curandState)));
  CUDA_CALL(cudaMalloc((void **)&dev_norm_rand, VECTOR_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_tdist_rand, VECTOR_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_pre_x_t, VECTOR_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_post_x_t, VECTOR_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_Q, COVMAT_SZ));
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to allocate memory on device");

  // double *host_norm_rand;
  double *host_Q;

  // host_norm_rand = (double *)malloc(VECTOR_SZ);
  host_Q = (double *)malloc(COVMAT_SZ);

  for (size_t i = 0; i < d; ++i)
    for (size_t j = 0; j < d; ++j)
      host_Q[i * d + j] = Q(i, j);

  CUDA_CALL(
      cudaMemcpy(dev_pre_x_t, host_x_t, VECTOR_SZ, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_Q, host_Q, COVMAT_SZ, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpyToSymbol(dev_d, &d, sizeof(dim_t)));
  CUDA_CALL(cudaMemcpyToSymbol(dev_N, &N, sizeof(dim_t)));
  CUDA_CALL(cudaMemcpyToSymbol(dev_df, &df, sizeof(float)));
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to transfer data to device");

  dim3 rand_blockDim(512);
  int rand_gridDim = ceil(double(N * d) / double(rand_blockDim.x));

  /* Setup prng states */
  setup_kernel<<<rand_gridDim, rand_blockDim>>>(devStates);
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to launch kernel: setup_kernel");

  // Generate random numbers
  tdist_rand_kernel<<<rand_gridDim, rand_blockDim>>>(dev_norm_rand, dev_tdist_rand, devStates);
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to launch kernel: tdist_rand_kernel");

  const unsigned int BLOCK_SIZE = TILE_SIZE;

  dim3 blockDim(1, BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(N, ceil(double(d) / double(BLOCK_SIZE)),
               ceil(double(d) / double(BLOCK_SIZE)));

  mvt_sample_kernel<<<gridDim, blockDim>>>(dev_post_x_t, dev_pre_x_t,
                                           dev_Q, dev_norm_rand, dev_tdist_rand);
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to launch kernel: mvt_sample_kernel");

  CUDA_CALL(
      cudaMemcpy(host_x_t, dev_post_x_t, VECTOR_SZ, cudaMemcpyDeviceToHost));
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to transfer data to host");

  // for (int i = 0; i < N; ++i)
  //   std::cout << host_x_t[i] << std::endl;
  for (unsigned i = 0; i < N; ++i)
    for (unsigned j = 0; j < d; ++j)
    {
      post_x_t[t][i][j] = host_x_t[i * d + j];

      // std::cout << host_x_t[i * d + j] << std::endl;
    }

  cudaFree(devStates);
  cudaFree(dev_pre_x_t);
  cudaFree(dev_post_x_t);
  cudaFree(dev_Q);
  free(host_x_t);
}

void reweight_G(Eigen::VectorXd *w_t, const Eigen::VectorXd *y_t,
                Eigen::VectorXd **post_x_t, const double norm,
                const Eigen::MatrixXd &E_inv, const Eigen::MatrixXd E,
                const Eigen::MatrixXd F, const dim_t N, const dim_t d,
                const dim_t t, const float df)
{
  cudaError_t cuda_ret;
  double *dev_w_t;
  double *dev_x_t;
  double *dev_y_t;
  double *dev_E_inv;
  double *dev_F;
  double *dev_Ealpha_t;
  double *dev_alpha_t;

  const size_t WEIGHTS_SZ = sizeof(double) * N;
  const size_t INPUT_SZ = sizeof(double) * d;
  const size_t PARTICLES_SZ = sizeof(double) * N * d;
  const size_t COVMAT_SZ = sizeof(double) * d * d;

  CUDA_CALL(cudaMalloc((void **)&dev_w_t, WEIGHTS_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_y_t, INPUT_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_x_t, PARTICLES_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_E_inv, COVMAT_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_F, COVMAT_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_Ealpha_t, PARTICLES_SZ));
  CUDA_CALL(cudaMalloc((void **)&dev_alpha_t, PARTICLES_SZ));
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to allocate memory on device");

  double *host_w_t;
  double *host_y_t;
  double *host_x_t;
  double *host_E_inv;
  double *host_F;

  host_w_t = (double *)malloc(WEIGHTS_SZ);
  host_y_t = (double *)malloc(INPUT_SZ);
  host_x_t = (double *)malloc(PARTICLES_SZ);
  host_E_inv = (double *)malloc(COVMAT_SZ);
  host_F = (double *)malloc(COVMAT_SZ);

  for (unsigned i = 0; i < N; ++i)
    for (unsigned j = 0; j < d; ++j)
    {
      host_y_t[j] = y_t[t][j];
      host_x_t[i * d + j] = post_x_t[t][i][j];
    }

  for (unsigned i = 0; i < d; ++i)
    for (unsigned j = 0; j < d; ++j)
      host_E_inv[i * d + j] = E_inv(i, j);

  for (unsigned i = 0; i < d; ++i)
    for (unsigned j = 0; j < d; ++j)
      host_F[i * d + j] = F(i, j);

  CUDA_CALL(cudaMemcpy(dev_y_t, host_y_t, INPUT_SZ, cudaMemcpyHostToDevice));
  CUDA_CALL(
      cudaMemcpy(dev_x_t, host_x_t, PARTICLES_SZ, cudaMemcpyHostToDevice));
  CUDA_CALL(
      cudaMemcpy(dev_E_inv, host_E_inv, COVMAT_SZ, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_F, host_F, COVMAT_SZ, cudaMemcpyHostToDevice));

  // Copy constants
  CUDA_CALL(cudaMemcpyToSymbol(dev_d, &d, sizeof(dim_t)));
  CUDA_CALL(cudaMemcpyToSymbol(dev_N, &N, sizeof(dim_t)));
  CUDA_CALL(cudaMemcpyToSymbol(dev_norm, &norm, sizeof(double)));
  CUDA_CALL(cudaMemcpyToSymbol(dev_df, &df, sizeof(float)));

  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to transfer data to device");

  // dim3 blockDim(2, 512);
  // int gridDim = ceil(double(N * 2) / double(blockDim.y));
  // // std::cout << "gridDim: " << gridDim << std::endl;
  // // std::cout << "blockDim: " << blockDim.x << ", " << blockDim.y <<
  // std::endl;

  // mvnpdf_kernel<<<gridDim, blockDim>>>(dev_w_t, dev_y_t, dev_x_t, dev_E_inv,
  //                                      dev_F);
  // cuda_ret = cudaDeviceSynchronize();
  // if (cuda_ret != cudaSuccess)
  //   FATAL("Unable to launch kernel: mvnpdf_kernel_xtra");

  const unsigned int BLOCK_SIZE = TILE_SIZE;

  dim3 blockDim(1, BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(N, ceil(double(d) / double(BLOCK_SIZE)),
               ceil(double(d) / double(BLOCK_SIZE)));

  mvtpdf_kernel_y_minus_Fmu<<<gridDim, blockDim>>>(dev_alpha_t, dev_y_t,
                                                   dev_x_t, dev_F);
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to launch kernel: mvtpdf_kernel_y_minus_Fmu");

  mvtpdf_kernel_Einv_alpha<<<gridDim, blockDim>>>(dev_Ealpha_t, dev_alpha_t,
                                                  dev_E_inv);
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to launch kernel: mvtpdf_kernel_Einv_alpha");

  mvtpdf_kernel_xtra<<<gridDim, blockDim>>>(dev_w_t, dev_alpha_t, dev_Ealpha_t);
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to launch kernel: mvtpdf_kernel_xtra");

  cudaMemcpy(host_w_t, dev_w_t, WEIGHTS_SZ, cudaMemcpyDeviceToHost);
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to transfer data to host");

  for (unsigned i = 0; i < N; ++i)
    w_t[t][i] = host_w_t[i];

  cudaDeviceReset();

  cudaFree(dev_w_t);
  cudaFree(dev_y_t);
  cudaFree(dev_x_t);
  cudaFree(dev_E_inv);

  free(host_w_t);
  free(host_x_t);
  free(host_y_t);
  free(host_E_inv);
}

// } // namespace Kernel
#endif