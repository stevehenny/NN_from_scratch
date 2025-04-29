#include "cudaKernels.cuh"

#define TILE_WIDTH BLOCK_SIZE
#define SHARED_ROWS (TILE_WIDTH + KERNEL_SIZE - 1)
#define SHARED_COLS (TILE_WIDTH + KERNEL_SIZE - 1)

__global__ void Convolution(float *A, float *B, float *C, int HA, int WA,
                            int HB, int WB, int HC, int WC, int input_channels,
                            int output_channels) {
  int col = blockIdx.x * (BLOCK_SIZE - WC + 1) + threadIdx.x;
  int row = blockIdx.y * (BLOCK_SIZE - WC + 1) + threadIdx.y;
  int row_i = row - WC + 1;
  int col_i = col - WC + 1;

  float tmp = 0.0;

  // Declare shared memory for a tile of A
  __shared__ float shm[BLOCK_SIZE][BLOCK_SIZE];

  if (row_i < WA && row_i >= 0 && col_i < WA && col_i >= 0) {
    shm[threadIdx.y][threadIdx.x] = A[col_i * WA + row_i];
  } else {
    shm[threadIdx.y][threadIdx.x] = 0.0f;
  }

  __syncthreads();

  if (threadIdx.y < (BLOCK_SIZE - WC + 1) &&
      threadIdx.x < (BLOCK_SIZE - WC + 1) && row < (WB - WC + 1) &&
      col < (WB - WC + 1)) {
    for (int i = 0; i < WC; i++) {
      for (int j = 0; j < WC; j++) {
        tmp += shm[threadIdx.y + i][threadIdx.x + j] * C[j * WC + i];
      }
    }
    B[col * WB + row] = tmp;
  }
}

__global__ void Convolution3D(float *A, float *B, float *C, int HA, int WA,
                              int HB, int WB, int HC, int WC,
                              int input_channels, int output_channels) {
  extern __shared__ float shm[]; // dynamically allocated shared memory
  float *tile = shm;             // alias for readability

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int block_output_x = blockIdx.x * (TILE_WIDTH);
  int block_output_y = blockIdx.y * (TILE_WIDTH);

  int out_col = block_output_x + tx;
  int out_row = block_output_y + ty;

  int out_channel = blockIdx.z;

  float tmp = 0.0f;

  // Loop over input channels
  for (int in_channel = 0; in_channel < input_channels; ++in_channel) {
    float *input = A + in_channel * HA * WA;
    float *kernel = C + (out_channel * input_channels + in_channel) * HC * WC;

    // Shared memory dimensions
    int shared_width = TILE_WIDTH + WC - 1;
    int shared_height = TILE_WIDTH + HC - 1;

    // Global input coordinates to load into shared memory
    int in_col = block_output_x + tx;
    int in_row = block_output_y + ty;

    // Load into shared memory
    if (ty < shared_height && tx < shared_width) {
      int shm_index = ty * shared_width + tx;
      if (in_row < HA && in_col < WA)
        tile[shm_index] = input[in_row * WA + in_col];
      else
        tile[shm_index] = 0.0f;
    }

    __syncthreads();

    // Only compute if this thread maps to a valid output location
    if (tx < TILE_WIDTH && ty < TILE_WIDTH && out_col < WB && out_row < HB) {

      for (int i = 0; i < HC; ++i) {
        for (int j = 0; j < WC; ++j) {
          int shm_row = ty + i;
          int shm_col = tx + j;
          tmp += tile[shm_row * shared_width + shm_col] * kernel[i * WC + j];
        }
      }
    }

    __syncthreads(); // safe barrier before next input channel
  }

  // Store result
  if (tx < TILE_WIDTH && ty < TILE_WIDTH && out_col < WB && out_row < HB) {
    B[out_channel * HB * WB + out_row * WB + out_col] = tmp;
  }
}
__global__ void maxPool2D(float *A, float *B, int HA, int WA, int HB, int WB,
                          int input_channels) {
  int out_col = blockIdx.x * blockDim.x + threadIdx.x;
  int out_row = blockIdx.y * blockDim.y + threadIdx.y;
  int input_channel = blockIdx.z;

  if (out_row >= HB || out_col >= WB)
    return; // bounds check

  int in_row = out_row * 2;
  int in_col = out_col * 2;

  float temp = -1000.0f;

  // 2x2 pooling window
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      int r = in_row + i;
      int c = in_col + j;
      float val = A[input_channel * HA * WA + r * WA + c];
      if (val > temp)
        temp = val;
    }
  }

  B[input_channel * HB * WB + out_row * WB + out_col] = temp;
}

__global__ void ReLU_kernel(float *B, int HB, int WB, int channels) {

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int input_channel = blockIdx.z;

  if (row >= HB || col >= WB)
    return;

  if (B[row * WB + col + input_channel * WB * HB] < 0) {
    B[row * WB + col + input_channel * WB * HB] = 0;
  }
}

// sgemm stands for single precision general matrix-matrix multiply
__global__ void sgemm(float *A, float *B, float *C, int HA, int WA, int HB,
                      int WB, int HC, int WC) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this lab
  __shared__ float ds_A[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float ds_B[BLOCK_SIZE][BLOCK_SIZE];

  // Calculate thread indexes
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int Col = blockIdx.x * BLOCK_SIZE + tx;
  int Row = blockIdx.y * BLOCK_SIZE + ty;

  // Initialize accumulation variable
  float Pvalue = 0;

  // Loop over tiles required for matrix multiplication
  for (int p = 0; p < (WA + BLOCK_SIZE - 1) / BLOCK_SIZE; p++) {
    // Load A's tile into shared memory
    if (Row < HA && (p * BLOCK_SIZE + tx) < WA) {
      ds_A[ty][tx] = A[Row * WA + (p * BLOCK_SIZE + tx)];
    } else {
      ds_A[ty][tx] = 0.0;
    }

    // Load B's tile into shared memory
    if ((p * BLOCK_SIZE + ty) < HB && Col < WB) {
      ds_B[ty][tx] = B[(p * BLOCK_SIZE + ty) * WB + Col];
    } else {
      ds_B[ty][tx] = 0.0;
    }

    // Synchronize threads to ensure tiles are loaded
    __syncthreads();

    // Multiply the two tiles and accumulate the result
    for (int i = 0; i < BLOCK_SIZE; i++) {
      Pvalue += ds_A[ty][i] * ds_B[i][tx];
    }

    // Synchronize threads before loading new tiles
    __syncthreads();
  }

  // Store the result in C, only if within bounds
  if (Row < HC && Col < WC) {
    C[Row * WC + Col] = Pvalue;
  }
}

__global__ void vecAdd(float *A_vec, float *B_vec, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < len)
    A_vec[i] += B_vec[i];
}

__global__ void matAdd(float *A, float *B, float *C, int rows, int cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y; // y-index
  int col = blockIdx.x * blockDim.x + threadIdx.x; // x-index

  int idx = row * cols + col;

  if (row < rows && col < cols) {
    C[idx] = A[idx] + B[idx];
  }
}

__global__ void softmaxKernel(const float *input, float *output, int len) {
  __shared__ float max_val;
  __shared__ float sum_exp;

  // Step 1: Find max value for numerical stability (single thread does it)
  if (threadIdx.x == 0) {
    float max_tmp = input[0];
    for (int i = 1; i < len; ++i) {
      if (input[i] > max_tmp)
        max_tmp = input[i];
    }
    max_val = max_tmp;
  }
  __syncthreads();

  // Step 2: Compute exp(x_i - max) and accumulate sum
  float local = 0.0f;
  for (int i = threadIdx.x; i < len; i += blockDim.x) {
    local += expf(input[i] - max_val);
  }

  // Use shared memory to sum partial results from each thread
  float thread_sum = local;
  __shared__ float block_sum[32]; // supports up to 1024 threads
  int lane = threadIdx.x;

  if (lane < 32)
    block_sum[lane] = 0;
  __syncthreads();

  atomicAdd(&block_sum[0], thread_sum);
  __syncthreads();

  if (threadIdx.x == 0)
    sum_exp = block_sum[0];
  __syncthreads();

  // Step 3: Compute softmax output
  for (int i = threadIdx.x; i < len; i += blockDim.x) {
    output[i] = expf(input[i] - max_val) / sum_exp;
  }
}
