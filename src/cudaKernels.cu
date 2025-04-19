#include "cudaKernels.cuh"
#define BLOCK_SIZE 32
#define SHARED_BLOCK_SIZE BLOCK_SIZE + 2

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
  int out_channel = blockIdx.z;

  // Global output location (row, col) this thread is responsible for
  int out_col = blockIdx.x * (BLOCK_SIZE - WC + 1) + threadIdx.x;
  int out_row = blockIdx.y * (BLOCK_SIZE - HC + 1) + threadIdx.y;

  __shared__ float shm[SHARED_BLOCK_SIZE][SHARED_BLOCK_SIZE];

  float tmp = 0.0f;

  for (int in_channel = 0; in_channel < input_channels; ++in_channel) {
    float *input = A + in_channel * HA * WA;
    float *kernel = C + (out_channel * input_channels + in_channel) * HC * WC;

    // Global input coordinates this thread will load into shared memory
    int in_col = blockIdx.x * (BLOCK_SIZE - WC + 1) + threadIdx.x;
    int in_row = blockIdx.y * (BLOCK_SIZE - HC + 1) + threadIdx.y;

    if (in_row < HA && in_col < WA && in_row >= 0 && in_col >= 0) {
      shm[threadIdx.y][threadIdx.x] = input[in_row * WA + in_col];
    } else {
      shm[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Compute output only from threads assigned to compute one pixel
    if (threadIdx.y < (BLOCK_SIZE - HC + 1) &&
        threadIdx.x < (BLOCK_SIZE - WC + 1) && out_row < HB && out_col < WB) {

      for (int i = 0; i < HC; ++i) {
        for (int j = 0; j < WC; ++j) {
          tmp += shm[threadIdx.y + i][threadIdx.x + j] * kernel[i * WC + j];
        }
      }

      B[out_channel * HB * WB + out_row * WB + out_col] = tmp;
    }

    __syncthreads();
  }
}

__global__ void maxPool2D(float *A, float *B, int HA, int WA, int HB, int WB,
                          int input_channels) {
  int out_col = blockIdx.x * blockDim.x + threadIdx.x;
  int out_row = blockIdx.y * blockDim.y + threadIdx.y;
  int in_col = out_col * 2;
  int in_row = out_row * 2;
  int input_channel = blockIdx.z;
  __shared__ float shm[BLOCK_SIZE][BLOCK_SIZE];

  // populate shared memory
  for (int i = in_row; i <= in_row + 1; ++i) {
    for (int j = in_col; j <= in_col + 1; ++j) {
      shm[(i + in_row) - in_row][(j + in_col) - in_col] =
          A[input_channel * WA * HA + i * WA + j];
    }
  }
  __syncthreads();

  // Take the max of the data and write to B
  float temp = -1000.0f;
  float cur;
  for (int i = in_row; i <= in_row + 1; ++i) {
    for (int j = in_col; j <= in_col + 1; ++j) {
      cur = shm[(i + in_row) - in_row][(j + in_col) - in_col];
      if (temp < cur)
        temp = cur;
    }
  }
  __syncthreads();
  B[input_channel * WB * HB + out_row * WB + out_col] = temp;
}
