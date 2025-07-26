#include "cudaKernels.cuh"
#define TILE_WIDTH BLOCK_SIZE
#define SHARED_ROWS (TILE_WIDTH + KERNEL_SIZE - 1)
#define SHARED_COLS (TILE_WIDTH + KERNEL_SIZE - 1)

__global__ void Convolution(float *A, float *B, float *C, int HA, int WA,
                            int HB, int WB, int HC, int WC, int input_channels,
                            int output_channels) {
  int output_block_size = BLOCK_SIZE - WC + 1;
  int col = blockIdx.x * output_block_size + (threadIdx.x / output_block_size);
  int row = blockIdx.y * output_block_size + (threadIdx.y % output_block_size);
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
  extern __shared__ float tile[]; // dynamically allocate memory for tile
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int out_col = blockIdx.x * TILE_WIDTH + tx;
  int out_row = blockIdx.y * TILE_WIDTH + ty;
  int out_channel = blockIdx.z;

  float tmp = 0.0f;

  int linear_tid = ty * TILE_WIDTH + tx;
  int tile_width = TILE_WIDTH + WC - 1;
  int tile_height = TILE_WIDTH + HC - 1;
  int tile_size = tile_width * tile_height;

  for (int in_channel = 0; in_channel < input_channels; ++in_channel) {
    float *input = A + in_channel * HA * WA;
    float *kernel = C + (out_channel * input_channels + in_channel) * HC * WC;

    // Coalesced load into shared memory
    for (int i = linear_tid; i < tile_size; i += TILE_WIDTH * TILE_WIDTH) {
      int row = blockIdx.y * TILE_WIDTH + (i / tile_width);
      int col = blockIdx.x * TILE_WIDTH + (i % tile_width);

      tile[i] = (row < HA && col < WA) ? input[row * WA + col] : 0.0f;
    }

    __syncthreads();

    // Compute output
    if (out_row < HB && out_col < WB) {
      for (int i = 0; i < HC; ++i) {
        for (int j = 0; j < WC; ++j) {
          tmp += tile[(ty + i) * tile_width + (tx + j)] * kernel[i * WC + j];
        }
      }
    }

    __syncthreads(); // sync before loading next input channel
  }

  // Write result
  if (out_row < HB && out_col < WB) {
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

// Kernel for Conv layers
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

// different kernel for MLP layers
__global__ void ReLU_kernel(float *B, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N && B[idx] < 0.0f) {
    B[idx] = 0.0f;
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

    __syncthreads();

    // Multiply the two tiles and accumulate the result
    for (int i = 0; i < BLOCK_SIZE; i++) {
      Pvalue += ds_A[ty][i] * ds_B[i][tx];
    }

    __syncthreads();
  }

  // Store the result in C, only if within bounds
  if (Row < HC && Col < WC) {
    C[Row * WC + Col] = Pvalue;
  }
}

__global__ void vecAdd(float *A_vec, float *B_vec, bool neg, int len) {
  int sign = neg ? -1 : 1;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < len)
    A_vec[i] += (B_vec[i] * sign);
}

__global__ void matAdd(float *A, float *B, float *C, int rows, int cols,
                       bool neg, float alpha) {
  int row = blockIdx.y * blockDim.y + threadIdx.y; // y-index
  int col = blockIdx.x * blockDim.x + threadIdx.x; // x-index

  int idx = row * cols + col;
  float sign = neg ? -1.0 : 1.0;

  if (row < rows && col < cols) {
    C[idx] = A[idx] + (B[idx] * sign * alpha);
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
  __shared__ float block_sum[32];
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

__device__ __host__ float computeCrossEntropyLoss(float *d_output,
                                                  float *d_target, int length) {
  float loss = 0.0f;
  for (int i = 0; i < length; ++i) {
    if (d_target[i] > 0) {
      loss = -logf(d_output[i] + 1e-8); // add epsilon to avoid log(0)
      break;
    }
  }
  return loss;
}

__device__ __host__ float computeCrossEntropyFromLogits(const float *logits,
                                                        int target_class,
                                                        int length) {
  float max_logit = logits[0];
  for (int i = 1; i < length; ++i) {
    if (logits[i] > max_logit)
      max_logit = logits[i];
  }

  float sum_exp = 0.0f;
  for (int i = 0; i < length; ++i) {
    sum_exp += expf(logits[i] - max_logit);
  }

  float log_sum_exp = logf(sum_exp) + max_logit;
  float loss = log_sum_exp - logits[target_class];

  return loss;
}

__global__ void softmaxCrossEntropyBackward(float *softmax_output, float *label,
                                            float *grad_output, int length) {
  int idx = threadIdx.x;
  if (idx < length) {
    grad_output[idx] =
        softmax_output[idx] - label[idx]; // ∇L/∇z for softmax + cross-entropy
  }
}

__global__ void outerProduct(float *d_out, float *input, float *dW,
                             int out_size, int in_size) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < out_size && col < in_size) {
    dW[row * in_size + col] = d_out[row] * input[col];
  }
}

__global__ void reluBackward(float *input, float *grad_output,
                             float *grad_input, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    grad_input[idx] = input[idx] > 0 ? grad_output[idx] : 0;
  }
}

__global__ void maxPoolBackward(float *d_out, int *max_indices, float *d_input,
                                int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int max_idx = max_indices[idx];
    d_input[max_idx] = d_out[idx]; // Only route gradient to max loc
  }
}

__global__ void sgdUpdate(float *weights, float *grad, float lr, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    weights[idx] -= lr * grad[idx];
  }
}
__global__ void tensorElementwiseMult(float *A, float *B, float *C,
                                      int totalElements) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < totalElements) {
    C[idx] = A[idx] * B[idx];
  }
}

// CUDA kernel to transpose a matrix A (HA x WA) into B (WA x HA)
__global__ void transposeKernel(float *A, float *B, int HA, int WA) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int size = HA * WA;

  if (idx < size) {
    int row = idx / WA;
    int col = idx % WA;

    // Transpose: B[col][row] = A[row][col]
    B[col * HA + row] = A[row * WA + col];
  }
}

template <const size_t shared_A>
__global__ void Convolution3D_1d_launch(float *A, float *B, float *C, int HA,
                                        int WA, int HB, int WB, int HC, int WC,
                                        int input_channels,
                                        int output_channels) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_outputs = output_channels * HB * WB;

  if (tid >= total_outputs)
    return;

  int out_channel = tid / (HB * WB);
  int rem = tid % (HB * WB);
  int out_row = rem / WB;
  int out_col = rem % WB;

  float tmp = 0.0f;

  for (int in_channel = 0; in_channel < input_channels; ++in_channel) {
    float *input = A + in_channel * HA * WA;
    float *kernel = C + (out_channel * input_channels + in_channel) * HC * WC;

    for (int i = 0; i < HC; ++i) {
      for (int j = 0; j < WC; ++j) {
        int r = out_row + i;
        int c = out_col + j;

        if (r < HA && c < WA) {
          tmp += input[r * WA + c] * kernel[i * WC + j];
        }
      }
    }
  }

  B[out_channel * HB * WB + out_row * WB + out_col] = tmp;
}
