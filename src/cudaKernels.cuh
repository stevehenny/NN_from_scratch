#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#define BLOCK_SIZE 16
#define TILE_WIDTH BLOCK_SIZE
#define SHARED_ROWS (TILE_WIDTH + KERNEL_SIZE - 1)
#define SHARED_COLS (TILE_WIDTH + KERNEL_SIZE - 1)

// CUDA kernel declarations

__global__ void Convolution(float *A, float *B, float *C, int HA, int WA,
                            int HB, int WB, int HC, int WC,
                            int input_channels, int output_channels);

__global__ void Convolution3D(float *A, float *B, float *C, int HA, int WA,
                              int HB, int WB, int HC, int WC,
                              int input_channels, int output_channels);

__global__ void maxPool2D(float *A, float *B, int HA, int WA, int HB, int WB,
                          int input_channels);

__global__ void ReLU_kernel(float *B, int HB, int WB, int channels);

__global__ void ReLU_kernel(float *B, int N);

__global__ void sgemm(float *A, float *B, float *C, int HA, int WA, int HB,
                      int WB, int HC, int WC);

template <const int block_size>
__global__ void sgemm_1d(float *A, float *B, float *C, int HA, int WA, int HB,
                         int WB, int HC, int WC) {
  // Shared memory as flat arrays
  __shared__ float ds_A[block_size * block_size];
  __shared__ float ds_B[block_size * block_size];

  // 1D thread and block index
  int thread_id = threadIdx.x;
  int block_id = blockIdx.x;

  // Map 1D thread index to 2D within the tile
  int tx = thread_id % block_size;
  int ty = thread_id / block_size;

  // Determine global output row and col
  int Col = (block_id % ((WC + block_size - 1) / block_size)) * block_size + tx;
  int Row = (block_id / ((WC + block_size - 1) / block_size)) * block_size + ty;

  float Pvalue = 0.0f;

  for (int p = 0; p < (WA + block_size - 1) / block_size; p++) {
    // Load A tile
    if (Row < HA && (p * block_size + tx) < WA) {
      ds_A[ty * block_size + tx] = A[Row * WA + (p * block_size + tx)];
    } else {
      ds_A[ty * block_size + tx] = 0.0f;
    }

    // Load B tile
    if ((p * block_size + ty) < HB && Col < WB) {
      ds_B[ty * block_size + tx] = B[(p * block_size + ty) * WB + Col];
    } else {
      ds_B[ty * block_size + tx] = 0.0f;
    }

    __syncthreads();

    for (int i = 0; i < block_size; i++) {
      Pvalue += ds_A[ty * block_size + i] * ds_B[i * block_size + tx];
    }

    __syncthreads();
  }

  if (Row < HC && Col < WC) {
    C[Row * WC + Col] = Pvalue;
  }
}

__global__ void vecAdd(float *A_vec, float *B_vec, bool neg, int len);

__global__ void matAdd(float *A, float *B, float *C, int rows, int cols,
                       bool neg, float alpha);

__global__ void softmaxKernel(const float *input, float *output, int len);

__device__ __host__ float computeCrossEntropyLoss(float *d_output,
                                                  float *d_target, int length);

__device__ __host__ float computeCrossEntropyFromLogits(const float *logits,
                                                        int target_class,
                                                        int length);

__global__ void softmaxCrossEntropyBackward(float *softmax_output, float *label,
                                            float *grad_output, int length);

__global__ void outerProduct(float *d_out, float *input, float *dW,
                             int out_size, int in_size);

__global__ void reluBackward(float *input, float *grad_output,
                             float *grad_input, int size);

__global__ void maxPoolBackward(float *d_out, int *max_indices, float *d_input,
                                int size);

__global__ void sgdUpdate(float *weights, float *grad, float lr, int size);

__global__ void tensorElementwiseMult(float *A, float *B, float *C,
                                      int totalElements);

__global__ void transposeKernel(float *A, float *B, int HA, int WA);

__global__ void Convolution3D_1d_launch(float *A, float *B, float *C, int HA,
                                        int WA, int HB, int WB, int HC, int WC,
                                        int input_channels,
                                        int output_channels);

#endif // CUDA_KERNELS_H
