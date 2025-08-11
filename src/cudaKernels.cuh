#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#define BLOCK_SIZE 16
#define TILE_WIDTH BLOCK_SIZE
#define SHARED_ROWS (TILE_WIDTH + KERNEL_SIZE - 1)
#define SHARED_COLS (TILE_WIDTH + KERNEL_SIZE - 1)

// CUDA kernel declarations

__global__ void convolution(float *a, float *b, float *c, int ha, int wa,
                            int hb, int wb, int hc, int wc, int input_channels,
                            int output_channels);

__global__ void convolution3d(float *a, float *b, float *c, int ha, int wa,
                              int hb, int wb, int hc, int wc,
                              int input_channels, int output_channels);

__global__ void max_pool2d(float *a, float *b, int ha, int wa, int hb, int wb,
                           int input_channels);

__global__ void relu_kernel(float *b, int hb, int wb, int channels);

__global__ void relu_kernel(float *b, int n);

__global__ void sgemm(float *a, float *b, float *c, int ha, int wa, int hb,
                      int wb, int hc, int wc);

template <const int block_size>
__global__ void sgemm_1d(float *A, float *B, float *C, int HA, int WA, int HB,
                         int WB, int HC, int WC) {
  __shared__ float ds_A[block_size * block_size];
  __shared__ float ds_B[block_size * block_size];

  // Thread & block index
  int thread_id = threadIdx.x;
  int block_id = blockIdx.x;

  // Map 1D thread index to 2D position in tile
  int tx = thread_id % block_size;
  int ty = thread_id / block_size;

  // Grid tile indexing
  int tiles_per_row = (WC + block_size - 1) / block_size;
  int Col = (block_id % tiles_per_row) * block_size + tx;
  int Row = (block_id / tiles_per_row) * block_size + ty;

  // --- Batch offset ---
  int batch_id = blockIdx.z; // z dimension is batch index
  size_t A_batch_offset = batch_id * (HA * WA);
  size_t B_batch_offset = batch_id * (HB * WB);
  size_t C_batch_offset = batch_id * (HC * WC);

  float Pvalue = 0.0f;

  // Loop over tiles
  for (int p = 0; p < (WA + block_size - 1) / block_size; p++) {
    // Load A tile
    if (Row < HA && (p * block_size + tx) < WA) {
      ds_A[ty * block_size + tx] =
          A[A_batch_offset + Row * WA + (p * block_size + tx)];
    } else {
      ds_A[ty * block_size + tx] = 0.0f;
    }

    // Load B tile
    if ((p * block_size + ty) < HB && Col < WB) {
      ds_B[ty * block_size + tx] =
          B[B_batch_offset + (p * block_size + ty) * WB + Col];
    } else {
      ds_B[ty * block_size + tx] = 0.0f;
    }

    __syncthreads();

    // Multiply and accumulate
    for (int i = 0; i < block_size; i++) {
      Pvalue += ds_A[ty * block_size + i] * ds_B[i * block_size + tx];
    }

    __syncthreads();
  }

  // Write result
  if (Row < HC && Col < WC) {
    C[C_batch_offset + Row * WC + Col] = Pvalue;
  }
}

__global__ void vec_add(float *a_vec, float *b_vec, float *c_vec, bool neg,
                        int len);

__global__ void mat_add(float *a, float *b, float *c, int rows, int cols,
                        bool neg, float alpha);

__global__ void softmax_kernel(const float *input, float *output, int len);

__device__ __host__ float
compute_cross_entropy_loss(float *d_output, float *d_target, int length);

__device__ __host__ float compute_cross_entropy_from_logits(const float *logits,
                                                            int target_class,
                                                            int length);

__global__ void softmax_cross_entropy_backward(float *softmax_output,
                                               float *label, float *grad_output,
                                               int length);

__global__ void outer_product(float *d_out, float *input, float *d_w,
                              int out_size, int in_size);

__global__ void relu_backward(float *input, float *grad_output,
                              float *grad_input, int size);

__global__ void max_pool_backward(float *d_out, int *max_indices,
                                  float *d_input, int size);

__global__ void sgd_update(float *weights, float *grad, float lr, int size);

__global__ void tensor_elementwise_mult(float *a, float *b, float *c,
                                        int total_elements);

__global__ void transpose_kernel(float *a, float *b, int ha, int wa);

__global__ void convolution3d_1d_launch(float *a, float *b, float *c, int ha,
                                        int wa, int hb, int wb, int hc, int wc,
                                        int input_channels,
                                        int output_channels);

#endif // CUDA_KERNELS_H
