#define BLOCK_SIZE 16
#define TILE_WIDTH BLOCK_SIZE
#define SHARED_ROWS (TILE_WIDTH + KERNEL_SIZE - 1)
#define SHARED_COLS (TILE_WIDTH + KERNEL_SIZE - 1)

__global__ void convolution(float *a, float *b, float *c, int ha, int wa,
                            int hb, int wb, int hc, int wc, int input_channels,
                            int output_channels) {
  int output_block_size = BLOCK_SIZE - wc + 1;
  int col = blockIdx.x * output_block_size + (threadIdx.x / output_block_size);
  int row = blockIdx.y * output_block_size + (threadIdx.y % output_block_size);
  int row_i = row - wc + 1;
  int col_i = col - wc + 1;

  float tmp = 0.0;

  __shared__ float shm[BLOCK_SIZE][BLOCK_SIZE];

  if (row_i < wa && row_i >= 0 && col_i < wa && col_i >= 0) {
    shm[threadIdx.y][threadIdx.x] = a[col_i * wa + row_i];
  } else {
    shm[threadIdx.y][threadIdx.x] = 0.0f;
  }

  __syncthreads();

  if (threadIdx.y < (BLOCK_SIZE - wc + 1) &&
      threadIdx.x < (BLOCK_SIZE - wc + 1) && row < (wb - wc + 1) &&
      col < (wb - wc + 1)) {
    for (int i = 0; i < wc; i++) {
      for (int j = 0; j < wc; j++) {
        tmp += shm[threadIdx.y + i][threadIdx.x + j] * c[j * wc + i];
      }
    }
    b[col * wb + row] = tmp;
  }
}

__global__ void convolution3d(float *a, float *b, float *c, int ha, int wa,
                              int hb, int wb, int hc, int wc,
                              int input_channels, int output_channels) {
  extern __shared__ float tile[];
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int out_col = blockIdx.x * TILE_WIDTH + tx;
  int out_row = blockIdx.y * TILE_WIDTH + ty;
  int out_channel = blockIdx.z;

  float tmp = 0.0f;

  int linear_tid = ty * TILE_WIDTH + tx;
  int tile_width = TILE_WIDTH + wc - 1;
  int tile_height = TILE_WIDTH + hc - 1;
  int tile_size = tile_width * tile_height;

  for (int in_channel = 0; in_channel < input_channels; ++in_channel) {
    float *input = a + in_channel * ha * wa;
    float *kernel = c + (out_channel * input_channels + in_channel) * hc * wc;

    for (int i = linear_tid; i < tile_size; i += TILE_WIDTH * TILE_WIDTH) {
      int row = blockIdx.y * TILE_WIDTH + (i / tile_width);
      int col = blockIdx.x * TILE_WIDTH + (i % tile_width);

      tile[i] = (row < ha && col < wa) ? input[row * wa + col] : 0.0f;
    }

    __syncthreads();

    if (out_row < hb && out_col < wb) {
      for (int i = 0; i < hc; ++i) {
        for (int j = 0; j < wc; ++j) {
          tmp += tile[(ty + i) * tile_width + (tx + j)] * kernel[i * wc + j];
        }
      }
    }

    __syncthreads();
  }

  if (out_row < hb && out_col < wb) {
    b[out_channel * hb * wb + out_row * wb + out_col] = tmp;
  }
}

__global__ void max_pool2d(float *a, float *b, int ha, int wa, int hb, int wb,
                           int input_channels) {
  int out_index = blockIdx.x * blockDim.x + threadIdx.x;
  int total_outputs = hb * wb * input_channels;
  if (out_index >= total_outputs)
    return;

  int out_row = (out_index / wb) % hb;
  int out_col = out_index % wb;
  int input_channel = out_index / (hb * wb);

  int in_row = out_row * 2;
  int in_col = out_col * 2;

  float temp = -1000.0f;

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      int r = in_row + i;
      int c = in_col + j;
      float val = a[input_channel * ha * wa + r * wa + c];
      if (val > temp)
        temp = val;
    }
  }

  b[input_channel * hb * wb + out_row * wb + out_col] = temp;
}

__global__ void relu_kernel(float *b, int hb, int wb, int channels) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = hb * wb * channels;

  if (tid >= total_elements)
    return;

  float &val = b[tid];
  if (val < 0.0f)
    val = 0.0f;
}

__global__ void relu_kernel(float *b, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n && b[idx] < 0.0f) {
    b[idx] = 0.0f;
  }
}

__global__ void sgemm(float *a, float *b, float *c, int ha, int wa, int hb,
                      int wb, int hc, int wc) {
  __shared__ float ds_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float ds_b[BLOCK_SIZE][BLOCK_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * BLOCK_SIZE + tx;
  int row = blockIdx.y * BLOCK_SIZE + ty;

  float p_value = 0;

  for (int p = 0; p < (wa + BLOCK_SIZE - 1) / BLOCK_SIZE; p++) {
    if (row < ha && (p * BLOCK_SIZE + tx) < wa) {
      ds_a[ty][tx] = a[row * wa + (p * BLOCK_SIZE + tx)];
    } else {
      ds_a[ty][tx] = 0.0;
    }

    if ((p * BLOCK_SIZE + ty) < hb && col < wb) {
      ds_b[ty][tx] = b[(p * BLOCK_SIZE + ty) * wb + col];
    } else {
      ds_b[ty][tx] = 0.0;
    }

    __syncthreads();

    for (int i = 0; i < BLOCK_SIZE; i++) {
      p_value += ds_a[ty][i] * ds_b[i][tx];
    }

    __syncthreads();
  }

  if (row < hc && col < wc) {
    c[row * wc + col] = p_value;
  }
}

__global__ void vec_add(float *a_vec, float *b_vec, bool neg, int len) {
  int sign = neg ? -1 : 1;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < len)
    a_vec[i] += (b_vec[i] * sign);
}

__global__ void mat_add(float *a, float *b, float *c, int rows, int cols,
                        bool neg, float alpha) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int idx = row * cols + col;
  float sign = neg ? -1.0 : 1.0;

  if (row < rows && col < cols) {
    c[idx] = a[idx] + (b[idx] * sign * alpha);
  }
}

__global__ void softmax_kernel(const float *input, float *output, int len) {
  __shared__ float max_val;
  __shared__ float sum_exp;

  if (threadIdx.x == 0) {
    float max_tmp = input[0];
    for (int i = 1; i < len; ++i) {
      if (input[i] > max_tmp)
        max_tmp = input[i];
    }
    max_val = max_tmp;
  }
  __syncthreads();

  float local = 0.0f;
  for (int i = threadIdx.x; i < len; i += blockDim.x) {
    local += expf(input[i] - max_val);
  }

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

  for (int i = threadIdx.x; i < len; i += blockDim.x) {
    output[i] = expf(input[i] - max_val) / sum_exp;
  }
}

__device__ __host__ float
compute_cross_entropy_loss(float *d_output, float *d_target, int length) {
  float loss = 0.0f;
  for (int i = 0; i < length; ++i) {
    if (d_target[i] > 0) {
      loss = -logf(d_output[i] + 1e-8);
      break;
    }
  }
  return loss;
}

__device__ __host__ float compute_cross_entropy_from_logits(const float *logits,
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

__global__ void softmax_cross_entropy_backward(float *softmax_output,
                                               float *label, float *grad_output,
                                               int length) {
  int idx = threadIdx.x;
  if (idx < length) {
    grad_output[idx] = softmax_output[idx] - label[idx];
  }
}

__global__ void outer_product(float *d_out, float *input, float *d_w,
                              int out_size, int in_size) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < out_size && col < in_size) {
    d_w[row * in_size + col] = d_out[row] * input[col];
  }
}

__global__ void relu_backward(float *input, float *grad_output,
                              float *grad_input, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    grad_input[idx] = input[idx] > 0 ? grad_output[idx] : 0;
  }
}

__global__ void max_pool_backward(float *d_out, int *max_indices,
                                  float *d_input, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int max_idx = max_indices[idx];
    d_input[max_idx] = d_out[idx];
  }
}

__global__ void sgd_update(float *weights, float *grad, float lr, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    weights[idx] -= lr * grad[idx];
  }
}

__global__ void tensor_elementwise_mult(float *a, float *b, float *c,
                                        int total_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    c[idx] = a[idx] * b[idx];
  }
}

__global__ void transpose_kernel(float *a, float *b, int ha, int wa) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int size = ha * wa;

  if (idx < size) {
    int row = idx / wa;
    int col = idx % wa;

    b[col * ha + row] = a[row * wa + col];
  }
}

__global__ void convolution3d_1d_launch(float *a, float *b, float *c, int ha,
                                        int wa, int hb, int wb, int hc, int wc,
                                        int input_channels,
                                        int output_channels) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_outputs = output_channels * hb * wb;

  if (tid >= total_outputs)
    return;

  int out_channel = tid / (hb * wb);
  int rem = tid % (hb * wb);
  int out_row = rem / wb;
  int out_col = rem % wb;

  float tmp = 0.0f;

  for (int in_channel = 0; in_channel < input_channels; ++in_channel) {
    float *input = a + in_channel * ha * wa;
    float *kernel = c + (out_channel * input_channels + in_channel) * hc * wc;

    for (int i = 0; i < hc; ++i) {
      for (int j = 0; j < wc; ++j) {
        int r = out_row + i;
        int col = out_col + j;

        if (r < ha && col < wa) {
          tmp += input[r * wa + col] * kernel[i * wc + j];
        }
      }
    }
  }

  b[out_channel * hb * wb + out_row * wb + out_col] = tmp;
}
