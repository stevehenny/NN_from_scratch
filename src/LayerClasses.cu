#include "CudaChecks.cuh"
#include "LayerClasses.cuh"
#include "cudaKernels.cuh"
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>

#define SHARED_BLOCK_SIZE BLOCK_SIZE + 2
#define POOL_BLOCK_SIZE 16
#define KERNEL_SIZE 3

Layer::~Layer() {}

// ConvLayer

ConvLayer::ConvLayer(ImageSize input_image_size, ImageSize output_image_size,
                     ImageSize kernel_size, uint8_t input_channels,
                     uint8_t output_channels)
    : input_channels(input_channels), output_channels(output_channels),
      ha(input_image_size.height), wa(input_image_size.width),
      hb(output_image_size.height), wb(output_image_size.width),
      hc(kernel_size.height), wc(kernel_size.width) {

  std::default_random_engine gen;
  float stddev =
      sqrtf(2.0f / (input_image_size.height * input_image_size.width));
  std::normal_distribution<float> dist(0.0f, stddev);

  kernels = (float *)malloc(output_channels * input_channels * KERNEL_SIZE *
                            KERNEL_SIZE * sizeof(float));

  for (int oc = 0; oc < output_channels; ++oc) {
    for (int ic = 0; ic < input_channels; ++ic) {
      for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; ++i) {
        kernels[((oc * input_channels + ic) * KERNEL_SIZE * KERNEL_SIZE) + i] =
            dist(gen);
      }
    }
  }

  cuda_check(cudaMalloc((void **)&d_kernels, input_channels * output_channels *
                                                 KERNEL_SIZE * KERNEL_SIZE *
                                                 sizeof(float)));

  cuda_check(cudaMemcpy(d_kernels, kernels,
                        input_channels * output_channels * KERNEL_SIZE *
                            KERNEL_SIZE * sizeof(float),
                        cudaMemcpyHostToDevice));
}

ConvLayer::~ConvLayer() {
  free(kernels);
  cudaFree(d_kernels);
}

int ConvLayer::get_num_outputs() { return output_channels * hb * wb; }

void ConvLayer::forward(float *d_input_image, float *d_output_image) {
  int total_outputs = output_channels * hb * wb;
  int threads_per_block = 256;
  int num_blocks = (total_outputs + threads_per_block - 1) / threads_per_block;

  convolution3d_1d_launch<<<num_blocks, threads_per_block>>>(
      d_input_image, d_output_image, d_kernels, ha, wa, hb, wb, hc, wc,
      input_channels, output_channels);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << cudaGetErrorString(error) << std::endl;
    throw std::runtime_error("Cuda kernel failed\n");
  }

  cuda_check(cudaDeviceSynchronize());
}

//TODO: Define this back_prop method. This is a place holder override for the Layer 
// virtual method
void ConvLayer::back_prop(float *d_input, float *d_grad_output, float alpha){}

void ConvLayer::relu(float *b) {
  int total_elements = output_channels * wb * hb;
  int threads_per_block = 256;
  int blocks_per_grid =
      (total_elements + threads_per_block - 1) / threads_per_block;

  relu_kernel<<<blocks_per_grid, threads_per_block>>>(b, hb, wb,
                                                      output_channels);
  cuda_check(cudaDeviceSynchronize());
}

// MaxPool

MaxPool::MaxPool(int ha, int wa, int hb, int wb, int input_channels)
    : ha(ha), wa(wa), hb(hb), wb(wb), input_channels(input_channels) {}

int MaxPool::get_num_outputs() { return input_channels * hb * wb; }

void MaxPool::forward(float *d_input, float *d_output){}

void MaxPool::forward(float *d_input, float *d_output, int *d_max_ind) {
  int total_outputs = hb * wb * input_channels;
  int block_size = POOL_BLOCK_SIZE * POOL_BLOCK_SIZE;
  int grid_size = (total_outputs + block_size - 1) / block_size;

  max_pool2d<<<grid_size, block_size>>>(d_input, d_output, ha, wa, hb, wb,
                                        input_channels);

  cuda_check(cudaPeekAtLastError());
  cuda_check(cudaDeviceSynchronize());
}


// TODO Define this method for conv layers
void MaxPool::back_prop(float *d_input, float *d_grad_output, float alpha){}

// MlpLayer

MlpLayer::MlpLayer(int input_size, int output_size)
    : input_size(input_size), output_size(output_size) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> pos_dist(0.0f, 1.0f);
  std::uniform_real_distribution<float> neg_dist(-1.0f, 0.0f);

  bias = (float *)malloc(output_size * sizeof(float));
  weights = (float *)malloc(output_size * input_size * sizeof(float));

  for (int i = 0; i < output_size; ++i) {
    bias[i] = (i % 2 == 0) ? pos_dist(gen) : neg_dist(gen);
  }

  float stddev = sqrtf(2.0f / input_size);
  std::normal_distribution<float> dist(0.0f, stddev);
  for (int i = 0; i < input_size * output_size; ++i) {
    weights[i] = dist(gen);
  }

  cuda_check(cudaMalloc((void **)&d_bias, output_size * sizeof(float)));
  cuda_check(cudaMalloc((void **)&d_weights,
                        input_size * output_size * sizeof(float)));
  cuda_check(
      cudaMalloc((void **)&dl_dw, input_size * output_size * sizeof(float)));
  cuda_check(cudaMalloc((void **)&dl_db, output_size * sizeof(float)));
  cuda_check(cudaMalloc((void **)&dl_dx, input_size * sizeof(float)));
  cuda_check(cudaMalloc((void **)&dl_dz, output_size * sizeof(float)));
  cuda_check(cudaMalloc((void **)&dy_dz, output_size * sizeof(float)));
  cuda_check(cudaMalloc((void **)&d_weights_transpose,
                        input_size * output_size * sizeof(float)));

  cuda_check(cudaMemcpy(d_bias, bias, output_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(d_weights, weights,
                        input_size * output_size * sizeof(float),
                        cudaMemcpyHostToDevice));
}

MlpLayer::~MlpLayer() {
  free(bias);
  free(weights);
  cuda_check(cudaFree(d_bias));
  cuda_check(cudaFree(d_weights));
  cuda_check(cudaFree(d_weights_transpose));
  cuda_check(cudaFree(dl_dw));
  cuda_check(cudaFree(dl_dx));
  cuda_check(cudaFree(dl_dz));
  cuda_check(cudaFree(dy_dz));
  cuda_check(cudaFree(dl_db));
}

int MlpLayer::get_num_outputs() { return output_size; }

void MlpLayer::forward(float *d_input, float *d_output) {
  const int block_size = 16;
  dim3 dim_block(BLOCK_SIZE * BLOCK_SIZE);
  int tiles_x = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int tiles_y = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 dim_grid(tiles_x * tiles_y);

  sgemm_1d<block_size><<<dim_grid, dim_block>>>(d_input, d_weights, d_output, 1,
                                                input_size, input_size,
                                                output_size, 1, output_size);
  cudaDeviceSynchronize();

  vec_add<<<(output_size + 255) / 256, 256>>>(d_output, d_bias, false,
                                              output_size);
  cuda_check(cudaDeviceSynchronize());
}

void MlpLayer::relu(float *d_input) {
  int threads_per_block = 256;
  int blocks_per_grid =
      (input_size + threads_per_block - 1) / threads_per_block;
  relu_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, input_size);
  cuda_check(cudaDeviceSynchronize());
}

void MlpLayer::compute_gradients(float *d_input, float *dl_dy) {
  int threads_per_block = 256;
  int blocks_per_grid =
      (output_size + threads_per_block - 1) / threads_per_block;

  relu_backward<<<blocks_per_grid, threads_per_block>>>(d_input, dy_dz, dl_dy,
                                                        output_size);
  cuda_check(cudaDeviceSynchronize());

  tensor_elementwise_mult<<<blocks_per_grid, threads_per_block>>>(
      dl_dy, dy_dz, dl_dz, output_size);
  cuda_check(cudaDeviceSynchronize());

  constexpr int block_size = 16;
  threads_per_block = 256;
  blocks_per_grid =
      (input_size * output_size + threads_per_block - 1) / threads_per_block;
  sgemm_1d<block_size><<<blocks_per_grid, threads_per_block>>>(
      d_input, dl_dz, dl_dw, input_size, 1, 1, output_size, input_size,
      output_size);
  cuda_check(cudaDeviceSynchronize());

  cuda_check(cudaMemcpy(dl_db, dl_dz, output_size * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  transpose_kernel<<<blocks_per_grid, threads_per_block>>>(
      d_weights, d_weights_transpose, input_size, output_size);
  cuda_check(cudaDeviceSynchronize());

  blocks_per_grid = (input_size + threads_per_block - 1) / threads_per_block;
  sgemm_1d<block_size><<<blocks_per_grid, threads_per_block>>>(
      dl_dz, d_weights_transpose, dl_dx, 1, output_size, output_size,
      input_size, 1, input_size);
  cuda_check(cudaDeviceSynchronize());
}

void MlpLayer::back_prop(float *d_input, float *dl_dy, float alpha) {
  compute_gradients(d_input, dl_dy);
  bool neg = true;

  int threads_per_block = 256;
  int blocks_per_grid =
      (output_size * input_size + threads_per_block - 1) / threads_per_block;
  mat_add<<<blocks_per_grid, threads_per_block>>>(
      d_weights, dl_dw, d_weights, input_size, output_size, neg, alpha);
  cuda_check(cudaDeviceSynchronize());

  blocks_per_grid = (output_size + threads_per_block - 1) / threads_per_block;
  mat_add<<<blocks_per_grid, threads_per_block>>>(d_bias, dl_db, d_bias, 1,
                                                  output_size, neg, alpha);
  cuda_check(cudaDeviceSynchronize());

  // return dl_dx;
}

float *MlpLayer::get_host_weights() { return weights; }
float *MlpLayer::get_host_bias() { return bias; }
float *MlpLayer::get_device_weights() { return d_weights; }
float *MlpLayer::get_device_bias() { return d_bias; }
float *MlpLayer::get_weight_grad() { return dl_dw; }
float *MlpLayer::get_input_grad() { return dl_dx; }
float *MlpLayer::get_bias_grad() { return dl_db; }
float *MlpLayer::get_output_grad() { return dl_dz; }

// SoftmaxLayer

SoftmaxLayer::SoftmaxLayer(int input_size, int output_size)
    : input_size(input_size), output_size(output_size) {
  cuda_check(cudaHostAlloc(&h_loss, sizeof(float), cudaHostAllocDefault));
  cuda_check(
      cudaHostAlloc(&y_hat, sizeof(float) * output_size, cudaHostAllocDefault));
  cuda_check(
      cudaHostAlloc(&y, sizeof(float) * output_size, cudaHostAllocDefault));
  cuda_check(cudaMalloc(&d_loss, sizeof(float)));
}

SoftmaxLayer::~SoftmaxLayer() {
  cuda_check(cudaFreeHost(h_loss));
  cuda_check(cudaFreeHost(y_hat));
  cuda_check(cudaFreeHost(y));
  cuda_check(cudaFree(d_loss));
}

int SoftmaxLayer::get_num_outputs() { return output_size; }

void SoftmaxLayer::softmax(float *d_input, float *d_output) {
  int block_size = 128;
  int grid_size = (output_size + block_size - 1) / block_size;
  softmax_kernel<<<grid_size, block_size>>>(d_input, d_output, output_size);
}

void SoftmaxLayer::forward(float *d_y_hat, float *d_y) {
  cuda_check(cudaMemcpy(y_hat, d_y_hat, sizeof(float) * output_size,
                        cudaMemcpyDeviceToHost));
  cuda_check(
      cudaMemcpy(y, d_y, sizeof(float) * output_size, cudaMemcpyDeviceToHost));
  *h_loss = compute_cross_entropy_loss(y_hat, y, output_size);
  cudaMemcpy(d_loss, h_loss, sizeof(float), cudaMemcpyHostToDevice);
}

void SoftmaxLayer::back_prop(float *d_y_hat, float *d_y, float alpha) {
  int threads_per_block = 256;
  int blocks_per_grid =
      (output_size + threads_per_block - 1) / threads_per_block;
  vec_add<<<blocks_per_grid, threads_per_block>>>(d_y_hat, d_y, true,
                                                  output_size);
  cuda_check(cudaDeviceSynchronize());
}
float SoftmaxLayer::get_loss(){
  return *h_loss;
}
