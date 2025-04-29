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

convLayer::convLayer(ImageSize inputImageSize, ImageSize outputImageSize,
                     ImageSize kernelSize, uint8_t input_channels,
                     uint8_t output_channels)
    : input_channels(input_channels), output_channels(output_channels),
      HA(inputImageSize.height), WA(inputImageSize.width),
      HB(outputImageSize.height), WB(outputImageSize.width),
      HC(kernelSize.height), WC(kernelSize.width) {

  // Random number generator setup
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> pos_dist(0.0f, 1.0f);
  std::uniform_real_distribution<float> neg_dist(-1.0f, 0.0f);
  kernels = (float *)malloc(output_channels * input_channels * KERNEL_SIZE *
                            KERNEL_SIZE * sizeof(float));
  for (int oc = 0; oc < output_channels; ++oc) {
    for (int ic = 0; ic < input_channels; ++ic) {
      for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; ++i) {
        kernels[((oc * input_channels + ic) * KERNEL_SIZE * KERNEL_SIZE) + i] =
            ((i + oc) % 2 == 0) ? pos_dist(gen) : neg_dist(gen);
      }
    }
  }
  cudaCheck(cudaMalloc((void **)&d_kernels, output_channels * KERNEL_SIZE *
                                                KERNEL_SIZE * sizeof(float)));
  cudaCheck(
      cudaMemcpy(d_kernels, kernels,
                 output_channels * KERNEL_SIZE * KERNEL_SIZE * sizeof(float),
                 cudaMemcpyHostToDevice));
}

convLayer::~convLayer() {
  free(kernels);
  cudaFree(d_kernels);
}

float *convLayer::forward(float *d_input_image, float *d_output_image) {

  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  int grid_x = (WB + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int grid_y = (HB + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 grid(grid_x, grid_y, output_channels);

  Convolution3D<<<grid, threads>>>(d_input_image, d_output_image, d_kernels, HA,
                                   WA, HB, WB, HC, WC, input_channels,
                                   output_channels);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << cudaGetErrorString(error) << std::endl;
    throw std::runtime_error("Cuda kernel failed\n");
  }

  cudaCheck(cudaDeviceSynchronize());
  return d_output_image;
}

void convLayer::ReLU(float *B) {

  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  int grid_x = (WB + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int grid_y = (HB + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 grid(grid_x, grid_y, output_channels);
  ReLU_kernel<<<grid, threads>>>(B, HB, WB, output_channels);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << cudaGetErrorString(error) << std::endl;
    throw std::runtime_error("Cuda kernel failed\n");
  }

  cudaCheck(cudaDeviceSynchronize());
}

maxPool::maxPool(int HA, int WA, int HB, int WB, int input_channels)
    : HA(HA), WA(WA), HB(HB), WB(WB), input_channels(input_channels) {}

float *maxPool::forward(float *d_input, float *d_output) {
  int grid_x = (WB + POOL_BLOCK_SIZE - 1) / POOL_BLOCK_SIZE;
  int grid_y = (HB + POOL_BLOCK_SIZE - 1) / POOL_BLOCK_SIZE;
  dim3 threads(POOL_BLOCK_SIZE, POOL_BLOCK_SIZE);
  dim3 grid(grid_x, grid_y, input_channels);
  maxPool2D<<<grid, threads>>>(d_input, d_output, HA, WA, HB, WB,
                               input_channels);
  cudaCheck(cudaPeekAtLastError());
  cudaCheck(cudaDeviceSynchronize());
  return d_output;
}

mlpLayer::mlpLayer(int input_size, int output_size)
    : input_size(input_size), output_size(output_size) {

  // Random number generator setup
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> pos_dist(0.0f, 1.0f);
  std::uniform_real_distribution<float> neg_dist(-1.0f, 0.0f);

  // Allocate host memory
  bias = (float *)malloc(output_size * sizeof(float));
  weights = (float *)malloc(output_size * input_size * sizeof(float));

  // Initialize bias values
  for (int i = 0; i < output_size; ++i) {
    bias[i] = (i % 2 == 0) ? pos_dist(gen) : neg_dist(gen);
  }

  // Initialize weight values
  for (int i = 0; i < input_size * output_size; ++i) {
    weights[i] = (i % 2 == 0) ? pos_dist(gen) : neg_dist(gen);
  }

  // Allocate device memory
  cudaCheck(cudaMalloc((void **)&d_bias, output_size * sizeof(float)));
  cudaCheck(cudaMalloc((void **)&d_weights,
                       input_size * output_size * sizeof(float)));

  // Copy host memory to device memory
  cudaCheck(cudaMemcpy(d_bias, bias, output_size * sizeof(float),
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_weights, weights,
                       input_size * output_size * sizeof(float),
                       cudaMemcpyHostToDevice));
}

mlpLayer::~mlpLayer() {
  free(bias);
  free(weights);
  cudaFree(d_bias);
  cudaFree(d_weights);
}

float *mlpLayer::forward(float *d_input, float *d_output) {
  dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 DimGrid((output_size + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

  // Launch sgemm
  sgemm<<<DimGrid, DimBlock>>>(
      d_input,     // A: input (1 x input_size)
      d_weights,   // B: weights (input_size x output_size)
      d_output,    // C: output (1 x output_size)
      1,           // HA
      input_size,  // WA
      input_size,  // HB
      output_size, // WB
      1,           // HC
      output_size  // WC
  );
  cudaCheck(cudaDeviceSynchronize());

  // add the bias
  vecAdd<<<(output_size + 255) / 256, 256>>>(d_output, d_bias, output_size);
  cudaCheck(cudaDeviceSynchronize());
  return d_output;
}

void mlpLayer::softMax(float *d_input, float *d_output) {

  int blockSize = 128;
  int gridSize = (output_size + blockSize - 1) / blockSize;
  softmaxKernel<<<gridSize, blockSize>>>(d_input, d_output, output_size);
}
