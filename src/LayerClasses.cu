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
  std::default_random_engine gen;
  float stddev =
      sqrtf(2.0f / (inputImageSize.height *
                    inputImageSize.width)); // He initialization for ReLU
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
  cudaCheck(cudaMalloc((void **)&d_kernels, input_channels * output_channels *
                                                KERNEL_SIZE * KERNEL_SIZE *
                                                sizeof(float)));
  cudaCheck(cudaMemcpy(d_kernels, kernels,
                       input_channels * output_channels * KERNEL_SIZE *
                           KERNEL_SIZE * sizeof(float),
                       cudaMemcpyHostToDevice));
}

convLayer::~convLayer() {
  free(kernels);
  cudaFree(d_kernels);
}

float *convLayer::forward(float *d_input_image, float *d_output_image) {

  int tile_output_width = BLOCK_SIZE - WC + 1;
  int tile_output_height = BLOCK_SIZE - HC + 1;

  int grid_x = (WB + tile_output_width - 1) / tile_output_width;
  int grid_y = (HB + tile_output_height - 1) / tile_output_height;

  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(grid_x, grid_y, output_channels);

  int shared_height = BLOCK_SIZE + HC - 1;
  int shared_width = BLOCK_SIZE + WC - 1;

  int tile_width = WC + 1;  // adjust this based on needed coverage
  int tile_height = HC + 1; // adjust this based on needed coverage
  int shared_mem_bytes = tile_width * tile_height * sizeof(float);

  int total_outputs = output_channels * HB * WB;
  int threads_per_block = 256;
  int num_blocks = (total_outputs + threads_per_block - 1) / threads_per_block;

  Convolution3D_1d_launch<<<num_blocks, threads_per_block>>>(
      d_input_image, d_output_image, d_kernels, HA, WA, HB, WB, HC, WC,
      input_channels, output_channels);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << cudaGetErrorString(error) << std::endl;
    throw std::runtime_error("Cuda kernel failed\n");
  }

  cudaCheck(cudaDeviceSynchronize());
  return d_output_image;
}

void convLayer::ReLU(float *B) {
  int total_elements = output_channels * WB * HB;
  int threads_per_block = 256;
  int blocks_per_grid =
      (total_elements + threads_per_block - 1) / threads_per_block;

  ReLU_kernel<<<blocks_per_grid, threads_per_block>>>(B, HB, WB,
                                                      output_channels);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << cudaGetErrorString(error) << std::endl;
    throw std::runtime_error("Cuda kernel failed\n");
  }

  cudaCheck(cudaDeviceSynchronize());
}

maxPool::maxPool(int HA, int WA, int HB, int WB, int input_channels)
    : HA(HA), WA(WA), HB(HB), WB(WB), input_channels(input_channels) {}

float *maxPool::forward(float *d_input, float *d_output, int *d_max_ind) {
  int total_outputs = HB * WB * input_channels;
  int block_size = POOL_BLOCK_SIZE * POOL_BLOCK_SIZE;
  int grid_size = (total_outputs + block_size - 1) / block_size;

  maxPool2D<<<grid_size, block_size>>>(d_input, d_output, HA, WA, HB, WB,
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

  float stddev = sqrtf(2.0f / input_size);
  std::normal_distribution<float> dist(0.0f, stddev);
  for (int i = 0; i < input_size * output_size; ++i) {
    weights[i] = dist(gen);
  }

  // Allocate device memory
  cudaCheck(cudaMalloc((void **)&d_bias, output_size * sizeof(float)));
  cudaCheck(cudaMalloc((void **)&d_weights,
                       input_size * output_size * sizeof(float)));
  // Copy host memory to device memory cudaCheck(cudaMemcpy(d_bias, bias,
  // output_size * sizeof(float), cudaMemcpyHostToDevice));
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
  vecAdd<<<(output_size + 255) / 256, 256>>>(d_output, d_bias, false,
                                             output_size);
  cudaCheck(cudaDeviceSynchronize());
  return d_output;
}

void mlpLayer::ReLU(float *d_input) {

  int threadsPerBlock = 256;
  int blocksPerGrid = (input_size + threadsPerBlock - 1) / threadsPerBlock;
  ReLU_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, input_size);
  cudaCheck(cudaDeviceSynchronize());
}

float *mlpLayer::backProp(float *input, float *dL_dy, float alpha) {

  // compute dy_dz
  int threadsPerBlock = 256;
  int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
  reluBackward<<<blocksPerGrid, threadsPerBlock>>>(input, dy_dz, dL_dy,
                                                   output_size);
  cudaCheck(cudaDeviceSynchronize());

  // elementwise operation of dL_dy and dy_dz to compute dL_dz
  tensorElementwiseMult<<<blocksPerGrid, threadsPerBlock>>>(dL_dy, dy_dz, dL_dz,
                                                            output_size);
  cudaCheck(cudaDeviceSynchronize());

  // compute dL_dW

  // compute dL_db

  // compute dL_dx
}

SoftmaxLayer::SoftmaxLayer(int input_size, int output_size)
    : input_size(input_size), output_size(output_size) {
  cudaCheck(cudaHostAlloc(&h_loss, sizeof(float), cudaHostAllocDefault));
  cudaCheck(
      cudaHostAlloc(&y_hat, sizeof(float) * output_size, cudaHostAllocDefault));
  cudaCheck(
      cudaHostAlloc(&y, sizeof(float) * output_size, cudaHostAllocDefault));
  cudaCheck(cudaMalloc(&d_loss, sizeof(float)));
}

SoftmaxLayer::~SoftmaxLayer() {
  cudaCheck(cudaFreeHost(h_loss));
  cudaCheck(cudaFreeHost(y_hat));
  cudaCheck(cudaFreeHost(y));
  cudaCheck(cudaFree(d_loss));
}

void SoftmaxLayer::softMax(float *d_input, float *d_output) {
  int blockSize = 128;
  int gridSize = (output_size + blockSize - 1) / blockSize;
  softmaxKernel<<<gridSize, blockSize>>>(d_input, d_output, output_size);
}

float SoftmaxLayer::computeLoss(float *d_y_hat, float *d_y) {
  cudaCheck(cudaMemcpy(y_hat, d_y_hat, sizeof(float) * output_size,
                       cudaMemcpyDeviceToHost));
  cudaCheck(
      cudaMemcpy(y, d_y, sizeof(float) * output_size, cudaMemcpyDeviceToHost));
  *h_loss = computeCrossEntropyLoss(y_hat, y, output_size);
  cudaMemcpy(d_loss, h_loss, sizeof(float), cudaMemcpyHostToDevice);
  return *h_loss;
}

float *SoftmaxLayer::backProp(float *d_y_hat, float *d_y, float alpha) {

  int threadsPerBlock = 256;
  int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blockDim(threadsPerBlock);
  dim3 DimGrid(blocksPerGrid);
  vecAdd<<<DimGrid, DimGrid>>>(d_y_hat, d_y, true, output_size);
  cudaCheck(cudaDeviceSynchronize());
  return d_y_hat;
}
