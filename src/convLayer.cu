#include "CudaChecks.cuh"
#include "convLayer.cuh"
#include <cstdint>
#include <iostream>
#include <stdexcept>

__global__ void Convolution(float *A, float *B, float *C, int HA, int WA,
                            int HB, int WB, int HC, int WC, int input_channels,
                            int output_channels) {
  int col = blockIdx.x * (BLOCK_SIZE - WC + 1) + threadIdx.x;
  int row = blockIdx.y * (BLOCK_SIZE - WC + 1) + threadIdx.y;
  int row_i = row - WC + 1;
  int col_i = col - WC + 1;

  float tmp = 0.0f;

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

__global__ void Convolution_3d(float *A, float *B, float *C, int HA, int WA,
                               int HB, int WB, int HC, int WC,
                               int input_channels, int output_channels) {

  int col = blockIdx.x * (BLOCK_SIZE - WC + 1) + threadIdx.x;
  int row = blockIdx.y * (BLOCK_SIZE - WC + 1) + threadIdx.y;
  int out_channel = blockIdx.z;

  int row_i = row - WC + 1;
  int col_i = col - WC + 1;

  float tmp = 0.0f;

  __shared__ float shm[BLOCK_SIZE][BLOCK_SIZE];

  for (int in_channel = 0; in_channel < input_channels; ++in_channel) {
    // Offset pointers for current input channel
    float *input = A + in_channel * HA * WA;
    float *kernel = C + (out_channel * input_channels + in_channel) * WC * HC;

    if (row_i >= 0 && col_i >= 0 && row_i < HA && col_i < WA) {
      shm[threadIdx.y][threadIdx.x] = input[col_i * WA + row_i];
    } else {
      shm[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    if (threadIdx.y < (BLOCK_SIZE - WC + 1) &&
        threadIdx.x < (BLOCK_SIZE - WC + 1) && row < (HB - WC + 1) &&
        col < (WB - WC + 1)) {
      for (int i = 0; i < WC; i++) {
        for (int j = 0; j < HC; j++) {
          tmp += shm[threadIdx.y + i][threadIdx.x + j] * kernel[j * WC + i];
        }
      }
    }
    __syncthreads();
  }

  // Write to output buffer
  if (threadIdx.y < (BLOCK_SIZE - WC + 1) &&
      threadIdx.x < (BLOCK_SIZE - WC + 1) && row < HB && col < WB) {
    B[out_channel * HB * WB + col * WB + row] = tmp;
  }
}

convLayer::convLayer(ImageSize inputImageSize, ImageSize outputImageSize,
                     ImageSize kernelSize, float *kernels,
                     uint8_t input_channels, uint8_t output_channels)
    : input_channels(input_channels), output_channels(output_channels),
      kernels(kernels), HA(inputImageSize.height), WA(inputImageSize.width),
      HB(outputImageSize.height), WB(outputImageSize.width),
      HC(kernelSize.height), WC(kernelSize.width) {
  cudaCheck(cudaMalloc((void **)&d_kernels, output_channels * KERNEL_SIZE *
                                                KERNEL_SIZE * sizeof(float)));
  cudaCheck(
      cudaMemcpy(d_kernels, kernels,
                 output_channels * KERNEL_SIZE * KERNEL_SIZE * sizeof(float),
                 cudaMemcpyHostToDevice));
}

convLayer::~convLayer() { cudaFree(d_kernels); }

void convLayer::forward(float *input_image, float *output_image) {
  float *d_input_image;
  float *d_output_image;
  cudaCheck(cudaMalloc((void **)&d_input_image,
                       input_channels * WA * HA * sizeof(float)));
  cudaCheck(cudaMalloc((void **)&d_output_image,
                       output_channels * WB * HB * sizeof(float)));
  cudaCheck(cudaMemcpy(d_input_image, input_image,
                       input_channels * WA * HA * sizeof(float),
                       cudaMemcpyHostToDevice));

  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((WB - 1) / (BLOCK_SIZE - WC + 1), (WB - 1) / (BLOCK_SIZE - WC + 1),
            output_channels);

  Convolution_3d<<<grid, threads>>>(d_input_image, d_output_image, d_kernels,
                                    HA, WA, HB, WB, HC, WC, input_channels,
                                    output_channels);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << cudaGetErrorString(error) << std::endl;
    throw std::runtime_error("Cuda kernel failed\n");
  }

  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaMemcpy(output_image, d_output_image,
                       output_channels * WB * HB * sizeof(float),
                       cudaMemcpyDeviceToHost));

  cudaCheck(cudaFree(d_input_image));
  cudaCheck(cudaFree(d_output_image));
}
