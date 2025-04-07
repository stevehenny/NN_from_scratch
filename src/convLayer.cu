#include "CudaChecks.cuh"
#include "convLayer.cuh"
#include "cudaKernels.cuh"
#include <cstdint>
#include <iostream>
#include <stdexcept>
#define SHARED_BLOCK_SIZE BLOCK_SIZE + 2
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

  int tile_output_size = BLOCK_SIZE - WC + 1;
  dim3 grid((WB + tile_output_size - 1) / tile_output_size,
            (HB + tile_output_size - 1) / tile_output_size, output_channels);

  Convolution3D<<<grid, threads>>>(d_input_image, d_output_image, d_kernels, HA,
                                   WA, HB, WB, HC, WC, input_channels,
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
