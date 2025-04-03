#include "convLayer.cuh"
#include <cstdint>
#include <iostream>
// 3D kernel. z - channels, x - width, y - height
__global__ void conv_forward_kernel(uint8_t *kernels, uint8_t input_channels,
                                    uint8_t output_channels, uint16_t height,
                                    uint16_t width, uint8_t *input_image,
                                    uint8_t *output_image) {
  // Compute the 3D index of the thread
  uint8_t out_ch = blockIdx.z; // Output channel (z dimension)
  uint16_t x = blockIdx.x;     // x-coordinate (width)
  uint16_t y = blockIdx.y;     // y-coordinate (height)

  // Make sure we are within bounds
  if (x >= width || y >= height || out_ch >= output_channels)
    return;

  // Access the kernel for the output channel
  uint8_t *kernel = kernels + (out_ch * input_channels * 3 *
                               3); // 3x3 kernel for each output channel

  // Initialize the result for this output pixel
  uint8_t result = 0;

  // Perform the convolution operation (assuming 3x3 kernel)
  for (int c = 0; c < input_channels; ++c) { // Iterate over input channels
    for (int kx = -1; kx <= 1; ++kx) {       // Iterate over kernel width
      for (int ky = -1; ky <= 1; ++ky) {     // Iterate over kernel height
        // Boundary checks for valid (x + kx, y + ky)
        int ix = x + kx;
        int iy = y + ky;
        if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
          // Access input pixel value (greyscale, so single value per channel)
          uint8_t input_pixel =
              input_image[(iy * width + ix) * input_channels + c];
          result += input_pixel * kernel[(c * 3 + (kx + 1)) * 3 + (ky + 1)];
        }
      }
    }
  }

  // Store the result in the output image (assuming output image memory is
  // allocated and available)
  output_image[(out_ch * height + y) * width + x] = result;
}

convLayer::convLayer(uint8_t *kernels, uint8_t input_channels,
                     uint8_t output_channels, uint16_t height, uint16_t width)
    : width(width), input_channels(input_channels),
      output_channels(output_channels), height(height), kernels(kernels) {

  cudaMalloc((void **)&d_kernels,
             sizeof(output_channels * KERNEL_SIZE * KERNEL_SIZE));
  cudaMemcpy(d_kernels, kernels, output_channels * KERNEL_SIZE * KERNEL_SIZE,
             cudaMemcpyHostToDevice);
}
convLayer::~convLayer() { cudaFree(d_kernels); }
void convLayer::forward(uint8_t *input_image, uint8_t *output_image) {
  uint8_t *d_input_image;
  uint8_t *d_output_image;
  cudaMalloc((void **)&d_input_image, 28 * 28 * sizeof(uint8_t));
  cudaMalloc((void **)&d_output_image, (28 - (KERNEL_SIZE - 1)) *
                                           (28 - (KERNEL_SIZE - 1)) *
                                           sizeof(uint8_t));
  cudaMemcpy(d_input_image, input_image, sizeof(input_image),
             cudaMemcpyHostToDevice);
  // Set up the block and grid dimensions
  dim3 blockDim(16, 16, 1); // 2D blocks (x, y)
  dim3 gridDim(
      (width + blockDim.x - 1) / blockDim.x,  // Grid size based on width
      (height + blockDim.y - 1) / blockDim.y, // Grid size based on height
      output_channels); // Grid size based on output channels (z dimension)

  // Launch the kernel
  conv_forward_kernel<<<gridDim, blockDim>>>(d_kernels, input_channels,
                                             output_channels, height, width,
                                             d_input_image, d_output_image);

  // Handle errors (check for CUDA errors)
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err)
              << std::endl;
  }

  // Optionally, synchronize and check the status of the kernel
  cudaDeviceSynchronize();
  cudaMemcpy(output_image, d_output_image, sizeof(output_image),
             cudaMemcpyDeviceToHost);
  cudaFree(d_input_image);
  cudaFree(d_output_image);
}
