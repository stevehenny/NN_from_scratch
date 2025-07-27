#include <algorithm>
#include <cmath>
#include <cudaKernels.cuh>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

constexpr int block_size = 4;

// Host reference implementation
void reference_gemm(const float *A, const float *B, float *C, int HA, int WA,
                    int WB) {
  for (int i = 0; i < HA; ++i) {
    for (int j = 0; j < WB; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < WA; ++k) {
        sum += A[i * WA + k] * B[k * WB + j];
      }
      C[i * WB + j] = sum;
    }
  }
}

// Reference CPU implementation for verification
void reference_convolution3d(const std::vector<float> &input,
                             const std::vector<float> &kernels,
                             std::vector<float> &output, int HA, int WA, int HB,
                             int WB, int HC, int WC, int input_channels,
                             int output_channels) {
  for (int oc = 0; oc < output_channels; ++oc) {
    for (int i = 0; i < HB; ++i) {
      for (int j = 0; j < WB; ++j) {
        float sum = 0.0f;
        for (int ic = 0; ic < input_channels; ++ic) {
          for (int ki = 0; ki < HC; ++ki) {
            for (int kj = 0; kj < WC; ++kj) {
              int in_i = i + ki;
              int in_j = j + kj;
              if (in_i < HA && in_j < WA) {
                float image_val = input[ic * HA * WA + in_i * WA + in_j];
                float kernel_val = kernels[oc * input_channels * HC * WC +
                                           ic * HC * WC + ki * WC + kj];
                sum += image_val * kernel_val;
              }
            }
          }
        }
        output[oc * HB * WB + i * WB + j] = sum;
      }
    }
  }
}

TEST(CudaKernelTests, test_sgemm) {
  // Matrix dimensions
  const int HA = 2;
  const int WA = 3;
  const int HB = 3;
  const int WB = 2;
  const int HC = HA;
  const int WC = WB;

  std::vector<float> h_A = {1, 2, 3, 4, 5, 6};

  std::vector<float> h_B = {7, 8, 9, 10, 11, 12};

  std::vector<float> h_C(HC * WC, 0.0f);          // For device result
  std::vector<float> h_C_expected(HC * WC, 0.0f); // For CPU result

  // Allocate device memory
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, h_A.size() * sizeof(float));
  cudaMalloc(&d_B, h_B.size() * sizeof(float));
  cudaMalloc(&d_C, h_C.size() * sizeof(float));

  // Copy input matrices to device
  cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  // Launch the kernel
  dim3 DimBlock(block_size * block_size); // 1D block
  int tilesX = (WC + block_size - 1) / block_size;
  int tilesY = (HC + block_size - 1) / block_size;
  dim3 DimGrid(tilesX * tilesY); // 1D grid

  sgemm_1d<block_size>
      <<<DimGrid, DimBlock>>>(d_A, d_B, d_C, HA, WA, HB, WB, HC, WC);
  cudaDeviceSynchronize();

  // Copy result back to host
  cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Compute reference result
  reference_gemm(h_A.data(), h_B.data(), h_C_expected.data(), HA, WA, WB);

  // Compare element-wise
  const float eps = 1e-4;
  for (int i = 0; i < HC * WC; ++i) {
    EXPECT_NEAR(h_C[i], h_C_expected[i], eps) << "Mismatch at index " << i;
  }

  // Cleanup
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

TEST(CudaKernelTests, test_conv3d_1d_launch) {

  // Dimensions
  const int HA = 4, WA = 4;
  const int HC = 2, WC = 2;
  const int HB = HA - HC + 1, WB = WA - WC + 1;
  const int input_channels = 1;
  const int output_channels = 1;

  const int input_size = input_channels * HA * WA;
  const int kernel_size = output_channels * input_channels * HC * WC;
  const int output_size = output_channels * HB * WB;

  // Host memory
  std::vector<float> h_input_image(input_size);
  std::vector<float> h_kernels(kernel_size);
  std::vector<float> h_output_image(output_size, 0.0f);
  std::vector<float> h_expected_output(output_size, 0.0f);

  // Fill input and kernel with known values
  std::iota(h_input_image.begin(), h_input_image.end(), 1.0f); // 1..16
  std::fill(h_kernels.begin(), h_kernels.end(), 1.0f); // all ones (simple sum)

  // Compute expected output on CPU
  reference_convolution3d(h_input_image, h_kernels, h_expected_output, HA, WA,
                          HB, WB, HC, WC, input_channels, output_channels);

  // Allocate device memory
  float *d_input_image, *d_output_image, *d_kernels;
  cudaMalloc(&d_input_image, input_size * sizeof(float));
  cudaMalloc(&d_kernels, kernel_size * sizeof(float));
  cudaMalloc(&d_output_image, output_size * sizeof(float));

  // Copy to device
  cudaMemcpy(d_input_image, h_input_image.data(), input_size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernels, h_kernels.data(), kernel_size * sizeof(float),
             cudaMemcpyHostToDevice);

  // Launch kernel
  int total_outputs = output_channels * HB * WB;
  int threads_per_block = 256;
  int num_blocks = (total_outputs + threads_per_block - 1) / threads_per_block;

  Convolution3D_1d_launch<<<num_blocks, threads_per_block>>>(
      d_input_image, d_output_image, d_kernels, HA, WA, HB, WB, HC, WC,
      input_channels, output_channels);

  cudaDeviceSynchronize();

  // Copy result back
  cudaMemcpy(h_output_image.data(), d_output_image, output_size * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Validate
  const float eps = 1e-4;
  for (int i = 0; i < output_size; ++i) {
    EXPECT_NEAR(h_output_image[i], h_expected_output[i], eps)
        << "Mismatch at index " << i;
  }

  // Cleanup
  cudaFree(d_input_image);
  cudaFree(d_output_image);
  cudaFree(d_kernels);
}
