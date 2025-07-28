#include <algorithm>
#include <cmath>
#include <CudaChecks.cuh>
#include <LayerClasses.cuh>
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

TEST(CudaKernelTests, test_mlpLayer_computeGradients_and_backProp) {
  const int input_size = 3;
  const int output_size = 2;

  // Construct layer
  mlpLayer layer(input_size, output_size);

  // Host input: x = [1.0, 2.0, 3.0]
  std::vector<float> h_input = {1.0f, 2.0f, 3.0f};
  std::vector<float> h_dL_dy = {0.1f,
                                -0.2f}; // example gradient from next layer

  // Allocate and copy input to device
  float *d_input, *d_dL_dy;
  cudaCheck(cudaMalloc(&d_input, input_size * sizeof(float)));
  cudaCheck(cudaMalloc(&d_dL_dy, output_size * sizeof(float)));

  cudaCheck(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float),
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_dL_dy, h_dL_dy.data(), output_size * sizeof(float),
                       cudaMemcpyHostToDevice));

  // Forward pass
  float *d_output;
  cudaCheck(cudaMalloc(&d_output, output_size * sizeof(float)));
  layer.forward(d_input, d_output);
  layer.ReLU(d_output); // Apply ReLU to simulate activation

  // Call computeGradients
  layer.computeGradients(d_input, d_dL_dy);

  // Retrieve gradients from device to check correctness
  std::vector<float> h_dL_dW(input_size * output_size);
  std::vector<float> h_dL_db(output_size);
  std::vector<float> h_dL_dx(input_size);

  cudaCheck(cudaMemcpy(h_dL_dW.data(), layer.getWeightGrad(),
                       input_size * output_size * sizeof(float),
                       cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(h_dL_db.data(), layer.getBiasGrad(), output_size * sizeof(float),
                       cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(h_dL_dx.data(), layer.backProp(d_input, d_dL_dy, 0.1f),
                       input_size * sizeof(float), cudaMemcpyDeviceToHost));

  // Check shapes and simple consistency
  for (float v : h_dL_db) {
    EXPECT_TRUE(std::isfinite(v)); // should be numbers
  }

  for (float v : h_dL_dx) {
    EXPECT_TRUE(std::isfinite(v)); // should be numbers
  }

  for (float v : h_dL_dW) {
    EXPECT_TRUE(std::isfinite(v)); // should be numbers
  }

  // Optional: Check bias was updated (forward bias - alpha * grad_bias)
  std::vector<float> h_bias_updated(output_size);
  cudaCheck(cudaMemcpy(h_bias_updated.data(), layer.getDeviceBias(),
                       output_size * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < output_size; ++i) {
    float expected = layer.getHostBias()[i] - 0.1f * h_dL_db[i]; // bias gradient applied
    EXPECT_NEAR(h_bias_updated[i], expected, 1e-3)
        << "Mismatch in updated bias at " << i;
  }

  cudaFree(d_input);
  cudaFree(d_dL_dy);
  cudaFree(d_output);
}
