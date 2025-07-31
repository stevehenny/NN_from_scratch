#include <CudaChecks.cuh>
#include <LayerClasses.cuh>
#include <algorithm>
#include <cudaKernels.cuh>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

constexpr int block_size = 4;

// Host reference implementation
void reference_gemm(const float *a, const float *b, float *c, int ha, int wa,
                    int wb) {
  for (int i = 0; i < ha; ++i) {
    for (int j = 0; j < wb; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < wa; ++k) {
        sum += a[i * wa + k] * b[k * wb + j];
      }
      c[i * wb + j] = sum;
    }
  }
}

// Reference CPU implementation for verification
void reference_convolution3d(const std::vector<float> &input,
                             const std::vector<float> &kernels,
                             std::vector<float> &output, int ha, int wa, int hb,
                             int wb, int hc, int wc, int input_channels,
                             int output_channels) {
  for (int oc = 0; oc < output_channels; ++oc) {
    for (int i = 0; i < hb; ++i) {
      for (int j = 0; j < wb; ++j) {
        float sum = 0.0f;
        for (int ic = 0; ic < input_channels; ++ic) {
          for (int ki = 0; ki < hc; ++ki) {
            for (int kj = 0; kj < wc; ++kj) {
              int in_i = i + ki;
              int in_j = j + kj;
              if (in_i < ha && in_j < wa) {
                float image_val = input[ic * ha * wa + in_i * wa + in_j];
                float kernel_val = kernels[oc * input_channels * hc * wc +
                                           ic * hc * wc + ki * wc + kj];
                sum += image_val * kernel_val;
              }
            }
          }
        }
        output[oc * hb * wb + i * wb + j] = sum;
      }
    }
  }
}

TEST(CudaKernelTests, test_sgemm) {
  const int ha = 2;
  const int wa = 3;
  const int hb = 3;
  const int wb = 2;
  const int hc = ha;
  const int wc = wb;

  std::vector<float> h_a = {1, 2, 3, 4, 5, 6};
  std::vector<float> h_b = {7, 8, 9, 10, 11, 12};
  std::vector<float> h_c(hc * wc, 0.0f);
  std::vector<float> h_c_expected(hc * wc, 0.0f);

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, h_a.size() * sizeof(float));
  cudaMalloc(&d_b, h_b.size() * sizeof(float));
  cudaMalloc(&d_c, h_c.size() * sizeof(float));

  cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 dim_block(block_size * block_size);
  int tiles_x = (wc + block_size - 1) / block_size;
  int tiles_y = (hc + block_size - 1) / block_size;
  dim3 dim_grid(tiles_x * tiles_y);

  sgemm_1d<block_size>
      <<<dim_grid, dim_block>>>(d_a, d_b, d_c, ha, wa, hb, wb, hc, wc);
  cudaDeviceSynchronize();

  cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  reference_gemm(h_a.data(), h_b.data(), h_c_expected.data(), ha, wa, wb);

  const float eps = 1e-4;
  for (int i = 0; i < hc * wc; ++i) {
    EXPECT_NEAR(h_c[i], h_c_expected[i], eps) << "Mismatch at index " << i;
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

TEST(CudaKernelTests, test_conv3d_1d_launch) {
  const int ha = 4, wa = 4;
  const int hc = 2, wc = 2;
  const int hb = ha - hc + 1, wb = wa - wc + 1;
  const int input_channels = 1;
  const int output_channels = 1;

  const int input_size = input_channels * ha * wa;
  const int kernel_size = output_channels * input_channels * hc * wc;
  const int output_size = output_channels * hb * wb;

  std::vector<float> h_input_image(input_size);
  std::vector<float> h_kernels(kernel_size);
  std::vector<float> h_output_image(output_size, 0.0f);
  std::vector<float> h_expected_output(output_size, 0.0f);

  std::iota(h_input_image.begin(), h_input_image.end(), 1.0f);
  std::fill(h_kernels.begin(), h_kernels.end(), 1.0f);

  reference_convolution3d(h_input_image, h_kernels, h_expected_output, ha, wa,
                          hb, wb, hc, wc, input_channels, output_channels);

  float *d_input_image, *d_output_image, *d_kernels;
  cudaMalloc(&d_input_image, input_size * sizeof(float));
  cudaMalloc(&d_kernels, kernel_size * sizeof(float));
  cudaMalloc(&d_output_image, output_size * sizeof(float));

  cudaMemcpy(d_input_image, h_input_image.data(), input_size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernels, h_kernels.data(), kernel_size * sizeof(float),
             cudaMemcpyHostToDevice);

  int total_outputs = output_channels * hb * wb;
  int threads_per_block = 256;
  int num_blocks = (total_outputs + threads_per_block - 1) / threads_per_block;

  convolution3d_1d_launch<<<num_blocks, threads_per_block>>>(
      d_input_image, d_output_image, d_kernels, ha, wa, hb, wb, hc, wc,
      input_channels, output_channels);

  cudaDeviceSynchronize();

  cudaMemcpy(h_output_image.data(), d_output_image, output_size * sizeof(float),
             cudaMemcpyDeviceToHost);

  const float eps = 1e-4;
  for (int i = 0; i < output_size; ++i) {
    EXPECT_NEAR(h_output_image[i], h_expected_output[i], eps)
        << "Mismatch at index " << i;
  }

  cudaFree(d_input_image);
  cudaFree(d_output_image);
  cudaFree(d_kernels);
}

TEST(CudaKernelTests, test_mlp_layer_compute_gradients_and_backprop) {
  const int input_size = 3;
  const int output_size = 2;

  MlpLayer layer(input_size, output_size);

  std::vector<float> h_input = {1.0f, 2.0f, 3.0f};
  std::vector<float> h_dL_dy = {0.1f, -0.2f};

  float *d_input, *d_dL_dy;
  cuda_check(cudaMalloc(&d_input, input_size * sizeof(float)));
  cuda_check(cudaMalloc(&d_dL_dy, output_size * sizeof(float)));

  cuda_check(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(d_dL_dy, h_dL_dy.data(), output_size * sizeof(float),
                        cudaMemcpyHostToDevice));

  float *d_output;
  cuda_check(cudaMalloc(&d_output, output_size * sizeof(float)));
  layer.forward(d_input, d_output);
  layer.relu(d_output);

  layer.compute_gradients(d_input, d_dL_dy);

  std::vector<float> h_dL_dW(input_size * output_size);
  std::vector<float> h_dL_db(output_size);
  std::vector<float> h_dL_dx(input_size);

  cuda_check(cudaMemcpy(h_dL_dW.data(), layer.get_weight_grad(),
                        input_size * output_size * sizeof(float),
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(h_dL_db.data(), layer.get_bias_grad(),
                        output_size * sizeof(float), cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(h_dL_dx.data(), layer.back_prop(d_input, d_dL_dy, 0.1f),
                        input_size * sizeof(float), cudaMemcpyDeviceToHost));

  for (float v : h_dL_db) {
    EXPECT_TRUE(std::isfinite(v));
  }

  for (float v : h_dL_dx) {
    EXPECT_TRUE(std::isfinite(v));
  }

  for (float v : h_dL_dW) {
    EXPECT_TRUE(std::isfinite(v));
  }

  std::vector<float> h_bias_updated(output_size);
  cuda_check(cudaMemcpy(h_bias_updated.data(), layer.get_device_bias(),
                        output_size * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < output_size; ++i) {
    float expected = layer.get_host_bias()[i] - 0.1f * h_dL_db[i];
    EXPECT_NEAR(h_bias_updated[i], expected, 1e-3)
        << "Mismatch in updated bias at " << i;
  }

  std::cout << "HELLO" << "\n";
  cudaFree(d_input);
  cudaFree(d_dL_dy);
  cudaFree(d_output);
}
