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
