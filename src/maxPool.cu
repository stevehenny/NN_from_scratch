#include "CudaChecks.cuh"
#include "cudaKernels.cuh"
#include "maxPool.cuh"
#define POOL_BLOCK_SIZE 16
maxPool::maxPool(int HA, int WA, int HB, int WB, int input_channels)
    : HA(HA), WA(WA), HB(HB), WB(WB), input_channels(input_channels) {}

void maxPool::forward(float *d_input, float *d_output) {
  int grid_x = (WB + POOL_BLOCK_SIZE - 1) / POOL_BLOCK_SIZE;
  int grid_y = (HB + POOL_BLOCK_SIZE - 1) / POOL_BLOCK_SIZE;
  dim3 threads(POOL_BLOCK_SIZE, POOL_BLOCK_SIZE);
  dim3 grid(grid_x, grid_y, input_channels);
  maxPool2D<<<grid, threads>>>(d_input, d_output, HA, WA, HB, WB,
                               input_channels);
  cudaCheck(cudaPeekAtLastError());
  cudaCheck(cudaDeviceSynchronize());
}
