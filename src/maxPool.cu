#include "cudaKernels.cuh"
#include "maxPool.cuh"
#include <cmath>
#define POOL_BLOCK_SIZE 16.0f
maxPool::maxPool(int HA, int WA, int HB, int WB, int input_channels)
    : HA(HA), WA(WA), HB(HB), WB(WB), input_channels(input_channels) {}

void maxPool::forward(float *d_input, float *d_output) {

  dim3 threads(POOL_BLOCK_SIZE, POOL_BLOCK_SIZE);
  dim3 grid(ceil(WB / POOL_BLOCK_SIZE), ceil(HB / POOL_BLOCK_SIZE),
            input_channels);
  maxPool2D<<<grid, threads>>>(d_input, d_output, HA, WA, HB, WB,
                               input_channels);
}
