#include "CudaChecks.cuh"
#include "cudaKernels.cuh"
#include "mlpLayer.cuh"

mlpLayer::mlpLayer(int input_size, int output_size, float *bias, float *weights)
    : input_size(input_size), output_size(output_size) {

  cudaCheck(cudaMalloc((void **)&d_bias, output_size * sizeof(float)));
  cudaCheck(cudaMalloc((void **)&d_weights,
                       input_size * output_size * sizeof(float)));

  cudaMemcpy(d_bias, bias, output_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_weights, weights, input_size * output_size * sizeof(float),
             cudaMemcpyHostToDevice);
}

mlpLayer::~mlpLayer() {
  cudaFree(d_bias);
  cudaFree(d_weights);
}

void mlpLayer::forward(float *d_input, float *d_output) {
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
  vecAdd<<<(output_size + 255) / 256, 256>>>(d_output, d_bias, output_size);
  cudaCheck(cudaDeviceSynchronize());
}
