#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H
#include <iostream>
#include <memory>

struct CudaDeleter {
  void operator()(float *ptr) const { cudaFree(ptr); }
};

using device_ptr = std::unique_ptr<float, CudaDeleter>;

#define cuda_check(stmt)                                                       \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << "\n";    \
      std::cerr << "Failed to run stmt: " << #stmt << "\n";                    \
      std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#endif // CUDA_CHECK_H
