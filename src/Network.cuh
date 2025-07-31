#pragma once
#include "CudaChecks.cuh"
#include "LayerClasses.cuh"
#include <memory>
#include <vector>

struct CudaDeleter {
  void operator()(float *ptr) const { cudaFree(ptr); }
};

using device_ptr = std::unique_ptr<float, CudaDeleter>;

class Network {
public:
  Network(std::vector<std::unique_ptr<Layer>> &&layer_list)
      : layers(std::move(layer_list)) {
    // Allocate device memory for each layer's output
    for (auto &layer : layers) {
      float *d_out = nullptr;
      int num_outputs = layer->get_num_outputs();
      cuda_check(cudaMalloc(&d_out, num_outputs * sizeof(float)));
      d_pointers.emplace_back(device_ptr(d_out));
    }
  }

  // Prevent accidental copies
  Network(const Network &) = delete;
  Network &operator=(const Network &) = delete;

  // Allow moves
  Network(Network &&) = default;
  Network &operator=(Network &&) = default;

  ~Network() = default;

private:
  std::vector<std::unique_ptr<Layer>> layers;
  std::vector<device_ptr> d_pointers;
};
