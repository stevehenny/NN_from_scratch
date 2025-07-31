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
  Network(std::vector<std::unique_ptr<Layer>> &&layer_list,
          std::unique_ptr<SoftmaxLayer> &&softmax_layer);
  // Prevent accidental copies:
  // copies will copy pointers and those pointers will be double freed
  Network(const Network &) = delete;
  Network &operator=(const Network &) = delete;

  ~Network() = default;

  void forward(float *d_input, float *d_label);
  float get_loss();

private:
  std::vector<std::unique_ptr<Layer>> layers;
  std::unique_ptr<SoftmaxLayer> softmax_layer;
  std::vector<device_ptr> d_pointers;
};
