#pragma once
#include "CudaChecks.cuh"
#include "LayerClasses.cuh"
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

// Base class for all layers

class Network {
public:
  // Main constructor: accepts arbitrary number of things
  template <typename... Args> Network(Args &&...args) {
    addLayers(std::forward<Args>(args)...);
    for (auto &layer : layers) {
      float *d_out = nullptr;
      int num_outputs = layer->get_num_outputs();
      cuda_check(cudaMalloc(&d_out, num_outputs * sizeof(float)));
      d_pointers.push_back(d_out);
    }
  }

  ~Network() {
    for (auto ptr : d_pointers) {
      cuda_check(cudaFree(ptr));
    }
  }
  void forward(float *d_input_image) {}

private:
  std::vector<std::unique_ptr<Layer>> layers;
  std::vector<float *> d_pointers;

  // Base case for recursion
  void addLayers() {}

  // Overload for raw pointer to a Layer
  template <typename T,
            typename = std::enable_if_t<std::is_base_of<Layer, T>::value>>
  void addLayers(T *ptr) {
    layers.emplace_back(ptr); // take ownership
  }

  // Overload for derived Layer rvalues or lvalues (objects)
  template <typename T, typename = std::enable_if_t<
                            std::is_base_of<Layer, std::decay_t<T>>::value>>
  void addLayers(T &&layer) {
    layers.emplace_back(
        std::make_unique<std::decay_t<T>>(std::forward<T>(layer)));
  }

  // Recursive case to handle multiple arguments
  template <typename First, typename... Rest>
  void addLayers(First &&first, Rest &&...rest) {
    addLayers(std::forward<First>(first));
    addLayers(std::forward<Rest>(rest)...);
  }
};
