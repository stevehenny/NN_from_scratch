#pragma once
#include "LayerClasses.cuh"
#include "cudaClasses.cuh"
#include <memory>
#include <vector>

class Network {
public:
  Network(std::vector<std::unique_ptr<Layer>> &&layer_list,
          std::unique_ptr<SoftmaxLayer> &&softmax_layer, device_ptr &&d_labels,
          float learning_rate);
  // Prevent accidental copies:
  // copies will copy pointers and those pointers will be double freed
  Network(const Network &) = delete;
  Network &operator=(const Network &) = delete;

  ~Network() = default;

  void forward(float *d_input, int label_num);
  void back_prop();
  float get_loss();
  float *get_label(int label_num);

private:
  std::vector<std::unique_ptr<Layer>> layers;
  std::unique_ptr<SoftmaxLayer> softmax_layer;
  device_ptr d_labels;
  std::vector<device_ptr> d_pointers_forward;
  std::vector<device_ptr> d_pointers_backprop;
  float *cur_label;
  float learning_rate;
};
