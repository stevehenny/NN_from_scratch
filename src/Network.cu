#include "LayerClasses.cuh"
#include "Network.cuh"
#include "gmock/gmock.h"

Network::Network(std::vector<std::unique_ptr<Layer>> &&layer_list,
                 std::unique_ptr<SoftmaxLayer> &&softmax_layer)
    : layers(std::move(layer_list)), softmax_layer(std::move(softmax_layer)) {
  // Allocate device memory for each layer's output
  for (int i = 0; i < layers.size(); ++i) {
    float *d_out = nullptr;
    int num_outputs = layers[i]->get_num_outputs();
    cuda_check(cudaMalloc(&d_out, num_outputs * sizeof(float)));
    d_pointers.emplace_back(device_ptr(d_out));
  }
  // allocate another pointer for softmax_layer
  float *d_out = nullptr;
  int num_outputs = this->softmax_layer->get_num_outputs();
  cuda_check(cudaMalloc(&d_out, num_outputs * sizeof(float)));
  d_pointers.emplace_back(device_ptr(d_out));
}

void Network::forward(float *d_input, float *d_label) {
  float *d_next_input = d_input;
  for (int i = 0; i < layers.size(); ++i) {
    layers[i]->forward(d_next_input, d_pointers[i].get());
    d_next_input = d_pointers[i].get();
  }

  float *d_softmax_output = d_pointers[d_pointers.size() - 1].get();
  this->softmax_layer->softmax(d_next_input, d_softmax_output);
  this->softmax_layer->forward(d_softmax_output, d_label);
}

float Network::get_loss() { return this->softmax_layer->get_loss(); }
