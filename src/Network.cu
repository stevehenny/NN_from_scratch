#include "LayerClasses.cuh"
#include "Network.cuh"
#include "cudaClasses.cuh"
#include <stdexcept>

#define SIZE_OF_LABEL 10
Network::Network(std::vector<std::unique_ptr<Layer>> &&layer_list,
                 std::unique_ptr<SoftmaxLayer> &&softmax_layer,
                 device_ptr &&d_labels, float learning_rate)
    : layers(std::move(layer_list)), softmax_layer(std::move(softmax_layer)),
      d_labels(std::move(d_labels)), learning_rate(learning_rate) {
  // Allocate device memory for each layer's output
  for (int i = 0; i < layers.size(); ++i) {
    float *d_out = nullptr;
    float *d_inp_grad = nullptr;
    cuda_check(
        cudaMalloc(&d_out, layers[i]->get_num_outputs() * sizeof(float)));
    cuda_check(
        cudaMalloc(&d_inp_grad, layers[i]->get_num_inputs() * sizeof(float)));

    d_pointers_forward.emplace_back(device_ptr(d_out));
    d_pointers_backprop.emplace_back(device_ptr(d_inp_grad));
  }
  // allocate another pointer for softmax_layer
  float *d_out = nullptr;
  float *d_inp_grad = nullptr;
  cuda_check(cudaMalloc(&d_out, this->softmax_layer->get_num_outputs() *
                                    sizeof(float)));
  cuda_check(cudaMalloc(&d_inp_grad,
                        this->softmax_layer->get_num_inputs() * sizeof(float)));
  d_pointers_forward.emplace_back(device_ptr(d_out));
  d_pointers_backprop.emplace_back(device_ptr(d_inp_grad));
}

void Network::forward(float *d_input, int label_num) {
  float *d_next_input = d_input;
  for (int i = 0; i < layers.size(); ++i) {
    layers[i]->forward(d_next_input, d_pointers_forward[i].get());
    d_next_input = d_pointers_forward[i].get();
  }

  float *d_label = get_label(label_num);
  this->cur_label = d_label;
  float *d_softmax_output =
      d_pointers_forward[d_pointers_forward.size() - 1].get();
  this->softmax_layer->softmax(d_next_input, d_softmax_output);
  this->softmax_layer->forward(d_softmax_output, d_label);
}

void Network::back_prop() {
  float *d_y_hat = d_pointers_forward[d_pointers_forward.size() - 1].get();
  if (cur_label == nullptr) {
    throw std::runtime_error("VARIABLE::cur_label::NULL\n");
  }
  this->softmax_layer->back_prop(d_y_hat, cur_label, learning_rate);

  // now go through mlp layers, backwards
  // first grad in d_y_hat
  float *d_next_grad = d_y_hat;
  for (int i = this->layers.size() - 1; i >= 0; --i) {
    this->layers[i]->back_prop(d_pointers_backprop[i].get(), d_next_grad,
                               learning_rate);
    d_next_grad = d_pointers_backprop[i].get();
  }
}

float Network::get_loss() { return this->softmax_layer->get_loss(); }

float *Network::get_label(int label_num) {
  return d_labels.get() + (SIZE_OF_LABEL * label_num);
}
