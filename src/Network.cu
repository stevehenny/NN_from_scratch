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

void Network::forward(float *d_input, int label_num, int batch_size) {
  this->d_initial_input = d_input;
  float *d_next_input = d_input;
  for (int i = 0; i < layers.size(); ++i) {
    layers[i]->forward(d_next_input, d_pointers_forward[i].get(), batch_size);
    // layers[i]->relu(d_pointers_forward[i].get());
    d_next_input = d_pointers_forward[i].get();
  }

  float *d_label = get_label(label_num);
  this->cur_label = d_label;
  float *d_softmax_output =
      d_pointers_forward[d_pointers_forward.size() - 1].get();
  this->softmax_layer->softmax(d_next_input, d_softmax_output, batch_size);
  this->softmax_layer->forward(d_softmax_output, d_label, batch_size);
}

void Network::back_prop() {
  float *d_y_hat = d_pointers_forward[d_pointers_forward.size() - 1].get();
  if (cur_label == nullptr) {
    throw std::runtime_error("VARIABLE::cur_label::NULL\n");
  }

  // Compute softmax + loss gradient
  this->softmax_layer->back_prop(d_y_hat, cur_label, learning_rate);

  // Grab the gradient that will be sent to the last hidden layer
  float *d_next_grad = this->softmax_layer->get_input_grad();

  // --- DEBUG: Print first 10 values of gradient from softmax ---
  int grad_size = this->softmax_layer->get_num_inputs();
  // std::vector<float> h_grad(grad_size);
  // cudaMemcpy(h_grad.data(), d_next_grad, grad_size * sizeof(float),
  //            cudaMemcpyDeviceToHost);
  //
  // std::cout << "[DEBUG] Softmax input_grad: ";
  // for (int j = 0; j < std::min(grad_size, 10); ++j) {
  //   std::cout << h_grad[j] << " ";
  // }
  // std::cout << "\n";
  // -------------------------------------------------------------

  // Now backprop through the rest of the layers
  for (int i = this->layers.size() - 1; i >= 0; --i) {
    float *forward_input =
        (i == 0) ? this->d_initial_input : d_pointers_forward[i - 1].get();

    this->layers[i]->back_prop(forward_input, d_next_grad, learning_rate);

    // // --- DEBUG: Print dl_dz (output_grad) ---
    // {
    //   float *d_dl_dz = this->layers[i]->get_output_grad();
    //   int dz_size = this->layers[i]->get_num_outputs();
    //   std::vector<float> h_dl_dz(dz_size);
    //   cudaMemcpy(h_dl_dz.data(), d_dl_dz, dz_size * sizeof(float),
    //              cudaMemcpyDeviceToHost);
    //
    //   std::cout << "[DEBUG] dl_dz for layer " << i << ": ";
    //   for (int j = 0; j < std::min(dz_size, 10); ++j) {
    //     std::cout << h_dl_dz[j] << " ";
    //   }
    //   std::cout << "\n";
    // }

    // // --- DEBUG: copy dl_dW from device to host for the last layer ---
    // if (i == (int)this->layers.size() - 1) {
    //   float *d_dl_dW = this->layers[i]->get_weight_grad();
    //   int num_weights = this->layers[i]->get_num_inputs() *
    //                     this->layers[i]->get_num_outputs();
    //
    //   std::vector<float> h_dl_dW(num_weights);
    //   cudaMemcpy(h_dl_dW.data(), d_dl_dW, num_weights * sizeof(float),
    //              cudaMemcpyDeviceToHost);
    //
    //   std::cout << "[DEBUG] dl_dW for last layer: ";
    //   for (int j = 0; j < std::min(num_weights, 10); ++j) {
    //     std::cout << h_dl_dW[j] << " ";
    //   }
    //   std::cout << "\n";
    // }
    // --------------------------------------------------------------

    d_next_grad = this->layers[i]->get_input_grad();
  }
}

float Network::get_loss() { return this->softmax_layer->get_loss(); }

float *Network::get_label(int label_num) {
  return d_labels.get() + (SIZE_OF_LABEL * label_num);
}
