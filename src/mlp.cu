#include "LayerClasses.cuh"
#include "LoadData.h"
#include "Network.cuh"
#include "cudaClasses.cuh"
#include "gmock/gmock.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdlib.h>

#define INPUT_IMAGE_ROWS 28
#define INPUT_IMAGE_COLS 28
#define NUM_OF_LABELS 10

float *getInputImage(float *images, int imageNum) {

  return (&images[0] + (INPUT_IMAGE_COLS * INPUT_IMAGE_ROWS * imageNum));
}

float *getLabel(float *labels, int label_num) {

  return (&labels[0] + (label_num));
}
void saveToFile(const float *data, size_t size, const char *filename) {
  FILE *file = fopen(filename, "wb");
  if (file) {
    fwrite(data, sizeof(float), size, file);
    fclose(file);
  } else {
    printf("Error saving to file %s\n", filename);
  }
}

template <typename... Layers>
std::vector<std::unique_ptr<Layer>> make_layer_vector(Layers &&...layers) {
  std::vector<std::unique_ptr<Layer>> v;
  (v.emplace_back(std::forward<Layers>(layers)), ...);
  return v;
}
int main(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "Usage: <imageFile> <labelFile>\n");
    return 1;
  }
  float alpha = 0.001;

  const char *imageFile = argv[1];
  const char *labelFile = argv[2];

  const int numImages = 60000;
  const int rows = 28, cols = 28;
  const int imageSize = rows * cols;

  const int input_layer_nodes = imageSize;
  const int hidden_layer_nodes = 100;
  const int output_layer_nodes = 10;
  const int batch_size = 1;
  const int num_epochs = 5;

  // Load MNIST
  uint8_t *images = loadMNISTImages(imageFile, numImages, rows, cols);
  uint8_t *labels = loadMNISTLabels(labelFile, numImages);

  // Normalize images to float in range [-1.0, 1.0]
  float *normalized_images =
      (float *)malloc(numImages * imageSize * sizeof(float));
  for (int img = 0; img < numImages; ++img) {
    for (int i = 0; i < imageSize; ++i) {
      uint8_t pixel = images[img * imageSize + i];
      normalized_images[img * imageSize + i] =
          ((float)pixel / 255.0f) * 2.0f - 1.0f;
    }
  }

  float *input_image = getInputImage(normalized_images, 0);
  float *d_input_image;
  cuda_check(cudaMalloc(&d_input_image, input_layer_nodes * sizeof(float)));
  cuda_check(cudaMemcpy(d_input_image, input_image,
                        input_layer_nodes * sizeof(float),
                        cudaMemcpyHostToDevice));
  float *label_output;
  // init label_outputs
  label_output =
      (float *)malloc(numImages * output_layer_nodes * sizeof(float));
  for (int img = 0; img < numImages; ++img) {
    for (int j = 0; j < output_layer_nodes; ++j) {
      label_output[img * output_layer_nodes + j] =
          (labels[img] == j) ? 1.0f : 0.0f;
    }
  }

  float *d_labels;
  cuda_check(
      cudaMalloc(&d_labels, numImages * output_layer_nodes * sizeof(float));
      cuda_check(cudaMemcpy(d_labels, label_output,
                            numImages * output_layer_nodes * sizeof(float),
                            cudaMemcpyHostToDevice)));
  // MlpLayer input_layer(input_layer_nodes, input_layer_nodes);
  // MlpLayer hidden_layer(input_layer_nodes, hidden_layer_nodes);
  // MlpLayer output_layer(hidden_layer_nodes, output_layer_nodes);
  // SoftmaxLayer softmax_layer(output_layer_nodes, output_layer_nodes);

  Network mlp(
      make_layer_vector(
          std::make_unique<MlpLayer>(input_layer_nodes, input_layer_nodes),
          std::make_unique<MlpLayer>(input_layer_nodes, hidden_layer_nodes),
          std::make_unique<MlpLayer>(hidden_layer_nodes, output_layer_nodes)),
      std::make_unique<SoftmaxLayer>(output_layer_nodes, output_layer_nodes),
      device_ptr(d_labels), 0.1f);
  // Print results
  printf("Label: %d\n", labels[0]);
  printImage(input_image, rows, cols);

  float *d_input_batch;

  cuda_check(cudaMalloc(&d_input_batch,
                        batch_size * input_layer_nodes * sizeof(float)));
  for (int epoch = 0; epoch < num_epochs; ++epoch) {
    float total_loss = 0.0f;

    for (int i = 0; i < numImages; i += batch_size) {

      // FIXME: This is here for when I support batch_sizes
      int current_batch_size = std::min(batch_size, numImages - i);

      float *image_ptr = getInputImage(normalized_images, i);
      cuda_check(
          cudaMemcpy(d_input_batch, image_ptr,
                     current_batch_size * input_layer_nodes * sizeof(float),
                     cudaMemcpyHostToDevice));

      // forward pass
      mlp.forward(d_input_batch, i);
      total_loss += mlp.get_loss();

      mlp.back_prop();
    }
    std::cout << "Epoch: " << (epoch + 1) << " - Avg Loss: "
              << (total_loss / (static_cast<float>(numImages) / batch_size))
              << "\n";
  }
  mlp.forward(d_input_image, 0);
  std::cout << "Loss: " << mlp.get_loss() << "\n";
  int length = 10;
  mlp.back_prop();
  float *output;
  // Cleanup
  free(images);
  free(labels);
  free(normalized_images);
  free(label_output);
  cudaFree(d_input_image);
  cudaFree(d_input_batch);
  return 0;
}
