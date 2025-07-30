#include "CudaChecks.cuh"
#include "LayerClasses.cuh"
#include "LoadData.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdlib.h>

#define INPUT_IMAGE_ROWS 28
#define INPUT_IMAGE_COLS 28

float *getInputImage(float *images, int imageNum) {

  return (&images[0] + (INPUT_IMAGE_COLS * INPUT_IMAGE_ROWS * imageNum));
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

  MlpLayer input_layer(input_layer_nodes, input_layer_nodes);
  MlpLayer hidden_layer(input_layer_nodes, output_layer_nodes);
  MlpLayer output_layer(hidden_layer_nodes, output_layer_nodes);
  SoftmaxLayer softmax_layer(output_layer_nodes, output_layer_nodes);
  float *input_image = getInputImage(normalized_images, 0);
  float *label_output;
  // init label_outputs
  label_output = (float *)malloc(output_layer_nodes * sizeof(float));
  for (int i = 0; i < output_layer_nodes; ++i) {
    if (labels[0] == i) {
      label_output[i] = 1.0f;
    } else {
      label_output[i] = 0.0f;
    }
  }

  // Print results
  printf("Label: %d\n", labels[0]);
  printImage(input_image, rows, cols);

  int length = 10;
  float *output;
  // Cleanup
  free(images);
  free(labels);
  free(normalized_images);
  free(label_output);
  return 0;
}
