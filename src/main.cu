#include "LoadData.h"
#include "convLayer.cuh"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

float randomFloat(float min, float max) {
  return min + ((float)rand() / RAND_MAX) * (max - min);
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

  const char *imageFile = argv[1];
  const char *labelFile = argv[2];

  const int numImages = 60000;
  const int rows = 28, cols = 28;
  const int imageSize = rows * cols;

  // Convolution setup
  const int kernel_size = KERNEL_SIZE; // usually 3
  const int input_channels = 1;
  const int output_channels = 16;
  const int out_rows = rows - (kernel_size - 1);
  const int out_cols = cols - (kernel_size - 1);
  const int output_size_per_channel = out_rows * out_cols;

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

  // Allocate kernels for all output channels
  float *kernels = (float *)malloc(output_channels * input_channels *
                                   kernel_size * kernel_size * sizeof(float));
  for (int oc = 0; oc < output_channels; ++oc) {
    for (int ic = 0; ic < input_channels; ++ic) {
      for (int i = 0; i < kernel_size * kernel_size; ++i) {
        // Different patterns for each output channel
        kernels[((oc * input_channels + ic) * kernel_size * kernel_size) + i] =
            ((i + oc) % 2 == 0) ? randomFloat(-1.0f, 1.0f) : 0;
      }
    }
  }

  // Initialize conv layer
  convLayer layer1 =
      convLayer(ImageSize(rows, cols), ImageSize(out_rows, out_cols),
                ImageSize(kernel_size, kernel_size), kernels, input_channels,
                output_channels);

  // Prepare input and output buffers
  float *input_image = &normalized_images[0];
  float *output_image = (float *)malloc(
      output_channels * output_size_per_channel * sizeof(float));

  // Run forward pass
  layer1.forward(input_image, output_image);

  // Save the input image to a binary file
  saveToFile(input_image, imageSize, "input_bin");

  // Save the output image to a binary file
  saveToFile(output_image, output_channels * output_size_per_channel,
             "output_bin");

  // Print results
  printf("Label: %d\n", labels[0]);
  printImage(input_image, rows, cols);

  for (int oc = 0; oc < output_channels; ++oc) {
    printf("=== Output Channel %d ===\n", oc);
    printImage(&output_image[oc * output_size_per_channel], out_rows, out_cols);
  }

  // Cleanup
  free(images);
  free(labels);
  free(normalized_images);
  free(kernels);
  free(output_image);

  return 0;
}
