#include "LoadData.h"
#include "convLayer.cuh"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

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
  const int outSize = (rows - (KERNEL_SIZE - 1)) * (cols - (KERNEL_SIZE - 1));

  // Load raw uint8_t MNIST data
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

  // Define simple float kernel (for example purposes)
  float *kernels = (float *)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
  for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; ++i) {
    kernels[i] = rand();
  }

  // Initialize conv layer
  convLayer layer1 = convLayer(ImageSize(28, 28), ImageSize(26, 26),
                               ImageSize(3, 3), kernels, 1, 1);

  // Process the first image
  float *input_image = &normalized_images[0];
  float *output_image = (float *)malloc(outSize * sizeof(float));
  layer1.forward(input_image, output_image);

  // Print results
  printf("Label: %d\n", labels[0]);
  printImage(input_image, rows, cols); // Assumes float-compatible printImage
  printImage(output_image, rows - (KERNEL_SIZE - 1), cols - (KERNEL_SIZE - 1));

  // Clean up
  free(images);
  free(labels);
  free(normalized_images);
  free(kernels);
  free(output_image);

  return 0;
}
