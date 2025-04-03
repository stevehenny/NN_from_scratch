#include "LoadData.h"
#include "convLayer.cuh"
#include <cstdint>
#include <cstdlib>
#include <stdio.h>
__global__ void forward(uint8_t *d_images, float **d_weights);

int main(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "Usage: <imageFile> <labels>");
    return 1;
  }

  const char *imageFile = argv[1];
  const char *labelFile = argv[2];

  int numImages = 60000;
  int rows = 28, cols = 28;

  uint8_t *images = loadMNISTImages(imageFile, numImages, rows, cols);
  // printf("Size of images: %zu\n", sizeof(images));
  uint8_t *labels = loadMNISTLabels(labelFile, numImages);
  uint8_t *d_images, *d_labels;
  uint8_t *kernels = (uint8_t *)malloc(KERNEL_SIZE * KERNEL_SIZE);
  for (int i = 0; i < 9; ++i) {
    if ((i & 1) == 0) {
      kernels[i] = 1;
    } else {
      kernels[i] = 0;
    }
  }
  convLayer layer1 = convLayer(kernels, 1, 1, 28, 28);
  uint8_t *input_image = &images[0];
  uint8_t *output_image =
      (uint8_t *)malloc((28 - (KERNEL_SIZE - 1)) * (28 - (KERNEL_SIZE - 1)));
  layer1.forward(input_image, output_image);
  // layer1.forward(uint8_t *input_image, uint8_t *output_image)
  // cudaMalloc((void **)&d_images, sizeof(images));
  // cudaMalloc((void **)&d_labels, sizeof(labels));
  // cudaMemcpy(d_images, images, sizeof(images), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_labels, labels, sizeof(labels), cudaMemcpyHostToDevice);

  // printf("Size of images: %zu\n", sizeof(labels));
  // Print first image
  printf("Label: %d\n", labels[0]);
  printImage(input_image, rows, cols);
  printImage(output_image, rows, cols);

  // Free allocated memory
  free(images);
  free(labels);
  free(output_image);
  return 0;
}
