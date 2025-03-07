#include "LoadData.h"
#include <stdio.h>

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
  uint32_t *dev_ptr;
  cudaMalloc((void **)&dev_ptr, sizeof(images));
  // printf("Size of images: %zu\n", sizeof(labels));
  // Print first image
  printf("Label: %d\n", labels[0]);
  printImage(images, rows, cols);

  // Free allocated memory
  free(images);
  free(labels);
  return 0;
}
