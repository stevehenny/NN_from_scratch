#include "LoadData.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
  const char *imageFile =
      "../dataset/train-images-idx3-ubyte/train-images-idx3-ubyte";
  const char *labelFile =
      "../dataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte";

  int numImages = 60000;
  int rows = 28, cols = 28;

  uint8_t *images = loadMNISTImages(imageFile, numImages, rows, cols);
  // printf("Size of images: %zu\n", sizeof(images));
  uint8_t *labels = loadMNISTLabels(labelFile, numImages);
  // printf("Size of images: %zu\n", sizeof(labels));
  // Print first image
  printf("Label: %d\n", labels[0]);
  printImage(images, rows, cols);

  // Free allocated memory
  free(images);
  free(labels);
  return 0;
}
