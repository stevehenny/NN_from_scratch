#include "LoadData.h"
#include <stddef.h>
#include <stdio.h>
#define IDX_LABEL_HEADER_SIZE 8
#define IDX_IMAGE_HEADER_SIZE 16

uint8_t *readIDXFile(const char *filename, size_t headerSize, size_t dataSize) {
  FILE *file = fopen(filename, "rb");
  // if the file is NULL
  if (!file) {
    perror("Failed to open file");
    exit(1);
  }

  fseek(file, 0, SEEK_END);
  long fileSize = ftell(file);
  fseek(file, headerSize, SEEK_SET);
  printf("File size: %ld, expected min size: %zu\n", fileSize,
         headerSize + dataSize);

  uint8_t *data = (uint8_t *)malloc(dataSize);
  if (!data) {
    perror("Failed to allocate memory");
    exit(1);
  }

  size_t dataRead = fread(data, 1, dataSize, file);
  printf("Data read: %zu\n", dataRead);
  fclose(file);
  return data;
}

uint8_t *loadMNISTImages(const char *filename, int numImages, int rows,
                         int cols) {
  int imageSize = rows * cols;
  int totalSize = numImages * imageSize;
  return readIDXFile(filename, IDX_IMAGE_HEADER_SIZE, totalSize);
}

uint8_t *loadMNISTLabels(const char *filename, int numLables) {
  return readIDXFile(filename, IDX_LABEL_HEADER_SIZE, numLables);
}

void printImage(float *image, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%c", image[i * cols + j] > 0 ? '#' : ' ');
    }
    printf("\n");
  }
}

void printImageValues(float *image, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%.2f, ", image[i * cols + j]);
    }
    printf("\n");
  }
}
