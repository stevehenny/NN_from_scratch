#include "CudaChecks.cuh"
#include "LoadData.h"
#include "convLayer.cuh"
#include "maxPool.cuh"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>

#define INPUT_IMAGE_ROWS 28
#define INPUT_IMAGE_COLS 28
#define IMAGE_NUMBER(ptr, num)                                                 \
  (return ((uint8_t *)&ptr[0] + (INPUT_IMAGE_ROWS * INPUT_IMAGE_COLS * num)))

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

  int pool_rows = out_rows / 2;
  int pool_cols = out_cols / 2;
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

  // Random number generator setup
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> pos_dist(0.0f, 1.0f);
  std::uniform_real_distribution<float> neg_dist(-1.0f, 0.0f);

  // Allocate and initialize kernels with actual randomness
  float *kernels = (float *)malloc(output_channels * input_channels *
                                   kernel_size * kernel_size * sizeof(float));
  for (int oc = 0; oc < output_channels; ++oc) {
    for (int ic = 0; ic < input_channels; ++ic) {
      for (int i = 0; i < kernel_size * kernel_size; ++i) {
        kernels[((oc * input_channels + ic) * kernel_size * kernel_size) + i] =
            ((i + oc) % 2 == 0) ? pos_dist(gen) : neg_dist(gen);
      }
    }
  }

  // Initialize conv layer
  convLayer layer1 =
      convLayer(ImageSize(rows, cols), ImageSize(out_rows, out_cols),
                ImageSize(kernel_size, kernel_size), kernels, input_channels,
                output_channels);

  maxPool poolLayer =
      maxPool(out_rows, out_cols, out_rows / 2, out_cols / 2, output_channels);
  // Prepare input and output buffers
  float *input_image = getInputImage(normalized_images, 3);
  float *output_image = (float *)malloc(
      output_channels * output_size_per_channel * sizeof(float));
  float *output_maxPool;

  output_maxPool =
      (float *)malloc(output_channels * pool_rows * pool_cols * sizeof(float));
  float *d_input_image;
  float *d_output_image;
  float *d_output_maxPool;
  cudaCheck(cudaMalloc((void **)&d_input_image,
                       input_channels * cols * rows * sizeof(float)));
  cudaCheck(cudaMalloc((void **)&d_output_image,
                       output_channels * out_cols * out_rows * sizeof(float)));
  cudaCheck(cudaMalloc((void **)&d_output_maxPool, output_channels * out_cols /
                                                       2 * out_rows / 2 *
                                                       sizeof(float)));
  cudaCheck(cudaMemcpy(d_input_image, input_image,
                       input_channels * cols * rows * sizeof(float),
                       cudaMemcpyHostToDevice));
  // Run forward pass
  layer1.forward(d_input_image, d_output_image);
  poolLayer.forward(d_output_image, d_output_maxPool);

  cudaCheck(cudaMemcpy(output_image, d_output_image,
                       output_channels * out_cols * out_rows * sizeof(float),
                       cudaMemcpyDeviceToHost));

  cudaCheck(cudaMemcpy(output_maxPool, d_output_maxPool,
                       output_channels * pool_rows * pool_cols * sizeof(float),
                       cudaMemcpyDeviceToHost));
  cudaCheck(cudaFree(d_input_image));
  cudaCheck(cudaFree(d_output_image));
  cudaCheck(cudaFree(d_output_maxPool));
  // Save the input image to a binary file
  saveToFile(input_image, imageSize, "input_bin");

  // Save the output image to a binary file
  saveToFile(output_image, output_channels * output_size_per_channel,
             "output_bin");

  saveToFile(output_maxPool, output_channels * pool_rows * pool_cols,
             "pool_bin");

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
