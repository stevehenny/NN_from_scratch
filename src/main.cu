#include "CudaChecks.cuh"
#include "LayerClasses.cuh"
#include "LoadData.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

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

  const int pool_rows = out_rows / 2;
  const int pool_cols = out_cols / 2;

  const int out2_rows = pool_rows - (kernel_size - 1);
  const int out2_cols = pool_cols - (kernel_size - 1);
  const int pool2_rows = out2_rows / 2;
  const int pool2_cols = out2_cols / 2;
  const int conv2_out_channels = 32;

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

  // Initialize conv layers
  convLayer layer1 = convLayer(
      ImageSize(rows, cols), ImageSize(out_rows, out_cols),
      ImageSize(kernel_size, kernel_size), input_channels, output_channels);

  maxPool poolLayer1 =
      maxPool(out_rows, out_cols, out_rows / 2, out_cols / 2, output_channels);

  convLayer layer2(
      ImageSize(pool_rows, pool_cols), ImageSize(out2_rows, out2_cols),
      ImageSize(kernel_size, kernel_size), output_channels, conv2_out_channels);
  maxPool pool2(out2_rows, out2_cols, pool2_rows, pool2_cols,
                conv2_out_channels);

  mlpLayer hidden_layer(pool2_cols * pool2_rows * conv2_out_channels,
                        hidden_layer_nodes);
  mlpLayer output_layer(hidden_layer_nodes, output_layer_nodes);
  float *input_image = getInputImage(normalized_images, 0);
  float *output_image = (float *)malloc(
      output_channels * output_size_per_channel * sizeof(float));
  float *output_maxPool;
  float *softmax_output;

  output_maxPool =
      (float *)malloc(output_channels * pool_rows * pool_cols * sizeof(float));
  softmax_output = (float *)malloc(output_layer_nodes * sizeof(float));

  float *d_input_image, *d_output_conv1, *d_output_pool1, *d_output_conv2,
      *d_output_pool2, *d_hidden_layer, *d_output_layer, *d_softmax;

  cudaCheck(cudaMalloc((void **)&d_input_image,
                       input_channels * rows * cols * sizeof(float)));
  cudaCheck(cudaMalloc((void **)&d_output_conv1,
                       output_channels * out_rows * out_cols * sizeof(float)));
  cudaCheck(
      cudaMalloc((void **)&d_output_pool1,
                 output_channels * pool_rows * pool_cols * sizeof(float)));
  cudaCheck(
      cudaMalloc((void **)&d_output_conv2,
                 conv2_out_channels * out2_rows * out2_cols * sizeof(float)));
  cudaCheck(
      cudaMalloc((void **)&d_output_pool2,
                 conv2_out_channels * pool2_rows * pool2_cols * sizeof(float)));
  cudaCheck(
      cudaMalloc((void **)&d_hidden_layer, hidden_layer_nodes * sizeof(float)));
  cudaCheck(
      cudaMalloc((void **)&d_output_layer, output_layer_nodes * sizeof(float)));
  cudaCheck(
      cudaMalloc((void **)&d_softmax, output_layer_nodes * sizeof(float)));

  cudaCheck(cudaMemcpy(d_input_image, input_image,
                       input_channels * cols * rows * sizeof(float),
                       cudaMemcpyHostToDevice));
  // Run forward pass
  d_output_conv1 = layer1.forward(d_input_image, d_output_conv1);
  layer1.ReLU(d_output_conv1);
  d_output_pool1 = poolLayer1.forward(d_output_conv1, d_output_pool1);
  d_output_conv2 = layer2.forward(d_output_pool1, d_output_conv2);
  layer2.ReLU(d_output_conv2);
  d_output_pool2 = pool2.forward(d_output_conv2, d_output_pool2);
  d_hidden_layer = hidden_layer.forward(d_output_pool2, d_hidden_layer);
  d_output_layer = output_layer.forward(d_hidden_layer, d_output_layer);
  output_layer.softMax(d_output_layer, d_softmax);

  // copy back to host
  cudaCheck(cudaMemcpy(output_image, d_output_conv1,
                       output_channels * out_cols * out_rows * sizeof(float),
                       cudaMemcpyDeviceToHost));

  cudaCheck(cudaMemcpy(output_maxPool, d_output_pool1,
                       output_channels * pool_rows * pool_cols * sizeof(float),
                       cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(softmax_output, d_softmax,
                       output_layer_nodes * sizeof(float),
                       cudaMemcpyDeviceToHost));
  cudaCheck(cudaFree(d_input_image));
  cudaCheck(cudaFree(d_output_conv1));
  cudaCheck(cudaFree(d_output_pool1));
  cudaCheck(cudaFree(d_output_conv2));
  cudaCheck(cudaFree(d_output_pool2));
  cudaCheck(cudaFree(d_hidden_layer));
  cudaCheck(cudaFree(d_output_layer));
  cudaCheck(cudaFree(d_softmax));
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

  // for (int oc = 0; oc < output_channels; ++oc) {
  //   printf("=== Output Channel %d ===\n", oc);
  //   printImage(&output_image[oc * output_size_per_channel], out_rows,
  //   out_cols);
  // }
  for (int i = 0; i < output_layer_nodes; ++i) {
    printf("Chance of %d: %.3f\n", i, softmax_output[i]);
  }
  // Cleanup
  free(images);
  free(labels);
  free(normalized_images);
  free(output_image);
  free(output_maxPool);
  free(softmax_output);
  return 0;
}
