#pragma once
#include <cstdint>

#define KERNEL_SIZE 3
#define IMAGE_SIZE 28

struct ImageSize {
  int width;
  int height;
  ImageSize(int width, int height) : width(width), height(height) {}
};
class convLayer {

public:
  convLayer(ImageSize inputImageSize, ImageSize outputImageSize,
            ImageSize kernelSize, float *kernels, uint8_t input_channels,
            uint8_t output_channels);
  ~convLayer();
  float *forward(float *input_image, float *output_image);
  void ReLU(float *B);

private:
  int HA, WA, HB, WB, HC, WC; // A - input, B - output, C - kernel
  float *d_kernels;
  float *kernels;
  uint8_t input_channels;
  uint8_t output_channels;
  uint8_t depth;
};

class maxPool {

public:
  maxPool(int HA, int WA, int HB, int WB, int input_channels);
  float *forward(float *d_input, float *d_output);

private:
  int HA, WA, HB, WB, input_channels;
};

class mlpLayer {

public:
  mlpLayer(int input_size, int output_size, float *bias, float *weights);
  ~mlpLayer();
  float *forward(float *d_input, float *d_output);
  void softMax(float *d_input, float *d_output);

private:
  int input_size, output_size;
  float *d_bias;    // vector
  float *d_weights; // matrix
};
