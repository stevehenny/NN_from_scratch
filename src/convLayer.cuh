#ifndef CONV_LAYER_H
#define CONV_LAYER_H
#include <cstdint>

#define KERNEL_SIZE 3
#define IMAGE_SIZE 28
#define BLOCK_SIZE 32

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
  void forward(float *input_image, float *output_image);

private:
  int HA, WA, HB, WB, HC, WC; // A - input, B - output, C - kernel
  float *d_kernels;
  float *kernels;
  uint8_t input_channels;
  uint8_t output_channels;
  uint8_t depth;
};

#endif // CONV_LAYER_H
