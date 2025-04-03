#ifndef CONV_LAYER_H
#define CONV_LAYER_H
#include <cstdint>

#define KERNEL_SIZE 3
#define IMAGE_SIZE 28
__global__ void conv_forward_kernel();
class convLayer {

public:
  convLayer(uint8_t *kernels, uint8_t input_channels, uint8_t output_channels,
            uint16_t height, uint16_t width);
  ~convLayer();
  void forward(uint8_t *input_image, uint8_t *output_image);

private:
  uint8_t *d_kernels;
  uint8_t *kernels;
  uint8_t input_channels;
  uint8_t output_channels;
  uint16_t height;
  uint16_t width;
  uint8_t depth;
};

#endif // CONV_LAYER_H
