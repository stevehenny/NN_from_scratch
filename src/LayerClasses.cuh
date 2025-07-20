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
            ImageSize kernelSize, uint8_t input_channels,
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
  float *dL_dy; // gradient carryover from previous layer
  float *dL_dk; // dl_dy cross-correlation x
  float *dL_dx; // dL_dy conv k^(180 degrees)
};

class maxPool {

public:
  maxPool(int HA, int WA, int HB, int WB, int input_channels);
  float *forward(float *d_input, float *d_output, int *d_max_ind);
  float *backProp(float alpha);

private:
  int HA, WA, HB, WB, input_channels;
  float *dL_dy;
  float *dL_dx;
};

class mlpLayer {

public:
  mlpLayer(int input_size, int output_size);
  ~mlpLayer();
  float *forward(float *d_input, float *d_output);
  void ReLU(float *d_input);
  float computeLoss(float *y_hat, float *y, float d_output, int length);
  float *backProp(float alpha);

private:
  int input_size, output_size;
  float *bias, *d_bias;       // vector
  float *weights, *d_weights; // matrix
  float *dL_dW;               // Local weight gradient -- dL_dy @ x^T
  float *dL_db;               // Local gradient -- dL_dy
  float *dL_dx;               // Local input gradient -- W^T @ dL_dy
  float *dz_dy;               // z = ReLU(y) -> dz_dy = (zi > 0) ? 1 : 0
                // IMPORTANT NOTE: dL_dy (loss of output of MLPLayer), is
                // dL_dz elementwize_mult dz_dy
};

class SoftmaxLayer {

public:
  SoftmaxLayer(int input_size, int output_size);
  ~SoftmaxLayer();
  void softMax(float *d_input, float *d_output);
  float computeLoss(float *y_hat, float *y, float d_output, int length);
  float *backProp(float alpha);

private:
  int input_size, output_size;
  float *dL_dz; // y_hat - y
  float *dL_dy; // d/dy(Loss(y))
};
