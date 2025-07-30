#pragma once
#include <cstdint>

#define KERNEL_SIZE 3
#define IMAGE_SIZE 28

struct ImageSize {
  int width;
  int height;
  ImageSize(int width, int height) : width(width), height(height) {}
};

#ifndef LAYER_H
#define LAYER_H

class Layer {
public:
  virtual ~Layer();
  virtual int get_num_outputs() = 0;
};

#endif

class ConvLayer : public Layer {
public:
  ConvLayer(ImageSize input_image_size, ImageSize output_image_size,
            ImageSize kernel_size, uint8_t input_channels,
            uint8_t output_channels);
  ~ConvLayer();
  float *forward(float *d_input_image, float *d_output_image);
  int get_num_outputs() override;
  void relu(float *b);

private:
  int ha, wa, hb, wb, hc, wc;
  float *d_input_image, *d_output_image;
  float *d_kernels;
  float *kernels;
  uint8_t input_channels;
  uint8_t output_channels;
  uint8_t depth;
  float *dl_dy;
  float *dl_dk;
  float *dl_dx;
};

class MaxPool : public Layer {
public:
  MaxPool(int ha, int wa, int hb, int wb, int input_channels);
  int get_num_outputs() override;
  float *forward(float *d_input, float *d_output, int *d_max_ind);
  float *back_prop(float alpha);

private:
  int ha, wa, hb, wb, input_channels;
  float *dl_dy;
  float *dl_dx;
};

class MlpLayer : public Layer {
public:
  MlpLayer(int input_size, int output_size);
  ~MlpLayer();
  int get_num_outputs() override;
  float *forward(float *d_input, float *d_output);
  void relu(float *d_input);
  void compute_gradients(float *d_input, float *dl_dy);
  float *back_prop(float *d_input, float *dl_dy, float alpha);
  float *get_host_weights();
  float *get_host_bias();
  float *get_device_weights();
  float *get_device_bias();
  float *get_weight_grad();
  float *get_input_grad();
  float *get_bias_grad();
  float *get_output_grad();

private:
  int input_size, output_size;
  float *bias, *d_bias;
  float *weights, *d_weights;
  float *d_weights_transpose;
  float *dl_dw;
  float *dl_db;
  float *dl_dx;
  float *dl_dz;
  float *dy_dz;
};

class SoftmaxLayer : public Layer {
public:
  SoftmaxLayer(int input_size, int output_size);
  ~SoftmaxLayer();
  int get_num_outputs() override;
  void softmax(float *d_input, float *d_output);
  float compute_loss(float *d_y_hat, float *d_y);
  float *back_prop(float *d_y_hat, float *d_y, float alpha);

private:
  int input_size, output_size;
  float *y_hat, *y;
  float *d_loss, *h_loss;
};
