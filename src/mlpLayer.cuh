class mlpLayer {

public:
  mlpLayer(int input_size, int output_size, float *bias, float *weights);
  ~mlpLayer();
  void forward(float *d_input, float *d_output);

private:
  int input_size, output_size;
  float *d_bias;    // vector
  float *d_weights; // matrix
};
