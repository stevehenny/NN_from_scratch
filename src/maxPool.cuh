
class maxPool {

public:
  maxPool(int HA, int WA, int HB, int WB, int input_channels);
  void forward(float *d_input, float *d_output);

private:
  int HA, WA, HB, WB, input_channels;
};
