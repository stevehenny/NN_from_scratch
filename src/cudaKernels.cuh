#define BLOCK_SIZE 16

__global__ void Convolution(float *A, float *B, float *C, int HA, int WA,
                            int HB, int WB, int HC, int WC, int input_channels,
                            int output_channels);

__global__ void Convolution3D(float *A, float *B, float *C, int HA, int WA,
                              int HB, int WB, int HC, int WC,
                              int input_channels, int output_channels);

__global__ void maxPool2D(float *A, float *B, int HA, int WA, int HB, int WB,
                          int input_channels);

__global__ void ReLU_kernel(float *A, int HA, int WA, int channels);

__global__ void sgemm(float *A, float *B, float *C, int HA, int WA, int HB,
                      int WB, int HC, int WC);

__global__ void vecAdd(float *A_vec, float *B_vec, int len);

__global__ void matAdd(float *A, float *B, float *C, int rows, int cols);

__global__ void softmaxKernel(const float *input, float *output, int len);
