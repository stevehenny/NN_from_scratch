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

__global__ void ReLU_kernel(float *B, int N);

__global__ void sgemm(float *A, float *B, float *C, int HA, int WA, int HB,
                      int WB, int HC, int WC);

__global__ void vecAdd(float *A_vec, float *B_vec, bool neg, int len);

__global__ void matAdd(float *A, float *B, float *C, int rows, int cols);

__global__ void softmaxKernel(const float *input, float *output, int len);

__global__ void softmaxCrossEntropyBackward(float *softmax_output, float *label,
                                            float *grad_output, int length);

__global__ void outerProduct(float *d_out, float *input, float *dW,
                             int out_size, int in_size);

__global__ void reluBackward(float *input, float *grad_output,
                             float *grad_input, int size);
__global__ void maxPoolBackward(float *d_out, int *max_indices, float *d_input,
                                int size);

__global__ void tensorElementwiseMult(float *A, float *B, float *C,
                                      int totalElements);
__global__ void sgdUpdate(float *weights, float *grad, float lr, int size);

__device__ __host__ float computeLoss(float *d_output, float *d_target,
                                      int length);

__device__ __host__ float computeCrossEntropyLoss(float *y_hat, float *y,
                                                  int length);
