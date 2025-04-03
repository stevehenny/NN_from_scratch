#include "Network.cuh"

Network::Network(int depth, int height, int width, int layer1_filters,
                 int layer2_filters, int layer3_filters, int filter_size,
                 int batch_size)
    : depth(depth), height(height), width(width), batch_size(batch_size),
      layer1_filters(layer1_filters), layer2_filters(layer2_filters),
      layer3_filters(layer3_filters), filter_size(filter_size) {
  size_t layer1_size = depth * height * width * batch_size;
  size_t layer2_size = depth * (height - (filter_size - 1)) *
                       (width - (filter_size - 1)) * layer1_filters *
                       batch_size;
  cudaMalloc((void **)&layer1_channels, layer1_size);
  cudaMalloc((void **)&layer2_channels, layer2_size);
}

Network::~Network() {
  cudaFree(layer1_channels);
  cudaFree(layer2_channels);
}
