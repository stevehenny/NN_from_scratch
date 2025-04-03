#ifndef NETWORK_H
#define NETWORK_H

#include <cstdint>
#include <sys/types.h>
class Network {
public:
  Network(int depth, int height, int width, int layer1_filters,
          int layer2_filters, int layer3_filters, int filter_size,
          int batch_size = 64);
  ~Network();

private:
  int depth, height, width, layer1_filters, layer2_filters, layer3_filters,
      filter_size, batch_size;

  uint8_t *layer1_channels; //  (height, width, depth)
  uint8_t *layer2_channels; //  (height, width, depth)
};

#endif // NETWORK_H
