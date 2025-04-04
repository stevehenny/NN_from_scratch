#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef LOAD_DATA_H
#define LOAD_DATA_H

// read the MNIST IDX file
uint8_t *readIDXFile(const char *filename, size_t headerSize, size_t dataSize);

// Load MNIST images
uint8_t *loadMNISTImages(const char *filename, int numImages, int rows,
                         int cols);

// load MNIST lables
uint8_t *loadMNISTLabels(const char *filename, int numLabels);

// prints image in ASCII
void printImage(float *image, int rows, int cols);

#endif // !LOAD_DATA_H
