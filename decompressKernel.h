#ifndef DECOMPRESS_KERNEL
#define DECOMPRESS_KERNEL 1

#include "huffman.h"
#include <cuda.h>

extern __constant__ TreeArrayNode deviceTree[512];
extern __constant__ int rootIndex;

__global__ void convertBitsToBytes(unsigned char *input,
                                   unsigned char *inputInBytes, unsigned size,
                                   unsigned blockSize);

__global__ void calculateNoOfTokensInBlock(unsigned char *input, unsigned size,
                                           unsigned blockSize,
                                           unsigned *outputSizes);

__global__ void writeOutput(unsigned char *input, unsigned char *output,
                            unsigned size, unsigned blockSize,
                            unsigned *offsets);

__global__ void printOffsets(unsigned *offsets, unsigned noOfElements);

#endif