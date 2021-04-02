#ifndef DECOMPRESS_KERNEL
#define DECOMPRESS_KERNEL 1

#include "constants.h"
#include "huffman.h"
#include <cuda.h>

__global__ void single_shot_decode(uint *encodedString,
                                   unsigned long long int encodedFileSize,
                                   unsigned char *treeToken, uint *treeLeft,
                                   uint *treeRight,
                                   volatile unsigned long long int *charOffset,
                                   uint *decodedString, uint *taskCounter,
                                   uint numNodes, uint numTasks);
#endif