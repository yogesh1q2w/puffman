#ifndef DECOMPRESS_KERNEL
#define DECOMPRESS_KERNEL 1

#include "constants.h"
#include "huffman.h"
#include <cuda.h>

__global__ void single_shot_decode(uint *encodedString, uint encodedFileSize,
                                   unsigned char *treeToken, uint *treeLeft,
                                   uint *treeRight, volatile uint *charOffset,
                                   uint numBlocksInEncodedString,
                                   uint *decodedString, uint sizeOfFile,
                                   uint *taskCounter, uint numNodes,
                                   uint numTasks);
#endif