#ifndef DECOMPRESS_KERNEL
#define DECOMPRESS_KERNEL 1

#include "constants.h"
#include "huffman.h"
#include <cuda.h>

__global__ void single_shot_decode(uint *encodedString, uint encodedFileSize,
                                   unsigned char *treeToken, uint *d_treeLeft,
                                   uint *d_treeRight,
                                   volatile uint *d_charOffset,
                                   uint numBlocksInEncodedString,
                                   uint *d_decodedString, uint sizeOfFile,
                                   uint *d_taskCounter, uint numNodes);
#endif