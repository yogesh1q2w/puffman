#include "huffman.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#ifndef KERNEL_CU
extern __constant__ codedict d_dictionary;
#endif

__global__ void updatefrequency(unsigned int fileSize, char *fileContent, unsigned long long int *frequency);

__global__ void genBitCompressed(unsigned int lastBlockIndex, char *dfileContent, unsigned int *dbitOffsets, unsigned char *dbitCompressedFile);

__global__ void encode(unsigned int bitCompressedFileSize, unsigned char *dbitCompressedFile, unsigned char *d_compressedFile);