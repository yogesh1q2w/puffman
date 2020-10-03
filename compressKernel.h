#include "huffman.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef KERNEL_CU
extern __constant__ unsigned char const_code[256*255];
extern __constant__ unsigned char const_codeSize[256];
#endif

__global__ void updatefrequency(unsigned int fileSize,
                                unsigned char *fileContent,
                                unsigned long long int *frequency);

__global__ void genBitCompressed(unsigned int lastBlockIndex,
                                 unsigned char *dfileContent,
                                 unsigned int *dbitOffsets,
                                 unsigned char *dbitCompressedFile);

__global__ void encode(unsigned int bitCompressedFileSize,
                       unsigned char *dbitCompressedFile,
                       unsigned char *d_compressedFile);

__global__ void skss_compress_with_shared(unsigned int lastBlockIndex,
                                          unsigned char *dfileContent,
                                          unsigned int *dbitOffsets,
                                          unsigned int *d_compressedFile,
                                          unsigned char maxCodeSize);

__global__ void skss_compress(unsigned int lastBlockIndex,
                              unsigned char *dfileContent,
                              unsigned int *dbitOffsets,
                              unsigned int *d_compressedFile,
                              unsigned char maxCodeSize);

__global__ void printDict(codedict &dict);