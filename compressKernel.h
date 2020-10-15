#include "huffman.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef KERNEL_CU
extern __constant__ unsigned char const_code[256 * 255];
extern __constant__ unsigned char const_codeSize[256];
#endif

__global__ void cu_histgram(unsigned int *d_PartialHistograms,
                            unsigned int *d_Data, unsigned int dataCount,
                            unsigned int byteCount);

__global__ void mergeHistogram(unsigned int *d_Histogram,
                               unsigned int *d_PartialHistograms);

__global__ void skss_compress_with_shared(unsigned int fileSize,
                                          unsigned int *dfileContent,
                                          unsigned int *dbitOffsets,
                                          unsigned int *d_compressedFile,
                                          unsigned char maxCodeSize);

__global__ void skss_compress(unsigned int fileSize, unsigned int *dfileContent,
                              unsigned int *dbitOffsets,
                              unsigned int *d_compressedFile,
                              unsigned char maxCodeSize);
