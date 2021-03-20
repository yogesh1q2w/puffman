#include "huffman.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct add_functor {
    uint searchValue;
    uint modifyIndex;
    uint *dbitOffsets;
    add_functor(uint _searchValue, uint _modifyIndex, uint *_dbitOffsets) {
        searchValue = _searchValue;
        modifyIndex = _modifyIndex;
        dbitOffsets = _dbitOffsets;
    }
    __device__ uint operator()(uint &x) {
        return x + (searchValue - (dbitOffsets[modifyIndex]));
    }
};

__global__ void cu_histgram(unsigned int *d_PartialHistograms,
                            unsigned int *d_Data, unsigned int dataCount,
                            unsigned int byteCount);

__global__ void mergeHistogram(unsigned int *d_Histogram,
                               unsigned int *d_PartialHistograms);

__global__ void encode(uint fileSize, uint *dfileContent, uint *dbitOffsets,
                       uint *d_boundary_index, uint *d_compressedFile,
                       uint *d_dictionary_code,
                       unsigned char *d_dictionary_codelens, uint *counter,
                       uint numTasks);

__global__ void print_dict(uint *d_dictionary_code,
                           unsigned char *d_dictionary_codelens);

__global__ void initOffsets(uint fileSize, uint *dfileContent,
                            uint *dbitOffsets,
                            unsigned char *d_dictionary_codelens, uint *counter,
                            uint numTasks);