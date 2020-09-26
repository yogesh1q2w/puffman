#define KERNEL_CU
#define PER_THREAD_PROC 8
#define SEGMENT_SIZE 32
#include "compressKernel.h"

__constant__ codedict d_dictionary;

__global__ void updatefrequency(unsigned int fileSize, char *fileContent, unsigned long long int *frequency) {
    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int index = (id%SEGMENT_SIZE) + (SEGMENT_SIZE * PER_THREAD_PROC * (id/SEGMENT_SIZE));
    for (unsigned int i = 0; i < PER_THREAD_PROC; i++) {
        if(index < fileSize) {
            atomicAdd(&frequency[fileContent[index]],1);
            index += SEGMENT_SIZE;
        }
        else {
            break;
        }
    }
}

__global__ void genBitCompressed(unsigned int lastBlockIndex, char *dfileContent, unsigned int *dbitOffsets, unsigned char *dbitCompressedFile) {
    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int index = (id%SEGMENT_SIZE) + (SEGMENT_SIZE * PER_THREAD_PROC * (id/SEGMENT_SIZE));
    for (unsigned char i = 0; i < PER_THREAD_PROC; i++) {
        if(index <= lastBlockIndex) {

            if(index < lastBlockIndex) {
                for(unsigned char j = 0; j < d_dictionary.codeSize[dfileContent[index]]; j++)
                    dbitCompressedFile[dbitOffsets[index] + j] = d_dictionary.code[dfileContent[index]][j];
            }
            
            if(index > 0 && dbitOffsets[index-1] + d_dictionary.codeSize[dfileContent[index-1]] != dbitOffsets[index]) {
                unsigned int start = dbitOffsets[index-1] + d_dictionary.codeSize[dfileContent[index-1]];
                for(unsigned int j = start; j < dbitOffsets[index]; j++)
                    dbitCompressedFile[j] = d_dictionary.code[dfileContent[index]][j - start];
            }
            index += SEGMENT_SIZE;
        }
        else {
            break;
        }
    }
}

__global__ void encode(unsigned int bitCompressedFileSize, unsigned char *dbitCompressedFile, unsigned char *d_compressedFile) {
    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int index = (id%SEGMENT_SIZE) + (SEGMENT_SIZE * PER_THREAD_PROC * (id/SEGMENT_SIZE));
    for (unsigned int i = 0; i < PER_THREAD_PROC; i++) {
        if(index < bitCompressedFileSize) {
            for(unsigned int j = 0; j < 8; j++) {
                if(dbitCompressedFile[index*8 + j])
                    d_compressedFile[index] = (d_compressedFile[index] << 1) | 1; 
                else
                    d_compressedFile[index] = d_compressedFile[index] << 1; 
            }
            index += SEGMENT_SIZE;
        }
        else {
            break;
        }
    }
}