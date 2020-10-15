#include <cstdio>
#define KERNEL_CU
#include "compressKernel.h"
#include "constants.h"

__constant__ unsigned char const_code[256 * 255];
__constant__ unsigned char const_codeSize[256];

inline __device__ void addByte(uint *s_WarpHist, unsigned char data) {
  atomicAdd(s_WarpHist + data, 1);
}

inline __device__ void addWord(uint *s_WarpHist, uint data) {
  addByte(s_WarpHist, (data >> 0) & 0xFFU);
  addByte(s_WarpHist, (data >> 8) & 0xFFU);
  addByte(s_WarpHist, (data >> 16) & 0xFFU);
  addByte(s_WarpHist, (data >> 24) & 0xFFU);
}

__global__ void cu_histgram(uint *d_PartialHistograms, uint *d_Data,
                            uint dataCount, uint byteCount) {

  __shared__ uint s_Hist[S_HIST_SIZE];
  uint *s_WarpHist = s_Hist + (threadIdx.x >> 5) * HIST_SIZE;
  uint warpLane = threadIdx.x & 31;

  for (uint i = warpLane; i < HIST_SIZE; i += WARP_SIZE) {
    s_WarpHist[i] = 0;
  }
  __syncthreads();

  uint pos = 0;
  for (pos = (blockIdx.x * blockDim.x) + threadIdx.x; pos < dataCount - 1;
       pos += (blockDim.x * gridDim.x)) {
    uint data = d_Data[pos];
    addWord(s_WarpHist, data);
  }

  if (pos == dataCount - 1) {
    uint data = d_Data[pos];
    switch (byteCount & 3) {
    case 1:
      addByte(s_WarpHist, (data >> 0) & 0xFFU);
      break;
    case 2:
      addByte(s_WarpHist, (data >> 0) & 0xFFU);
      addByte(s_WarpHist, (data >> 8) & 0xFFU);
      break;
    case 3:
      addByte(s_WarpHist, (data >> 0) & 0xFFU);
      addByte(s_WarpHist, (data >> 8) & 0xFFU);
      addByte(s_WarpHist, (data >> 16) & 0xFFU);
      break;
    default:
      addByte(s_WarpHist, (data >> 0) & 0xFFU);
      addByte(s_WarpHist, (data >> 8) & 0xFFU);
      addByte(s_WarpHist, (data >> 16) & 0xFFU);
      addByte(s_WarpHist, (data >> 24) & 0xFFU);
    }
  }

  __syncthreads();

  //
  for (uint bin = threadIdx.x; bin < HIST_SIZE; bin += HIST_THREADS) {
    uint sum = 0;
    for (uint i = 0; i < WARP_COUNT; i++) {
      sum += s_Hist[bin + i * HIST_SIZE];
    }
    d_PartialHistograms[blockIdx.x * HIST_SIZE + bin] = sum;
  }
}

__global__ void mergeHistogram(uint *d_Histogram, uint *d_PartialHistograms) {

  uint val = d_PartialHistograms[blockIdx.x * HIST_SIZE + threadIdx.x];
  atomicAdd(d_Histogram + threadIdx.x, val);
}

__device__ inline unsigned char getcharAt(uint *dfileContent, uint pos) {
  return (dfileContent[pos >> 2] >> ((pos & 3U) << 3)) & 0xFFU;
}

__global__ void skss_compress_with_shared(uint fileSize, uint *dfileContent,
                                          uint *dbitOffsets,
                                          uint *d_compressedFile,
                                          unsigned char maxCodeSize) {

  extern __shared__ unsigned char sh_dictionary[];
  sh_dictionary[threadIdx.x] = const_codeSize[threadIdx.x];
  for (ushort i = 0; i < maxCodeSize; i++)
    sh_dictionary[((i + 1) << 8) + threadIdx.x] =
        const_code[(i << 8) + threadIdx.x];
  __syncthreads();

  uint id = blockIdx.x * blockDim.x + threadIdx.x;
  uint stepSize = blockDim.x * gridDim.x;

  while (id <= fileSize) {
    if (id < fileSize) {
      uint start = dbitOffsets[id];
      uint end = start + sh_dictionary[getcharAt(dfileContent, id)];
      for (uint j = start; j < end; j++) {
        uint compressedFilePos = j >> 5;
        uint modifyIndex = j & 31U;
        modifyIndex = ((modifyIndex >> 3) << 3) + 7U - (modifyIndex & 7U);
        uint mask = 1 << modifyIndex;
        if (sh_dictionary[getcharAt(dfileContent, id) +
                          ((j - start + 1) << 8)]) {
          atomicOr(&d_compressedFile[compressedFilePos], mask);
        } else {
          atomicAnd(&d_compressedFile[compressedFilePos], ~mask);
        }
      }
    }

    if (id > 0 &&
        dbitOffsets[id - 1] + sh_dictionary[getcharAt(dfileContent, id - 1)] !=
            dbitOffsets[id]) {
      uint start =
          dbitOffsets[id - 1] + sh_dictionary[getcharAt(dfileContent, id - 1)];
      for (uint j = start; j < dbitOffsets[id]; j++) {
        uint compressedFilePos = j >> 5;
        uint modifyIndex = j & 31U;
        modifyIndex = ((modifyIndex >> 3) << 3) + 7U - (modifyIndex & 7U);
        uint mask = 1 << modifyIndex;
        if (sh_dictionary[getcharAt(dfileContent, id) +
                          ((j + 1 - start) << 8)]) {
          atomicOr(&d_compressedFile[compressedFilePos], mask);
        } else {
          atomicAnd(&d_compressedFile[compressedFilePos], ~mask);
        }
      }
    }
    id += stepSize;
  }
}

__global__ void skss_compress(uint fileSize, uint *dfileContent,
                              uint *dbitOffsets, uint *d_compressedFile,
                              unsigned char maxCodeSize) {
  uint id = blockIdx.x * blockDim.x + threadIdx.x;
  uint stepSize = blockDim.x * gridDim.x;

  while (id <= fileSize) {
    if (id < fileSize) {
      uint start = dbitOffsets[id];
      uint end = start + const_codeSize[getcharAt(dfileContent, id)];
      for (uint j = start; j < end; j++) {
        uint compressedFilePos = j >> 5;
        uint modifyIndex = j & 31U;
        modifyIndex = ((modifyIndex >> 3) << 3) + 7U - (modifyIndex & 7U);
        uint mask = 1 << modifyIndex;
        if (const_code[getcharAt(dfileContent, id) + ((j - start) << 8)]) {
          atomicOr(&d_compressedFile[compressedFilePos], mask);
        } else {
          atomicAnd(&d_compressedFile[compressedFilePos], ~mask);
        }
      }
    }

    if (id > 0 &&
        dbitOffsets[id - 1] + const_codeSize[getcharAt(dfileContent, id - 1)] !=
            dbitOffsets[id]) {
      uint start =
          dbitOffsets[id - 1] + const_codeSize[getcharAt(dfileContent, id - 1)];
      for (uint j = start; j < dbitOffsets[id]; j++) {
        uint compressedFilePos = j >> 5;
        uint modifyIndex = j & 31U;
        modifyIndex = ((modifyIndex >> 3) << 3) + 7U - (modifyIndex & 7U);
        uint mask = 1 << modifyIndex;
        if (const_code[getcharAt(dfileContent, id) + ((j - start) << 8)]) {
          atomicOr(&d_compressedFile[compressedFilePos], mask);
        } else {
          atomicAnd(&d_compressedFile[compressedFilePos], ~mask);
        }
      }
    }
    id += stepSize;
  }
}