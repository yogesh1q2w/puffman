#include "huffman.h"
#include <cstdio>
#define KERNEL_CU
#include "compressKernel.h"
#include "constants.h"
//---------------------------------HISTOGRAM-------------------------------------

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

//-----------------------------------------------------------------------------------------------

__device__ inline unsigned char getcharAt(uint *dfileContent, uint pos) {
  return (dfileContent[pos >> 2] >> ((pos & 3U) << 3)) & 0xFFU;
}

__global__ void encode(uint fileSize, uint *dfileContent, uint *dbitOffsets,
                       uint *d_boundary_index, uint *d_compressedFile,
                       uint *d_dictionary_code,
                       unsigned char *d_dictionary_codelens, uint *counter,
                       uint numTasks) {
  uint task_idx = 0;
  uint threadInput_idx = 0;
  uint *threadInput, *threadBoundaryIndex;
  __shared__ struct codedict sh_dictionary;
  __shared__ unsigned int shared_task_idx;

  sh_dictionary.code[threadIdx.x] = d_dictionary_code[threadIdx.x];
  sh_dictionary.codeSize[threadIdx.x] = d_dictionary_codelens[threadIdx.x];

  // if (sh_dictionary.codeSize[threadIdx.x] != 0)
  //   printf("code_len %c= %d\n", threadIdx.x,
  //          sh_dictionary.codeSize[threadIdx.x]);

  if (threadIdx.x == 0) {
    shared_task_idx = atomicAdd(counter, 1);
  }
  __syncthreads();

  task_idx = shared_task_idx;
  threadInput_idx = (task_idx * blockDim.x + threadIdx.x) * PER_THREAD_PROC;

  while (task_idx < numTasks) {
    threadInput = dfileContent + (threadInput_idx / 4);
    threadBoundaryIndex = d_boundary_index + threadInput_idx;
    uint inputPosInThreadTask = 0;
    uint outputPos = (d_boundary_index[threadInput_idx] == 0)
                         ? dbitOffsets[threadInput_idx] / 32
                         : d_boundary_index[threadInput_idx] / 32;
    uint startPosInOutputWord = (d_boundary_index[threadInput_idx] == 0)
                                    ? dbitOffsets[threadInput_idx] % 32
                                    : d_boundary_index[threadInput_idx] % 32;
    uint outputWord = 0;
    uint input = 0;
    uint pendingBitsFromPreviousCode;
    uint remain_code = 0;
    while (inputPosInThreadTask < PER_THREAD_PROC &&
           threadInput_idx + inputPosInThreadTask < fileSize) {
      if ((inputPosInThreadTask & 3) == 0)
        input = threadInput[inputPosInThreadTask / 4];
      uint code = sh_dictionary.code[GET_CHAR(input, inputPosInThreadTask & 3)];
      unsigned char code_length =
          sh_dictionary.codeSize[GET_CHAR(input, inputPosInThreadTask & 3)];
      code >>= (32 - code_length);
      uint boundary_pos = threadBoundaryIndex[inputPosInThreadTask];
      if (boundary_pos != 0) {
        code >>=
            (code_length -
             (BLOCK_SIZE * ((uint)ceil(boundary_pos / (1. * BLOCK_SIZE)))) +
             boundary_pos);
        code_length =
            BLOCK_SIZE * ((uint)ceil(boundary_pos / (1. * BLOCK_SIZE))) -
            boundary_pos;
        threadBoundaryIndex[inputPosInThreadTask] = 0;
        inputPosInThreadTask--;
      }
      if (32 - startPosInOutputWord >= code_length) {
        code <<= (32 - startPosInOutputWord - code_length);
        remain_code = 0;
        pendingBitsFromPreviousCode = 0;
        startPosInOutputWord += code_length;
      } else {
        // printf("It also reaches here :)\n");
        remain_code = code << (32 - code_length + 32 - startPosInOutputWord);
        pendingBitsFromPreviousCode = (code_length - 32 + startPosInOutputWord);
        code >>= pendingBitsFromPreviousCode;
      }
      outputWord |= code;
      if (pendingBitsFromPreviousCode) {
        // printf("writing %u to output\n", outputWord);
        atomicOr(&d_compressedFile[outputPos++], outputWord);
        outputWord = remain_code;
        startPosInOutputWord = pendingBitsFromPreviousCode;
      }
      inputPosInThreadTask++;
    }
    atomicOr(&d_compressedFile[outputPos++], outputWord);
    if (threadIdx.x == 0) {
      shared_task_idx = atomicAdd(counter, 1);
    }
    // return;

    __syncthreads();

    task_idx = shared_task_idx;
    threadInput_idx = (task_idx * blockDim.x + threadIdx.x) * PER_THREAD_PROC;
  }
}

__global__ void print_dict(uint *d_dictionary_code,
                           unsigned char *d_dictionary_codelens) {
  for (int i = 0; i < 256; i++) {
    printf("%c\t|\t%d\t|\t", i, d_dictionary_codelens[i]);
    for (unsigned char j = 0; j < d_dictionary_codelens[i]; j++)
      printf("%d", (1 & (d_dictionary_code[i] >> (31 - j))));
    printf("\n");
  }
}