#define KERNEL_CU
#define PER_THREAD_PROC 8
#define SEGMENT_SIZE 256
#include "compressKernel.h"

__constant__ unsigned char const_code[256 * 255];
__constant__ unsigned char const_codeSize[256];

__global__ void updatefrequency(unsigned int fileSize,
                                unsigned char *fileContent,
                                unsigned long long int *frequency) {
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int index = (id % SEGMENT_SIZE) +
                       (SEGMENT_SIZE * PER_THREAD_PROC * (id / SEGMENT_SIZE));
  for (unsigned int i = 0; i < PER_THREAD_PROC; i++) {
    if (index < fileSize) {
      atomicAdd(&frequency[fileContent[index]], 1);
      index += SEGMENT_SIZE;
    } else {
      break;
    }
  }
}

// __global__ void genBitCompressed(unsigned int lastBlockIndex,
//                                  unsigned char *dfileContent,
//                                  unsigned int *dbitOffsets,
//                                  unsigned char *dbitCompressedFile) {
//   unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
//   unsigned int index = (id % SEGMENT_SIZE) +
//                        (SEGMENT_SIZE * PER_THREAD_PROC * (id /
//                        SEGMENT_SIZE));
//   for (unsigned char i = 0; i < PER_THREAD_PROC; i++) {
//     if (index <= lastBlockIndex) {

//       if (index < lastBlockIndex) {
//         for (unsigned char j = 0; j < const_codeSize[dfileContent[index]];
//         j++)
//           dbitCompressedFile[dbitOffsets[index] + j] =
//               const_code[dfileContent[index]][j];
//       }

//       if (index > 0 &&
//           dbitOffsets[index - 1] + const_codeSize[dfileContent[index - 1]] !=
//               dbitOffsets[index]) {
//         unsigned int start =
//             dbitOffsets[index - 1] + const_codeSize[dfileContent[index - 1]];
//         for (unsigned int j = start; j < dbitOffsets[index]; j++)
//           dbitCompressedFile[j] = const_code[dfileContent[index]][j - start];
//       }
//       index += SEGMENT_SIZE;
//     } else {
//       break;
//     }
//   }
// }

// __global__ void encode(unsigned int bitCompressedFileSize,
//                        unsigned char *dbitCompressedFile,
//                        unsigned char *d_compressedFile) {
//   unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
//   unsigned int index = (id % SEGMENT_SIZE) +
//                        (SEGMENT_SIZE * PER_THREAD_PROC * (id /
//                        SEGMENT_SIZE));
//   for (unsigned int i = 0; i < PER_THREAD_PROC; i++) {
//     if (index < bitCompressedFileSize) {
//       for (unsigned int j = 0; j < 8; j++) {
//         if (dbitCompressedFile[index * 8 + j])
//           d_compressedFile[index] = (d_compressedFile[index] << 1) | 1;
//         else
//           d_compressedFile[index] = d_compressedFile[index] << 1;
//       }
//       index += SEGMENT_SIZE;
//     } else {
//       break;
//     }
//   }
// }

__global__ void skss_compress_with_shared(unsigned int lastBlockIndex,
                                          unsigned char *dfileContent,
                                          unsigned int *dbitOffsets,
                                          unsigned int *d_compressedFile,
                                          unsigned char maxCodeSize) {
  extern __shared__ unsigned char sh_dictionary[];
  memcpy(sh_dictionary, const_codeSize, 256);
  memcpy(&sh_dictionary[256], const_code, maxCodeSize * 256);
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int index = (id % SEGMENT_SIZE) +
                       (SEGMENT_SIZE * PER_THREAD_PROC * (id / SEGMENT_SIZE));
  for (unsigned int i = 0; i < PER_THREAD_PROC; i++) {
    if (index <= lastBlockIndex) {
      if (index < lastBlockIndex) {
        for (unsigned int j = 0; j < sh_dictionary[dfileContent[index]]; j++) {
          unsigned int compressedFilePos =
              (dbitOffsets[index] + j) / (8. * sizeof(unsigned int));
          unsigned int modifyIndex =
              ((dbitOffsets[index] + j) % (8 * sizeof(unsigned int)));
          modifyIndex = 8 * (modifyIndex / 8) + 7 - (modifyIndex % 8);
          unsigned int mask = 1 << modifyIndex;
          if (sh_dictionary[dfileContent[index] * maxCodeSize + j + 256]) {
            atomicOr(&d_compressedFile[compressedFilePos], mask);
          } else {
            atomicAnd(&d_compressedFile[compressedFilePos], ~mask);
          }
        }
      }

      if (index > 0 &&
          dbitOffsets[index - 1] + sh_dictionary[dfileContent[index - 1]] !=
              dbitOffsets[index]) {
        unsigned int start =
            dbitOffsets[index - 1] + sh_dictionary[dfileContent[index - 1]];
        for (unsigned int j = start; j < dbitOffsets[index]; j++) {
          unsigned int compressedFilePos = (j / (8. * sizeof(unsigned int)));
          unsigned int modifyIndex = (j % (8 * sizeof(unsigned int)));
          modifyIndex = 8 * (modifyIndex / 8) + 7 - (modifyIndex % 8);
          unsigned int mask = 1 << modifyIndex;
          if (sh_dictionary[dfileContent[index] * maxCodeSize + j - start +
                            256]) {
            atomicOr(&d_compressedFile[compressedFilePos], mask);
          } else {
            atomicAnd(&d_compressedFile[compressedFilePos], ~mask);
          }
        }
      }
      index += SEGMENT_SIZE;
    } else {
      break;
    }
  }
}

__global__ void skss_compress(unsigned int lastBlockIndex,
                              unsigned char *dfileContent,
                              unsigned int *dbitOffsets,
                              unsigned int *d_compressedFile,
                              unsigned char maxCodeSize) {
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int index = (id % SEGMENT_SIZE) +
                       (SEGMENT_SIZE * PER_THREAD_PROC * (id / SEGMENT_SIZE));
  for (unsigned int i = 0; i < PER_THREAD_PROC; i++) {
    if (index <= lastBlockIndex) {
      if (index < lastBlockIndex) {
        for (unsigned int j = 0; j < const_codeSize[dfileContent[index]]; j++) {
          unsigned int compressedFilePos =
              (dbitOffsets[index] + j) / (8. * sizeof(unsigned int));
          unsigned int modifyIndex =
              ((dbitOffsets[index] + j) % (8 * sizeof(unsigned int)));
          modifyIndex = 8 * (modifyIndex / 8) + 7 - (modifyIndex % 8);
          unsigned int mask = 1 << modifyIndex;
          if (const_code[dfileContent[index] * maxCodeSize + j]) {
            atomicOr(&d_compressedFile[compressedFilePos], mask);
          } else {
            atomicAnd(&d_compressedFile[compressedFilePos], ~mask);
          }
        }
      }

      if (index > 0 &&
          dbitOffsets[index - 1] + const_codeSize[dfileContent[index - 1]] !=
              dbitOffsets[index]) {
        unsigned int start =
            dbitOffsets[index - 1] + const_codeSize[dfileContent[index - 1]];
        for (unsigned int j = start; j < dbitOffsets[index]; j++) {
          unsigned int compressedFilePos = (j / (8. * sizeof(unsigned int)));
          unsigned int modifyIndex = (j % (8 * sizeof(unsigned int)));
          modifyIndex = 8 * (modifyIndex / 8) + 7 - (modifyIndex % 8);
          unsigned int mask = 1 << modifyIndex;
          if (const_code[dfileContent[index] * maxCodeSize + j - start]) {
            atomicOr(&d_compressedFile[compressedFilePos], mask);
          } else {
            atomicAnd(&d_compressedFile[compressedFilePos], ~mask);
          }
        }
      }
      index += SEGMENT_SIZE;
    } else {
      break;
    }
  }
}

__global__ void printDict(codedict &dict) {
  printf("MaxcodeSize = %u", dict.maxCodeSize);
  for (int i = 0; i < 256; i++) {
    if (dict.codeSize[i] > 0) {
      printf("%c-> ", i);
      for (int j = 0; j < dict.codeSize[i]; j++)
        printf("%u", dict.code[i * dict.maxCodeSize + j]);
      printf("\n");
    }
  }
}