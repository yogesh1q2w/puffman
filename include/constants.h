#ifndef CONSTANTS_
#define CONSTANTS_
#define CUERROR                                                                \
  {                                                                            \
    cudaError_t cuError;                                                       \
    cuError = cudaGetLastError();                                              \
    if (cuError != cudaSuccess) {                                              \
      printf("Error: %s : at %s line %d\n", cudaGetErrorString(cuError),       \
             __FILE__, __LINE__);                                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define GET_CHAR(value, shift) ((value >> ((shift)*8)) & 0xFF)
#define BLOCK_SIZE 4096
#define PER_THREAD_PROC 32
#define BLOCK_NUM 8 * 3
#define NUM_THREADS 256
#define INT_BITS 32

#define HIST_THREADS 192
#define WARP_SIZE 32
#define WARP_COUNT (HIST_THREADS / WARP_SIZE)
#define HIST_SIZE 256
#define S_HIST_SIZE (WARP_COUNT * HIST_SIZE)
#define HIST_BLOCK 240

#define FLAG_A 0x0100000000000000
#define FLAG_P 0x8000000000000000
#define FLAG_MASK 0xFF00000000000000

#define MAX_FILE_NAME_SIZE 100
#define MAX_THREADS_TO_USE 65536
#define MAX_THREADS 1024

#define UINT_OUT(symbols, symbol, pos)                                         \
  symbols = (symbols | (symbol << (pos * 8)));

#define THREAD_DIV_WARP NUM_THREADS / WARP_SIZE
typedef unsigned long long int ull;
#endif