static double sum_of_time = 0;
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
#define TIMER_START(id)                                                        \
  cudaEvent_t start##id, stop##id;                                             \
  cudaEventCreate(&start##id);                                                 \
  cudaEventCreate(&stop##id);                                                  \
  cudaEventRecord(start##id);

#define TIMER_STOP(id)                                                         \
  cudaEventRecord(stop##id);                                                   \
  cudaEventSynchronize(stop##id);                                              \
  float time##id;                                                              \
  cudaEventElapsedTime(&time##id, start##id, stop##id);                        \
  cudaEventDestroy(start##id);                                                 \
  cudaEventDestroy(stop##id);                                                  \
  sum_of_time += time##id;                                                     \
  printf("%s -> %f\n", #id, time##id);

#define GET_CHAR(value, shift) ((value >> ((shift)*8)) & 0xFF)
#define BLOCK_SIZE 256
#define PER_THREAD_PROC 16
#define BLOCK_NUM 2
#define NUM_THREADS 256

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