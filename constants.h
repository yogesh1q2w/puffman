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

#define MAX_THREADS 1024
#define BLOCK_SIZE 256
#define PER_THREAD_PROC 8
#define SEGMENT_SIZE 256
#define BLOCK_NUM 8 * 80

#define HIST_THREADS 192
#define WARP_SIZE 32
#define WARP_COUNT (HIST_THREADS / WARP_SIZE)
#define HIST_SIZE 256
#define S_HIST_SIZE (WARP_COUNT * HIST_SIZE)
#define HIST_BLOCK 240