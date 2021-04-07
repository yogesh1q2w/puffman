#ifndef TIMER
#define TIMER

#include <time.h>

double get_ms(struct timespec start, struct timespec stop);

#define GPU_TIMER_START(id)                                                    \
  cudaEvent_t start##id, stop##id;                                             \
  cudaEventCreate(&start##id);                                                 \
  cudaEventCreate(&stop##id);                                                  \
  cudaEventRecord(start##id);

#define GPU_TIMER_STOP(id)                                                     \
  cudaEventRecord(stop##id);                                                   \
  cudaEventSynchronize(stop##id);                                              \
  float time##id;                                                              \
  cudaEventElapsedTime(&time##id, start##id, stop##id);                        \
  cudaEventDestroy(start##id);                                                 \
  cudaEventDestroy(stop##id);                                                  \
  printf("Time (%s) : %f [ms]\n", #id, time##id);

#define CPU_TIMER_START(id)                                                    \
  struct timespec start##id, stop##id;                                         \
  clock_gettime(CLOCK_REALTIME, &start##id);

#define CPU_TIMER_STOP(id)                                                     \
  clock_gettime(CLOCK_REALTIME, &stop##id);                                    \
  double time##id = get_ms(start##id, stop##id);                                      \
  printf("Time - cpu (%s) : %f [ms]\n", #id, get_ms(start##id, stop##id));

#endif
